"""Basket-MLM with Purchase History Context
Example:
python shopping_embedding_clean.py --model_dir ./results/ --output_data_dir ./results/ --context_dir ./data/ --batch_size 256 --vocab_size 2501
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class BasketMLMConfig:
    """Configuration for Basket-MLM model"""

    # Data parameters
    vocab_size: int = 2501
    store_vocab_size: int = 15001
    time_vocab_size: int = 29

    # Context parameters
    max_history_length: int = 10
    max_context_items: int = 20

    # Model architecture
    hidden_size: int = 64
    num_attention_heads: int = 2
    num_hidden_layers: int = 2
    intermediate_size: int = 256
    max_position_embeddings: int = 128
    dropout_prob: float = 0.1

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epoch: int = 100
    mask_prob: float = 0.15


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Basket-MLM training")

    parser.add_argument("--model_dir", default="./models", help="Model save directory")
    parser.add_argument(
        "--output_data_dir", default="./output", help="Output directory"
    )
    parser.add_argument("--context_dir", default="./data", help="Input data directory")
    parser.add_argument("--context_fname", default="data.parquet", help="Data filename")

    # Model parameters
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--vocab_size", default=2501, type=int)
    parser.add_argument("--store_vocab_size", default=15001, type=int)
    parser.add_argument("--time_vocab_size", default=29, type=int)
    parser.add_argument("--max_history_length", default=10, type=int)
    parser.add_argument("--max_context_items", default=20, type=int)

    # Training parameters
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--mask_prob", default=0.15, type=float)
    parser.add_argument("--patience", default=5, type=int)

    return parser.parse_args()


def load_basket_dataset(filename: str) -> pd.DataFrame:
    """Load basket dataset from parquet file"""
    df = pd.read_parquet(filename)

    # Map column names if needed
    column_mapping = {
        "basket_id": "basket_id",
        "user_id": "user_id",
        "store_id": "store_id",
        "seasonality": "time_bucket",
        "item_id_list": "items",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    return df[["user_id", "basket_id", "store_id", "time_bucket", "items"]].copy()


def preprocess_basket_data(data: pd.DataFrame, val_ratio: float = 0.1):
    """Split data into train and validation sets"""
    random.seed(42)

    data = data.sort_values(["user_id", "basket_id"]).reset_index(drop=True)

    unique_users = data["user_id"].unique()
    n_val_users = int(len(unique_users) * val_ratio)
    val_users = random.sample(list(unique_users), n_val_users)
    train_users = list(set(unique_users) - set(val_users))

    train_data = data[data["user_id"].isin(train_users)].reset_index(drop=True)
    val_data = data[data["user_id"].isin(val_users)].reset_index(drop=True)

    return train_data, val_data


class BasketDataset(Dataset):
    """Dataset for basket-level MLM learning with purchase history"""

    def __init__(self, data: pd.DataFrame, config: BasketMLMConfig):
        self.data = data
        self.config = config
        self.special_tokens = {
            "CLS": config.vocab_size,
            "MASK": config.vocab_size + 1,
            "PAD": config.vocab_size + 2,
            "SEP": config.vocab_size + 3,
        }

        # Group baskets by user
        self.user_baskets = {}
        for user_id, group in data.groupby("user_id"):
            self.user_baskets[user_id] = group.sort_values("basket_id").to_dict(
                "records"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basket_row = self.data.iloc[idx]
        user_id = basket_row["user_id"]
        current_basket_id = basket_row["basket_id"]
        current_basket = basket_row.to_dict()

        # Get user's history before current basket
        user_history = self.user_baskets.get(user_id, [])
        history_baskets = [
            b for b in user_history if b["basket_id"] < current_basket_id
        ]

        # Limit history length
        if len(history_baskets) > self.config.max_history_length:
            history_baskets = history_baskets[-self.config.max_history_length :]

        # Create masked sequence
        masked_sequence, original_items = self._create_masked_sequence(
            history_baskets,
            current_basket,
        )

        return {
            "masked_sequence": masked_sequence,
            "original_items": original_items,
            "basket_id": current_basket_id,
            "user_id": user_id,
        }

    def _normalize_items(self, items):
        """Normalize item IDs to list format"""
        if not isinstance(items, list):
            items = [items]

        normalized = []
        for item in items:
            if isinstance(item, np.ndarray):
                normalized.extend(item.tolist() if item.ndim > 0 else [item.item()])
            elif isinstance(item, list):
                normalized.extend(item)
            else:
                try:
                    item_id = int(item)
                    if 0 <= item_id < self.config.vocab_size:
                        normalized.append(item_id)
                except (ValueError, TypeError):
                    continue

        return normalized

    def _create_masked_sequence(self, history_baskets, current_basket):
        """Create masked sequence with history context"""
        all_tokens = []
        all_store_tokens = []
        all_time_tokens = []
        original_items = []

        # Add CLS token
        all_tokens.append(self.special_tokens["CLS"])
        all_store_tokens.append(0)
        all_time_tokens.append(0)
        original_items.append(-100)

        # Add history baskets (not masked)
        for hist_basket in history_baskets:
            hist_items = self._normalize_items(hist_basket["items"])
            hist_items = hist_items[: self.config.max_context_items]

            hist_store_id = int(hist_basket["store_id"]) % self.config.store_vocab_size
            hist_time_bucket = (
                int(hist_basket["time_bucket"]) % self.config.time_vocab_size
            )

            for item in hist_items:
                if item < self.config.vocab_size:
                    all_tokens.append(item)
                    all_store_tokens.append(hist_store_id)
                    all_time_tokens.append(hist_time_bucket)
                    original_items.append(-100)

            # Add separator
            all_tokens.append(self.special_tokens["SEP"])
            all_store_tokens.append(hist_store_id)
            all_time_tokens.append(hist_time_bucket)
            original_items.append(-100)

        # Add current basket (masked)
        current_items = self._normalize_items(current_basket["items"])
        current_store_id = (
            int(current_basket["store_id"]) % self.config.store_vocab_size
        )
        current_time_bucket = (
            int(current_basket["time_bucket"]) % self.config.time_vocab_size
        )

        for item in current_items:
            if item < self.config.vocab_size:
                if random.random() < self.config.mask_prob:
                    all_tokens.append(self.special_tokens["MASK"])
                    original_items.append(item)
                else:
                    all_tokens.append(item)
                    original_items.append(-100)

                all_store_tokens.append(current_store_id)
                all_time_tokens.append(current_time_bucket)

        masked_sequence = {
            "tokens": all_tokens,
            "store_tokens": all_store_tokens,
            "time_tokens": all_time_tokens,
        }

        return masked_sequence, original_items


def collate_basket_fn(batch, config):
    """Collate function for DataLoader"""
    max_len = min(
        max(len(item["masked_sequence"]["tokens"]) for item in batch),
        config.max_position_embeddings,
    )

    def pad_sequence(sequences, max_len, pad_value=0):
        """Pad sequences to max length"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [pad_value] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded.append(seq)
        return torch.tensor(padded, dtype=torch.long)

    pad_token_id = config.vocab_size + 2

    tokens_list = [item["masked_sequence"]["tokens"] for item in batch]
    store_list = [item["masked_sequence"]["store_tokens"] for item in batch]
    time_list = [item["masked_sequence"]["time_tokens"] for item in batch]
    original_list = [item["original_items"] for item in batch]

    return {
        "masked_sequence": {
            "tokens": pad_sequence(tokens_list, max_len, pad_token_id),
            "store_tokens": pad_sequence(store_list, max_len),
            "time_tokens": pad_sequence(time_list, max_len),
        },
        "original_tokens": pad_sequence(original_list, max_len, -100),
        "basket_ids": [item["basket_id"] for item in batch],
        "user_ids": [item["user_id"] for item in batch],
    }


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""

    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x):
        """Reshape tensor for multi-head attention"""
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """Forward pass"""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TransformerLayer(nn.Module):
    """Single Transformer encoder layer"""

    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        """Forward pass"""
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + self.dropout(attention_output))

        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        hidden_states = self.layernorm2(hidden_states + self.dropout(layer_output))

        return hidden_states


class BasketEncoder(nn.Module):
    """Transformer-based basket encoder"""

    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size + 4, config.hidden_size)
        self.store_embeddings = nn.Embedding(
            config.store_vocab_size, config.hidden_size
        )
        self.time_embeddings = nn.Embedding(config.time_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, tokens, store_tokens, time_tokens, attention_mask=None):
        """Forward pass"""
        seq_length = tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        # Compute embeddings
        token_embeds = self.token_embeddings(tokens)
        store_embeds = self.store_embeddings(store_tokens)
        time_embeds = self.time_embeddings(time_tokens)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + store_embeds + time_embeds + position_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create attention mask for padding
        if attention_mask is None:
            pad_token_id = self.config.vocab_size + 2
            attention_mask = (tokens != pad_token_id).float()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class LightningBasketMLM(pl.LightningModule):
    """PyTorch Lightning wrapper for Basket-MLM model"""

    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.encoder = BasketEncoder(config)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(self, masked_sequence, original_tokens=None):
        """Forward pass"""
        hidden = self.encoder(
            masked_sequence["tokens"],
            masked_sequence["store_tokens"],
            masked_sequence["time_tokens"],
        )

        mlm_logits = self.mlm_head(hidden)

        if original_tokens is not None:
            return {"mlm_logits": mlm_logits, "original_tokens": original_tokens}
        return {"mlm_logits": mlm_logits}

    def compute_mlm_loss(self, mlm_logits, original_tokens):
        """Compute MLM loss for masked tokens only"""
        batch_size, seq_len, vocab_size = mlm_logits.shape

        active_loss = original_tokens.view(-1) != -100
        active_indices = torch.nonzero(active_loss, as_tuple=False).squeeze(-1)

        if len(active_indices) == 0:
            return torch.tensor(0.0, device=mlm_logits.device, requires_grad=True)

        active_logits = mlm_logits.view(-1, vocab_size)[active_indices]
        active_labels = original_tokens.view(-1)[active_indices]

        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(active_logits, active_labels)

    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self.forward(batch["masked_sequence"], batch["original_tokens"])
        loss = self.compute_mlm_loss(outputs["mlm_logits"], outputs["original_tokens"])

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self.forward(batch["masked_sequence"], batch["original_tokens"])
        loss = self.compute_mlm_loss(outputs["mlm_logits"], outputs["original_tokens"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epoch,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def extract_basket_embedding(self, tokens, store_tokens, time_tokens):
        """Extract basket embeddings for inference"""
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(tokens, store_tokens, time_tokens)
            basket_embedding = hidden[:, 0, :]
        return basket_embedding


def train_model(config, train_loader, val_loader, checkpoint_path):
    """Train the model"""
    model = LightningBasketMLM(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename="basket_mlm-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer, model


def main():
    """Main training function"""
    args = parse_args()

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # Load and preprocess data
    data_path = os.path.join(args.context_dir, args.context_fname)
    print(f"Loading data from {data_path}")
    data = load_basket_dataset(data_path)

    print("Splitting data into train/val sets")
    train_data, val_data = preprocess_basket_data(data)

    # Create config
    config = BasketMLMConfig(
        vocab_size=args.vocab_size,
        store_vocab_size=args.store_vocab_size,
        time_vocab_size=args.time_vocab_size,
        max_history_length=args.max_history_length,
        max_context_items=args.max_context_items,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        mask_prob=args.mask_prob,
    )

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders")
    train_dataset = BasketDataset(train_data, config)
    val_dataset = BasketDataset(val_data, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: collate_basket_fn(x, config),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: collate_basket_fn(x, config),
    )

    # Train model
    print("Starting training")
    trainer, model = train_model(config, train_loader, val_loader, args.model_dir)

    # Save model
    print("Saving model")
    model_path = os.path.join(args.model_dir, "basket_mlm_model.pth")
    torch.save(model.state_dict(), model_path)

    # Extract sample embeddings
    print("Extracting sample embeddings")
    model.eval()
    sample_embeddings = []
    sample_basket_ids = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:
                break

            device = next(model.parameters()).device
            tokens = batch["masked_sequence"]["tokens"].to(device)
            store_tokens = batch["masked_sequence"]["store_tokens"].to(device)
            time_tokens = batch["masked_sequence"]["time_tokens"].to(device)

            embeddings = model.extract_basket_embedding(
                tokens, store_tokens, time_tokens
            )
            sample_embeddings.append(embeddings.cpu().numpy())
            sample_basket_ids.extend(batch["basket_ids"])

    # Save embeddings
    embeddings_dict = {
        "embeddings": np.vstack(sample_embeddings),
        "basket_ids": sample_basket_ids,
        "config": vars(config),
    }

    embeddings_path = os.path.join(
        args.output_data_dir, "sample_basket_embeddings.pickle"
    )
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    # Save metadata
    metadata = {
        "vocab_size": config.vocab_size,
        "store_vocab_size": config.store_vocab_size,
        "time_vocab_size": config.time_vocab_size,
        "hidden_size": config.hidden_size,
    }

    metadata_path = os.path.join(args.output_data_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
