"""Example:
python inference_shopping_embedding.py \
        --model_path /path/to/basket_mlm_model.pth \
        --input_data /path/to/transaction_data.parquet \
        --output_path /path/to/embeddings.parquet \
        --vocab_size 2501 \
        --max_history_length 10 \
        --max_context_items 20
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BasketMLMConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 2501)
        self.store_vocab_size = kwargs.get("store_vocab_size", 15001)
        self.time_vocab_size = kwargs.get("time_vocab_size", 29)
        self.max_history_length = kwargs.get("max_history_length", 10)
        self.max_context_items = kwargs.get("max_context_items", 20)
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_attention_heads = kwargs.get("num_attention_heads", 2)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 2)
        self.intermediate_size = kwargs.get("intermediate_size", 256)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 128)
        self.dropout_prob = kwargs.get("dropout_prob", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

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
    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + self.dropout(attention_output))

        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        hidden_states = self.layernorm2(hidden_states + self.dropout(layer_output))

        return hidden_states


class BasketEncoder(nn.Module):
    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.config = config

        # +4は特殊トークン（CLS, MASK, PAD, SEP）用
        self.token_embeddings = nn.Embedding(config.vocab_size + 4, config.hidden_size)
        self.store_embeddings = nn.Embedding(
            config.store_vocab_size,
            config.hidden_size,
        )
        self.time_embeddings = nn.Embedding(config.time_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)],
        )
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        tokens: torch.Tensor,
        store_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        token_embeds = self.token_embeddings(tokens)
        store_embeds = self.store_embeddings(store_tokens)
        time_embeds = self.time_embeddings(time_tokens)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + store_embeds + time_embeds + position_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is None:
            pad_token_id = self.config.vocab_size + 2
            attention_mask = (tokens != pad_token_id).float()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class BasketMLMModel(nn.Module):
    def __init__(self, config: BasketMLMConfig):
        super().__init__()
        self.config = config
        self.encoder = BasketEncoder(config)

        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

        self.special_tokens = {
            "CLS": config.vocab_size,
            "MASK": config.vocab_size + 1,
            "PAD": config.vocab_size + 2,
            "SEP": config.vocab_size + 3,
        }

    def extract_basket_embedding(
        self,
        tokens: torch.Tensor,
        store_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """バスケット埋め込みを抽出

        CLSトークン（履歴コンテキスト）と対象取引の埋め込みを結合して返す
        """
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(tokens, store_tokens, time_tokens)
            cls_embedding = hidden[:, 0, :]

            batch_size = tokens.shape[0]
            sep_token_id = self.special_tokens["SEP"]
            pad_token_id = self.special_tokens["PAD"]
            cls_token_id = self.special_tokens["CLS"]

            basket_embeddings = []

            for i in range(batch_size):
                valid_mask = tokens[i] != pad_token_id
                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    combined_emb = torch.cat(
                        [cls_embedding[i], cls_embedding[i]],
                        dim=0,
                    )
                    basket_embeddings.append(combined_emb)
                    continue

                sep_positions = torch.where(tokens[i] == sep_token_id)[0]

                if len(sep_positions) > 0:
                    target_start_idx = sep_positions[-1].item() + 1
                else:
                    target_start_idx = 1

                seq_len = tokens.shape[1]
                pos_indices = torch.arange(seq_len, device=tokens.device)
                target_mask = (
                    (pos_indices >= target_start_idx)
                    & valid_mask
                    & (tokens[i] != cls_token_id)
                    & (tokens[i] != sep_token_id)
                )
                target_indices = torch.where(target_mask)[0]

                if len(target_indices) > 0:
                    target_basket_emb = hidden[i, target_indices, :].mean(dim=0)
                    combined_emb = torch.cat(
                        [cls_embedding[i], target_basket_emb],
                        dim=0,
                    )
                    basket_embeddings.append(combined_emb)
                else:
                    combined_emb = torch.cat(
                        [cls_embedding[i], cls_embedding[i]],
                        dim=0,
                    )
                    basket_embeddings.append(combined_emb)

            basket_embedding = torch.stack(basket_embeddings, dim=0)

        return basket_embedding

    def predict_masked_items(
        self,
        tokens: torch.Tensor,
        store_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(tokens, store_tokens, time_tokens)
            mlm_logits = self.mlm_head(hidden)
        return mlm_logits


class TransactionDatasetWithHistory(Dataset):
    def __init__(self, data: pd.DataFrame, config: BasketMLMConfig):
        self.data = data
        self.config = config
        self.special_tokens = {
            "CLS": config.vocab_size,
            "MASK": config.vocab_size + 1,
            "PAD": config.vocab_size + 2,
            "SEP": config.vocab_size + 3,
        }

        self._preprocess_data()
        self._group_user_baskets()

    def _preprocess_data(self):
        required_columns = [
            "user_id",
            "basket_id",
            "items",
            "store_id",
            "time_bucket",
        ]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        normalized_items_list = []
        for idx in range(len(self.data)):
            items = self.data.iloc[idx]["items"]
            if not isinstance(items, list):
                items = [items]

            items = [x.tolist() if isinstance(x, np.ndarray) else x for x in items]
            items = [
                item
                for sublist in items
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]

            normalized_items = []
            for item in items:
                try:
                    item_id = int(item)
                    if 0 <= item_id < self.config.vocab_size:
                        normalized_items.append(item_id)
                except (ValueError, TypeError):
                    continue

            normalized_items_list.append(normalized_items)

        self.data["items"] = normalized_items_list

    def _group_user_baskets(self):
        self.user_baskets = {}
        unique_users = self.data["user_id"].unique()

        for user_id in unique_users:
            user_data = self.data[self.data["user_id"] == user_id]
            sorted_group = user_data.sort_values("basket_id").to_dict("records")
            self.user_baskets[user_id] = sorted_group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basket_row = self.data.iloc[idx]
        current_basket = basket_row.to_dict()
        user_id = current_basket["user_id"]
        current_basket_id = current_basket["basket_id"]

        user_history = self.user_baskets.get(user_id, [])
        history_baskets = [
            basket for basket in user_history if basket["basket_id"] < current_basket_id
        ]

        if len(history_baskets) > self.config.max_history_length:
            history_baskets = history_baskets[-self.config.max_history_length :]

        masked_sequence = self._create_sequence_with_history(
            history_baskets,
            current_basket,
        )

        return {
            "masked_sequence": masked_sequence,
            "basket_id": current_basket_id,
            "user_id": user_id,
            "num_history_baskets": len(history_baskets),
        }

    def _create_sequence_with_history(
        self,
        history_baskets: list,
        current_basket: dict,
    ) -> dict:
        all_tokens = []
        all_store_tokens = []
        all_time_tokens = []

        # add CLS token
        all_tokens.append(self.special_tokens["CLS"])
        all_store_tokens.append(0)
        all_time_tokens.append(0)

        # add history baskets
        for hist_basket in history_baskets:
            hist_items = hist_basket["items"][: self.config.max_context_items]
            hist_store_id = int(hist_basket["store_id"]) % self.config.store_vocab_size
            hist_time_bucket = (
                int(hist_basket["time_bucket"]) % self.config.time_vocab_size
            )

            for item in hist_items:
                if item < self.config.vocab_size:
                    all_tokens.append(item)
                    all_store_tokens.append(hist_store_id)
                    all_time_tokens.append(hist_time_bucket)

            all_tokens.append(self.special_tokens["SEP"])
            all_store_tokens.append(hist_store_id)
            all_time_tokens.append(hist_time_bucket)

        # add current basket
        current_items = current_basket["items"]
        current_store_id = (
            int(current_basket["store_id"]) % self.config.store_vocab_size
        )
        current_time_bucket = (
            int(current_basket["time_bucket"]) % self.config.time_vocab_size
        )

        for item in current_items:
            if item < self.config.vocab_size:
                all_tokens.append(item)
                all_store_tokens.append(current_store_id)
                all_time_tokens.append(current_time_bucket)

        return {
            "tokens": all_tokens,
            "store_tokens": all_store_tokens,
            "time_tokens": all_time_tokens,
        }


def collate_inference_fn_with_history(
    batch: list[dict],
    config: BasketMLMConfig,
) -> dict:
    max_len = min(
        max(len(item["masked_sequence"]["tokens"]) for item in batch),
        config.max_position_embeddings,
    )

    def pad_sequence(
        sequences: list[list[int]],
        max_len: int,
        pad_value: int = 0,
    ) -> torch.Tensor:
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [pad_value] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded.append(seq)
        return torch.tensor(padded, dtype=torch.long)

    tokens_list = [item["masked_sequence"]["tokens"] for item in batch]
    store_list = [item["masked_sequence"]["store_tokens"] for item in batch]
    time_list = [item["masked_sequence"]["time_tokens"] for item in batch]
    basket_ids_list = [item["basket_id"] for item in batch]
    user_ids_list = [item["user_id"] for item in batch]
    num_history_list = [item["num_history_baskets"] for item in batch]

    pad_token_id = config.vocab_size + 2
    masked_tokens = pad_sequence(tokens_list, max_len, pad_value=pad_token_id)
    masked_store = pad_sequence(store_list, max_len)
    masked_time = pad_sequence(time_list, max_len)

    return {
        "masked_sequence": {
            "tokens": masked_tokens,
            "store_tokens": masked_store,
            "time_tokens": masked_time,
        },
        "basket_ids": basket_ids_list,
        "user_ids": user_ids_list,
        "num_history_baskets": num_history_list,
    }


class BasketMLMInference:
    def __init__(self, model_path: str, config: BasketMLMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BasketMLMModel(self.config).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def preprocess_transaction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "seasonality": "time_bucket",
            "item_id_list": "items",
            "item_ids": "items",
            "item_list": "items",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and new_col not in data.columns:
                data = data.rename(columns={old_col: new_col})

        required_columns = [
            "user_id",
            "basket_id",
            "items",
            "store_id",
            "time_bucket",
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data["user_id"] = (
            pd.to_numeric(data["user_id"], errors="coerce").fillna(0).astype(int)
        )
        data["store_id"] = (
            pd.to_numeric(data["store_id"], errors="coerce").fillna(0).astype(int)
        )
        data["time_bucket"] = (
            pd.to_numeric(data["time_bucket"], errors="coerce").fillna(0).astype(int)
        )

        return data

    def extract_features(
        self,
        data: str | pd.DataFrame,
        batch_size: int = 256,
        output_type: str = "embedding",
    ) -> pd.DataFrame:
        if isinstance(data, str):
            if data.endswith(".parquet"):
                df = pd.read_parquet(data)
            elif data.endswith(".csv"):
                df = pd.read_csv(data)
            else:
                raise ValueError("Unsupported file format. Use .parquet or .csv")
        else:
            df = data.copy()

        df = self.preprocess_transaction_data(df)

        dataset = TransactionDatasetWithHistory(df, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_inference_fn_with_history(x, self.config),
        )

        all_embeddings = []
        all_basket_ids = []
        all_user_ids = []
        all_num_history = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                tokens = batch["masked_sequence"]["tokens"].to(self.device)
                store_tokens = batch["masked_sequence"]["store_tokens"].to(self.device)
                time_tokens = batch["masked_sequence"]["time_tokens"].to(self.device)

                if output_type == "mlm_prediction":
                    mlm_logits = self.model.predict_masked_items(
                        tokens,
                        store_tokens,
                        time_tokens,
                    )
                    embeddings = F.softmax(mlm_logits, dim=-1)
                    embeddings = embeddings[:, 0, :]
                else:
                    embeddings = self.model.extract_basket_embedding(
                        tokens,
                        store_tokens,
                        time_tokens,
                    )

                all_embeddings.append(embeddings.cpu().numpy())
                all_basket_ids.extend(batch["basket_ids"])
                all_user_ids.extend(batch["user_ids"])
                all_num_history.extend(batch["num_history_baskets"])

        embeddings_array = np.vstack(all_embeddings)

        if output_type == "mlm_prediction":
            embedding_columns = [
                f"item_prob_{i}" for i in range(embeddings_array.shape[1])
            ]
        else:
            embedding_columns = [
                f"embedding_{i}" for i in range(embeddings_array.shape[1])
            ]

        result_df = pd.DataFrame(embeddings_array, columns=embedding_columns)
        result_df["basket_id"] = all_basket_ids
        result_df["user_id"] = all_user_ids
        result_df["num_history_baskets"] = all_num_history

        return result_df

    def save_features(self, features_df: pd.DataFrame, output_path: str):
        if output_path.endswith(".parquet"):
            features_df.to_parquet(output_path, index=False)
        elif output_path.endswith(".csv"):
            features_df.to_csv(output_path, index=False)
        elif output_path.endswith(".pickle"):
            features_df.to_pickle(output_path)
        else:
            msg = "Unsupported output format. Use .parquet, .csv, or .pickle"
            raise ValueError(msg)


def load_basket_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    column_mapping = {
        "seasonality": "time_bucket",
        "item_id_list": "items",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    return df[["user_id", "basket_id", "store_id", "time_bucket", "items"]].copy()


def main():
    parser = argparse.ArgumentParser(description="Basket-MLM Feature Extraction")

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2501,
        help="Vocabulary size for items",
    )
    parser.add_argument(
        "--store_vocab_size",
        type=int,
        default=15001,
        help="Vocabulary size for stores",
    )
    parser.add_argument(
        "--time_vocab_size",
        type=int,
        default=29,
        help="Vocabulary size for time buckets",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=2,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=256,
        help="Intermediate size of the model",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=128,
        help="Maximum position embeddings",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--input_data",
        required=True,
        help="Path to input transaction data",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save extracted features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output_type",
        choices=["embedding", "mlm_prediction"],
        default="embedding",
        help="Type of features to extract",
    )
    parser.add_argument(
        "--max_history_length",
        type=int,
        default=10,
        help="Maximum number of history baskets to use",
    )
    parser.add_argument(
        "--max_context_items",
        type=int,
        default=20,
        help="Maximum number of items per history basket",
    )

    args = parser.parse_args()

    input_data = load_basket_dataset(args.input_data)

    config = BasketMLMConfig(
        vocab_size=args.vocab_size,
        store_vocab_size=args.store_vocab_size,
        time_vocab_size=args.time_vocab_size,
        max_history_length=args.max_history_length,
        max_context_items=args.max_context_items,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        dropout_prob=args.dropout_prob,
    )

    model_path = os.path.join(args.model_dir, "basket_mlm_model.pth")
    inference = BasketMLMInference(model_path, config)

    features_df = inference.extract_features(
        input_data,
        batch_size=args.batch_size,
        output_type=args.output_type,
    )

    if not args.output_path.endswith((".parquet", ".csv", ".pickle")):
        args.output_path = os.path.join(args.output_path, "features.parquet")

    inference.save_features(features_df, args.output_path)


if __name__ == "__main__":
    main()
