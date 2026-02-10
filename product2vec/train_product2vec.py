from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

tqdm.pandas()

import pickle
import random

import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger  # , TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

os.system("ulimit -u unlimited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for product2vec model training"""

    data_name: str
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 4096
    # n_neg: int
    method_name: str = "product2vec"
    max_epoch: int = 500
    n_sample_per_epoch: int | None = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_name(self) -> str:
        """Generate model name from parameters"""
        params = [
            self.data_name,
            self.method_name,
            self.hidden_dim,
            self.learning_rate,
            self.batch_size,
            # self.n_neg,
        ]
        return "_".join(map(str, params))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nrows", default=None)
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--output_data_dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR"),
    )
    parser.add_argument("--context_dir", default=os.environ.get("SM_CHANNEL_CONTEXT"))
    parser.add_argument("--data_name", default="test")
    parser.add_argument("--hidden_dim", default=256)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--batch_size", default=4096)
    parser.add_argument("--n_neg", default=5)
    parser.add_argument("--max_epoch", default=1)
    parser.add_argument("--chunk_size", default=0)
    parser.add_argument("--progress_bar_interval", default=0)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--patience", default=3)
    args = parser.parse_args()
    if args.nrows is not None:
        args.nrows = int(args.nrows)
    args.hidden_dim = int(args.hidden_dim)
    args.learning_rate = float(args.learning_rate)
    args.batch_size = int(args.batch_size)
    args.n_neg = int(args.n_neg)
    args.max_epoch = int(args.max_epoch)
    args.chunk_size = int(args.chunk_size)
    args.progress_bar_interval = int(args.progress_bar_interval)

    return args


def load_dataset(
    filename: str,
    chunk_size: int = 0,
    trial=False,
    neg_size: int = 10,
) -> pd.DataFrame:
    if chunk_size == 0:
        return pd.read_parquet(filename)
    parquet_file = pq.ParquetFile(filename)
    batch_df = []
    # all columns: ['basket_id','user_id','price','item_1','item_2','item_x']
    cols = ["user_id", "item_x", "item_1", "item_2"]
    for itr, batch in tqdm(
        enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=cols)),
    ):
        tmp_batch_df = batch.to_pandas()
        batch_df.append(tmp_batch_df)
        if trial:
            break

    del batch, tmp_batch_df, parquet_file
    return pd.concat(batch_df)


def pad_sequences_torch(
    sequences: list[list] | list[np.ndarray],
    maxlen: int | None = None,
    padding: str = "post",
    truncating: str = "post",
    value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Pads sequences to the same length (similar to Keras pad_sequence)."""
    # Convert all sequences to numpy arrays if they aren't already
    sequences = [
        np.asarray(seq) if not isinstance(seq, np.ndarray) else seq for seq in sequences
    ]

    # Get lengths of each sequence
    lengths = [len(seq) for seq in sequences]

    # Use longest sequence length if maxlen is not specified
    if maxlen is None:
        maxlen = max(lengths)

    # Create output array
    num_samples = len(sequences)
    x = np.full((num_samples, maxlen), value, dtype=dtype)

    for idx, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        # Truncate
        if len(seq) > maxlen:
            if truncating == "pre":
                seq = seq[-maxlen:]
            else:  # truncating == 'post'
                seq = seq[:maxlen]

        # Pad
        trunc = seq.astype(dtype)
        if padding == "post":
            x[idx, : len(trunc)] = trunc
        else:  # padding == 'pre'
            x[idx, -len(trunc) :] = trunc

    return x


def preprocess(data: pd.DataFrame):
    random.seed(0)
    user_num = data.user_id.max()

    validation_user_list = random.sample(
        list(range(1, user_num + 1)),
        round(0.1 * user_num),
    )
    train_user_list = list(
        set(list(range(1, user_num + 1))) - set(validation_user_list),
    )

    train_X = data.query("user_id in @train_user_list").drop(
        ["item_2", "user_id"],
        axis=1,
    )
    train_y = data.query("user_id in @train_user_list")["item_2"].astype("int64")

    validation_X = data.query("user_id in @validation_user_list").drop(
        ["item_2", "user_id"],
        axis=1,
    )
    validation_y = data.query("user_id in @validation_user_list")["item_2"].astype(
        "int64",
    )

    return train_X, train_y, validation_X, validation_y


class Product2VecDataset(Dataset):
    def __init__(self, item_ids, context_ids, negative_sample: list[list[int]]):
        self.item_ids = torch.LongTensor(item_ids)
        self.context_ids = torch.LongTensor(context_ids)
        self.negative_sample = torch.LongTensor(negative_sample)

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return {
            "item": self.item_ids[idx],
            "context": self.context_ids[idx],
            "negative_sample": self.negative_sample[idx],
        }


def calculate_hr_at_k(predictions, ground_truth, k=10):
    hits = 0
    total = len(ground_truth)

    for pred, true in zip(predictions, ground_truth):
        if true in pred[:k]:
            hits += 1

    return hits / total if total > 0 else 0


def calculate_ndcg_at_k(predictions, ground_truth, k=10):

    def dcg_at_k(pred, true, k):
        if true not in pred[:k]:
            return 0
        rank = np.where(pred[:k] == true)[0][0] + 1
        return 1 / np.log2(rank + 1)

    ndcg = 0
    total = len(ground_truth)

    for pred, true in zip(predictions, ground_truth):
        ndcg += dcg_at_k(pred, true, k)

    return ndcg / total if total > 0 else 0


class LightningProduct2Vec(pl.LightningModule):
    def __init__(self, n_items: int, embedding_dim: int, learning_rate: float = 1e-3):
        """Product2Vec model implementation
        Args:
            n_items: Total number of unique products
            embedding_dim: Dimension of the embedding vectors
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.input_embeddings = nn.Embedding(n_items, embedding_dim)
        self.output_embeddings = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings using uniform distribution
        self.input_embeddings.weight.data.uniform_(
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
        )
        self.output_embeddings.weight.data.uniform_(
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
        )

    def forward(
        self,
        target_products: torch.Tensor,
        context_products: torch.Tensor,
        negative_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model
        Args:
            target_products: Tensor of target product indices
            context_products: Tensor of context product indices
            negative_samples: Tensor of negative sample indices
        Returns:
            loss: Binary cross entropy loss
        """
        # Get embeddings
        target_embeds = self.input_embeddings(
            target_products,
        )  # [batch_size, embedding_dim]
        context_embeds = self.output_embeddings(
            context_products,
        )  # [batch_size, embedding_dim]
        neg_embeds = self.output_embeddings(
            negative_samples,
        )  # [batch_size, n_neg, embedding_dim]

        # Compute positive and negative scores
        pos_score = torch.sum(target_embeds * context_embeds, dim=1)  # [batch_size]

        # Reshape context_embeds for batch matrix multiplication
        context_embeds = context_embeds.unsqueeze(2)  # [batch_size, embedding_dim, 1]
        neg_score = torch.bmm(neg_embeds, context_embeds).squeeze(
            -1,
        )  # [batch_size, n_neg]

        # Calculate loss using binary cross entropy
        pos_loss = F.logsigmoid(pos_score)  # [batch_size]
        neg_loss = F.logsigmoid(-neg_score).sum(1)  # [batch_size]

        return -(pos_loss + neg_loss).mean()

    def training_step(self, batch, batch_idx):
        target_products = batch["item"]
        context_products = batch["context"]
        negative_samples = batch["negative_sample"]

        loss = self(target_products, context_products, negative_samples)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.global_step % 2000 == 0:
            logger.info(f"Step {self.global_step} train loss: {loss:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        target_products = batch["item"]
        context_products = batch["context"]
        negative_samples = batch["negative_sample"]

        loss = self(target_products, context_products, negative_samples)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        logger.info(f"Epoch {self.current_epoch} valid loss: {loss:.4f}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def extract_embedding(self):
        w_input = self.input_embeddings.weight.to("cpu").detach().numpy().copy()
        w_output = self.output_embeddings.weight.to("cpu").detach().numpy().copy()
        return {
            "input_emb": w_input,
            "output_emb": w_output,
        }

    def evaluate_recommendations(self, val_loader):
        self.eval()
        all_predictions = []
        all_ground_truth = []
        all_negative_ground_truth = []
        with torch.no_grad():
            for batch in val_loader:
                target_products = batch["item"]
                context_products = batch["context"]
                negative_target_products = context_products[0:1]

                target_embeds = self.input_embeddings(target_products)
                negative_target_embeds = self.input_embeddings(negative_target_products)
                all_embeds = self.output_embeddings.weight
                scores = torch.matmul(target_embeds, all_embeds.t())

                _, top_k_indices = torch.topk(scores, k=10, dim=1)

                predictions = top_k_indices.cpu().numpy()
                ground_truth = target_products.cpu().numpy()
                negative_ground_truth = negative_target_products.cpu().numpy()

                all_predictions.extend(predictions)
                all_ground_truth.extend(ground_truth)
                all_negative_ground_truth.extend(negative_ground_truth)

        hr_at_10 = calculate_hr_at_k(all_predictions, all_ground_truth, k=10)
        ndcg_at_10 = calculate_ndcg_at_k(all_predictions, all_ground_truth, k=10)
        hr_at_10_negative = calculate_hr_at_k(
            all_predictions,
            all_negative_ground_truth,
            k=10,
        )
        ndcg_at_10_negative = calculate_ndcg_at_k(
            all_predictions,
            all_negative_ground_truth,
            k=10,
        )

        return hr_at_10, ndcg_at_10, hr_at_10_negative, ndcg_at_10_negative


def setup_training(
    model,
    train_loader,
    val_loader,
    max_epochs=10,
    patience=3,
    min_delta=1e-4,
    refresh_rate=1000,
    model_name="test",
):
    os.makedirs(os.environ.get("SM_MODEL_DIR"), exist_ok=True)

    checkpoint_path = "/opt/ml/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.environ.get("SM_MODEL_DIR"),
        filename="product2vec-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )
    # logger
    logger_csv = CSVLogger(save_dir=checkpoint_path, name="ln_log_csv")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            TQDMProgressBar(refresh_rate=refresh_rate),
        ],
        accelerator="auto",  # GPU if available, else CPU
        devices=1,
        logger=[logger_csv],
        log_every_n_steps=refresh_rate,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    return trainer, model


def main():
    args = parse_args()
    parquet_filename = os.path.join(args.context_dir, "proc_data_subst2vec.parquet")
    logger.info(f"Loading training data...{parquet_filename}")

    data = load_dataset(
        parquet_filename,
        chunk_size=args.chunk_size,
        trial=(args.data_name == "test"),
        neg_size=int(args.n_neg),
    )
    logger.info("split training data...")
    train_X, train_y, validation_X, validation_y = preprocess(data)

    print("memory: ", psutil.virtual_memory())
    config = ModelConfig(
        data_name=args.data_name,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        # n_neg=args.n_neg,
        max_epoch=args.max_epoch,
    )
    train_X["item_x"] = train_X["item_x"].apply(lambda x: x[: args.n_neg])
    train_negatives = pad_sequences_torch(
        train_X["item_x"],
        maxlen=20,
        padding="post",
        value=0,
    )
    del train_X["item_x"]
    validation_negatives = pad_sequences_torch(
        validation_X["item_x"],
        maxlen=20,
        padding="post",
        value=0,
    )
    del validation_X["item_x"]

    logger.info("make dataloader...")
    train_dataset = Product2VecDataset(
        item_ids=train_y.values,
        context_ids=train_X["item_1"].astype("int64").values,
        negative_sample=train_negatives,
    )
    del train_X, train_y

    val_dataset = Product2VecDataset(
        item_ids=validation_y.values,
        context_ids=validation_X["item_1"].astype("int64").values,
        negative_sample=validation_negatives,
    )
    s_user = 2500
    s_item = 1000
    print(s_user, s_item)
    del data
    del validation_X, validation_y
    print("memory: ", psutil.virtual_memory())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=8)

    del train_dataset, val_dataset
    model = LightningProduct2Vec(
        n_items=s_item,
        embedding_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
    )

    logger.info("start training...")
    print("memory: ", psutil.virtual_memory())

    trainer, trained_model = setup_training(
        model,
        train_loader,
        val_loader,
        max_epochs=config.max_epoch,
        patience=int(args.patience),
        refresh_rate=args.progress_bar_interval,
        model_name=config.method_name,
    )
    (
        final_hr_at_10,
        final_ndcg_at_10,
        final_hr_at_10_negative,
        final_ndcg_at_10_negative,
    ) = trained_model.evaluate_recommendations(
        val_loader,
    )
    logger.info(f"Final HR@10 = {(final_hr_at_10 * (1 - final_hr_at_10_negative)):.4f}")
    logger.info(
        f"Final NDCG@10 = {(final_ndcg_at_10 * (1 - final_ndcg_at_10_negative)):.4f}",
    )

    if args.save_model:
        torch.save(
            trained_model.state_dict(),
            f"{args.model_dir}/product2vec_model.pth",
        )
        emb_dict = trained_model.extract_embedding()

        # metadata about the embeddings
        metadata = {
            "n_users": int(s_user),
            "n_items": int(s_item),
            "embedding_dim": str(config.hidden_dim),
            "model_name": config.model_name,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        item_emb_path = os.path.join(
            os.environ.get("SM_OUTPUT_DATA_DIR"),
            "embeddings_dict.pickle.gz",
        )
        with open(item_emb_path, "wb") as f:
            pickle.dump(emb_dict, f)
        logger.info(f"Saved item embeddings to {item_emb_path}")

        metadata_path = os.path.join(
            os.environ.get("SM_OUTPUT_DATA_DIR"),
            "metadata.json",
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    logger.info("=== Starting Training ===")
    main()
    logger.info("=== Training Complete ===")
