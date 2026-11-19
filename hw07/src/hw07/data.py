"""Banking77 text embeddings and metadata for Assignment 7 SAE experiments.

This keeps the Assignment 5 OpenVINO embedding pipeline while preserving the
original text and intent labels alongside each embedding. Labels are metadata
for post-hoc analysis; they are not inputs to SAE training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForFeatureExtraction
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .config import DataSettings


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token vectors while ignoring padding tokens."""
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(
        token_embeddings.dtype
    )
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding row, avoiding division by zero."""
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def extract_openvino_embeddings(
    texts: list[str],
    tokenizer,
    ov_model,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract Assignment 5's normalized, mean-pooled sentence embeddings."""
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        outputs = ov_model(**encoded)
        token_embeddings = (
            outputs[0]
            if isinstance(outputs, (tuple, list))
            else outputs.get("last_hidden_state", outputs)
        )
        pooled = mean_pooling(token_embeddings, encoded["attention_mask"])
        all_embeddings.append(normalize_l2(pooled))

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(all_embeddings)


@dataclass
class IntentDataset:
    """Aligned embeddings, labels, and source texts for a dataset split."""

    x: np.ndarray
    y: np.ndarray
    texts: list[str] | None = None
    num_samples: int = field(init=False)
    indices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int32)
        if self.x.ndim != 2:
            raise ValueError(f"Expected x to have shape (N, D), got {self.x.shape}")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of examples")

        if self.texts is None:
            # Keeps older code that constructs IntentDataset(x, y) working.
            self.texts = [""] * len(self.y)
        else:
            self.texts = list(self.texts)
        if len(self.texts) != len(self.y):
            raise ValueError("texts, x, and y must contain the same number of examples")

        self.num_samples = len(self.y)
        self.indices = np.arange(self.num_samples)

    def get_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        truncate_dim: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a random (embedding, label) batch; text remains available by index."""
        if self.num_samples == 0:
            raise ValueError("Cannot sample a batch from an empty dataset")
        if not 0 < batch_size <= self.num_samples:
            raise ValueError(
                f"batch_size must be in [1, {self.num_samples}], got {batch_size}"
            )

        choices = rng.choice(self.indices, size=batch_size, replace=False)
        batch_x = self.x[choices]
        if truncate_dim is not None:
            batch_x = batch_x[:, :truncate_dim]
        return batch_x, self.y[choices]


class Banking77Data:
    """Load Banking77 and keep text/embedding/label alignment across splits."""

    def __init__(self, settings: DataSettings):
        self.settings = settings
        raw_dataset = load_dataset(settings.dataset_name)

        train_texts = list(raw_dataset["train"]["text"])
        train_labels = np.asarray(raw_dataset["train"]["label"], dtype=np.int32)
        test_texts = list(raw_dataset["test"]["text"])
        test_labels = np.asarray(raw_dataset["test"]["label"], dtype=np.int32)
        label_feature = raw_dataset["train"].features.get("label")
        self.label_names = list(getattr(label_feature, "names", []) or [])

        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        model_dir = Path("artifacts/openvino_embedding_model")
        if model_dir.exists():
            ov_model = OVModelForFeatureExtraction.from_pretrained(str(model_dir))
        else:
            ov_model = OVModelForFeatureExtraction.from_pretrained(
                settings.embedding_model,
                export=True,
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            ov_model.save_pretrained(str(model_dir))

        train_x = extract_openvino_embeddings(
            train_texts,
            tokenizer,
            ov_model,
            batch_size=settings.embedding_batch_size,
        )
        test_x = extract_openvino_embeddings(
            test_texts,
            tokenizer,
            ov_model,
            batch_size=settings.embedding_batch_size,
        )

        train_indices, val_indices = train_test_split(
            np.arange(len(train_labels)),
            test_size=0.2,
            random_state=31415,
            stratify=train_labels,
        )

        # Index all three arrays with the same split indices so examples stay paired.
        self.train_texts = [train_texts[i] for i in train_indices]
        self.val_texts = [train_texts[i] for i in val_indices]
        self.test_texts = test_texts

        self.train_dataset = IntentDataset(
            x=train_x[train_indices],
            y=train_labels[train_indices],
            texts=self.train_texts,
        )
        self.val_dataset = IntentDataset(
            x=train_x[val_indices],
            y=train_labels[val_indices],
            texts=self.val_texts,
        )
        self.test_dataset = IntentDataset(
            x=test_x,
            y=test_labels,
            texts=self.test_texts,
        )

        self.train_x = self.train_dataset.x
        self.val_x = self.val_dataset.x
        self.test_x = self.test_dataset.x
        self.train_labels = self.train_dataset.y
        self.val_labels = self.val_dataset.y
        self.test_labels = self.test_dataset.y

        print("Train feature shape:", self.train_x.shape)
        print("Validation feature shape:", self.val_x.shape)
        print("Test feature shape:", self.test_x.shape)
        print("Dataset ready for SAE training and latent inspection.")
