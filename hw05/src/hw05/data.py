from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForFeatureExtraction
from transformers import AutoTokenizer

from .config import DataSettings


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Computes mean pooling over token embeddings weighted by attention mask."""
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(token_embeddings.dtype)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """Applies L2 normalization to embedding vectors across rows."""
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def extract_openvino_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Extracts L2-normalized dense embeddings using OpenVINO runtime."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ov_model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True)

    all_embeddings = []
    num_samples = len(texts)

    for i in range(0, num_samples, batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        outputs = ov_model(**encoded)
        token_embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.get("last_hidden_state", outputs)
        pooled = mean_pooling(token_embeddings, encoded["attention_mask"])
        normalized = normalize_l2(pooled)
        all_embeddings.append(normalized)

    return np.vstack(all_embeddings)


@dataclass
class IntentDataset:
    """Container for processed intent embeddings and label targets."""
    x: np.ndarray
    y: np.ndarray
    num_samples: int = field(init=False)
    indices: np.ndarray = field(init=False)

    def __post_init__(self):
        self.num_samples = len(self.y)
        self.indices = np.arange(self.num_samples)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int, truncate_dim: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects a random mini-batch of normalized features and integer labels."""
        choices = rng.choice(self.indices, size=batch_size, replace=False)
        batch_x = self.x[choices]
        if truncate_dim is not None:
            batch_x = batch_x[:, :truncate_dim]
        batch_y = self.y[choices]
        return batch_x, batch_y


class Banking77Data:
    """Handles loading mteb/banking77 and preparing extracted embeddings for splits."""

    def __init__(self, settings: DataSettings):
        self.settings = settings
        print("Loading dataset", settings.dataset_name)
        raw_dataset = load_dataset(settings.dataset_name)

        # Process training and test splits
        train_texts = raw_dataset["train"]["text"]
        self.train_labels = np.array(raw_dataset["train"]["label"], dtype=np.int32)

        test_texts = raw_dataset["test"]["text"]
        self.test_labels = np.array(raw_dataset["test"]["label"], dtype=np.int32)
        self.train_x = extract_openvino_embeddings(train_texts, settings.embedding_model)
        self.test_x = extract_openvino_embeddings(test_texts, settings.embedding_model)

        # 80/20 Train-Validation Split for checkpoint selection
        val_size = int(0.2 * len(self.train_x))
        self.val_dataset = IntentDataset(x=self.train_x[:val_size], y=self.train_labels[:val_size])
        self.train_dataset = IntentDataset(x=self.train_x[val_size:], y=self.train_labels[val_size:])
        self.test_dataset = IntentDataset(x=self.test_x, y=self.test_labels)

        print("dataset finished!")