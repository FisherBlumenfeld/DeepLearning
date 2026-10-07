from dataclasses import InitVar, dataclass, field
from typing import Literal, Generator
import numpy as np
from datasets import load_dataset
import logging

# Suppress HTTP request logging from urllib3 / requests / huggingface_hub
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

@dataclass
class Data:
    """Imports CIFAR 10 dataset from hugging face datasets, gives batches of each subset"""

    rng: InitVar[np.random.Generator]
    num_train: int = 40_000
    num_val: int = 10_000
    num_test: int = 10_000

    raw_train_x: np.ndarray = field(init=False)
    raw_train_y: np.ndarray = field(init=False)
    augmented_train_x: np.ndarray = field(init=False) #with flipping and padded cropping
    augmented_train_y: np.ndarray = field(init=False)
    val_x: np.ndarray = field(init=False)
    val_y: np.ndarray = field(init=False)
    test_x: np.ndarray = field(init=False)
    test_y: np.ndarray = field(init=False)

    _train_perm: np.ndarray = field(init=False)
    _train_cursor: int = field(init=False, default=0)

    def __post_init__(self, rng: np.random.Generator):
        print("Loading Data")
        dataset = load_dataset("uoft-cs/cifar10")
        dataset_np = dataset.with_format("numpy")

        raw_x_training = np.asarray(dataset_np["train"]["img"], dtype = np.int32) / 255.0
        raw_y_training = np.asarray(dataset_np["train"]["label"], dtype = np.int32)

        total_train_samples = len(raw_x_training)

        perm = rng.permutation(total_train_samples)
        train_idx = perm[: self.num_train]
        val_idx = perm[self.num_train : self.num_train + self.num_val]

        self.raw_train_x, self.raw_train_y = raw_x_training[train_idx], raw_y_training[train_idx]
        self.val_x, self.val_y = raw_x_training[val_idx], raw_y_training[val_idx]

        raw_x_test = np.asarray(dataset_np["test"]["img"], dtype = np.int32) / 255.0
        raw_y_test = np.asarray(dataset_np["test"]["label"], dtype = np.int32)

        self.test_x = raw_x_test[: self.num_test]
        self.test_y = raw_y_test[: self.num_test]

        self._train_perm = np.arange(self.num_train)
        self._train_cursor = 0

        self.on_epoch_start(rng, augment=True)

    def on_epoch_start(self, rng: np.random.Generator, augment: bool = True):
        """
        Shuffles samples, applies 4-pixel padded cropping, and horizontal flips.
        """
        perm = rng.permutation(self.num_train)
        x_shuffled = self.raw_train_x[perm]
        y_shuffled = self.raw_train_y[perm]

        if not augment:
            self.augmented_train_x = x_shuffled
            self.augmented_train_y = y_shuffled
            self._train_cursor = 0
            return

        flips = rng.uniform(size=self.num_train) < 0.5
        x_shuffled[flips] = x_shuffled[flips, :, ::-1, :]

        padded_x = np.pad(
            x_shuffled,
            pad_width=((0, 0), (4, 4), (4, 4), (0, 0)),
            mode="reflect",
        )

        crop_y = rng.integers(0, 9, size=self.num_train)
        crop_x = rng.integers(0, 9, size=self.num_train)

        batch_idx = np.arange(self.num_train)[:, None, None]
        grid_y = crop_y[:, None, None] + np.arange(32)[None, :, None]
        grid_x = crop_x[:, None, None] + np.arange(32)[None, None, :]

        self.augmented_train_x = padded_x[batch_idx, grid_y, grid_x, :]
        self.augmented_train_y = y_shuffled
        self._train_cursor = 0

    def _get_pool(
        self, split: Literal["train", "validation", "test"]
    ) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            return self.augmented_train_x, self.augmented_train_y
        elif split == "validation":
            return self.val_x, self.val_y
        elif split == "test":
            return self.test_x, self.test_y
        else:
            print("tried to call invalid split")
            return 0

    def get_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        split: Literal["train", "validation", "test"] = "train",
        sequential: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:

        x_pool, y_pool = self._get_pool(split)

        if (
            sequential and split == "train"
        ):  # will go thru entire training set, and shuffle once complete
            if self._train_cursor + batch_size > len(x_pool):
                self._train_cursor = 0

            batch_idx = slice(self._train_cursor, self._train_cursor + batch_size)
            self._train_cursor += batch_size
            return x_pool[batch_idx], y_pool[batch_idx]

        indices = rng.choice(len(x_pool), size=batch_size, replace=False)
        return x_pool[indices], y_pool[indices]

    def get_epoch_batches(
        self,
        rng: np.random.Generator,
        batch_size: int,
        split: Literal["train", "validation", "test"] = "train",
        shuffle: bool = True,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yields every example in the split exactly once in batches."""
        x_pool, y_pool = self._get_pool(split)
        num_samples = len(x_pool)

        indices = rng.permutation(num_samples) if shuffle else np.arange(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_idx = indices[start_idx:end_idx]
            yield x_pool[batch_idx], y_pool[batch_idx]
