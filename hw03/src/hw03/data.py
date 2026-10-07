from dataclasses import InitVar, dataclass, field
from typing import Literal, Generator
import numpy as np
from sklearn.datasets import fetch_openml
# from .model import LinearModel


@dataclass
class Data:
    """Imports MNIST dataset from sklearn fetch_openml, gives batches of each data set"""

    rng: InitVar[np.random.Generator]
    num_train: int = 50_000
    num_val: int = 10_000
    num_test: int = 10_000

    train_x: np.ndarray = field(init=False)
    train_y: np.ndarray = field(init=False)
    val_x: np.ndarray = field(init=False)
    val_y: np.ndarray = field(init=False)
    test_x: np.ndarray = field(init=False)
    test_y: np.ndarray = field(init=False)

    _train_perm: np.ndarray = field(init=False)
    _train_cursor: int = field(init=False, default=0)

    def __post_init__(self, rng: np.random.Generator):
        print("Fetching MNIST via fetch_openml (cached after first run)...")
        raw_x, raw_y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )

        x = raw_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
        y = raw_y.astype(np.int32)

        # Shuffle indices before partitioning to ensure balanced class distributions
        total_samples = len(x)
        assert self.num_train + self.num_val + self.num_test <= total_samples, (
            f"Total requested samples exceed available dataset size ({total_samples})."
        )

        perm = rng.permutation(total_samples)
        train_idx = perm[: self.num_train]
        val_idx = perm[self.num_train : self.num_train + self.num_val]
        test_idx = perm[
            self.num_train + self.num_val : self.num_train
            + self.num_val
            + self.num_test
        ]

        self.train_x, self.train_y = x[train_idx], y[train_idx]
        self.val_x, self.val_y = x[val_idx], y[val_idx]
        self.test_x, self.test_y = x[test_idx], y[test_idx]

        self._train_perm = rng.permutation(self.num_train)
        self._train_cursor = 0

    def _get_pool(
        self, split: Literal["train", "validation", "test"]
    ) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            return self.train_x, self.train_y
        elif split == "validation":
            return self.val_x, self.val_y
        elif split == "test":
            return self.test_x, self.test_y
        else:
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train', 'validation', or 'test'."
            )

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
                self._train_perm = rng.permutation(len(x_pool))
                self._train_cursor = 0

            batch_idx = self._train_perm[
                self._train_cursor : self._train_cursor + batch_size
            ]
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
