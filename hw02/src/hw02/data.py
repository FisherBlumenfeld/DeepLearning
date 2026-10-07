from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Spiral_Data:
    """Handles generation of dual spirals for training data."""

    rng: InitVar[np.random.Generator]
    num_samples_A: int
    num_samples_B: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)
    is_class_a: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data for dual spirals."""
        num_samples = self.num_samples_A + self.num_samples_B
        self.index = np.arange(num_samples)
        # parametrizing spiral curve
        Theta = rng.uniform(np.pi, np.pi * 9.0 / 2.0, size=num_samples)
        self.is_class_a = np.concatenate(
            [
                np.ones(self.num_samples_A, dtype=int),
                np.zeros(self.num_samples_B, dtype=int),
            ]
        )
        multiplier = np.where(self.is_class_a, 1, -1)
        clean_x = -multiplier * Theta * np.cos(Theta)
        clean_y = multiplier * Theta * np.sin(Theta)
        self.x = rng.normal(loc=clean_x, scale=self.sigma)
        self.y = rng.normal(loc=clean_y, scale=self.sigma)
        for i in range(len(self.x)):
            print("x:", self.x[i], "y: ", self.y[i])

    def get_batch(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        """Select a random subset containing examples from both classes."""
        choices = np.sort(rng.choice(self.index, size=batch_size, replace=False))
        return np.ndarray(
            [
                self.x[choices],
                self.y[choices],
                self.is_class_a[choices],
            ]
        )
