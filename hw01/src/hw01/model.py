from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float


@dataclass
class GaussianModel:
    """represents gaussian model"""

    sigmas: np.ndarray
    mus: np.ndarray
    weights: np.ndarray
    bias: float
    size: int


class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model."""

    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.num_features = num_features
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (self.num_features, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return jnp.squeeze(x @ self.w.value + self.b.value, axis=-1)

    @property
    def model(self) -> LinearModel:
        """Returns the underlying simple linear model."""
        return LinearModel(
            weights=np.array(self.w.value).reshape([self.num_features]),
            bias=float(np.array(self.b.value).squeeze()),
        )


class NNXGaussianModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, num_gaussians: int):
        self.num_gaussians = num_gaussians
        key_mu = rngs.params()
        key_sigma = rngs.params()
        key_w = rngs.params()
        self.mu = nnx.Param(
            jax.random.uniform(key_mu, (num_gaussians, 1), minval=0, maxval=1)
        )
        self.sigma = nnx.Param(
            jnp.abs(jax.random.normal(key_sigma, (self.num_gaussians, 1)))
        )
        self.w = nnx.Param(jax.random.normal(key_w, (self.num_gaussians, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        x = x.reshape(-1, 1)
        mu = self.mu.value.reshape(1, -1)
        sigma = self.sigma.value.reshape(1, -1)
        w = self.w.value.reshape(1, -1)

        phi = jnp.exp(-((x - mu) ** 2) / (sigma**2))
        gaussians = w * phi
        yhat = jnp.sum(gaussians, axis=-1, keepdims=True) + self.b[0][0]

        return jnp.squeeze(yhat, axis=-1)

    @property
    def model(self) -> GaussianModel:
        """Returns the underlying simple linear model."""
        return GaussianModel(
            mus=np.array(self.mu.value.reshape(1, -1)),
            sigmas=np.array(self.sigma.value.reshape(1, -1)),
            weights=np.array(self.w.value.reshape(1, -1)),
            bias=float(np.array(self.b.value).squeeze()),
            size=self.num_gaussians,
        )
