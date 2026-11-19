"""Sparse autoencoder implemented with Flax NNX."""

import jax
import jax.numpy as jnp
from flax import nnx


class SparseAutoencoder(nnx.Module):
    """Reconstruct dense embeddings from sparse, overcomplete activations."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        input_dim: int = 384,
        latent_dim: int = 768,
    ):
        super().__init__()
        if latent_dim < input_dim:
            raise ValueError("latent_dim should be >= input_dim for an overcomplete SAE")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pre_bias = nnx.Param(jnp.zeros((input_dim,), dtype=jnp.float32))
        self.encoder = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, input_dim, rngs=rngs)
        self.normalize_decoder_weights()

    def encode(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.encoder(x - self.pre_bias.value))

    def decode(self, activations: jax.Array) -> jax.Array:
        return self.decoder(activations)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        activations = self.encode(x)
        reconstruction = self.decode(activations)
        return reconstruction, activations

    def normalize_decoder_weights(self) -> None:
        """Keep each latent's decoder vector unit length to control SAE scaling."""
        kernel = self.decoder.kernel.value
        norms = jnp.linalg.norm(kernel, axis=1, keepdims=True)
        self.decoder.kernel.value = kernel / jnp.maximum(norms, 1e-8)
