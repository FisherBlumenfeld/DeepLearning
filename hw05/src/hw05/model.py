"""Model architecture for the Banking77 intent classification task using Flax NNX."""

from typing import Sequence
import jax
import jax.numpy as jnp
from flax import nnx


class MLPClassifier(nnx.Module):
    """Multi-Layer Perceptron (MLP) intent classifier built with Flax NNX.
    
    Supports dynamic input embedding dimensions to accommodate Matryoshka
    Representation Learning evaluation across truncated dimensions d in {384, 256, 128, 64}.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        input_dim: int = 384,
        num_classes: int = 77,
        hidden_dims: Sequence[int] = (256, 128),
        dropout_rate: float = 0.1,
    ):
        """Initializes the MLP classifier layers.
        
        Args:
            rngs: Flax NNX PRNG key generator.
            input_dim: Input embedding feature dimension (default: 384).
            num_classes: Number of output target intent classes (default: 77).
            hidden_dims: Sequence of hidden layer node counts.
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        dropouts = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, h_dim, rngs=rngs))
            dropouts.append(nnx.Dropout(dropout_rate, rngs=rngs) if dropout_rate > 0 else None)
            prev_dim = h_dim

        self.layers = layers
        self.dropouts = dropouts
        self.head = nnx.Linear(prev_dim, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        """Forward pass computing class logits.
        
        Args:
            x: Input feature array of shape (batch_size, input_dim).
            deterministic: If True, disables dropout for evaluation.
            
        Returns:
            Unnormalized logits of shape (batch_size, num_classes).
        """
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = nnx.gelu(x)
            if dropout is not None:
                x = dropout(x, deterministic=deterministic)

        logits = self.head(x)
        return logits


# Alias for backward compatibility with earlier project templates
NNXIntentClassifier = MLPClassifier
NNXLinearModel = MLPClassifier
