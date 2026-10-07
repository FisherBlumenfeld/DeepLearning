from typing import Literal
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .data import Data
from .model import ResNetClassifier

@nnx.jit
def eval_step(
    model: ResNetClassifier, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes validation/test accuracy and loss WITHOUT gradient updates."""
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return acc, loss


def evaluate(
    model: ResNetClassifier,
    data: Data,
    split: Literal["train", "validation", "test"],
    batch_size: int,
    np_rng: np.random.Generator,
) -> tuple[float, float]:
    """Evaluates the model over the full split."""
    total_loss, total_acc = 0.0, 0.0
    num_batches = 0

    # shuffle=False for deterministic evaluation passes
    for x_np, y_np in data.get_epoch_batches(
        np_rng, batch_size=batch_size, split=split, shuffle=False
    ):
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)
        acc, loss = eval_step(model, x, y)
        total_loss += float(loss)
        total_acc += float(acc)
        num_batches += 1

    return total_acc / num_batches, total_loss / num_batches