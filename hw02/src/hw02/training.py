import jax
import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Spiral_Data

log = structlog.get_logger()


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, batch: np.ndarray):
    """Performs a single training step."""

    def loss_fn(model, batch):
        x = jnp.stack([batch[0], batch[1]], axis=1)
        y = jnp.asarray(batch[2], dtype=jnp.int32)

        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)  # In-place update of model parameters
    return loss


def train(
    model,
    optimizer: nnx.Optimizer,
    data: Spiral_Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
):
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    losses = []

    for i in bar:
        batch = data.get_batch(np_rng, settings.batch_size)

        loss = train_step(model, optimizer, batch)
        losses.append(loss)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")
    return losses
