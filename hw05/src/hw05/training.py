"""Training loop and epoch checkpointing for Flax NNX intent classification."""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import structlog

from .config import TrainingSettings
from .data import IntentDataset
from .model import MLPClassifier

log = structlog.get_logger()


def compute_loss_and_accuracy(
    model: MLPClassifier,
    x: jax.Array,
    y: jax.Array,
    deterministic: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Computes cross-entropy loss and classification accuracy."""
    logits = model(x, deterministic=deterministic)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == y)
    return loss, accuracy


@nnx.jit
def train_step(
    model: MLPClassifier,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Executes a single JIT-compiled training step with gradient updates."""
    def loss_fn(m: MLPClassifier):
        return compute_loss_and_accuracy(m, x, y, deterministic=False)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = grad_fn(model)
    optimizer.update(grads)
    return loss, accuracy


@nnx.jit
def eval_step(
    model: MLPClassifier,
    x: jax.Array,
    y: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluates loss and accuracy deterministically without dropout."""
    return compute_loss_and_accuracy(model, x, y, deterministic=True)


def evaluate_dataset(
    model: MLPClassifier,
    dataset: IntentDataset,
    batch_size: int = 128,
    truncate_dim: int | None = None,
) -> Tuple[float, float]:
    """Computes mean loss and accuracy over an entire dataset."""
    num_samples = dataset.num_samples
    if num_samples == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_acc = 0.0

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_x = dataset.x[i:end_idx]
        if truncate_dim is not None:
            batch_x = batch_x[:, :truncate_dim]
        batch_y = dataset.y[i:end_idx]

        x_jax = jnp.array(batch_x)
        y_jax = jnp.array(batch_y)

        loss, acc = eval_step(model, x_jax, y_jax)
        batch_count = end_idx - i
        total_loss += float(loss) * batch_count
        total_acc += float(acc) * batch_count

    return total_loss / num_samples, total_acc / num_samples


def save_checkpoint(
    model: MLPClassifier,
    epoch: int,
    checkpoint_dir: Path,
) -> Path:
    """Saves model parameters and state for a given epoch as an npz artifact."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.npz"

    # Extract NNX graph state
    _, state = nnx.split(model)
    flat_state = state.to_flat_dict()

    # Convert JAX state arrays to portable NumPy dictionary
    np_state = {
        "/".join(map(str, k)): np.array(v)
        for k, v in flat_state.items()
    }
    np.savez(checkpoint_path, **np_state)
    log.info("Checkpoint saved", epoch=epoch, path=str(checkpoint_path))
    return checkpoint_path


def train(
    model: MLPClassifier,
    optimizer: nnx.Optimizer,
    train_dataset: IntentDataset,
    val_dataset: IntentDataset,
    settings: TrainingSettings,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Runs epoch-based training loop with validation tracking and checkpointing."""
    history = []
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, train_dataset.num_samples // settings.batch_size)
    log.info(
        "Starting training loop",
        epochs=settings.num_epochs,
        batch_size=settings.batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    for epoch in range(1, settings.num_epochs + 1):
        epoch_losses = []
        epoch_accuracies = []

        for _ in range(steps_per_epoch):
            batch_x, batch_y = train_dataset.get_batch(rng, settings.batch_size)
            x_jax = jnp.array(batch_x)
            y_jax = jnp.array(batch_y)

            loss, acc = train_step(model, optimizer, x_jax, y_jax)
            epoch_losses.append(float(loss))
            epoch_accuracies.append(float(acc))

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accuracies))

        # Evaluate on validation split
        val_loss, val_acc = evaluate_dataset(model, val_dataset, batch_size=settings.batch_size)

        # Save epoch checkpoint
        ckpt_path = save_checkpoint(model, epoch, settings.checkpoint_dir)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "checkpoint_path": str(ckpt_path),
        }
        history.append(epoch_metrics)

        log.info(
            "Epoch completed",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
        )

    return history
