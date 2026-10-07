import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange
from typing import Literal
import optax

from .config import TrainingSettings
from .data import Data
from .model import Classifier

log = structlog.get_logger()


@nnx.jit
def train_step(
    model: Classifier, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Performs a single training step."""

    def loss_fn(model: Classifier):
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y
        ).mean()
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)  # In-place update of model parameters
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return acc, loss


@nnx.jit
def eval_step(
    model: Classifier, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes validation/test accuracy and loss WITHOUT gradient updates."""
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return acc, loss


def evaluate(
    model: Classifier,
    data: Data,
    split: Literal["train", "validation", "test"],
    batch_size: int,
    np_rng: np.random.Generator,
) -> tuple[float, float]:
    """Evaluates the model over the full split."""
    total_loss, total_acc = 0.0, 0.0
    num_batches = 0

    # Shuffle=False for deterministic evaluation passes
    for x_np, y_np in data.get_epoch_batches(
        np_rng, batch_size=batch_size, split=split, shuffle=False
    ):
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)
        acc, loss = eval_step(model, x, y)
        total_loss += float(loss)
        total_acc += float(acc)
        num_batches += 1

    return total_acc / num_batches, total_loss / num_batches


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    num_epochs: int,
) -> None:
    """Train the model across epochs and report progress using print statements."""

    total_batches = int(np.ceil(data.num_train / settings.batch_size))

    print(f"\nTraining for {num_epochs} epochs ({total_batches} batches per epoch)...")

    for epoch in range(1, num_epochs + 1):
        epoch_loss, epoch_acc = 0.0, 0.0
        num_train_batches = 0

        for step, (x_np, y_np) in enumerate(
            data.get_epoch_batches(
                np_rng, batch_size=settings.batch_size, split="train", shuffle=True
            ),
            start=1,
        ):
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)

            acc, loss = train_step(model, optimizer, x, y)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            num_train_batches += 1

            # Print intermediate progress every 100 batches and last batch
            if step % 100 == 0 or step == total_batches:
                print(
                    f"Epoch {epoch}/{num_epochs} [Batch {step:03d}/{total_batches}] - "
                    f"Loss: {float(loss):.4f} | Acc: {float(acc) * 100:.2f}%"
                )

        train_acc = epoch_acc / num_train_batches
        train_loss = epoch_loss / num_train_batches

        # run validation pass at epoch end
        val_acc, val_loss = evaluate(
            model,
            data,
            split="validation",
            batch_size=settings.batch_size,
            np_rng=np_rng,
        )

        print(
            f"\n>>> Epoch {epoch:02d} Summary <<<\n"
            f"  Train Acc: {train_acc * 100:.2f}% | Train Loss: {train_loss:.4f}\n"
            f"  Val   Acc: {val_acc * 100:.2f}% | Val   Loss: {val_loss:.4f}\n"
        )

    # final test on unseen test set
    print("Evaluating on test set...")
    test_acc, test_loss = evaluate(
        model, data, split="test", batch_size=settings.batch_size, np_rng=np_rng
    )

    print("========================================")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test Loss:     {test_loss:.4f}")
    print("========================================\n")
