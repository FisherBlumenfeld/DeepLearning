from typing import Literal
import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx

from .config import TrainingSettings
from .data import Data
from .evaluate import evaluate
from .model import ResNetClassifier

log = structlog.get_logger()

@nnx.jit
def train_step(
    model: ResNetClassifier,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Performs a single training step."""

    def loss_fn(model: ResNetClassifier):
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y
        ).mean()
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return acc, loss


def train(
    model: ResNetClassifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    num_epochs: int,
    augment: bool = True,
) -> dict[str, list[float]]:
    """Train the model across epochs and report progress.

    Returns history dictionary for plotting learning curves.
    """
    total_batches = int(np.ceil(data.num_train / settings.batch_size))
    print(f"\nTraining for {num_epochs} epochs ({total_batches} batches per epoch)...")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        data.on_epoch_start(np_rng, augment=augment)

        epoch_loss, epoch_acc = 0.0, 0.0
        num_train_batches = 0

        for step, (x_np, y_np) in enumerate(
            data.get_epoch_batches(
                np_rng,
                batch_size=settings.batch_size,
                split="train",
                shuffle=False,
            ),
            start=1,
        ):
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)

            acc, loss = train_step(model, optimizer, x, y)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            num_train_batches += 1

            if step % 100 == 0 or step == total_batches:
                print(
                    f"Epoch {epoch}/{num_epochs} [Batch {step:03d}/{total_batches}] - "
                    f"Loss: {float(loss):.4f} | Acc: {float(acc) * 100:.2f}%"
                )

        train_acc = epoch_acc / num_train_batches
        train_loss = epoch_loss / num_train_batches

        # evaluate on validation set after each epoch
        val_acc, val_loss = evaluate(
            model,
            data,
            split="validation",
            batch_size=settings.batch_size,
            np_rng=np_rng,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"\n>>> Epoch {epoch:02d} Summary <<<\n"
            f"  Train Acc: {train_acc * 100:.2f}% | Train Loss: {train_loss:.4f}\n"
            f"  Val   Acc: {val_acc * 100:.2f}% | Val   Loss: {val_loss:.4f}\n"
        )

    #evaluate on test data to find actual accuracy
    print("Evaluating on test set...")
    test_acc, test_loss = evaluate(
        model, data, split="test", batch_size=settings.batch_size, np_rng=np_rng
    )

    print("========================================")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test Loss:     {test_loss:.4f}")
    print("========================================\n")

    return history