"""Training and checkpoint utilities for the sparse autoencoder."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from .model import SparseAutoencoder


@nnx.jit
def train_step(
    model: SparseAutoencoder,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    l1_coefficient: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def loss_fn(m: SparseAutoencoder):
        reconstruction, activations = m(x)
        reconstruction_loss = jnp.mean(jnp.square(x - reconstruction))
        sparsity_loss = jnp.mean(jnp.sum(jnp.abs(activations), axis=-1))
        return reconstruction_loss + l1_coefficient * sparsity_loss, (
            reconstruction_loss,
            sparsity_loss,
        )

    (loss, (reconstruction_loss, sparsity_loss)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model)
    optimizer.update(model, grads)
    model.normalize_decoder_weights()
    return loss, reconstruction_loss, sparsity_loss


@nnx.jit
def eval_step(
    model: SparseAutoencoder,
    x: jax.Array,
    l1_coefficient: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    reconstruction, activations = model(x)
    reconstruction_loss = jnp.mean(jnp.square(x - reconstruction))
    sparsity_loss = jnp.mean(jnp.sum(jnp.abs(activations), axis=-1))
    return (
        reconstruction_loss + l1_coefficient * sparsity_loss,
        reconstruction_loss,
        sparsity_loss,
    )


def _dataset_losses(
    model: SparseAutoencoder,
    x: np.ndarray,
    batch_size: int,
    l1_coefficient: float,
) -> tuple[float, float, float]:
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(x), batch_size):
        batch = jnp.asarray(x[start : start + batch_size])
        values = eval_step(model, batch, l1_coefficient)
        n = len(batch)
        totals += np.asarray([float(v) for v in values]) * n
        count += n
    if count == 0:
        return 0.0, 0.0, 0.0
    return tuple((totals / count).tolist())


def save_checkpoint(model: SparseAutoencoder, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, state = nnx.split(model)
    pure_state = nnx.to_pure_dict(state)
    arrays: dict[str, np.ndarray] = {}

    def flatten(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}/{key}" if prefix else str(key))
        else:
            arrays[prefix] = np.asarray(value)

    flatten(pure_state)
    np.savez(path, **arrays)
    return path


def load_checkpoint(model: SparseAutoencoder, path: Path) -> SparseAutoencoder:
    _, state = nnx.split(model)
    template = nnx.to_pure_dict(state)
    with np.load(path, allow_pickle=False) as saved_flat:
        saved_nested: dict[str, Any] = {}
        for key in saved_flat.files:
            parts = key.split("/")
            current = saved_nested
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = saved_flat[key]

    def restore(current: Any, saved: Any) -> Any:
        if isinstance(current, dict):
            return {
                key: restore(value, saved[key]) if key in saved else value
                for key, value in current.items()
            }
        return jnp.asarray(saved, dtype=getattr(current, "dtype", None))

    state.replace_by_pure_dict(restore(template, saved_nested))
    nnx.update(model, state)
    return model


def train_autoencoder(
    model: SparseAutoencoder,
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    l1_coefficient: float,
    checkpoint_path: Path,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=learning_rate),
        wrt=nnx.Param,
    )
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        order = rng.permutation(len(train_x))
        train_totals = np.zeros(3, dtype=np.float64)
        train_count = 0
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            values = train_step(
                model,
                optimizer,
                jnp.asarray(train_x[ids]),
                l1_coefficient,
            )
            n = len(ids)
            train_totals += np.asarray([float(v) for v in values]) * n
            train_count += n

        train_values = train_totals / max(train_count, 1)
        val_values = _dataset_losses(model, val_x, batch_size, l1_coefficient)
        record = {
            "epoch": float(epoch),
            "train_total": float(train_values[0]),
            "train_reconstruction": float(train_values[1]),
            "train_sparsity": float(train_values[2]),
            "val_total": val_values[0],
            "val_reconstruction": val_values[1],
            "val_sparsity": val_values[2],
        }
        history.append(record)
        if val_values[0] < best_val_loss:
            best_val_loss = val_values[0]
            save_checkpoint(model, checkpoint_path)

    load_checkpoint(model, checkpoint_path)
    return history
