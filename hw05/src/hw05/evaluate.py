"""Evaluation module for Banking77 intent classification and Matryoshka dimension tradeoff analysis."""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import jax.numpy as jnp
import numpy as np
from flax import nnx
import structlog

from .config import AppSettings
from .data import IntentDataset, Banking77Data
from .model import MLPClassifier
from .training import evaluate_dataset

log = structlog.get_logger()


def load_checkpoint(model: MLPClassifier, checkpoint_path: Path) -> MLPClassifier:
    """Loads model parameters from an NPZ checkpoint."""
    import jax
    import jax.numpy as jnp

    log.info("Loading model checkpoint", path=str(checkpoint_path))

    with np.load(checkpoint_path, allow_pickle=False) as data:
        flat_state = {key: data[key] for key in data.files}

    # Reconstruct nested dictionaries from slash-separated NPZ keys.
    nested_state = {}
    for key, value in flat_state.items():
        parts = key.split("/")
        current = nested_state
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    # Use the current model's state as a structural template.
    graphdef, state = nnx.split(model)
    pure_state = nnx.to_pure_dict(state)

    def restore(template, saved):
        if isinstance(template, dict):
            return {
                key: restore(value, saved[key]) if key in saved else value
                for key, value in template.items()
            }

        if isinstance(saved, dict):
            return template

        # Restore PRNG keys to their original typed representation.
        if hasattr(template, "dtype") and jax.dtypes.issubdtype(
            template.dtype, jax.dtypes.prng_key
        ):
            return jax.random.wrap_key_data(jnp.asarray(saved, dtype=jnp.uint32))

        return jnp.asarray(saved, dtype=getattr(template, "dtype", None))

    restored = restore(pure_state, nested_state)
    _, model_state = nnx.split(model)
    model_state.replace_by_pure_dict(restored)
    nnx.update(model, model_state)

    return model


def select_best_checkpoint(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Selects the epoch checkpoint with the highest validation accuracy.

    Args:
        history: List of epoch metric dictionaries from training history.

    Returns:
        Metric dictionary corresponding to the best validation performance epoch.
    """
    best_epoch_metrics = max(history, key=lambda x: x["val_accuracy"])
    log.info(
        "Best validation checkpoint selected",
        epoch=best_epoch_metrics["epoch"],
        val_accuracy=round(best_epoch_metrics["val_accuracy"], 4),
        val_loss=round(best_epoch_metrics["val_loss"], 4),
        checkpoint_path=best_epoch_metrics["checkpoint_path"],
    )
    return best_epoch_metrics


def evaluate_matryoshka_dimensions(
    model: MLPClassifier,
    test_dataset: IntentDataset,
    matryoshka_dims: List[int] = [384, 256, 128, 64],
    full_dim: int = 384,
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    """Evaluates test performance across truncated Matryoshka embedding dimensions.

    Args:
        model: Evaluated MLPClassifier model.
        test_dataset: Test set IntentDataset container.
        matryoshka_dims: List of truncated dimensions to evaluate (e.g. [384, 256, 128, 64]).
        full_dim: Original full embedding feature dimension (default: 384).
        batch_size: Evaluation batch size.

    Returns:
        List of dictionaries containing evaluation metrics per dimension.
    """
    results = []

    for d in matryoshka_dims:
        if d > full_dim:
            log.warning(
                "Truncated dimension exceeds full dimension, skipping",
                d=d,
                full_dim=full_dim,
            )
            continue

        test_loss, test_acc = evaluate_dataset(
            model=model,
            dataset=test_dataset,
            batch_size=batch_size,
            truncate_dim=d,
        )

        compression_factor = full_dim / d

        dim_metrics = {
            "dimension": d,
            "compression_factor": float(compression_factor),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        }
        results.append(dim_metrics)

        log.info(
            "Matryoshka evaluation completed for dimension",
            dimension=d,
            compression_factor=f"{compression_factor:.2f}x",
            test_accuracy=f"{test_acc * 100:.2f}%",
            test_loss=round(test_loss, 4),
        )

    return results


def run_evaluation(
    model: MLPClassifier,
    data: Banking77Data,
    history: List[Dict[str, Any]],
    settings: AppSettings,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Coordinates full evaluation workflow: checkpoint selection, restoration, and Matryoshka benchmarking.

    Args:
        model: Initialized MLPClassifier model.
        data: Prepared Banking77Data instance.
        history: Training metric history list.
        settings: Application settings configuration.

    Returns:
        Tuple of (best_epoch_metrics, matryoshka_results).
    """
    best_metrics = select_best_checkpoint(history)
    ckpt_path = Path(best_metrics["checkpoint_path"])

    # Restore model parameters from best validation checkpoint
    model = load_checkpoint(model, ckpt_path)

    # Benchmark across Matryoshka dimensions
    matryoshka_results = evaluate_matryoshka_dimensions(
        model=model,
        test_dataset=data.test_dataset,
        matryoshka_dims=settings.data.matryoshka_dims,
        full_dim=settings.data.embedding_dim,
        batch_size=settings.training.batch_size,
    )

    return best_metrics, matryoshka_results
