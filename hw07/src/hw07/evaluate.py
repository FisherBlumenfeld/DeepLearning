"""Held-out reconstruction, sparsity, dead-latent, and feature-example analysis."""

from collections import Counter
from typing import Any

import jax.numpy as jnp
import numpy as np

from .data import IntentDataset
from .model import SparseAutoencoder


def evaluate_autoencoder(
    model: SparseAutoencoder,
    dataset: IntentDataset,
    *,
    batch_size: int = 256,
    top_k: int = 8,
    activity_threshold: float = 1e-6,
    near_dead_fraction_threshold: float = 0.001,
    label_names: list[str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a split and return aggregate metrics plus top text examples."""
    if dataset.num_samples == 0:
        raise ValueError("Cannot evaluate an empty split")

    all_acts = []
    squared_error = 0.0
    total_values = 0
    for start in range(0, dataset.num_samples, batch_size):
        stop = min(start + batch_size, dataset.num_samples)
        x = jnp.asarray(dataset.x[start:stop])
        reconstruction, activations = model(x)
        residual = np.asarray(x - reconstruction)
        squared_error += float(np.square(residual).sum())
        total_values += residual.size
        all_acts.append(np.asarray(activations))

    activations = np.concatenate(all_acts, axis=0)
    active = activations > activity_threshold
    active_counts = active.sum(axis=0)
    active_fractions = active_counts / dataset.num_samples
    dead_fraction = float(np.mean(active_counts == 0))
    near_dead_fraction = float(np.mean(active_fractions < near_dead_fraction_threshold))
    metrics = {
        "reconstruction_mse": squared_error / max(total_values, 1),
        "mean_active_latents": float(active.sum(axis=1).mean()),
        "mean_active_fraction": float(active.mean()),
        "dead_latent_fraction": dead_fraction,
        "near_dead_latent_fraction": near_dead_fraction,
        "latent_dim": float(activations.shape[1]),
        "num_examples": float(dataset.num_samples),
    }

    examples: list[dict[str, Any]] = []
    y = np.asarray(dataset.y)
    available_labels = max(len(np.unique(y)), 1)
    for latent_id in range(activations.shape[1]):
        values = activations[:, latent_id]
        k = min(top_k, len(values))
        top_indices = np.argsort(values)[-k:][::-1]
        top_labels = y[top_indices]
        counts = Counter(int(label) for label in top_labels)
        probs = np.asarray(list(counts.values()), dtype=np.float64) / k
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        normalized_entropy = entropy / np.log(available_labels) if available_labels > 1 else 0.0
        majority_label, majority_count = counts.most_common(1)[0]
        if label_names and 0 <= majority_label < len(label_names):
            majority_name = label_names[majority_label]
        else:
            majority_name = str(majority_label)

        examples.append(
            {
                "latent_id": latent_id,
                "activation_rate": float(active_fractions[latent_id]),
                "top_label": majority_name,
                "top_label_purity": float(majority_count / k),
                "top_label_entropy": normalized_entropy,
                "top_examples": [
                    {
                        "activation": float(values[i]),
                        "label": (
                            label_names[int(y[i])]
                            if label_names and 0 <= int(y[i]) < len(label_names)
                            else int(y[i])
                        ),
                        "text": dataset.texts[i],
                    }
                    for i in top_indices
                ],
            }
        )

    return metrics, examples
