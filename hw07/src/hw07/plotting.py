"""Plots for comparing SAE dictionary size and training sample capacity."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PlottingSettings


def plot_experiment_results(
    results: list[dict[str, Any]], settings: PlottingSettings
) -> list[Path]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = sorted({int(row["sample_size"]) for row in results})
    fig, axes = plt.subplots(1, 3, figsize=settings.figsize, dpi=settings.dpi)
    metrics = [
        ("reconstruction_mse", "Validation reconstruction MSE"),
        ("mean_active_latents", "Mean active latents per example"),
        ("near_dead_latent_fraction", "Near-dead latent fraction"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        for sample_size in sample_sizes:
            rows = sorted(
                (r for r in results if int(r["sample_size"]) == sample_size),
                key=lambda r: int(r["latent_dim"]),
            )
            ax.plot(
                [int(r["latent_dim"]) for r in rows],
                [float(r[key]) for r in rows],
                marker="o",
                label=f"{sample_size:,} train examples",
            )
        ax.set_xlabel("Latent dictionary dimension")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        path = settings.output_dir / f"sae_capacity.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
