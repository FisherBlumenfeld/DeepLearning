"""Plotting utilities for Banking77 intent classification and Matryoshka dimension performance trade-offs."""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/sandbox environment
import matplotlib.pyplot as plt
import structlog

from .config import PlottingSettings

log = structlog.get_logger()


def plot_matryoshka_performance(
    results: List[Dict[str, Any]],
    settings: PlottingSettings,
    filename_prefix: str = "matryoshka_tradeoff",
) -> List[Path]:
    """Generates dual-axis performance curve comparing Test Accuracy and Compression Factor against Embedding Dimension.
    
    Args:
        results: List of dictionaries containing evaluation metrics per dimension.
                 Keys required: 'dimension', 'compression_factor', 'test_accuracy', 'test_loss'.
        settings: PlottingSettings object with output_dir, figsize, and dpi.
        filename_prefix: Prefix for saved plot file artifacts.
        
    Returns:
        List of Path objects for saved plot files (SVG and PDF).
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort results by dimension descending (384 -> 64)
    sorted_results = sorted(results, key=lambda x: x["dimension"], reverse=True)
    dims = [r["dimension"] for r in sorted_results]
    accuracies = [r["test_accuracy"] * 100.0 for r in sorted_results]  # Convert to percentage
    compressions = [r["compression_factor"] for r in sorted_results]
    
    fig, ax1 = plt.subplots(figsize=settings.figsize, dpi=settings.dpi)
    
    # Primary axis: Test Accuracy (%)
    color_acc = "#1f77b4"  # Deep blue
    ax1.set_xlabel("Embedding Dimension (d)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    line1 = ax1.plot(
        dims,
        accuracies,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=color_acc,
        label="Test Accuracy (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(dims)
    ax1.invert_xaxis()  # 384 -> 64 left-to-right representation
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Annotate accuracy points
    for d, acc, comp in zip(dims, accuracies, compressions):
        ax1.annotate(
            f"{acc:.2f}%\n({comp:.1f}x)",
            xy=(d, acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_acc, lw=0.8, alpha=0.8),
        )

    # Secondary axis: Compression Factor
    ax2 = ax1.twinx()
    color_comp = "#ff7f0e"  # Vivid orange
    ax2.set_ylabel("Compression Factor (x)", color=color_comp, fontsize=11, fontweight="bold")
    line2 = ax2.plot(
        dims,
        compressions,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=7,
        color=color_comp,
        label="Compression Factor",
    )
    ax2.tick_params(axis="y", labelcolor=color_comp)
    
    # Title & Layout
    plt.title("Matryoshka Representation Learning: Accuracy vs. Compression", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    
    saved_paths = []
    for ext in ["svg", "pdf", "png"]:
        out_path = settings.output_dir / f"{filename_prefix}.{ext}"
        plt.savefig(out_path, format=ext, dpi=settings.dpi, bbox_inches="tight")
        saved_paths.append(out_path)
        log.info("Saved Matryoshka tradeoff plot", path=str(out_path))
        
    plt.close(fig)
    return saved_paths


def plot_training_history(
    history: List[Dict[str, Any]],
    settings: PlottingSettings,
    filename_prefix: str = "training_curves",
) -> List[Path]:
    """Generates loss and accuracy progression curves across training epochs.
    
    Args:
        history: List of epoch metric dictionaries containing 'epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'.
        settings: PlottingSettings configuration.
        filename_prefix: Output file prefix.
        
    Returns:
        List of Path objects for saved plot files.
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] * 100.0 for h in history]
    val_acc = [h["val_accuracy"] * 100.0 for h in history]
    
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(settings.figsize[0] * 2, settings.figsize[1]), dpi=settings.dpi)
    
    # Loss subplot
    ax_loss.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.0)
    ax_loss.plot(epochs, val_loss, label="Val Loss", color="#d62728", linestyle="--", linewidth=2.0)
    ax_loss.set_xlabel("Epoch", fontweight="bold")
    ax_loss.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax_loss.set_title("Training & Validation Loss", fontweight="bold")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend()
    
    # Accuracy subplot
    ax_acc.plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", linewidth=2.0)
    ax_acc.plot(epochs, val_acc, label="Val Accuracy", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax_acc.set_xlabel("Epoch", fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontweight="bold")
    ax_acc.set_title("Training & Validation Accuracy", fontweight="bold")
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend()
    
    plt.suptitle("Banking77 MLP Classifier Training Progress", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    saved_paths = []
    for ext in ["svg", "pdf", "png"]:
        out_path = settings.output_dir / f"{filename_prefix}.{ext}"
        plt.savefig(out_path, format=ext, dpi=settings.dpi, bbox_inches="tight")
        saved_paths.append(out_path)
        log.info("Saved training history plot", path=str(out_path))
        
    plt.close(fig)
    return saved_paths
