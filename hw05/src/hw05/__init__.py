"""Main entry point for Banking77 intent classification with Flax NNX and Matryoshka Representation Learning."""

from pathlib import Path
import numpy as np
import optax
from flax import nnx
import structlog

from .config import load_settings
from .data import Banking77Data
from .evaluate import run_evaluation
from .app_logging import setup_logging
from .model import MLPClassifier
from .plotting import plot_matryoshka_performance, plot_training_history
from .training import train

log = structlog.get_logger()


def main() -> None:
    """Executes the complete Assignment 5 pipeline:
    1. Loads configuration settings and initializes logging.
    2. Loads Banking77 dataset and extracts OpenVINO dense embeddings.
    3. Initializes Flax NNX MLP classifier and Optax Adam optimizer.
    4. Trains classifier across epochs with validation tracking and checkpointing.
    5. Plots training loss and accuracy progression curves.
    6. Selects optimal validation checkpoint and evaluates test set performance
       across truncated Matryoshka embedding dimensions d in {384, 256, 128, 64}.
    7. Plots test accuracy vs. embedding dimension and compression factors.
    """
    # 1. Configuration & Logging Initialization
    settings = load_settings()
    setup_logging(debug=settings.debug)
    log.info("Starting Assignment 5 execution pipeline", seed=settings.random_seed)

    # Seed random number generators
    rng = np.random.default_rng(settings.random_seed)
    nnx_rngs = nnx.Rngs(settings.random_seed)

    # 2. Dataset Loading & OpenVINO Embedding Extraction
    data = Banking77Data(settings.data)

    # 3. Model & Optimizer Initialization
    model = MLPClassifier(
        rngs=nnx_rngs,
        input_dim=settings.data.embedding_dim,
        num_classes=settings.data.num_classes,
        dropout_rate=0.1,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=settings.training.learning_rate),
    )

    # 4. Model Training & Epoch Checkpointing
    history = train(
        model=model,
        optimizer=optimizer,
        train_dataset=data.train_dataset,
        val_dataset=data.val_dataset,
        settings=settings.training,
        rng=rng,
    )

    # 5. Plot Training History Curves
    training_plot_paths = plot_training_history(history, settings.plotting)
    log.info("Saved training curves", paths=[str(p) for p in training_plot_paths])

    # 6. Checkpoint Selection & Matryoshka Evaluation
    best_val_metrics, matryoshka_results = run_evaluation(
        model=model,
        data=data,
        history=history,
        settings=settings,
    )

    # 7. Plot Matryoshka Performance Trade-Off Curves
    tradeoff_plot_paths = plot_matryoshka_performance(matryoshka_results, settings.plotting)
    log.info("Saved Matryoshka tradeoff plots", paths=[str(p) for p in tradeoff_plot_paths])

    log.info(
        "Assignment 5 pipeline completed successfully!",
        best_val_epoch=best_val_metrics["epoch"],
        best_val_accuracy=f"{best_val_metrics['val_accuracy'] * 100:.2f}%",
        full_dim_test_accuracy=f"{matryoshka_results[0]['test_accuracy'] * 100:.2f}%" if matryoshka_results else "N/A",
    )


if __name__ == "__main__":
    main()
