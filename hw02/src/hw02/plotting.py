import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
import jax.numpy as jnp
import numpy as np
import structlog
from sklearn.inspection import DecisionBoundaryDisplay

from .config import PlottingSettings, TrainingSettings
from .data import Spiral_Data
from .model import MLP, SklearnModelWrapper

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)
plt.rcParams["svg.fonttype"] = "none"


def plot_spiral_data(
    settings: PlottingSettings,
    data: Spiral_Data,
):
    log.info("Plotting Spiral")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Spiral Data")
    ax.set_xlabel("x")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    for i in range(data.num_samples_A):
        plt.scatter(data.x[i], data.y[i], s=10, color="blue")

    for i in range(data.num_samples_A, data.num_samples_A + data.num_samples_B):
        plt.scatter(data.x[i], data.y[i], s=10, color="red")

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = settings.output_dir / "spiral data.svg"
    pdf_path = settings.output_dir / "spiral data.pdf"
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))
    pass


def plot_spiral_decision_boundary(
    settings: PlottingSettings,
    data: Spiral_Data,
    model,
    title,
):
    log.info("Plotting Decision Boundary")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title(str(title))
    ax.set_xlabel("x")

    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    X = np.stack([data.x, data.y], axis=1)
    y = data.is_class_a

    wrapped_model = SklearnModelWrapper(model)
    wrapped_model.fit()

    DecisionBoundaryDisplay.from_estimator(
        wrapped_model,
        np.asarray(X),
        response_method="prediction",
        class_of_interest=0,  # P(t=1 | x)
        alpha=0.5,
        grid_resolution=100,
        cmap="RdBu",
        ax=ax,
        eps=10,
    )

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = settings.output_dir / f"{title}.svg"
    pdf_path = settings.output_dir / f"{title}.pdf"
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))


def plot_losses(
    losses_1,
    losses_2,
    training_settings=None,
    plotting_settings=None,
    loss_1_label="regular MLP loss",
    loss_2_label="fourier MLP loss",
    title="log(loss) vs iterations",
):
    training_settings = training_settings or TrainingSettings()
    plotting_settings = plotting_settings or PlottingSettings()
    fig, ax = plt.subplots(
        1, 1, figsize=plotting_settings.figsize, dpi=plotting_settings.dpi
    )
    losses_1 = np.asarray(losses_1)
    losses_2 = np.asarray(losses_2)
    iterations = np.arange(len(losses_1))
    ax.plot(iterations, np.log10(losses_1), label=loss_1_label)
    ax.plot(iterations, np.log10(losses_2), label=loss_1_label)

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log(loss)")
    ax.set_ylim(-6, 2)
    ax.set_xlim(0, len(losses_1))

    fig.tight_layout()

    plotting_settings.output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = plotting_settings.output_dir / f"{title}.svg"
    pdf_path = plotting_settings.output_dir / f"{title}.pdf"

    fig.savefig(svg_path)
    fig.savefig(pdf_path)

    plt.close(fig)
