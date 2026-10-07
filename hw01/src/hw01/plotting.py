import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import structlog

from .config import PlottingSettings, TrainingSettings
from .data import Data
from .model import GaussianModel, LinearModel, NNXLinearModel, NNXGaussianModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)
plt.rcParams["svg.fonttype"] = "none"


def compare_linear_models(a: LinearModel, b: LinearModel):
    """Prints a comparison of two linear models."""
    log.info("Comparing models", true=a, estimated=b)
    print("w,    w_hat")
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


def plot_fit(
    model: NNXLinearModel | NNXGaussianModel,
    data: Data,
    settings: PlottingSettings,
    title,
):
    """Plots the fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylim(np.amin(data.y), np.amax(data.y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 5000)
    xs = xs[:, np.newaxis]
    ax.plot(
        xs,
        np.squeeze(model(jnp.asarray(xs))),
        "-",
        color="blue",
        label="fitted curve",
    )
    ax.plot(np.squeeze(data.x), data.y, "o", color="blue", label="noisy data")
    if not isinstance(model, NNXLinearModel):
        ax.plot(
            xs,
            np.sin(2 * np.pi * xs),
            color="red",
            linestyle="--",
            label="real sine curve",
        )
    ax.legend(fontsize=8)

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = settings.output_dir / str(title + ".svg")
    pdf_path = settings.output_dir / str(title + ".pdf")
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))


def plot_basis(
    model: GaussianModel, settings: PlottingSettings, title="bases for trained model"
):
    log.info("Plotting bases")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylim(0, 1.25)
    muMin = model.mus[0][0]
    muMax = model.mus[0][0]
    for i in range(model.size):
        if model.mus[0][i] < muMin:
            muMin = model.mus[0][i]
        elif model.mus[0][i] > muMax:
            muMax = model.mus[0][i]
    ax.set_xlim(muMin - 1, muMax + 1)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)
    x = np.linspace(muMin - 1, muMax + 1, int(500 * (muMax - muMin)))
    for i in range(model.size):
        yi = np.exp(-((x - model.mus[0][i]) ** 2) / (2 * model.sigmas[0][i] ** 2))

        plt.plot(x, yi)

    plt.tight_layout()
    svg_path = settings.output_dir / str(title + ".svg")
    pdf_path = settings.output_dir / str(title + ".pdf")
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))


def plot_losses(
    losses,
    training_settings=None,
    plotting_settings=None,
    title="log(loss) vs iterations",
):
    training_settings = training_settings or TrainingSettings()
    plotting_settings = plotting_settings or PlottingSettings()
    fig, ax = plt.subplots(
        1, 1, figsize=plotting_settings.figsize, dpi=plotting_settings.dpi
    )
    losses = np.asarray(losses)
    iterations = np.arange(len(losses))
    print("losses: ", losses[0:10])
    ax.plot(iterations, np.log10(losses))

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log(loss)")
    ax.set_ylim(-3.5, -1)
    ax.set_xlim(0, len(losses))

    fig.tight_layout()

    plotting_settings.output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = plotting_settings.output_dir / f"{title}.svg"
    pdf_path = plotting_settings.output_dir / f"{title}.pdf"

    fig.savefig(svg_path)
    fig.savefig(pdf_path)

    plt.close(fig)
