import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Spiral_Data
from .logging import configure_logging
from .model import MLP, MLP_model, FourierMLP
from .plotting import plot_losses, plot_spiral_data, plot_spiral_decision_boundary
from .training import train


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.key(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))

    num_samples_A = settings.data.num_samples // 2
    data = Spiral_Data(
        rng=np_rng,
        num_samples_A=num_samples_A,
        num_samples_B=settings.data.num_samples - num_samples_A,
        sigma=settings.data.sigma_noise,
    )

    # regular MLP

    model = MLP(
        num_inputs=2,
        num_outputs=2,
        num_hidden_layers=3,
        hidden_layer_width=64,
        rngs=nnx.Rngs(params=model_key),
    )

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    MLP_losses = train(
        model=model,
        optimizer=optimizer,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
    )

    plot_spiral_data(settings.plotting, data)
    plot_spiral_decision_boundary(
        settings.plotting, data, model, title="MLP decision boundary"
    )

    # fourier MLP
    fourier_model = FourierMLP(
        num_inputs=2,
        num_outputs=2,
        num_hidden_layers=3,
        hidden_layer_width=32,
        rng=np_rng,
        rngs=nnx.Rngs(params=model_key),
        sigma=0.1,
        fourier_features=4,
    )

    fourier_optimizer = nnx.Optimizer(
        fourier_model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    fourier_losses = train(
        model=fourier_model,
        optimizer=fourier_optimizer,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
    )

    plot_spiral_decision_boundary(
        settings.plotting, data, fourier_model, title="Fourier MLP decision boundary"
    )
    plot_losses(MLP_losses, fourier_losses, settings.training, settings.plotting)
