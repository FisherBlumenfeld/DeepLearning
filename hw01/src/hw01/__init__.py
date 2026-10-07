import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import GaussianModel as GaussianModel
from .model import LinearModel, NNXGaussianModel, NNXLinearModel
from .plotting import plot_basis, plot_fit, plot_losses
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

    data_generating_model = LinearModel(
        weights=np_rng.integers(low=0, high=5, size=(settings.data.num_features)),
        bias=2,
    )
    log.debug("Data generating model", model=data_generating_model)

    data = Data(
        model=data_generating_model,
        rng=np_rng,
        num_features=settings.data.num_features,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )

    GaussModel = NNXGaussianModel(
        rngs=nnx.Rngs(params=model_key), num_gaussians=settings.training.num_gaussians
    )

    log.debug("Initial model", model=GaussModel.model)

    optimizer = nnx.Optimizer(
        GaussModel, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    losses = train(GaussModel, optimizer, data, settings.training, np_rng)

    log.debug("Trained sinusoidal model", model=GaussModel.model)

    plot_basis(GaussModel.model, settings.plotting, title="Trained sinusoidal model")
    plot_losses(losses, title="Trained sinusoidal model log(losses) vs iterations")
    plot_fit(GaussModel, data, settings.plotting, title="trained sinusoidal model")

    # linear model
    linear_model = NNXLinearModel(
        num_features=settings.data.num_features,
        rngs=nnx.Rngs(params=model_key),
    )

    linear_data = Data(
        model=data_generating_model,
        rng=np_rng,
        num_features=settings.data.num_features,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )
    linear_data.y = linear_data.generate_linear(np_rng)

    log.debug("Initial linear model", model=linear_model.model)

    if settings.data.num_features == 1:
        plot_fit(
            linear_model,
            linear_data,
            settings.plotting,
            title="trained linear model",
        )

    # print("calculated Mus: ", GaussModel.model.mus)
    # print("calculated Sigmas: ", GaussModel.model.sigmas)
    # print("calculated Weights: ", GaussModel.model.weights)
    # print("calculated Bias: ", GaussModel.model.bias)
