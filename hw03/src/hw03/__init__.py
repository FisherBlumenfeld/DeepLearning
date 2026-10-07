import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import Classifier
from .training import train

EPOCHS = 5


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
    nnx_rng = nnx.Rngs(model_key)

    data = Data(rng=np_rng)  # imports the MNIST dataset thru sklearn

    model = Classifier(
        input_channels=1,
        layer_channels=[32, 64, 128],
        kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        strides=[1, 2, 2],
        num_classes=10,
        rngs=nnx_rng,
    )

    log.debug("Classifier Initialized")

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng, EPOCHS)
