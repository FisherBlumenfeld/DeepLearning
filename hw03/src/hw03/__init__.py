import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import ResNetClassifier
from .training import train

EPOCHS = 15


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # Seed initialization
    key = jax.random.key(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))
    nnx_rng = nnx.Rngs(model_key)

    # 1. Load CIFAR-10 data
    data = Data(rng=np_rng)

    # 2. Initialize ResNet with 3 input channels for CIFAR-10 RGB
    model = ResNetClassifier(
        input_channels=3,
        stage_channels=(16, 32, 64),
        blocks_per_stage=(2, 2, 2),
        num_classes=10,
        num_groups=8,
        rngs=nnx_rng,
    )
    log.debug("ResNetClassifier Initialized")

    # 3. Configure Warmup Cosine Decay Schedule
    steps_per_epoch = int(np.ceil(data.num_train / settings.training.batch_size))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch  # 1 epoch linear warmup

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=settings.training.learning_rate,  # e.g., 1e-3 or 3e-3
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )

    # AdamW with cosine decay and light weight decay
    optimizer_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )
    optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)

    # 4. Train the model
    train(
        model=model,
        optimizer=optimizer,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )


if __name__ == "__main__":
    main()