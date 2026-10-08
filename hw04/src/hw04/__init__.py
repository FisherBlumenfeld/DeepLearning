import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import ResNetClassifier, BaselineClassifier
from .training import train

EPOCHS = 15

def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    key = jax.random.key(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))
    nnx_rng = nnx.Rngs(model_key)

    data = Data(rng=np_rng)
    non_augmenting_data = Data(rng=np_rng, augment=False)

    baseline_model = BaselineClassifier(
        input_channels=1,
        layer_channels=[32, 64, 128],
        kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        strides=[1, 2, 2],
        num_classes=10,
        rngs=nnx_rng,
    )

    ResNetModel = ResNetClassifier(
        input_channels=3,
        stage_channels=(16, 32, 64),
        blocks_per_stage=(2, 2, 2),
        num_classes=10,
        num_groups=8,
        rngs=nnx_rng,
    )
    log.debug("ResNetClassifier Initialized")

    steps_per_epoch = int(np.ceil(data.num_train / settings.training.batch_size))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = 1 * steps_per_epoch 

    cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=settings.training.learning_rate,  
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )

    optimizer_tx_cosine = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cosine_schedule, weight_decay=1e-4),
    )

    optimizer_tx_fixed_rate = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=settings.training.learning_rate),
    )

    resnet_optimizer_cosine = nnx.Optimizer(ResNetModel, optimizer_tx_cosine, wrt=nnx.Param)
    baseline_optimizer = nnx.Optimizer(baseline_model, optimizer_tx_fixed_rate, wrt=nnx.Param)
    resnet_optimizer_fixed = nnx.Optimizer(ResNetModel, optimizer_tx_fixed_rate, wrt=nnx.Param)

    resnet_acc = train( # regular residual CNN
        model=ResNetModel,
        optimizer=resnet_optimizer_cosine,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )

    baseline_acc = train(# • Baseline CNN (from Assignment 3) vs. Residual CNN.
        model=ResNetModel,
        optimizer=baseline_optimizer,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )

    no_augment_acc = train( # • No augmentation vs. data augmentation.
        model=ResNetModel,
        optimizer=optimizer_tx_cosine,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=False,
    )

    fixed_acc = train( # • Fixed learning rate vs. cosine decay schedule
        model=ResNetModel,
        optimizer=resnet_optimizer_fixed,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )

    print("Ablation table")
    print(f"regular resnet test accuracy {resnet_acc * 100:.2f}")
    print(f"baseline CNN test accuracy {baseline_acc * 100:.2f}")
    print(f"non-augmented training data resnet test accuracy {no_augment_acc * 100:.2f}")
    print(f"fixed learning rate resnet test accuracy {fixed_acc * 100:.2f}")
