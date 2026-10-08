import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import BaselineClassifier, ResNetClassifier
from .training import train

EPOCHS = 15


def make_resnet(rngs: nnx.Rngs) -> ResNetClassifier:
    return ResNetClassifier(
        input_channels=3,
        stage_channels=(16, 32, 64),
        blocks_per_stage=(2, 2, 2),
        num_classes=10,
        num_groups=8,
        rngs=rngs,
    )


def make_baseline(rngs: nnx.Rngs) -> BaselineClassifier:
    return BaselineClassifier(
        input_channels=3,
        layer_channels=[32, 64, 128],
        kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        strides=[1, 2, 2],
        num_classes=10,
        rngs=rngs,
    )


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    key = jax.random.key(settings.random_seed)
    data_key, run_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.asarray(jax.random.key_data(data_key)))

    data = Data(rng=np_rng)

    # Calculate step budgets for schedules
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

    optimizer_tx_fixed = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=settings.training.learning_rate, weight_decay=1e-4),
    )

    # Run 1: Anchor Model (Residual CNN + Augmentation + Cosine Schedule)
    k1, run_key = jax.random.split(run_key)
    print("\n" + "=" * 50)
    print("Run 1/4: Anchor (ResNet + Aug + Cosine)")
    print("=" * 50)
    m1 = make_resnet(nnx.Rngs(k1))
    opt1 = nnx.Optimizer(m1, optimizer_tx_cosine, wrt=nnx.Param)
    res_anchor = train(
        model=m1,
        optimizer=opt1,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )[1]
    resnet_acc = (
        res_anchor["val_acc"][-1] if isinstance(res_anchor, dict) else res_anchor
    )

    # Run 2: Axis 1 - Architecture (Baseline CNN + Aug + Cosine)
    k2, run_key = jax.random.split(run_key)
    print("\n" + "=" * 50)
    print("Run 2/4: Baseline CNN (Baseline + Aug + Cosine)")
    print("=" * 50)
    m2 = make_baseline(nnx.Rngs(k2))
    opt2 = nnx.Optimizer(m2, optimizer_tx_cosine, wrt=nnx.Param)
    res_baseline = train(
        model=m2,
        optimizer=opt2,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )[1]
    baseline_acc = (
        res_baseline["val_acc"][-1] if isinstance(res_baseline, dict) else res_baseline
    )

    # Run 3: Axis 2 - Augmentation (ResNet + NO Aug + Cosine)
    k3, run_key = jax.random.split(run_key)
    print("\n" + "=" * 50)
    print("Run 3/4: No Augmentation (ResNet + No Aug + Cosine)")
    print("=" * 50)
    m3 = make_resnet(nnx.Rngs(k3))
    opt3 = nnx.Optimizer(m3, optimizer_tx_cosine, wrt=nnx.Param)
    res_noaug = train(
        model=m3,
        optimizer=opt3,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=False,
    )[1]
    no_augment_acc = (
        res_noaug["val_acc"][-1] if isinstance(res_noaug, dict) else res_noaug
    )

    # Run 4: Axis 3 - Schedule (ResNet + Aug + Fixed LR)
    k4, run_key = jax.random.split(run_key)
    print("\n" + "=" * 50)
    print("Run 4/4: Fixed LR (ResNet + Aug + Fixed LR)")
    print("=" * 50)
    m4 = make_resnet(nnx.Rngs(k4))
    opt4 = nnx.Optimizer(m4, optimizer_tx_fixed, wrt=nnx.Param)
    res_fixed = train(
        model=m4,
        optimizer=opt4,
        data=data,
        settings=settings.training,
        np_rng=np_rng,
        num_epochs=EPOCHS,
        augment=True,
    )[1]
    fixed_acc = res_fixed["val_acc"][-1] if isinstance(res_fixed, dict) else res_fixed

    # summary table
    print("\n" + "=" * 60)
    print("                     ABLATION SUMMARY TABLE")
    print("=" * 60)
    print(f"1. Anchor (ResNet, Aug, Cosine):       {resnet_acc * 100:.2f}%")
    print(f"2. Baseline Architecture (Baseline):   {baseline_acc * 100:.2f}%")
    print(f"3. No Augmentation:                    {no_augment_acc * 100:.2f}%")
    print(f"4. Fixed Learning Rate:                {fixed_acc * 100:.2f}%")
    print("=" * 60 + "\n")
