import jax.numpy as jnp
from flax import nnx
from typing import Sequence, Union


class GroupNorm(nnx.Module):
    def __init__(
        self,
        num_features: int,
        num_groups: int = 16,
        epsilon: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if num_features % num_groups != 0:
            print("incompadible num_features and num_group values")

        self.num_features = num_features
        self.num_groups = num_groups
        self.epsilon = epsilon

        self.scale = nnx.Param(jnp.ones((1, 1, 1, num_features), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((1, 1, 1, num_features), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        N, H, W, C = x.shape
        G = self.num_groups
        channels_per_group = C // G

        x_grouped = x.reshape((N, H, W, G, channels_per_group))

        mean = jnp.mean(x_grouped, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x_grouped, axis=(1, 2, 4), keepdims=True)

        x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
        x_norm = x_norm.reshape((N, H, W, C))

        return x_norm * self.scale.value + self.bias.value


class ResidualBlock(nnx.Module):
    """
    Standard residual block with two 3x3 Conv2d layers and GroupNorm, including ID shortcut
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        num_groups: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = GroupNorm(
            num_features=out_features, num_groups=num_groups, rngs=rngs
        )

        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = GroupNorm(
            num_features=out_features, num_groups=num_groups, rngs=rngs
        )

        if stride != 1 or in_features != out_features:
            self.shortcut = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=(stride, stride),
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            self.shortcut_norm = GroupNorm(
                num_features=out_features, num_groups=num_groups, rngs=rngs
            )
        else:
            self.shortcut = None
            self.shortcut_norm = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            if self.shortcut_norm is not None:
                identity = self.shortcut_norm(identity)

        return nnx.relu(out + identity)


class ResNetClassifier(nnx.Module):
    """Residual network classifier for CIFAR-10.
    Accepts (N, 32, 32, 3) inputs and classifies into 10 classes.
    """

    def __init__(
        self,
        input_channels: int = 3,
        stage_channels: tuple[int, ...] = (16, 32, 64),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2),
        num_classes: int = 10,
        num_groups: int = 8,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        stem_channels = stage_channels[0]
        self.stem_conv = nnx.Conv(
            in_features=input_channels,
            out_features=stem_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.stem_norm = GroupNorm(
            num_features=stem_channels, num_groups=num_groups, rngs=rngs
        )

        blocks = []
        in_c = stem_channels

        for stage_idx, (out_c, num_b) in enumerate(
            zip(stage_channels, blocks_per_stage)
        ):
            for b_idx in range(num_b):
                stride = 2 if (stage_idx > 0 and b_idx == 0) else 1
                blocks.append(
                    ResidualBlock(
                        in_features=in_c,
                        out_features=out_c,
                        stride=stride,
                        num_groups=num_groups,
                        rngs=rngs,
                    )
                )
                in_c = out_c

        self.blocks = nnx.List(blocks)

        self.head = nnx.Linear(
            in_features=stage_channels[-1],
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = nnx.relu(x)

        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        return self.head(x)


class Conv2d(nnx.Module):
    """A convolutional layer inheriting nnx.Conv. Uses ReLU as activation function"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]],
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        return nnx.relu(x)


class BaselineClassifier(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        layer_channels: list[int],
        kernel_sizes: list[tuple[int, int]],
        strides: list[int],
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        num_layers = len(layer_channels)

        layers = []
        in_c = input_channels
        for i in range(num_layers):
            out_c = int(layer_channels[i])
            k_size = kernel_sizes[i]
            stride = strides[i]

            if isinstance(k_size, int):
                k_size = (k_size, k_size)
            if isinstance(stride, int):
                stride = [stride, stride]
            layers.append(
                Conv2d(
                    in_features=int(in_c),
                    out_features=out_c,
                    kernel_size=k_size,
                    strides=stride,
                    rngs=rngs,
                )
            )
            in_c = int(
                layer_channels[i]
            )  # ensure next layer's input matches previous layer's output

        self.layers = nnx.List(layers)

        self.head = nnx.Linear(
            in_features=layer_channels[-1],
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling

        logits = self.head(x)
        return logits
