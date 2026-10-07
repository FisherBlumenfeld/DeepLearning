import jax.numpy as jnp
from flax import nnx
from typing import Sequence, Union


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


class Classifier(nnx.Module):
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

        self.layers = layers

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
