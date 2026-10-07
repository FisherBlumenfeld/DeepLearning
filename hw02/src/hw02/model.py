from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin


import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class MLP_model:
    num_inputs: int
    num_outputs: int
    num_hidden_layers: int
    hidden_layer_width: int
    hidden_activation = nnx.gelu
    output_activation = nnx.identity


class MLP(nnx.Module):
    """Basic MLP model with dynamic layer count and width to approximate spiral data"""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_layer_width: int,
        rngs: nnx.Rngs,
        hidden_activation=nnx.gelu,
        output_activation=nnx.identity,
    ):
        self.layers = nnx.List([])
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        last_width = num_inputs
        for _ in range(num_hidden_layers):
            self.layers.append(nnx.Linear(last_width, hidden_layer_width, rngs=rngs))
            last_width = hidden_layer_width
        self.lastLayer = nnx.Linear(last_width, num_outputs, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        for layer in self.layers:
            x = self.hidden_activation(layer(x))
        return self.output_activation(self.lastLayer(x))

    def predict(self, X):
        X = jnp.asarray(X, dtype=jnp.float32)
        logits = self.__call__(X)
        return np.asarray(jnp.argmax(logits, axis=1))


class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """allows sklearn tools to work"""

    def __init__(self, model):
        self.model = model

    def fit(self):
        self.classes_ = np.array([0, 1])
        return self

    def prediction(self, X):
        X = jnp.asarray(X, dtype=jnp.float32)
        predictions = jax.nn.softmax(self.model(X), axis=-1)
        return np.asarray(predictions)

    def predict(self, X):
        return np.argmax(self.prediction(X), axis=-1)


class FourierMapping:
    def __init__(
        self,
        input_dim: int,
        sigma: float,
        rng: np.random.Generator,
        num_features: int,
    ):
        self.B = rng.normal(0, sigma, size=(num_features, input_dim))

    def __call__(self, x):
        projection = 2 * np.pi * x @ self.B.T

        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)], axis=-1)


class FourierMLP(MLP):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_layer_width: int,
        rng,
        rngs: nnx.Rngs,
        sigma: float,
        fourier_features: int,  # how many frequencies the fourier layer spits out
    ):
        self.fourier = FourierMapping(
            input_dim=num_inputs, sigma=sigma, rng=rng, num_features=fourier_features
        )

        super().__init__(
            num_inputs=2 * fourier_features,
            num_outputs=num_outputs,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_width=hidden_layer_width,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.fourier(x)
        x = jnp.asarray(x, dtype=jnp.float32)
        return super().__call__(x)
