# Reference Linear Regression Package

This package provides a reference implementation of a linear regression model implemented in Flax NNX and JAX. It serves as an architectural model for homework implementations in the course.

---

## Running the Model

Execute the package from the repository root:

```bash
just run example
```

This builds the dedicated virtual environment, installs dependencies, trains the model on synthetic data, logs evaluation metrics, and generates vector plots.

---

## Architecture and Execution Pipeline

The package executes the following stages:

1. **Synthetic Data Generation** (`data.py`): Generates synthetic linear data with additive Gaussian noise from a known ground-truth model.
2. **Model Initialization** (`model.py`): Instantiates an `NNXLinearModel` (subclassing `nnx.Module`) with weights and bias initialized as `nnx.Param`.
3. **Optimization** (`training.py`): Optimizes parameter values using JIT-compiled gradient steps (`nnx.jit`, `nnx.value_and_grad`) with Optax Adam.
4. **Parameter Comparison** (`plotting.py`): Compares learned model parameters against ground-truth data-generating parameters.
5. **Vector Visualization** (`plotting.py`): For single-feature models, generates publication-ready vector plots (`fit.svg` and `fit.pdf`) in `artifacts/` with `plt.rcParams["svg.fonttype"] = "none"`.

---

## Configuration

Settings are loaded from `src/example/config.toml` via `pydantic-settings`:

- `data.num_features`: Feature dimension.
- `data.num_samples`: Number of synthetic training points.
- `data.sigma_noise`: Standard deviation of additive Gaussian noise.
- `training.num_iters`: Number of gradient descent optimization steps.
- `training.learning_rate`: Adam optimizer learning rate.

### Log Level Configuration

Runtime log verbosity is managed via the `LOG_LEVEL` environment variable:

```bash
LOG_LEVEL=DEBUG just run example
```
