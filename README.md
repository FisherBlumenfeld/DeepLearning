# ECE472 Starter Repository

This repository provides the development environment, baseline templates, and submission tooling for programming assignments in **ECE472 (Deep Learning)** at The Cooper Union. It uses a pure-Python workflow managed with `uv`, task automation via `just`, modular Flax NNX / JAX package templates, and an automated HTML submission generator with embedded vector SVG plots.

---

> **Note on Repository Usage and Portfolio Presentation**  
> This document functions as an operational guide for setup, assignment lifecycle, and submission workflows during the semester. If you make this repository public as part of your technical portfolio, you are encouraged to replace this root `README.md` with a project overview that documents your implementations, experimental results, loss curves, and architectural comparisons across the coursework.

---

## Platform Requirements

All assignments are designed for local CPU execution via `jax[cpu]`. Supported operating systems:

- **Linux**: Supported natively out of the box.
- **macOS (Apple Silicon & Intel)**: Supported natively.
- **Windows**: **WSL2 (Windows Subsystem for Linux 2) is required.** JAX does not distribute native Windows wheels. Install WSL2 via PowerShell (`wsl --install`), launch an Ubuntu shell, clone the repository within WSL, and follow the Linux instructions below.

---

## Environment Setup

### 1. Install Required Tooling

The workflow requires three system-level tools:

1. **`uv`** (Python package and environment manager):
   - macOS / Linux / WSL2:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - macOS (Homebrew): `brew install uv`
   - Additional installation methods: [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. **`just`** (Command runner):
   - macOS: `brew install just`
   - Linux / WSL2: `sudo apt install just` or `cargo install just`
   - Additional installation methods: [just documentation](https://just.systems/man/en/pre-built-binaries.html).

3. **`pango`** (Native C library for automated PDF compilation via `just submission`):
   - macOS (Homebrew): `brew install pango`
   - Linux / WSL2: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0`
### 2. Configure Local Git Identity

Configure your full name and Cooper Union email locally within this repository:

```bash
git config user.name "Your Full Name"
git config user.email "your.name@cooper.edu"
```
Assignment scaffolding (`just new`) and submission compilation (`just html` / `just pdf`) use these settings to attribute assignments in `pyproject.toml` and watermark generated submission reports with running headers containing your name and email on every page. The build pipeline will reject submissions if placeholder values or non-Cooper email addresses are detected.
*(Optional)* **Pre-commit Hooks**: Install pre-commit hooks to automatically format and lint code on commit:
```bash
uvx pre-commit install
```

---

## Assignment Lifecycle

Each assignment follows a standardized workflow:

### 1. Scaffold a New Assignment
```bash
just new hw01
```
Generates a new, independent Python package in `hw01/` via Copier with typed configuration (`config.toml`), synthetic data loading, Flax NNX module scaffolding, training loop, and SVG plotting routines. The command immediately executes the scaffolded project to verify the environment.

### 2. Implement and Run
```bash
just run hw01
```
Executes the assignment entry point (`hw01:main`) inside its dedicated virtual environment. Training metrics stream to stdout, and output figures are written to `hw01/artifacts/` as vector SVGs.

To run with debug logging:
```bash
LOG_LEVEL=DEBUG just run hw01
```

### 3. Check Code Quality and Types
```bash
just check hw01
```
Runs code formatting checks (`ruff format --check`), linter diagnostics (`ruff check`), and static type analysis (`ty check`).

### 4. Compile Submission Reports (HTML & PDF)
```bash
# Generate both HTML and PDF submission reports:
just submission hw01
```

Automatically formats source files with `ruff`, verifies static type safety with `ty`, and compiles submission documents in the dedicated, git-ignored `submissions/` directory (`submissions/hw01.html` and `submissions/hw01.pdf`).

The submission report includes:
- Deterministic module ordering (`pyproject.toml`, `config.toml`, `README.md`, followed by package source files and output plots).
- Running headers on every printed/PDF page containing your student name, Cooper Union email, homework title, and page numbering.
- Syntax-highlighted source code with line numbering.
- Collapsible file sections for interactive grading review.
- Inline vector SVG figures with selectable, scalable text.
### 5. Digital and Hard-Copy Submission
- **Digital Submission**: Submit the generated `submissions/hw01.html` through the course submission portal.
- **Printed Hard Copy**: Print the generated `submissions/hw01.pdf` for in-person physical submission.
---

## Engineering Conventions

### Adding Dependencies
Each homework directory is an independent Python package. Add external dependencies using `uv add`:

```bash
# From the repository root:
uv add scikit-learn --project hw01

# Or from within the homework directory:
cd hw01 && uv add scikit-learn
```

Always use `uv add` instead of `pip install` to ensure `pyproject.toml` and `uv.lock` remain synchronized.

### Typed Configuration (`config.toml`)
Settings are declared via `pydantic-settings` models in `config.py` and loaded from `config.toml`:

```toml
debug = false
random_seed = 31415

[data]
num_features = 1
num_samples = 50

[training]
batch_size = 16
learning_rate = 0.1
```

### Vector Plotting Standards
When creating figures with Matplotlib, save vector SVG files with text preserved as text elements rather than paths:

```python
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

fig, ax = plt.subplots()
ax.plot(x, y)
plt.savefig("artifacts/my_plot.svg")
plt.close(fig)
```

### Structured Logging
Assignments use `structlog` for structured, key-value console logging:

```python
import structlog

log = structlog.get_logger()
log.info("Training started", batch_size=16, lr=0.01)
log.debug("Model state", weights=w)
```

---

## Hosting Your Repository on GitHub

To track your commit history remotely and host your repository on GitHub:

1. Create a private repository on [GitHub](https://github.com) (do not initialize with a README, `.gitignore`, or license).
2. Configure your remote origin and push the `main` branch:
   ```bash
   git remote add origin <YOUR_REPOSITORY_URL>
   git push -u origin main
   ```
