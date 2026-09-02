set shell := ["bash", "-c"]

# Lists available commands.
default:
    @just --list

# Creates a new homework assignment from the template if it doesn't exist,
# then runs it.
#
# Usage:
#   just new hw01
new hw_name:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d "{{hw_name}}" ]; then
        GIT_NAME="$(git config user.name || true)"
        GIT_EMAIL="$(git config user.email || true)"
        if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ] || [ "$GIT_NAME" = "Student" ] || [ "$GIT_EMAIL" = "noreply@cooper.edu" ]; then
            echo "❌ Error: Git user.name and user.email must be configured before scaffolding homework." >&2
            echo "" >&2
            echo "Please set your full name and Cooper Union email in this repository:" >&2
            echo "    git config user.name \"Your Full Name\"" >&2
            echo "    git config user.email \"your.name@cooper.edu\"" >&2
            echo "" >&2
            exit 1
        fi
        echo "==> Creating new homework '{{hw_name}}' for $GIT_NAME <$GIT_EMAIL>..."
        uvx copier copy --defaults \
            --data project_name="{{hw_name}}" \
            --data author_name="$GIT_NAME" \
            --data author_email="$GIT_EMAIL" \
            hw-template "{{hw_name}}"
        just run "{{hw_name}}"
    else
        echo "==> Homework '{{hw_name}}' already exists. Skipping creation."
    fi

# Runs a specific homework assignment.
#
# Usage:
#   just run hw01
run hw_name:
    @echo "==> Running homework '{{hw_name}}'..."
    @uv run --directory {{hw_name}} {{hw_name}}

# Evaluates a specific homework assignment (for assignments configured with an evaluation script).
#
# Usage:
#   just eval hw03
eval hw_name:
    @echo "==> Evaluating homework '{{hw_name}}'..."
    @uv run --directory {{hw_name}} {{hw_name}}-evaluate
# Checks formatting, linting, and types for a specific homework assignment.
#
# Usage:
#   just check hw01
check hw_name:
    @echo "==> Syncing environment for '{{hw_name}}'..."
    @uv sync --project {{hw_name}} --quiet
    @echo "==> Checking formatting with ruff..."
    @uvx ruff format --check {{hw_name}}
    @echo "==> Linting with ruff..."
    @uvx ruff check {{hw_name}}
    @echo "==> Type checking with ty..."
    @uvx ty check --project {{hw_name}} --python {{hw_name}}/.venv
# Automatically formats source code with ruff and enforces ty type checking.
#
# Usage:
#   just submission hw01
submission hw_name:
    @uv run scripts/build_submission.py {{hw_name}} --pdf

# Generates a self-contained HTML submission report in submissions/<hw_name>.html.
#
# Usage:
#   just html hw01
html hw_name:
    @uv run scripts/build_submission.py {{hw_name}}

# Generates HTML and PDF submission reports in submissions/<hw_name>.pdf via WeasyPrint.
#
# Usage:
#   just pdf hw01
pdf hw_name:
    @uv run scripts/build_submission.py {{hw_name}} --pdf
