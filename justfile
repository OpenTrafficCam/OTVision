# torch and ultralytics come from this extra, tests and mypy need them
with_inference := "uv run --group dev --extra inference_cpu"

# List the available recipes
default:
    @just --list

# Install git hooks, including pre-push (not installed by plain `pre-commit install`)
install-hooks:
    uv run --group dev pre-commit install --hook-type pre-commit --hook-type pre-push

test:
    {{ with_inference }} pytest

lint:
    uv run --group dev ruff check

fix:
    uv run --group dev ruff check --fix

format:
    uv run --group dev ruff format

typecheck:
    {{ with_inference }} mypy .
