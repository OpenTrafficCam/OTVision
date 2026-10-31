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

super-lint:
    docker run --rm -i --platform linux/amd64 \
        -e RUN_LOCAL=true \
        -e SAVE_SUPER_LINTER_SUMMARY=true \
        -e SUPER_LINTER_OUTPUT_DIRECTORY_NAME=.super-linter \
        -e SUPER_LINTER_SUMMARY_FILE_NAME=SUMMARY.md \
        --env-file .github/super-linter.env \
        --env-file .github/super-linter-fix.env \
        -v {{ justfile_directory() }}:/tmp/lint ghcr.io/super-linter/super-linter@sha256:c95c714f746edc70e54926a69e229c834ffcdec2450bd3475f7865164d749a56 # slim-v8.7.0
