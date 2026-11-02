# Dependency Management

OTVision's dependencies are managed with `uv`.

- Dependencies are declared in `pyproject.toml`. Runtime dependencies are in
  `[project] dependencies`, the inference libraries in `[project.optional-dependencies]`
  and development tools in the `dev` group of `[dependency-groups]`.
- The locked versions are kept in `uv.lock` for reproducible environments on Windows,
  macOS and Linux.
- `exclude-newer = "3 days"` in `[tool.uv]` only lets uv resolve releases that are at
  least three days old.

## Development environment

```sh
uv sync --extra inference_cpu
```

This installs the runtime and development requirements into a virtual environment.

## Adding or updating requirements

Pin versions exactly (`==`), like the existing entries:

```sh
uv add "<mypackage>==<version>"                 # runtime dependency
uv add --group dev "<mypackage>==<version>"     # development dependency
```

For updating packages:

```sh
# update all packages
uv lock --upgrade

# update a specific package
uv lock --upgrade-package <mypackage>
```

## Type stubs

mypy runs from the project environment, so third-party type stubs are declared in the
`dev` group, e.g. `types-requests`. Only add a stub if the library does not ship its
own types (no `py.typed` marker or `*-stubs` directory in the installed package), and
only from a source you trust: typeshed's stubs are published as `types-<library>`, and
anyone can register a `types-*` name on PyPI. Pin it, as with any other dependency.
