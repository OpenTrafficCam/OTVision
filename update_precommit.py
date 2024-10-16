#!/usr/bin/env python3

import re
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import requests
import yaml

REPOSITORIES = "repos"
REPOSITORY = "repo"
MYPY_REPOSITORY = "https://github.com/pre-commit/mirrors-mypy"
HOOKS = "hooks"
ADDITIONAL_DEPENDENCIES = "additional_dependencies"


class CustomDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super(CustomDumper, self).increase_indent(flow, False)


def get_type_stub_package(package_name: str) -> str:
    """Check if a type stub exists for a given package name and return it."""
    types_package_name = f"types-{package_name}"
    if __check_types_for_package_exists(types_package_name):
        return types_package_name
    else:
        # Some packages already provide type stubs with their package
        # If they don't pre-commit mypy won't fail
        return package_name


def __check_types_for_package_exists(package_name: str) -> bool:
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    return response.status_code == 200


def extract_package_name(requirement_line: str) -> str | None:
    """Extract package name from a requirement line using regex."""
    # Regex pattern to capture the package name, ignoring version specifiers
    pattern = re.compile(r"^([a-zA-Z0-9_\-\.]+)(?:[<>=~!]+\S*)?")
    match = pattern.match(requirement_line)
    if match:
        return match.group(1).strip()
    return None


def parse_requirements(requirements_file: Path) -> set[str]:
    """Parse requirements.txt and extract package names using regex."""
    with open(requirements_file, "r") as file:
        lines = file.readlines()

    packages = set()
    for line in lines:
        line = line.strip()
        if (
            line and not line.startswith("#") and line != "-r requirements.txt"
        ):  # Ignore empty lines, comments '-r requirements.txt'
            package_name = extract_package_name(line)
            if package_name:
                packages.add(package_name)

    return packages


def parse_multiple_requirements(files: Iterable[Path]) -> set[str]:
    packages: set[str] = set()
    for _file in files:
        packages.update(parse_requirements(_file))
    return packages


def retrieve_type_stubs(packages: Iterable[str]) -> list[str]:
    """Generate a list of type stubs for the given list of packages."""
    type_stubs = []
    for package in packages:
        if package_name := get_type_stub_package(package):
            type_stubs.append(package_name)
    return sorted(type_stubs)


def read_precommit_file(precommit_file: Path) -> dict:
    with open(precommit_file, "r") as stream:
        yaml_config = yaml.safe_load(stream)
    return yaml_config


def update_precommit_config(config: dict, type_stubs: list[str]) -> dict:
    updated_config = deepcopy(config)
    for repo in updated_config[REPOSITORIES]:
        if repo[REPOSITORY] == MYPY_REPOSITORY:
            repo[HOOKS][0][ADDITIONAL_DEPENDENCIES] = type_stubs
            break
    return updated_config


def save_precommit_config(config: dict, save_path: Path) -> None:
    with open(save_path, "w") as yaml_file:
        yaml.dump(
            data=config,
            stream=yaml_file,
            Dumper=CustomDumper,
            explicit_start=True,
            default_flow_style=False,
            sort_keys=False,
        )


def display_available_type_stubs(type_stubs: list[str]) -> None:
    if type_stubs:
        print("\nType stubs that can be added to your pre-commit configuration:")
        for stub in type_stubs:
            print(f"- {stub}")
    else:
        print("\n No type stubs to be added to your pre-commit configuration.")


def type_stubs_have_changed(actual: dict, to_compare: dict) -> bool:
    return actual != to_compare


def main() -> None:
    requirements_file = Path("requirements.txt")
    requirements_dev_file = Path("requirements-dev.txt")
    precommit_file = Path(".pre-commit-config.yaml")

    packages = parse_multiple_requirements([requirements_file, requirements_dev_file])
    type_stubs = retrieve_type_stubs(packages)
    precommit_config = read_precommit_file(precommit_file)
    updated_precommit_config = update_precommit_config(precommit_config, type_stubs)
    save_precommit_config(updated_precommit_config, precommit_file)


if __name__ == "__main__":
    main()
