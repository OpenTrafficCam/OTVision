#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import yaml

REPOSITORIES = "repos"
REPOSITORY = "repo"
MYPY_REPOSITORY = "https://github.com/pre-commit/mirrors-mypy"
HOOKS = "hooks"
ADDITIONAL_DEPENDENCIES = "additional_dependencies"

CAPTURE_GROUP_URL = "url"
CAPTURE_GROUP_PACKAGE = "package"
CAPTURE_GROUP_VERSION = "version"


class AdditionalMypyDependency(ABC):

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> str:
        raise NotImplementedError


class Package(AdditionalMypyDependency):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str | None:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Package):
            return False
        return (self.name, self.version) == (other.name, other.version)


class TypeStubPackage(Package):
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def __init__(self, name: str, version: str | None) -> None:
        self._name = name
        self._version = version

    def serialize(self) -> str:
        return self.name


class NormalPackage(Package):
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def __init__(self, name: str, version: str | None) -> None:
        self._name = name
        self._version = version

    def serialize(self) -> str:
        if self.version:
            return self.name + "==" + self.version
        return self.name


@dataclass
class ExtraIndexUrl(AdditionalMypyDependency):
    url: str

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtraIndexUrl):
            return False
        return self.url == other.url

    def serialize(self) -> str:
        return f"--extra-index-url={self.url}"


class CustomDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super(CustomDumper, self).increase_indent(flow, False)


def parse_multiple_requirements_file(
    files: Iterable[Path],
) -> set[AdditionalMypyDependency]:
    packages = set()
    for _file in files:
        packages.update(parse_requirements_file(_file))
    return packages


def parse_requirements_file(requirements_file: Path) -> set[AdditionalMypyDependency]:
    """Parse requirements.txt and extract package names using regex."""
    with open(requirements_file, "r") as file:
        lines = file.readlines()

    packages = set()
    for line in lines:
        line = line.strip()
        if (
            line and not line.startswith("#") and line != "-r requirements.txt"
        ):  # Ignore empty lines, comments '-r requirements.txt'
            package_name = parse_requirement(line)
            if package_name:
                packages.add(package_name)

    return packages


pattern_package = re.compile(
    r"^(?!--extra-index-url)(?P<package>[a-zA-Z0-9_\-\.]+)(?:[<>=~!]+(?P<version>\S*))?"
)
pattern_extra_index_url = re.compile(r"^--extra-index-url\s+(?P<url>\S+)")


def parse_requirement(requirement_line: str) -> AdditionalMypyDependency | None:
    """Extract package name from a requirement line using regex."""
    # Regex pattern to capture the package name, ignoring version specifiers
    if match_extra_index_url := pattern_extra_index_url.match(requirement_line):
        return create_extra_index_url(
            match_extra_index_url.group(CAPTURE_GROUP_URL).strip()
        )

    match = pattern_package.match(requirement_line)
    if not match:
        return None

    package_name = match.group(CAPTURE_GROUP_PACKAGE).strip()
    if package_version := match.group(CAPTURE_GROUP_VERSION):
        package_version = package_version.strip()

    return create_package(name=package_name, version=package_version)


def create_extra_index_url(url: str) -> AdditionalMypyDependency:
    return ExtraIndexUrl(url)


def create_package(name: str, version: str | None) -> AdditionalMypyDependency:
    """Check if a type stub exists for a given package name and return it."""
    types_package_name = f"types-{name}"
    if __check_types_for_package_exists(types_package_name):
        return create_type_stub_package(name=types_package_name, version=version)

    # Some packages already provide type stubs with their package
    # If they don't pre-commit mypy won't fail
    return create_normal_package(name=name, version=version)


def __check_types_for_package_exists(package_name: str) -> bool:
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    return response.status_code == 200


def create_type_stub_package(
    name: str, version: str | None
) -> AdditionalMypyDependency:
    return TypeStubPackage(name=name, version=version)


def create_normal_package(name: str, version: str | None) -> AdditionalMypyDependency:
    return NormalPackage(name=name, version=version)


def serialize_packages(packages: Iterable[AdditionalMypyDependency]) -> list[str]:
    """Converts packages to a serializable format."""
    return sorted([package.serialize() for package in packages])


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

    additional_dependencies = parse_multiple_requirements_file(
        [requirements_file, requirements_dev_file]
    )
    serializable_dependencies = serialize_packages(additional_dependencies)
    precommit_config = read_precommit_file(precommit_file)
    updated_precommit_config = update_precommit_config(
        precommit_config, serializable_dependencies
    )
    save_precommit_config(updated_precommit_config, precommit_file)


if __name__ == "__main__":
    main()
