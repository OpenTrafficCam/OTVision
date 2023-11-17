import os
import shutil
import zipfile
import yaml
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from yaml.loader import SafeLoader
from datetime import datetime


class File(ABC):
    @abstractmethod
    def source(self) -> Path:
        pass

    @abstractmethod
    def destination(self) -> Path:
        pass


@dataclass(frozen=True)
class SameNameFile(File):
    _path: Path

    def source(self) -> Path:
        return self._path

    def destination(self) -> Path:
        return self._path


@dataclass(frozen=True)
class DifferentNameFile(File):
    _source: Path
    _destination: Path

    def source(self) -> Path:
        return self._source

    def destination(self) -> Path:
        return self._destination


ENCODING: str = "UTF-8"


@dataclass(frozen=True)
class CliArguments:
    package_version: str


class CliArgumentParser:
    """Build command line interface argument parser.

    Acts as a wrapper to `argparse.ArgumentParser`.

    Args:
        arg_parser (ArgumentParser, optional): the argument parser.
            Defaults to ArgumentParser("build").
    """

    def __init__(self, arg_parser: ArgumentParser = ArgumentParser("build")) -> None:
        self._parser = arg_parser
        self._setup()

    def _setup(self) -> None:
        """Sets up the argument parser by defining the command line arguments."""
        self._parser.add_argument(
            "--package_version",
            type=str,
            help="Version of the package. Defaults to 0.0",
            required=False,
            default="0.0",
        )

    def parse(self) -> CliArguments:
        """Parse and checks for cli arg

        Returns:
            CliArguments: _description_
        """
        args = self._parser.parse_args()
        return CliArguments(args.package_version)


def collect_files(
        base_path: Path,
        file_extension: str = ".py",
) -> list[File]:
    """Collect all files in the file tree.

    Args:
        base_path (Path): base path to start searching files
        file_extension (str, optional): only files with this extension will be
            collected. Defaults to ".py".

    Returns:
        list[Path]: list of collected files.
    """
    paths: list[File] = []
    for folder_name, subfolders, file_names in os.walk(base_path):
        for file_name in file_names:
            if file_name.endswith(file_extension):
                file_path = Path(folder_name, file_name)
                paths.append(SameNameFile(file_path))
    return paths


def zip_output_folder(input_folder: Path, zip_file: Path) -> None:
    """Zip a whole directory into a zip file. The directory structure (subdirectories)
        will remain in the zip file.

    Args:
        input_folder (Path): folder to zip
        zip_file (Path): zipped folder as file
    """
    with ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as output:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = Path(root, file)
                base_path = Path(input_folder, ".")
                output_path = file_path.relative_to(base_path)
                output.write(file_path, output_path)


def clean_directory(directory: Path) -> None:
    """Clean the given directory.

    Args:
        directory (Path): directory to clean.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Configuration:
    """Configuration for deployment. It contains all files required on the given
    platform (suffix)
    """

    _package_version: str
    _package_path: Path
    _files: list[File]
    _additional_files: list[File]
    _suffix: str
    _additional_requirements: list[File]

    def create_zip(
            self, base_file_name: str, output_directory: Path, temp_directory: Path
    ) -> None:
        """Create a zip file with the given name in the given output directory.
        Store temporal artifacts in the temp_folder.

        Args:
            base_file_name (str): base name of the zip file.
            output_directory (Path): directory to store the zip file in.
            temp_directory (Path): directory for temporal artifacts.
        """
        clean_directory(build_path)
        zip_file = Path(output_directory, self._build_file_name(base_file_name))
        self._copy_to_output_directory(temp_directory)
        self._apply_additional_requirements(temp_directory)
        if self._package_version != "":
            self._update_version_file_in(temp_directory)
        zip_output_folder(temp_directory, zip_file)
        clean_directory(build_path)

    def _build_file_name(self, base_file_name: str) -> str:

        if self._package_version == "nightly":
            dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
            package_version = f"nightly-{dt_string}"
        else:
            package_version = self._package_version
        return f"{base_file_name}-{self._suffix}-{package_version}.zip"

    def _copy_to_output_directory(self, output_directory: Path) -> None:
        files = collect_files(base_path=self._package_path, file_extension=".py")
        self._copy_files(files, output_directory=output_directory)

        self._copy_files(self._files, output_directory=output_directory)
        self._copy_files(self._additional_files, output_directory=output_directory)

    def _copy_files(self, files: list[File], output_directory: Path) -> None:
        """Copies all files into the output directory. The directory structure will
        remain.

        Args:
            files (list[Path]): files to copy.
            output_directory (Path): directory to copy the files to.
        """
        for file in files:
            output_file = Path(output_directory, file.destination())
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file.source(), output_file)

    def _update_version_file_in(self, directory: Path) -> None:
        version_file = Path(directory, self._package_path, "version.py")
        if self._package_version == "nightly":
            dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
            content = f'__version__ = "{dt_string}"'
        else:
            content = f'__version__ = "{self._package_version}"'

        with open(version_file, "wt", encoding=ENCODING) as file:
            file.write(content)

    def _apply_additional_requirements(self, directory: Path) -> None:
        if (self._additional_requirements[0] != ""):
            for additional_requirements in self._additional_requirements:
                requirements_file = Path(directory, "requirements.txt")
                requirements = self._read_requirements(requirements_file)
                with open(requirements_file, "w") as output:
                    output.write("--extra-index-url " + additional_requirements + "\n")
                    output.write(requirements)

    def _read_requirements(self, requirements_file: Path) -> str:
        """Read content of existing requirements file.

        Args:
            requirements_file (Path): path to requirements.txt

        Returns:
            str: content of requirements.txt
        """
        with open(requirements_file, "r") as requirements:
            return requirements.read()


cli_args = CliArgumentParser().parse()

build_path = Path("build")
distribution_path = Path("dist")

clean_directory(distribution_path)

with open("config.yml") as f:
    configurationsDic = yaml.load(f, Loader=SafeLoader)

configurations = []
files = []
for file in configurationsDic["base_files"]:
    files.append(SameNameFile(Path(file)))

for data in configurationsDic["configurations"]:
    additional_files = []
    for file in data["files"]:
        if "destination" in file.keys():
            additional_files.append(
                DifferentNameFile(Path(file["source"]), Path(file["destination"])))
        else:
            additional_files.append(SameNameFile(Path(file["source"])))
    output_file_name = data["output_file_name"]
    package_version = cli_args.package_version
    package_path = Path(data["package_path"])
    suffix = data["suffix"]
    additional_requirements = data["additional_requirements"]
    configuration = Configuration(
        package_version,
        package_path,
        files,
        additional_files,
        suffix,
        additional_requirements,
    )
    configurations.append(configuration)

for configuration in configurations:
    configuration.create_zip(
        output_file_name,
        output_directory=distribution_path,
        temp_directory=build_path,
    )
