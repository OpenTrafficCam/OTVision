#!/usr/bin/env python
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import copy2, move, rmtree
from typing import Any, Literal

import torch
import yaml
from fire import Fire
from ultralytics import YOLO

PATTERN_MODEL_NAME = (
    r"^(?P<core>.*?)(?:_(?P<imgsz_prefix>imgsz)(?P<digits>[0-9]+))?\.pt$"
)
GROUP_CORE = "core"
GROUP_DIGITS = "digits"
TEMP_FOLDER = Path.home() / ".yolo_exporter_temp"
AVAILABLE_EXPORT_TYPES = {".engine", ".onnx", ".mlpackage"}


class ParseError(Exception):
    pass


class ExportFormat(StrEnum):
    ENGINE = "engine"
    ONNX = "onnx"
    COREML = "coreml"


class Quantization(StrEnum):
    INT8 = "int8"
    FP16 = "fp16"


@dataclass
class ModelInfo:
    """Provides information to the model to be exported.

    Attributes:
        original_path(Path): the path to the model weights pt file.
        temp_path (Path): the path to the temporary weights pt file.
        core (str): the model core name without the image size quantization suffix.
        imagesize (int | None): the image size. None if no image size is specified in
            model name.
    """

    original_path: Path
    temp_path: Path
    core: str
    imagesize: int | None


@dataclass
class ModelExportSpecification:
    """Holds all relevant information for model export.

    Attributes:
        model_info (ModelInfo): Information about the model's weights file.
        formats (list[ExportFormat]): The desired export formats
            (e.g., ONNX, CoreML, or TensorRT engine).
        quantization (Quantization | None): Quantization type (e.g., INT8, FP16),
            or None if not specified.
    """

    model_info: ModelInfo
    formats: list[ExportFormat]
    quantization: Quantization | None = None

    @property
    def imagesize(self) -> int | None:
        """Gets the model's image size.

        Returns:
            int | None: The image size, or None if unspecified.
        """

        return self.model_info.imagesize

    @property
    def model_path(self) -> Path:
        """Gets the path to the model weights file.

        Returns:
            Path: The path to the model's weights file.
        """

        return self.model_info.original_path

    def generate_file_stem(self) -> Path:
        """Generates a new file stem for the exported model
        based on its core, image size, and quantization.

        Returns:
            Path: The new file stem for the model export.
        """

        new_name = (
            f"{self.model_info.core}{self.__image_size_suffix()}"
            f"{self.__quantization_suffix()}"
        )
        return self.model_info.original_path.parent / new_name

    def __image_size_suffix(self) -> str:
        if self.imagesize is not None:
            return f"_imgsz{self.imagesize}"
        return ""

    def __quantization_suffix(self) -> str:
        if self.quantization is not None:
            return f"_{self.quantization}"
        return ""


class ModelExportSpecificationParser:
    """Parses the model export specification from input arguments."""

    def parser(
        self,
        model: Path,
        formats: list[str],
        quantization: str | None = None,
    ) -> "ModelExportSpecification":
        """Parses input arguments to create a ModelExportSpecification.

        Args:
            model (Path): The path to the model weights file.
            formats (list[str]): Desired export formats
                (e.g., "onnx", "engine", "coreml").
            quantization (str | None): Quantization type, if any (e.g., "int8", "fp16").

        Returns:
            ModelExportSpecification: The constructed export specification object.
        """

        model_info = self.__parse_model_info(pt_model_path=model)
        export_formats = [ExportFormat(_format) for _format in formats]

        return ModelExportSpecification(
            model_info=model_info,
            formats=export_formats,
            quantization=Quantization(quantization) if quantization else None,
        )

    def __parse_model_info(self, pt_model_path: Path) -> ModelInfo:
        match = re.match(PATTERN_MODEL_NAME, pt_model_path.name)
        if match:
            core = match.group(GROUP_CORE)
            if (raw_imagesize := match.group(GROUP_DIGITS)) is not None:
                imagesize = int(raw_imagesize)
            else:
                imagesize = None

            original_path = Path(pt_model_path)
            temp_path = TEMP_FOLDER / original_path.name
            return ModelInfo(
                original_path=original_path,
                temp_path=temp_path,
                core=core,
                imagesize=imagesize,
            )
        raise ParseError(f"Unable to parse model name '{pt_model_path.name}'.")


class PreExportAction:
    """Performs pre-export actions such as creating a temporary folder for the model.

    1. Creates a temporary folder where all model exports are located.
    2. Copies the model to be exported to the temporary folder.
    3. Returns an updated specification with the new model location.
    """

    def execute(self, spec: ModelExportSpecification) -> None:
        """Executes pre-export actions to prepare the model for export.

        1. Creates a temporary folder where all model exports are located.
        2. Copies the model to be exported to the temporary folder.
        3. Returns an updated specification with the new model location.

        Args:
            spec (ModelExportSpecification): The model export specification.

        Returns:
            ModelExportSpecification: Updated specification with a new model location.
        """

        self.__create_temp_folder(TEMP_FOLDER)
        self.__copy_model_to_temp_folder(
            src=spec.model_info.original_path, dst=spec.model_info.temp_path
        )

    def __create_temp_folder(self, temp_folder: Path) -> Path:
        try:
            temp_folder.mkdir(exist_ok=False)
        except FileExistsError:
            print(f"Temporary folder {temp_folder} already exists. Clearing it.")
            rmtree(temp_folder)
            temp_folder.mkdir(exist_ok=False)

        return temp_folder

    def __copy_model_to_temp_folder(self, src: Path, dst: Path) -> None:
        copy2(src=src, dst=dst)


class PostExportAction:
    """Performs post-export actions.

    1. Moves exported models to their original location, that is, the parent folder of
        original weights pt file.
    2. Removes the temporary folder.
    """

    def execute(self, spec: ModelExportSpecification) -> None:
        """Executes post-export operations.

        1. Moves exported models to their original location, that is,
            the parent folder of original weights pt file.
        2. Removes the temporary folder.

        Args:
            spec (ModelExportSpecification): The model export specification.
        """

        self.__remove_pt_model_from_temp_folder(pt_model=spec.model_info.temp_path)
        self.__move_exported_models_to_original_location(spec=spec)
        self.__remove_temp_folder(temp_folder=TEMP_FOLDER)

    def __remove_pt_model_from_temp_folder(self, pt_model: Path) -> None:
        if self.__is_in_temp_folder(file=pt_model):
            pt_model.unlink()
            return
        print(f"File '{pt_model}' is not in temp folder. Skipping removal.")

    def __move_exported_models_to_original_location(
        self, spec: ModelExportSpecification
    ) -> None:
        for exported_model in TEMP_FOLDER.iterdir():
            if self.__is_model(exported_model):
                dst = Path(f"{spec.generate_file_stem()}{exported_model.suffix}")
                if dst.is_dir():
                    # Replace does not work on existing destinations that are
                    # directories. In our case .mlpackage is a directory.
                    rmtree(dst)

                move(src=exported_model, dst=dst)
                print(f"Model '{spec.model_path}' exported to '{dst}'")

    def __remove_temp_folder(self, temp_folder: Path) -> None:
        if self.__is_temp_folder(temp_folder):
            rmtree(temp_folder)
            return
        print(f"Folder '{temp_folder}' is not a temp folder. Skipping removal.")

    def __is_in_temp_folder(self, file: Path) -> bool:
        return self.__is_temp_folder(file.parent)

    def __is_temp_folder(self, temp_folder: Path) -> bool:
        return temp_folder == TEMP_FOLDER

    def __is_model(self, model_path: Path) -> bool:
        return model_path.suffix in AVAILABLE_EXPORT_TYPES


@dataclass
class ExportConfig:
    specifications: list[ModelExportSpecification]


class ConfigParser:
    def __init__(self, parser: ModelExportSpecificationParser) -> None:
        self._parser = parser

    def parse(self, config_file: Path) -> ExportConfig:
        data = self.__read_yaml(config_file)
        specs = [self.__parse_specification(spec) for spec in data["specifications"]]
        return ExportConfig(specifications=specs)

    def __read_yaml(self, config_file: Path) -> dict:
        with open(config_file, "r") as stream:
            return yaml.safe_load(stream)

    def __parse_specification(self, data: dict) -> ModelExportSpecification:
        model = Path(data["model"])
        formats = [_format.lower() for _format in data["formats"]]
        quantization = self.__parse_quantization(data)
        return self._parser.parser(
            model=model, formats=formats, quantization=quantization
        )

    def __parse_quantization(self, data: dict) -> str | None:
        quantization: str | None = data.get("quantization", None)
        if quantization is not None:
            return quantization.lower()
        return None


class YoloModelExporter:
    """Handles the export process for YOLO models.

    Args:
        pre_export_action (PreExportAction): Pre-export actions.
        post_export_action (PostExportAction): Post-export actions.
    """

    def __init__(
        self, pre_export_action: PreExportAction, post_export_action: PostExportAction
    ) -> None:
        self._pre_export_action = pre_export_action
        self._post_export_action = post_export_action

    def export(self, export_config: ExportConfig) -> None:
        """Executes the complete export process for a YOLO model.

        The process involves:
        1. Performing pre-export actions (e.g., creating a temporary folder).
        2. Exporting the model to the desired format.
        3. Performing post-export actions (e.g., cleaning up temporary files).

        Args:
            export_config (ExportConfig): The export config.
        """
        for spec in export_config.specifications:
            self.__export_specification(spec=spec)

    def __export_specification(self, spec: ModelExportSpecification) -> None:
        self._pre_export_action.execute(spec=spec)
        self.__export_model(spec=spec)
        self._post_export_action.execute(spec=spec)

    def __export_model(self, spec: ModelExportSpecification) -> None:
        for export_format in spec.formats:
            self.__export_model_for(spec=spec, export_format=export_format)

    def __export_model_for(
        self, spec: ModelExportSpecification, export_format: ExportFormat
    ) -> None:
        model = YOLO(model=spec.model_info.temp_path)
        kwargs = self.__create_kwargs_from(
            export_format=export_format,
            imagesize=spec.imagesize,
            quantization=spec.quantization,
        )
        try:
            print(
                f"Exporting model '{spec.model_path.name}' "
                f"with following options {kwargs}"
            )
            model.export(**kwargs)
            print("Exporting model successful")
        except Exception as cause:
            print(
                "Error occurred during export. "
                f"Skipping {spec.formats} export of model '{spec.model_path.name}'"
            )
            print(cause)

    def __create_kwargs_from(
        self,
        export_format: ExportFormat,
        imagesize: int | None,
        quantization: Quantization | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"format": export_format.value}

        if quantization is not None:
            quantization_arg = self.__get_quantization_kwarg(quantization)
            kwargs[quantization_arg] = True

        if imagesize is not None:
            kwargs["imgsz"] = imagesize
        return kwargs

    def __get_quantization_kwarg(self, quantization: Quantization) -> str:
        if quantization == "int8":
            return "int8"
        return "half"


def main(
    formats: list[str] | str | None = None,
    model: str | None = None,
    quantization: Literal["int8", "fp16"] | None = None,
    config: str | None = None,
) -> None:
    """CLI Tool for Exporting YOLO Models.

    IMPORTANT: If exporting to a .engine model fails due to a missing tensorrt-bindings
    module, clear the cache (pip cache purge), delete the virtual environment, and
    rerun install.sh.

    Usage:
        python export_models.py [OPTIONS]

    Options:
        --formats (Required if --config is not provided):
            Export format(s) for the model. Supported values:
            - onnx (ONNX format)
            - engine (TensorRT Engine format)
            - coreml (CoreML format)
            Multiple formats can be provided as: "[onnx,engine,coreml]"

        --model (Required if --config is not provided):
            Path to the YOLO model file (e.g., model.pt).

        --quantization (Optional):
            Apply quantization to optimize the model.
            Supported values: int8, fp16.

        --config (Optional):
            Path to a YAML configuration file for batch export of multiple models
            with specific formats and settings.

    Examples:
    1. Export a model to ONNX:
        python export_models.py --model path/to/model.pt --formats onnx

    2. Export to multiple formats:
        python export_models.py --model path/to/model.pt --formats "[onnx,engine]"

    3. Export with FP16 quantization:
        python export_models.py --model path/to/model.pt --formats engine --quantization
        fp16

    4. Export using a config file:
        python export_models.py --config config.yaml

    YAML Config Example:
    specifications:
      - model: models/model1.pt
        formats: [onnx,engine]
        quantization: int8
      - model: models/model2.pt
        formats: [engine,coreml]
        quantization: fp16

    Notes:
    - Use [] for multiple formats with `--formats`.
    - Temporary files are cleaned after export.
    - Supported formats: onnx, engine, coreml.

    Args:
        model (str, optional): the path to the model weights.
        formats (list[str] | str, optional ): The export formats.
            Supported values: onnx, engine, coreml.
        quantization (Literal["int8","fp16"], optional): enable INT-8 or FP-16
            quantization. Supported values: int8, fp16.
        config (str, optional): the path to the export configuration file.

    """
    print(f"CUDA is available: {torch.cuda.is_available()}")
    model_export_spec_parser = ModelExportSpecificationParser()
    exporter = YoloModelExporter(PreExportAction(), PostExportAction())
    if config is not None:
        config_parser = ConfigParser(model_export_spec_parser)
        export_config = config_parser.parse(config_file=Path(config))
        exporter.export(export_config)
    else:
        if formats is None:
            raise ParseError("--formats must be specified.")
        if isinstance(formats, str):
            _formats = [formats]
        else:
            _formats = formats

        if model is None:
            raise ParseError("--model must be specified.")

        spec = model_export_spec_parser.parser(
            model=Path(model), formats=_formats, quantization=quantization
        )
        exporter.export(ExportConfig([spec]))


if __name__ == "__main__":
    Fire(main)
