from argparse import ArgumentParser
from functools import cached_property

from OTVision.application.configure_logger import ConfigureLogger
from OTVision.application.detect.get_detect_cli_args import GetDetectCliArgs
from OTVision.application.detect.update_detect_config_with_cli_args import (
    UpdateDetectConfigWithCliArgs,
)
from OTVision.application.get_config import GetConfig
from OTVision.config import ConfigParser
from OTVision.detect.cli import ArgparseDetectCliParser
from OTVision.detect.otdet import OtdetBuilder
from OTVision.domain.cli import DetectCliParser


class DetectBuilder:
    @cached_property
    def get_config(self) -> GetConfig:
        return GetConfig(self.config_parser)

    @cached_property
    def config_parser(self) -> ConfigParser:
        return ConfigParser()

    @cached_property
    def get_detect_cli_args(self) -> GetDetectCliArgs:
        return GetDetectCliArgs(self.detect_cli_parser)

    @cached_property
    def detect_cli_parser(self) -> DetectCliParser:
        return ArgparseDetectCliParser(
            parser=ArgumentParser("Detect objects in videos or images"), argv=self.argv
        )

    @cached_property
    def update_detect_config_with_ci_args(self) -> UpdateDetectConfigWithCliArgs:
        return UpdateDetectConfigWithCliArgs(self.get_detect_cli_args)

    @cached_property
    def configure_logger(self) -> ConfigureLogger:
        return ConfigureLogger()

    @cached_property
    def otdet_builder(self) -> OtdetBuilder:
        return OtdetBuilder()

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv
