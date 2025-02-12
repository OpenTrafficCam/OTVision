from OTVision.domain.cli import DetectCliArgs, DetectCliParser


class GetDetectCliArgs:
    def __init__(self, cli_parser: DetectCliParser) -> None:
        self._cli_parser = cli_parser

    def get(self) -> DetectCliArgs:
        return self._cli_parser.parse()
