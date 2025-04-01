from OTVision.domain.cli import TrackCliArgs, TrackCliParser


class GetTrackCliArgs:
    """Use case to retrieve the CLI arguments for OTVision track.

    Args:
        cli_parser (TrackCliParser): the CLI parser to parse the track CLI arguments.
    """

    def __init__(self, cli_parser: TrackCliParser) -> None:
        self._cli_parser = cli_parser

    def get(self) -> TrackCliArgs:
        """Get the track CLI arguments.

        Returns:
            TrackCliArgs: the track CLI arguments.
        """
        return self._cli_parser.parse()
