from pathlib import Path

from OTVision.application.config_parser import ConfigParser


class FileSourceConfigParser(ConfigParser):
    """Parses file source configurations.

    This class extends ConfigParser to handle and parse file source
    configurations by accepting a list of sources, expanding user paths,
    and converting them to absolute paths.

    Attributes:
        No additional attributes apart from those inherited from the
        ConfigParser class.
    """

    def parse_sources(self, sources: list[str]) -> list[str]:
        return [str(Path(source).expanduser()) for source in sources]
