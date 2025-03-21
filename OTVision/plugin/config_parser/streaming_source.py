from OTVision.application.config_parser import ConfigParser


class StreamingSourceConfigParser(ConfigParser):
    def parse_sources(self, sources: list[str]) -> list[str]:
        return sources
