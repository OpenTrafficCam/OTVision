from argparse import ArgumentParser
from functools import cached_property

from OTVision.application.config import Config
from OTVision.application.config_parser import ConfigParser
from OTVision.application.configure_logger import ConfigureLogger
from OTVision.application.get_config import GetConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.track.get_track_cli_args import GetTrackCliArgs
from OTVision.application.track.tracking_run_id import StrIdGenerator
from OTVision.application.track.update_current_track_config import (
    UpdateCurrentTrackConfig,
)
from OTVision.application.track.update_track_config_with_cli_args import (
    UpdateTrackConfigWithCliArgs,
)
from OTVision.application.update_current_config import UpdateCurrentConfig
from OTVision.domain.cli import TrackCliParser
from OTVision.domain.current_config import CurrentConfig
from OTVision.domain.serialization import Deserializer
from OTVision.domain.tracker import TrackerType
from OTVision.plugin.yaml_serialization import YamlDeserializer
from OTVision.track.cli import ArgparseTrackCliParser
from OTVision.track.exporter.filebased_exporter import FinishedChunkTrackExporter
from OTVision.track.id_generator import track_id_generator, tracking_run_uuid_generator
from OTVision.track.model.filebased.frame_chunk import ChunkParser
from OTVision.track.model.filebased.frame_group import FrameGroupParser
from OTVision.track.model.track_exporter import FinishedTracksExporter
from OTVision.track.model.tracking_interfaces import Tracker
from OTVision.track.parser.chunk_parser_plugins import JsonChunkParser
from OTVision.track.parser.frame_group_parser_plugins import (
    TimeThresholdFrameGroupParser,
)
from OTVision.track.track import OtvisionTrack
from OTVision.track.tracker.filebased_tracking import (
    GroupedFilesTracker,
    UnfinishedChunksBuffer,
)
from OTVision.track.tracker.tracker_plugin_botsort import BotsortTracker
from OTVision.track.tracker.tracker_plugin_iou import IouTracker


class TrackBuilder:
    @cached_property
    def get_config(self) -> GetConfig:
        return GetConfig(self.config_parser)

    @cached_property
    def config_parser(self) -> ConfigParser:
        return ConfigParser(self.yaml_deserializer)

    @cached_property
    def yaml_deserializer(self) -> Deserializer:
        return YamlDeserializer()

    @cached_property
    def get_track_cli_args(self) -> GetTrackCliArgs:
        return GetTrackCliArgs(self.track_cli_parser)

    @cached_property
    def track_cli_parser(self) -> TrackCliParser:
        return ArgparseTrackCliParser(
            parser=ArgumentParser("Track objects through detections"), argv=self.argv
        )

    @cached_property
    def update_track_config_with_cli_args(self) -> UpdateTrackConfigWithCliArgs:
        return UpdateTrackConfigWithCliArgs(self.get_track_cli_args)

    @cached_property
    def configure_logger(self) -> ConfigureLogger:
        return ConfigureLogger()

    @cached_property
    def update_current_config(self) -> UpdateCurrentConfig:
        return UpdateCurrentConfig(self.current_config)

    @cached_property
    def get_current_config(self) -> GetCurrentConfig:
        return GetCurrentConfig(self.current_config)

    @cached_property
    def current_config(self) -> CurrentConfig:
        return CurrentConfig(Config())

    @cached_property
    def chunk_parser(self) -> ChunkParser:
        return JsonChunkParser()

    def _create_tracker(self, tracker_type: TrackerType) -> Tracker:
        """Create the tracker implementation named by ``tracker_type``.

        Args:
            tracker_type (TrackerType): Selected tracker.

        Returns:
            Tracker: Tracker instance.

        Raises:
            ValueError: If the tracker type has no implementation.
        """
        match tracker_type:
            case TrackerType.IOU:
                return IouTracker(get_current_config=self.get_current_config)
            case TrackerType.BOTSORT:
                return BotsortTracker(get_current_config=self.get_current_config)
        raise ValueError(f"No tracker implementation for '{tracker_type}'.")

    @cached_property
    def frame_group_parser(self) -> FrameGroupParser:
        return TimeThresholdFrameGroupParser(self.get_current_config)

    @cached_property
    def tracker(self) -> GroupedFilesTracker:
        tracker = self._create_tracker(self.get_current_config.get().track.tracker_type)
        return GroupedFilesTracker(
            tracker=tracker,
            chunk_parser=self.chunk_parser,
            frame_group_parser=self.frame_group_parser,
            id_generator_factory=lambda _: track_id_generator(),
        )

    @cached_property
    def unfinished_chunks_buffer(self) -> UnfinishedChunksBuffer:
        # Discarded (sub-t_min) tracks must not reach the .ottrk: the file
        # format has no discard flag, so kept rows are indistinguishable from
        # real tracks downstream, and OTAnalytics applies no length filter of
        # its own.
        return UnfinishedChunksBuffer(tracker=self.tracker, keep_discarded=False)

    @cached_property
    def track_exporter(self) -> FinishedTracksExporter:
        return FinishedChunkTrackExporter()

    @cached_property
    def update_current_track_config(self) -> UpdateCurrentTrackConfig:
        return UpdateCurrentTrackConfig(
            get_current_config=self.get_current_config,
            update_current_config=self.update_current_config,
        )

    @cached_property
    def tracking_run_id_generator(self) -> StrIdGenerator:
        return tracking_run_uuid_generator

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv

    def build(self) -> OtvisionTrack:
        return OtvisionTrack(
            get_current_config=self.get_current_config,
            track_exporter=self.track_exporter,
            unfinished_chunks_buffer=self.unfinished_chunks_buffer,
            tracking_run_id_generator=self.tracking_run_id_generator,
        )
