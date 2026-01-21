import logging
from argparse import ArgumentParser
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

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
from OTVision.plugin.yaml_serialization import YamlDeserializer
from OTVision.track.boxmot_utils import extract_fps_from_metadata
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
from OTVision.track.tracker.tracker_plugin_iou import IouTracker
from OTVision.track.video_frame_provider import (
    SequentialVideoFrameProvider,
    VideoFrameProvider,
    resolve_video_path_from_otdet,
)

# Optional BOXMOT support
BOXMOT_AVAILABLE: bool
APPEARANCE_TRACKERS: set[str]

try:
    from OTVision.track.tracker.tracker_plugin_boxmot import (
        APPEARANCE_TRACKERS as _APPEARANCE_TRACKERS,
    )
    from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

    BOXMOT_AVAILABLE = True
    APPEARANCE_TRACKERS = _APPEARANCE_TRACKERS
except ImportError:
    BOXMOT_AVAILABLE = False
    APPEARANCE_TRACKERS = set()  # Empty set when BOXMOT not available

logger = logging.getLogger(__name__)


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
        config = self.get_current_config.get().track

        # Check if appearance-based tracker is enabled
        needs_video_frames = (
            BOXMOT_AVAILABLE
            and config.boxmot.enabled
            and config.boxmot.tracker_type.lower() in APPEARANCE_TRACKERS
        )

        if needs_video_frames:
            logger.info(
                f"Appearance-based tracker '{config.boxmot.tracker_type}' enabled, "
                "video frames will be loaded for tracking"
            )

            def video_provider_factory(
                otdet_file: Path, metadata: dict
            ) -> VideoFrameProvider:
                video_path = resolve_video_path_from_otdet(otdet_file, metadata)
                return SequentialVideoFrameProvider(video_path)

            return JsonChunkParser(video_frame_provider_factory=video_provider_factory)

        return JsonChunkParser()

    @cached_property
    def frame_group_parser(self) -> FrameGroupParser:
        return TimeThresholdFrameGroupParser(self.get_current_config)

    def _create_boxmot_tracker_factory(
        self,
    ) -> "Callable[[dict[str, Any]], Tracker]":
        """Create factory for BOXMOT tracker with deferred FPS detection.

        The factory receives the first file's metadata and uses it to auto-detect
        frame_rate if not explicitly configured in tracker_params.

        Returns:
            Factory function that creates a BoxmotTrackerAdapter from metadata
        """
        config = self.get_current_config.get().track

        def factory(metadata: dict[str, Any]) -> Tracker:
            tracker_params = dict(config.boxmot.tracker_params)

            # Auto-detect frame_rate if not explicitly configured
            if "frame_rate" not in tracker_params:
                fps = extract_fps_from_metadata(metadata)
                if fps is not None:
                    tracker_params["frame_rate"] = fps
                    logger.info(f"Auto-detected frame_rate from OTDET metadata: {fps}")
                else:
                    logger.warning(
                        "Could not auto-detect frame_rate from OTDET metadata. "
                        "Using BOXMOT default (30). Consider setting in config."
                    )
            else:
                logger.info(
                    f"Using configured frame_rate: {tracker_params['frame_rate']}"
                )

            reid_weights = (
                Path(config.boxmot.reid_weights) if config.boxmot.reid_weights else None
            )

            return BoxmotTrackerAdapter(
                tracker_type=config.boxmot.tracker_type,
                reid_weights=reid_weights,
                device=config.boxmot.device,
                half=config.boxmot.half_precision,
                get_current_config=self.get_current_config,
                tracker_params=tracker_params,
            )

        return factory

    @cached_property
    def tracker(self) -> GroupedFilesTracker:
        config = self.get_current_config.get().track

        # Check if BOXMOT is enabled in configuration
        if config.boxmot.enabled:
            if not BOXMOT_AVAILABLE:
                raise ImportError(
                    "BOXMOT is enabled in configuration but not installed. "
                    "Install with: uv pip install -e .[tracking_boxmot]"
                )

            logger.info(
                f"Using BOXMOT tracker: {config.boxmot.tracker_type} "
                f"on device: {config.boxmot.device}"
            )

            return GroupedFilesTracker(
                tracker_factory=self._create_boxmot_tracker_factory(),
                chunk_parser=self.chunk_parser,
                frame_group_parser=self.frame_group_parser,
                id_generator_factory=lambda _: track_id_generator(),
            )
        else:
            logger.info("Using IOU tracker")
            base_tracker = IouTracker(get_current_config=self.get_current_config)

            return GroupedFilesTracker(
                tracker=base_tracker,
                chunk_parser=self.chunk_parser,
                frame_group_parser=self.frame_group_parser,
                id_generator_factory=lambda _: track_id_generator(),
            )

    @cached_property
    def unfinished_chunks_buffer(self) -> UnfinishedChunksBuffer:
        return UnfinishedChunksBuffer(tracker=self.tracker, keep_discarded=True)

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
