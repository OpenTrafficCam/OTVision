from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OTVision.abstraction.observer import Observer, Subject
from OTVision.application.buffer import Buffer
from OTVision.application.config import Config, TrackConfig
from OTVision.application.configure_logger import logger
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.otvision_save_path_provider import OtvisionSavePathProvider
from OTVision.application.track.ottrk import OttrkBuilder, OttrkBuilderConfig
from OTVision.application.track.tracking_run_id import GetCurrentTrackingRunId
from OTVision.detect.otdet import OtdetBuilderConfig
from OTVision.detect.otdet_file_writer import OtdetFileWrittenEvent
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import TrackedFrame
from OTVision.helpers.files import write_json

STREAMING_FRAME_GROUP_ID = 0


@dataclass(frozen=True)
class OttrkFileWrittenEvent:
    save_location: Path


class StreamOttrkFileWriter(Buffer[TrackedFrame, OtdetFileWrittenEvent]):
    @property
    def config(self) -> Config:
        return self._get_current_config.get()

    @property
    def track_config(self) -> TrackConfig:
        return self.config.track

    @property
    def build_condition_fulfilled(self) -> bool:
        return len(self._ottrk_unfinished_tracks) == 0

    @property
    def current_output_file(self) -> Path:
        if self._current_output_file is None:
            raise ValueError("Output file has not been set yet.")
        return self._current_output_file

    def __init__(
        self,
        subject: Subject[OttrkFileWrittenEvent],
        builder: OttrkBuilder,
        get_current_config: GetCurrentConfig,
        get_current_tracking_run_id: GetCurrentTrackingRunId,
        save_path_provider: OtvisionSavePathProvider,
    ) -> None:
        Buffer.__init__(self)
        self._subject = subject
        self._builder = builder
        self._get_current_config = get_current_config
        self._current_tracking_run_id = get_current_tracking_run_id
        self._save_path_provider = save_path_provider

        self._in_writing_state: bool = False
        self._ottrk_unfinished_tracks: set[TrackId] = set()
        self._current_output_file: Path | None = None

    def on_flush(self, event: OtdetFileWrittenEvent) -> None:
        tracked_frames = self._get_buffered_elements()
        if not tracked_frames:
            return

        self._in_writing_state = True
        self._current_output_file = self._save_path_provider.provide(
            event.otdet_builder_config.source, self.config.filetypes.track
        )
        builder_config = self._create_ottrk_builder_config(
            event.otdet_builder_config, event.number_of_frames
        )
        self._builder.set_config(builder_config)
        last_frame = tracked_frames[-1]
        self._builder.add_tracked_frames(tracked_frames)
        self._ottrk_unfinished_tracks = last_frame.unfinished_tracks
        self.reset()

    def _create_ottrk_builder_config(
        self,
        otdet_builder_config: OtdetBuilderConfig,
        number_of_frames: int,
    ) -> OttrkBuilderConfig:
        return OttrkBuilderConfig(
            otdet_builder_config=otdet_builder_config,
            number_of_frames=number_of_frames,
            sigma_l=self.track_config.sigma_l,
            sigma_h=self.track_config.sigma_h,
            sigma_iou=self.track_config.sigma_iou,
            t_min=self.track_config.t_min,
            t_miss_max=self.track_config.t_miss_max,
            tracking_run_id=self._current_tracking_run_id.get(),
            frame_group=STREAMING_FRAME_GROUP_ID,
        )

    def reset(self) -> None:
        self._reset_buffer()

    def buffer(self, to_buffer: TrackedFrame) -> None:
        self._buffer.append(to_buffer.without_image())

        if self._in_writing_state:
            self._builder.finish_tracks(to_buffer.finished_tracks)
            self._builder.discard_tracks(to_buffer.discarded_tracks)
            self._ottrk_unfinished_tracks = (
                self._ottrk_unfinished_tracks.difference(to_buffer.unfinished_tracks)
                .difference(to_buffer.finished_tracks)
                .difference(to_buffer.discarded_tracks)
            )
            logger().warning(f"Unfinished tracks: {self._ottrk_unfinished_tracks}")
            if self.build_condition_fulfilled:
                self._create_ottrk()

    def _create_ottrk(self) -> None:
        ottrk_data = self._builder.build()
        self.write(ottrk_data)
        self.full_reset()

    def full_reset(self) -> None:
        self._in_writing_state = False
        self._builder.reset()
        self._ottrk_unfinished_tracks = set()
        self._current_output_file = None

    def write(self, ottrk: dict) -> None:
        current_output_file = self.current_output_file
        write_json(
            dict_to_write=ottrk,
            file=current_output_file,
            filetype=self.config.filetypes.track,
            overwrite=True,
        )
        self._notify_ottrk_file_written(save_location=current_output_file)

    def force_flush(self, _: Any) -> None:
        self._create_ottrk()

    def register_observers(self, observer: Observer[OttrkFileWrittenEvent]) -> None:
        self._subject.register(observer)

    def _notify_ottrk_file_written(self, save_location: Path) -> None:
        self._subject.notify(OttrkFileWrittenEvent(save_location=save_location))
