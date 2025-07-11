from OTVision.application.buffer import Buffer
from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.track.ottrk import OttrkBuilder, OttrkBuilderConfig
from OTVision.application.track.tracking_run_id import GetCurrentTrackingRunId
from OTVision.detect.otdet import OtdetBuilderConfig
from OTVision.detect.otdet_file_writer import OtdetFileWrittenEvent
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import TrackedFrame

STREAMING_FRAME_GROUP_ID = 0

# Finished -> ✅
# Unfinished -> Track ids -> Wait -> Id-1 -> Löschen, finished id-2, discarded id,
# Discarded -> Remove
# Nebenbedingung config: flush_buffersize > tmin und t_miss_max < flush_buffer_size
# überprüfe beim Parsen einer Config Datei. Streaming Parser


class StreamOttrkFileWriter(Buffer[TrackedFrame, OtdetFileWrittenEvent]):

    @property
    def track_config(self) -> TrackConfig:
        return self._get_current_config.get().track

    @property
    def build_condition_fulfilled(self) -> bool:
        return len(self._ottrk_unfinished_tracks) == 0

    def __init__(
        self,
        builder: OttrkBuilder,
        get_current_config: GetCurrentConfig,
        get_current_tracking_run_id: GetCurrentTrackingRunId,
    ) -> None:
        Buffer.__init__(self)
        self._builder = builder
        self._get_current_config = get_current_config
        self._current_tracking_run_id = get_current_tracking_run_id

        self._unfinished_tracks: set[TrackId] = set()
        self._notify_tracking_info_observers: bool = False
        self.__in_writing_state: bool = False
        self._ottrk_unfinished_tracks: set[TrackId] = set()

    def on_flush(self, event: OtdetFileWrittenEvent) -> None:
        self.__in_writing_state = True
        builder_config = self._create_ottrk_builder_config(
            event.otdet_builder_config, event.number_of_frames
        )
        self._builder.add_config(builder_config)
        self._builder.add_tracked_frames(self._get_buffered_elements())
        self._ottrk_unfinished_tracks = self._unfinished_tracks
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
        self._reset_unfinished_tracks()
        self._reset_buffer()

    def _reset_unfinished_tracks(self) -> None:
        self._unfinished_tracks = set()

    def buffer(self, to_buffer: TrackedFrame) -> None:
        self._buffer.append(to_buffer.without_image())
        self._unfinished_tracks.update(
            to_buffer.unfinished_tracks.difference(
                to_buffer.discarded_tracks
            ).difference(to_buffer.finished_tracks)
        )

        if self.__in_writing_state:
            self._builder.finish_tracks(to_buffer.finished_tracks)
            self._builder.discard_tracks(to_buffer.discarded_tracks)
            self._ottrk_unfinished_tracks = (
                self._ottrk_unfinished_tracks.difference(to_buffer.unfinished_tracks)
                .difference(to_buffer.finished_tracks)
                .difference(to_buffer.discarded_tracks)
            )
            if self.build_condition_fulfilled:
                self._builder.build()
                self.__in_writing_state = False
