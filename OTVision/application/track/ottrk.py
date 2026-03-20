from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from OTVision import dataformat, version
from OTVision.application.config import TrackConfig
from OTVision.detect.otdet import (
    OtdetBuilderConfig,
    OtdetMetadataBuilder,
    extract_expected_duration_from_otdet,
    extract_start_date_from_otdet,
)
from OTVision.domain.detection import TrackedDetection, TrackId
from OTVision.domain.frame import FrameNo, TrackedFrame
from OTVision.track.tracker_metadata import (
    BoxmotTrackerMetadata,
    IouTrackerMetadata,
    TrackerMetadata,
    build_tracker_metadata,
)


@dataclass
class OttrkBuilderConfig:
    otdet_builder_config: OtdetBuilderConfig
    number_of_frames: int
    sigma_l: float
    sigma_h: float
    sigma_iou: float
    t_min: int
    t_miss_max: int
    tracking_run_id: str
    frame_group: int
    track_config: TrackConfig


class OttrkBuilderError(Exception):
    pass


class OttrkBuilder:
    @property
    def config(self) -> OttrkBuilderConfig:
        if self._config is None:
            raise OttrkBuilderError("Ottrk builder config is not set")
        return self._config

    def __init__(self, otdet_metadata_builder: OtdetMetadataBuilder) -> None:
        self._config: OttrkBuilderConfig | None = None
        self._otdet_metadata_builder = otdet_metadata_builder
        self._tracked_detections: dict[TrackId, list[dict]] = defaultdict(list)

    def build(self) -> dict:
        result = {
            dataformat.METADATA: self.build_metadata(),
            dataformat.DATA: {
                dataformat.DETECTIONS: self._build_data(),
            },
        }
        self.reset()
        return result

    def build_metadata(self) -> dict:
        otdet_metadata = self._otdet_metadata_builder.build(
            self.config.number_of_frames
        )
        start_date = extract_start_date_from_otdet(otdet_metadata)
        duration = extract_expected_duration_from_otdet(otdet_metadata)
        end_date = start_date + duration
        ottrk_metadata = self._build_track_metadata(start_date, end_date)
        return {
            **otdet_metadata,
            **ottrk_metadata,
        }

    def _build_track_metadata(self, start_date: datetime, end_date: datetime) -> dict:
        tracker_metadata = build_tracker_metadata(
            self.config.track_config,
            otdet_builder_config=self.config.otdet_builder_config,
        )
        result = create_ottrk_metadata_entry(
            start_date=start_date,
            end_date=end_date,
            tracker_metadata=tracker_metadata,
        )
        tracking_metadata = result[dataformat.TRACKING]
        tracking_metadata[dataformat.TRACKING_RUN_ID] = self.config.tracking_run_id
        tracking_metadata[dataformat.FRAME_GROUP] = self.config.frame_group
        return result

    def _build_data(self) -> list[dict]:
        return sorted(
            (
                detection
                for detections in self._tracked_detections.values()
                for detection in detections
            ),
            key=lambda detection: (
                detection[dataformat.FRAME],
                detection[dataformat.OCCURRENCE],
            ),
        )

    def set_config(self, config: OttrkBuilderConfig) -> Self:
        self._config = config
        self._otdet_metadata_builder.add_config(config.otdet_builder_config)
        return self

    def add_tracked_frames(self, tracked_frames: list[TrackedFrame]) -> Self:
        finished_tracks = set()
        discarded_tracks = set()

        for tracked_frame in tracked_frames:
            finished_tracks.update(tracked_frame.finished_tracks)
            discarded_tracks.update(tracked_frame.discarded_tracks)

            for detection in tracked_frame.detections:
                self._tracked_detections[detection.track_id].append(
                    self._serialize_tracked_detection(
                        frame_no=tracked_frame.no,
                        detection=detection,
                        occurrence=tracked_frame.occurrence,
                    )
                )
        self.__sort_detections()
        self.discard_tracks(discarded_tracks)
        self.finish_tracks(finished_tracks)
        return self

    def _serialize_tracked_detection(
        self,
        frame_no: FrameNo,
        detection: TrackedDetection,
        occurrence: datetime,
    ) -> dict:
        result = detection.to_otdet()
        result[dataformat.INTERPOLATED_DETECTION] = False
        result[dataformat.FIRST] = detection.is_first
        result[dataformat.FINISHED] = False
        result[dataformat.TRACK_ID] = detection.track_id
        result[dataformat.FRAME] = frame_no
        result[dataformat.OCCURRENCE] = occurrence.timestamp()
        result[dataformat.INPUT_FILE_PATH] = self.config.otdet_builder_config.source
        return result

    def __sort_detections(self) -> None:
        for detections in self._tracked_detections.values():
            detections.sort(key=lambda detection: (detection[dataformat.OCCURRENCE]))

    def reset(self) -> Self:
        self._config = None
        self._otdet_metadata_builder.reset()
        self._tracked_detections = defaultdict(list)
        return self

    def discard_tracks(self, tracks: set[TrackId]) -> Self:
        for track_id in tracks:
            self.discard_track(track_id)
        return self

    def discard_track(self, track_id: TrackId) -> Self:
        try:
            del self._tracked_detections[track_id]
        except KeyError:
            pass
        return self

    def finish_tracks(self, tracks: set[TrackId]) -> Self:
        for track_id in tracks:
            self.finish_track(track_id)
        return self

    def finish_track(self, track_id: TrackId) -> Self:
        if track_id in self._tracked_detections:
            self._tracked_detections[track_id][-1][dataformat.FINISHED] = True
        return self


def create_ottrk_metadata_entry(
    start_date: datetime,
    end_date: datetime,
    tracker_metadata: TrackerMetadata,
) -> dict:
    return {
        dataformat.OTTRACK_VERSION: version.ottrack_version(),
        dataformat.TRACKING: {
            dataformat.OTVISION_VERSION: version.otvision_version(),
            dataformat.FIRST_TRACKED_VIDEO_START: start_date.timestamp(),
            dataformat.LAST_TRACKED_VIDEO_END: end_date.timestamp(),
            dataformat.TRACKER: create_tracker_metadata(tracker_metadata),
        },
    }


def create_tracker_metadata(tracker_metadata: TrackerMetadata) -> dict:
    if isinstance(tracker_metadata, IouTrackerMetadata):
        return {
            dataformat.NAME: tracker_metadata.name,
            dataformat.SIGMA_L: tracker_metadata.sigma_l,
            dataformat.SIGMA_H: tracker_metadata.sigma_h,
            dataformat.SIGMA_IOU: tracker_metadata.sigma_iou,
            dataformat.T_MIN: tracker_metadata.t_min,
            dataformat.T_MISS_MAX: tracker_metadata.t_miss_max,
        }

    if isinstance(tracker_metadata, BoxmotTrackerMetadata):
        result = {
            dataformat.NAME: tracker_metadata.name,
            dataformat.TRACKER_DEVICE: tracker_metadata.device,
            dataformat.HALF_PRECISION: tracker_metadata.half_precision,
        }
        if tracker_metadata.reid_weights is not None:
            result[dataformat.TRACKER_REID_WEIGHTS] = tracker_metadata.reid_weights
        if tracker_metadata.tracker_params:
            result[dataformat.TRACKER_PARAMS] = tracker_metadata.tracker_params
        return result

    raise TypeError(f"Unsupported tracker metadata: {type(tracker_metadata)!r}")
