from dataclasses import dataclass
from typing import Any

from OTVision.application.config import TrackConfig, _TrackBoxmotConfig
from OTVision.detect.otdet import OtdetBuilderConfig
from OTVision.track.boxmot_utils import extract_fps_from_metadata

BOXMOT_DEFAULT_FRAME_RATE = 30.0


@dataclass(frozen=True)
class IouTrackerMetadata:
    name: str
    sigma_l: float
    sigma_h: float
    sigma_iou: float
    t_min: int
    t_miss_max: int


@dataclass(frozen=True)
class BoxmotTrackerMetadata:
    name: str
    device: str
    half_precision: bool
    reid_weights: str | None
    tracker_params: dict[str, Any]


TrackerMetadata = IouTrackerMetadata | BoxmotTrackerMetadata


def build_tracker_metadata(
    track_config: TrackConfig,
    metadata: dict[str, Any] | None = None,
    otdet_builder_config: OtdetBuilderConfig | None = None,
) -> TrackerMetadata:
    if not track_config.boxmot.enabled:
        return IouTrackerMetadata(
            name="IOU",
            sigma_l=track_config.sigma_l,
            sigma_h=track_config.sigma_h,
            sigma_iou=track_config.sigma_iou,
            t_min=track_config.t_min,
            t_miss_max=track_config.t_miss_max,
        )

    return BoxmotTrackerMetadata(
        name=track_config.boxmot.tracker_type.lower(),
        device=track_config.boxmot.device,
        half_precision=track_config.boxmot.half_precision,
        reid_weights=track_config.boxmot.reid_weights,
        tracker_params=resolve_effective_boxmot_tracker_params(
            track_config.boxmot,
            metadata=metadata,
            otdet_builder_config=otdet_builder_config,
        ),
    )


def resolve_effective_boxmot_tracker_params(
    boxmot_config: _TrackBoxmotConfig,
    metadata: dict[str, Any] | None = None,
    otdet_builder_config: OtdetBuilderConfig | None = None,
) -> dict[str, Any]:
    tracker_params = dict(boxmot_config.tracker_params)

    if "frame_rate" in tracker_params:
        return tracker_params

    frame_rate = None
    if metadata is not None:
        frame_rate = extract_fps_from_metadata(metadata)

    if frame_rate is None and otdet_builder_config is not None:
        frame_rate = extract_fps_from_otdet_builder_config(otdet_builder_config)

    if frame_rate is not None:
        tracker_params["frame_rate"] = frame_rate
    else:
        tracker_params["frame_rate"] = BOXMOT_DEFAULT_FRAME_RATE

    return tracker_params


def extract_fps_from_otdet_builder_config(
    otdet_builder_config: OtdetBuilderConfig,
) -> float | None:
    actual_fps = _normalize_positive_float(otdet_builder_config.actual_fps)
    if actual_fps is not None:
        return actual_fps

    return _normalize_positive_float(otdet_builder_config.recorded_fps)


def _normalize_positive_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None

    if normalized <= 0:
        return None

    return normalized
