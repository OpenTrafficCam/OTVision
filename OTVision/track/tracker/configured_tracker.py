from collections.abc import Callable

from OTVision.application.config_parser import validate_tracker_type
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker
from OTVision.track.tracker.tracker_plugin_iou import IouTracker

_TRACKER_FACTORIES: dict[str, Callable[[GetCurrentConfig], Tracker]] = {
    "iou": IouTracker,
}


def create_tracker(get_current_config: GetCurrentConfig) -> Tracker:
    """Return a tracker that selects its implementation on first use and on reset."""
    return _ConfiguredTracker(get_current_config)


class _ConfiguredTracker(Tracker):
    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        self._get_current_config = get_current_config
        self._tracker_type: str | None = None
        self._tracker: Tracker | None = None

    def reset(self) -> None:
        self._resolve_tracker().reset()

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        tracker = self._tracker
        if tracker is None:
            tracker = self._resolve_tracker()
        return tracker.track_frame(frame, id_generator)

    def _resolve_tracker(self) -> Tracker:
        tracker_type = self._get_current_config.get().track.tracker_type
        if self._tracker is None or self._tracker_type != tracker_type:
            validate_tracker_type(tracker_type)
            self._tracker = _TRACKER_FACTORIES[tracker_type](self._get_current_config)
            self._tracker_type = tracker_type
        return self._tracker
