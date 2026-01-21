"""BOXMOT tracker adapter for OTVision.

This module provides an adapter that implements OTVision's Tracker interface
using BOXMOT's state-of-the-art multi-object tracking algorithms.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import TrackId
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.track.boxmot_utils import (
    ClassLabelMapper,
    boxmot_tracks_to_detections,
    detections_to_boxmot_array,
)
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker

# Import will fail if BOXMOT not installed - this is intentional
# Users must install with optional dependency group
try:
    from boxmot import (
        BoostTrack,
        BotSort,
        ByteTrack,
        DeepOcSort,
        HybridSort,
        OcSort,
        StrongSort,
    )

    BOXMOT_AVAILABLE = True

    # Mapping of tracker type strings to BOXMOT tracker classes
    # Only defined when BOXMOT is available
    TRACKER_CLASSES = {
        "bytetrack": ByteTrack,
        "botsort": BotSort,
        "boosttrack": BoostTrack,
        "strongsort": StrongSort,
        "ocsort": OcSort,
        "deepocsort": DeepOcSort,
        "hybridsort": HybridSort,
    }
except ImportError:
    BOXMOT_AVAILABLE = False
    TRACKER_CLASSES = {}  # Empty dict when BOXMOT not available

# Trackers that require ReID weights for appearance features
APPEARANCE_TRACKERS = {
    "botsort",
    "boosttrack",
    "strongsort",
    "deepocsort",
    "hybridsort",
}

# Motion-only trackers that don't need images
MOTION_ONLY_TRACKERS = {"bytetrack", "ocsort"}

logger = logging.getLogger(__name__)


class BoxmotTrackerAdapter(Tracker):
    """Adapter for BOXMOT trackers to OTVision's Tracker interface.

    This adapter bridges the API and data format differences between
    BOXMOT and OTVision's tracking system.

    Attributes:
        _boxmot_tracker: The underlying BOXMOT tracker instance
        _class_mapper: Bidirectional class label mapper
        _track_id_mapping: Maps BOXMOT track IDs to OTVision TrackIds
        _previous_track_ids: Set of BOXMOT track IDs from previous frame
        _get_current_config: Function to get current configuration
        _frame_width: Width of video frame (captured from first frame with image)
        _frame_height: Height of video frame (captured from first frame with image)
    """

    def __init__(
        self,
        tracker_type: str,
        reid_weights: Optional[Path] = None,
        device: str = "cpu",
        half: bool = False,
        get_current_config: Optional[GetCurrentConfig] = None,
        tracker_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the BOXMOT tracker adapter.

        Args:
            tracker_type: Type of BOXMOT tracker to use
                (e.g., 'bytetrack', 'botsort', 'ocsort')
            reid_weights: Optional path to ReID model weights
                (required for appearance-based trackers)
            device: Device to run tracker on ('cpu', 'cuda:0', etc.)
            half: Whether to use FP16 precision
            get_current_config: Optional config getter for OTVision settings
            tracker_params: Additional parameters to pass to BOXMOT tracker.
                Can include frame_rate, track_buffer, track_thresh, match_thresh, etc.
                These parameters override the defaults.

        Raises:
            ImportError: If BOXMOT is not installed
            ValueError: If tracker_type is not supported
        """
        super().__init__()

        if not BOXMOT_AVAILABLE:
            raise ImportError(
                "BOXMOT is not installed. "
                "Install with: uv pip install -e .[tracking_boxmot]"
            )

        tracker_type_lower = tracker_type.lower()
        if tracker_type_lower not in TRACKER_CLASSES:
            raise ValueError(
                f"Unknown tracker type: {tracker_type}. "
                f"Supported types: {list(TRACKER_CLASSES.keys())}"
            )

        # Validate ReID weights for appearance-based trackers
        if tracker_type_lower in APPEARANCE_TRACKERS and reid_weights is None:
            raise ValueError(
                f"Tracker type '{tracker_type}' is an appearance-based tracker "
                "and requires reid_weights to be specified. "
                f"Motion-only trackers: {list(MOTION_ONLY_TRACKERS)}"
            )

        tracker_class = TRACKER_CLASSES[tracker_type_lower]

        # Build kwargs for BOXMOT tracker
        kwargs: dict[str, Any] = {"device": device, "half": half}
        if reid_weights is not None:
            kwargs["reid_weights"] = reid_weights

        # Merge user-provided tracker_params (override defaults)
        if tracker_params:
            kwargs.update(tracker_params)

        logger.info(
            f"Initializing BOXMOT tracker '{tracker_type}' with params: {kwargs}"
        )

        # Initialize BOXMOT tracker with error handling
        try:
            self._boxmot_tracker = tracker_class(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize BOXMOT tracker '{tracker_type}' "
                f"on device '{device}': {str(e)}"
            ) from e

        logger.info(f"Initialized BOXMOT tracker: {tracker_type} on device: {device}")

        self._class_mapper = ClassLabelMapper()
        self._track_id_mapping: dict[int, TrackId] = {}
        self._previous_track_ids: set[int] = set()
        self._get_current_config = get_current_config
        self._frame_width: int | None = None
        self._frame_height: int | None = None

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        """Track detections in a single frame.

        Converts OTVision DetectedFrame to BOXMOT format, performs tracking,
        and converts results back to OTVision TrackedFrame format.

        Args:
            frame: Frame with untracked detections
            id_generator: Generator for new track IDs

        Returns:
            TrackedFrame with tracking information

        Raises:
            ValueError: If frame.image is None for appearance-based trackers
        """
        # Check if image is required and available
        if self._requires_image() and frame.image is None:
            raise ValueError(
                f"Tracker {type(self._boxmot_tracker).__name__} requires "
                "frame images for appearance features, but frame.image is None"
            )

        # Convert detections to BOXMOT format
        # Build mapping on-the-fly for any new labels
        class_mapping = self._class_mapper.get_forward_mapping()
        has_new_labels = False
        for det in frame.detections:
            if det.label not in class_mapping:
                self._class_mapper.get_id(det.label)
                has_new_labels = True

        # Only retrieve mapping again if we added new labels
        if has_new_labels:
            class_mapping = self._class_mapper.get_forward_mapping()

        detections_array = detections_to_boxmot_array(
            list(frame.detections), class_mapping
        )

        # Prepare image (use empty array if not available for motion-only trackers)
        if frame.image is not None:
            image = frame.image
            # Capture frame dimensions from first frame with image
            if self._frame_width is None:
                self._frame_height, self._frame_width = image.shape[:2]
        else:
            # Create dummy image for motion-only trackers
            image = np.zeros((1, 1, 3), dtype=np.uint8)

        # Run BOXMOT tracker
        if len(detections_array) > 0:
            tracks = self._boxmot_tracker.update(detections_array, image)
        else:
            tracks = np.empty((0, 8), dtype=np.float32)

        # Convert BOXMOT tracks to OTVision TrackedDetections
        class_mapping_reverse = self._class_mapper.get_reverse_mapping()
        tracked_detections, current_track_ids, updated_mapping = (
            boxmot_tracks_to_detections(
                tracks,
                class_mapping_reverse,
                self._track_id_mapping,
                id_generator,
                self._previous_track_ids,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
            )
        )
        self._track_id_mapping = updated_mapping

        # Detect finished tracks (tracks that disappeared)
        finished_boxmot_ids = self._previous_track_ids - current_track_ids
        finished_track_ids = {
            self._track_id_mapping[boxmot_id]
            for boxmot_id in finished_boxmot_ids
            if boxmot_id in self._track_id_mapping
        }

        # Clean up finished tracks from mapping to prevent memory leak
        for boxmot_id in finished_boxmot_ids:
            self._track_id_mapping.pop(boxmot_id, None)

        # Update previous track IDs for next frame
        self._previous_track_ids = current_track_ids

        # For now, we don't explicitly mark tracks as discarded
        # BOXMOT handles track lifecycle internally
        discarded_track_ids: set[TrackId] = set()

        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=tracked_detections,
            image=frame.image,
            finished_tracks=finished_track_ids,
            discarded_tracks=discarded_track_ids,
        )

    def _requires_image(self) -> bool:
        """Check if the tracker requires image data (appearance-based trackers).

        Returns:
            True if tracker needs images, False for motion-only trackers
        """
        # Check tracker type against known motion-only trackers
        tracker_type = type(self._boxmot_tracker).__name__.lower()
        return tracker_type not in MOTION_ONLY_TRACKERS

    def reset(self) -> None:
        """Reset tracker state.

        Clears all active tracks and resets internal mappings.
        Should be called between tracking different videos/sequences.
        """
        # BOXMOT trackers have a reset() method
        if hasattr(self._boxmot_tracker, "reset"):
            self._boxmot_tracker.reset()

        # Reset our internal state
        self._track_id_mapping.clear()
        self._previous_track_ids.clear()
        self._class_mapper = ClassLabelMapper()  # Reset class mapping for new video
        self._frame_width = None
        self._frame_height = None
