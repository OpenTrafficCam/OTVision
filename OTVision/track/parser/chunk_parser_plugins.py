import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DETECTIONS,
    OCCURRENCE,
    H,
    W,
    X,
    Y,
)
from OTVision.domain.detection import Detection
from OTVision.domain.frame import DetectedFrame
from OTVision.helpers.date import parse_datetime
from OTVision.helpers.files import denormalize_bbox, read_json
from OTVision.track.model.filebased.frame_chunk import ChunkParser, FrameChunk
from OTVision.track.model.filebased.frame_group import FrameGroup
from OTVision.track.video_frame_provider import VideoFrameProvider

logger = logging.getLogger(__name__)

# Type alias for video frame provider factory
VideoFrameProviderFactory = Callable[[Path, dict], VideoFrameProvider]


class JsonChunkParser(ChunkParser):
    """Parser for JSON-based OTDET detection files.

    Optionally loads video frames when a video frame provider factory is given,
    which is required for appearance-based trackers.

    Attributes:
        _video_frame_provider_factory: Optional factory for creating video frame
            providers. When provided, video frames will be loaded for each frame.
    """

    def __init__(
        self,
        video_frame_provider_factory: VideoFrameProviderFactory | None = None,
    ) -> None:
        """Initialize the JSON chunk parser.

        Args:
            video_frame_provider_factory: Optional factory function that creates
                a VideoFrameProvider given an OTDET file path and its metadata.
                When provided, frames will be loaded from video for appearance
                tracking. When None, frames will have image=None.
        """
        self._video_frame_provider_factory = video_frame_provider_factory

    def parse(
        self, file: Path, frame_group: FrameGroup, frame_offset: int = 0
    ) -> FrameChunk:
        json = read_json(file)
        metadata: dict = frame_group.metadata_by_file[file]

        denormalized = denormalize_bbox(
            json, file, metadata={file.as_posix(): metadata}
        )
        input: dict[int, dict[str, Any]] = denormalized[DATA]

        # Create video frame provider if factory is provided
        video_provider: VideoFrameProvider | None = None
        if self._video_frame_provider_factory is not None:
            try:
                video_provider = self._video_frame_provider_factory(file, metadata)
                logger.debug(f"Created video frame provider for {file}")
            except Exception as e:
                # Re-raise with context about which file failed
                raise RuntimeError(
                    f"Failed to create video frame provider for {file}: {e}"
                ) from e

        try:
            frames = self.convert(file, frame_offset, input, video_provider)
        finally:
            # Always close the video provider to release resources
            if video_provider is not None:
                video_provider.close()

        frames.sort(key=lambda frame: (frame.occurrence, frame.no))
        return FrameChunk(file, metadata, frames, frame_group.id)

    def convert(
        self,
        file: Path,
        frame_offset: int,
        input: dict[int, dict[str, Any]],
        video_provider: VideoFrameProvider | None = None,
    ) -> list[DetectedFrame]:
        detection_parser = DetectionParser()
        frames = []

        input_progress = tqdm(
            input.items(), desc="parse Frames", total=len(input), leave=False
        )
        for key, value in input_progress:
            occurrence: datetime = parse_datetime(value[OCCURRENCE])
            data_detections = value[DETECTIONS]
            detections = detection_parser.convert(data_detections)

            # Load frame image if video provider is available
            frame_no = int(key) + frame_offset
            image = None
            if video_provider is not None:
                # Frame numbers in OTDET are typically 1-indexed
                # The key in the JSON is the frame number from detection
                image = video_provider.get_frame(int(key))
                if image is None:
                    logger.warning(f"Could not load frame {key} from video for {file}")

            parsed_frame = DetectedFrame(
                no=frame_no,
                occurrence=occurrence,
                source=str(file),
                output=str(file),
                detections=detections,
                image=image,
            )
            frames.append(parsed_frame)
        return frames


class DetectionParser:
    def convert(self, detection_data: list[dict[str, str]]) -> list[Detection]:
        detections: list[Detection] = []
        for detection in detection_data:
            detected_item = Detection(
                detection[CLASS],
                float(detection[CONFIDENCE]),
                float(detection[X]),
                float(detection[Y]),
                float(detection[W]),
                float(detection[H]),
            )
            detections.append(detected_item)
        return detections
