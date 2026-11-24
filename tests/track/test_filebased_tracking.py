from datetime import timedelta
from pathlib import Path

import pytest

from OTVision.dataformat import (
    DETECTION,
    HEIGHT,
    METADATA,
    NORMALIZED_BBOX,
    VIDEO,
    WIDTH,
)
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from OTVision.helpers.files import write_json
from OTVision.track.builder import TrackBuilder
from OTVision.track.id_generator import track_id_generator
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker
from OTVision.track.tracker.filebased_tracking import (
    GroupedFilesTracker,
    UnfinishedChunksBuffer,
)
from tests.track.helper.data_builder import DEFAULT_START_DATE, DataBuilder


class RecordingTracker(Tracker):
    def __init__(self) -> None:
        self.events: list[str] = []

    def reset(self) -> None:
        self.events.append("reset")

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        self.events.append(Path(frame.source).name)
        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=[],
            finished_tracks=set(),
            discarded_tracks=set(),
        )


@pytest.mark.asyncio
async def test_resets_tracker_once_before_each_frame_group(tmp_path: Path) -> None:
    filenames = [
        "camera_2022-05-04_00-00-00.otdet",
        "camera_2022-05-04_00-00-01.otdet",
        "camera_2022-05-04_00-05-00.otdet",
    ]
    files = [tmp_path / name for name in filenames]
    for file, offset in zip(files, (0, 1, 300)):
        start = DEFAULT_START_DATE + timedelta(seconds=offset)
        otdet = DataBuilder(file, start).batch_append_non_classified_frame(2)
        data = otdet.build_ot_det()
        data[METADATA][VIDEO].update({WIDTH: 1024, HEIGHT: 512})
        data[METADATA][DETECTION] = {NORMALIZED_BBOX: False}
        write_json(data, file, filetype=".otdet")

    recording_tracker = RecordingTracker()
    builder = TrackBuilder()
    tracker = GroupedFilesTracker(
        tracker=recording_tracker,
        chunk_parser=builder.chunk_parser,
        frame_group_parser=builder.frame_group_parser,
        id_generator_factory=lambda _: track_id_generator(),
    )
    pipeline = UnfinishedChunksBuffer(tracker)

    async for _ in pipeline.group_and_track(files):
        pass

    first, second, third = filenames
    assert recording_tracker.events == [
        "reset",
        first,
        first,
        second,
        second,
        "reset",
        third,
        third,
    ]
