from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

from OTVision.dataformat import DATA, DETECTIONS, FRAME
from OTVision.domain.detection import FinishedDetection
from OTVision.domain.frame import FinishedFrame
from OTVision.track.exporter.filebased_exporter import FinishedChunkTrackExporter
from OTVision.track.model.filebased.frame_chunk import FinishedChunk

VIDEO_START = datetime(2026, 6, 24, 10, 0, 0)
FPS = 20.0
FIRST_GLOBAL_FRAME_NO = 101  # chunk is not the first file of its frame group


def create_detection(track_id: int) -> FinishedDetection:
    return FinishedDetection(
        label="car",
        conf=0.9,
        x=1.0,
        y=2.0,
        w=3.0,
        h=4.0,
        is_first=False,
        track_id=track_id,
        is_last=False,
        is_discarded=False,
    )


def create_frame(
    global_no: int, detections: Sequence[FinishedDetection]
) -> FinishedFrame:
    local_frame_index = global_no - FIRST_GLOBAL_FRAME_NO
    return FinishedFrame(
        no=global_no,
        occurrence=VIDEO_START + timedelta(seconds=local_frame_index / FPS),
        source="video.otdet",
        output="video.otdet",
        detections=detections,
        finished_tracks=set(),
        discarded_tracks=set(),
    )


class TestFinishedChunkTrackExporter:
    @patch("OTVision.track.model.track_exporter.write_json")
    def test_reindex_rebases_frames_to_video_not_to_first_detection(
        self, mock_write_json: MagicMock
    ) -> None:
        """Detection frame numbers in the written ottrk must match the source
        video's frame numbering. The chunk's first frames have no detections,
        so rebasing by the first detection would wrongly shift all frame
        numbers towards 1 while occurrence stays at wall clock time."""
        chunk = FinishedChunk(
            file=Path("video.otdet"),
            metadata={"tracking": {}},
            is_last_chunk=False,
            frames=[
                create_frame(101, []),  # video frame 1: no detections
                create_frame(102, [create_detection(1)]),  # video frame 2
                create_frame(103, [create_detection(1)]),  # video frame 3
            ],
            frame_group_id=1,
        )

        FinishedChunkTrackExporter().export_frames(
            chunk, tracking_run_id="run", overwrite=True
        )

        written = mock_write_json.call_args.kwargs["dict_to_write"]
        written_frame_nos = [det[FRAME] for det in written[DATA][DETECTIONS]]
        assert written_frame_nos == [2, 3]

    @patch("OTVision.track.model.track_exporter.write_json")
    def test_export_chunk_without_detections_writes_empty_detections(
        self, mock_write_json: MagicMock
    ) -> None:
        chunk = FinishedChunk(
            file=Path("video.otdet"),
            metadata={"tracking": {}},
            is_last_chunk=False,
            frames=[create_frame(101, [])],
            frame_group_id=1,
        )

        FinishedChunkTrackExporter().export_frames(
            chunk, tracking_run_id="run", overwrite=True
        )

        written = mock_write_json.call_args.kwargs["dict_to_write"]
        assert written[DATA][DETECTIONS] == []
