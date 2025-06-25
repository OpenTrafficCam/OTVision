import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from OTVision.dataformat import FRAME, INPUT_FILE_PATH, OCCURRENCE
from OTVision.domain.detection import FinishedDetection, TrackedDetection
from OTVision.domain.frame import TrackedFrame
from OTVision.track.model.filebased.frame_chunk import (
    FinishedChunk,
    FrameChunk,
    TrackedChunk,
)
from tests.track.helper.data_builder import DEFAULT_START_DATE


class TestFrameChunk:

    def _mock_file(self, is_file: bool) -> MagicMock:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = is_file
        return mock_file

    @patch("OTVision.track.model.filebased.frame_chunk.get_output_file")
    def test_check_output_file_exists(self, mock_get_output_files: Any) -> None:
        mock_file = self._mock_file(True)
        mock_get_output_files.return_value = mock_file

        instance = FrameChunk(
            file=Path("/mock/path"),
            metadata={"test": 42},
            frames=tuple(),
            frame_group_id=1,
        )

        result = instance.check_output_file_exists(".exmpl")
        assert result

    @patch("OTVision.track.model.filebased.frame_chunk.get_output_file")
    def test_check_output_file_exists_false(self, mock_get_output_files: Any) -> None:
        mock_file = self._mock_file(False)
        mock_get_output_files.return_value = mock_file

        print(mock_get_output_files())

        instance = FrameChunk(
            file=Path("/mock/path"),
            metadata={"test": 42},
            frames=tuple(),
            frame_group_id=1,
        )

        result = instance.check_output_file_exists(".exmpl")
        assert not result


class TestTrackedChunk(unittest.TestCase):

    def _mock_detection(self, track_id: int) -> TrackedDetection:
        return TrackedDetection(
            label="car", conf=1.0, x=5, y=5, w=5, h=5, is_first=False, track_id=track_id
        )

    def _mock_occurrence(self, frame: int) -> datetime:
        return DEFAULT_START_DATE + timedelta(microseconds=frame)

    def _mock_frame(
        self, no: int, observed: set[int], finished: set[int], discarded: set[int]
    ) -> TrackedFrame:
        return TrackedFrame(
            no=no,
            occurrence=self._mock_occurrence(no),
            source=str(self.mock_file),
            output=str(self.mock_file),
            detections=[self._mock_detection(track_id=i) for i in observed],
            image=None,
            finished_tracks=finished,
            discarded_tracks=discarded,
        )

    def _is_last(self, frame_no: int, track_id: int) -> bool:
        return (track_id, frame_no) in [(1, 1), (2, 1), (3, 3), (4, 3)]

    def setUp(self) -> None:
        self.mock_file = Path("/mock/path")

        self.mock_tracked_frame_1 = self._mock_frame(
            no=1,
            observed={1, 2, 3},
            finished=set(),
            discarded=set(),
        )

        self.mock_tracked_frame_2 = self._mock_frame(
            no=2,
            observed={3, 4},
            finished=set(),
            discarded={1},
        )

        self.mock_tracked_frame_3 = self._mock_frame(
            no=3,
            observed={3, 4},
            finished={2},
            discarded=set(),
        )

        self.frames = [
            self.mock_tracked_frame_1,
            self.mock_tracked_frame_2,
            self.mock_tracked_frame_3,
        ]

    def test_init_last_chunk(self) -> None:
        metadata = {"test": 42}
        chunk = TrackedChunk(
            file=self.mock_file,
            metadata=metadata,
            is_last_chunk=True,
            frames=self.frames,
            frame_group_id=1,
        )

        assert chunk.file == self.mock_file
        assert chunk.metadata == metadata
        assert chunk.observed_tracks == {1, 2, 3, 4}
        assert chunk.finished_tracks == {2, 3, 4}
        assert chunk.discarded_tracks == {1}
        assert chunk.unfinished_tracks == set()

        assert chunk.last_track_frame == {1: 1, 2: 1, 3: 3, 4: 3}

        assert chunk.frames[-1].unfinished_tracks == set()
        assert chunk.frames[-1].finished_tracks == {2, 3, 4}
        assert chunk.frames[-1].discarded_tracks == set()
        assert chunk.frames[-1].observed_tracks == {3, 4}

    def test_init_not_last_chunk(self) -> None:
        metadata = {"test": 42}
        chunk = TrackedChunk(
            file=self.mock_file,
            metadata=metadata,
            is_last_chunk=False,
            frames=self.frames,
            frame_group_id=1,
        )

        assert chunk.file == self.mock_file
        assert chunk.metadata == metadata
        assert chunk.observed_tracks == {1, 2, 3, 4}
        assert chunk.finished_tracks == {2}
        assert chunk.discarded_tracks == {1}
        assert chunk.unfinished_tracks == {3, 4}

        assert chunk.last_track_frame == {1: 1, 2: 1, 3: 3, 4: 3}

    def test_finish_keep_discarded(self) -> None:
        metadata = {"test": 42}
        chunk = TrackedChunk(
            file=self.mock_file,
            metadata=metadata,
            is_last_chunk=True,
            frames=self.frames,
            frame_group_id=1,
        )

        finished = chunk.finish(
            self._is_last,
            discarded_tracks={1},
            keep_discarded=True,
        )

        assert finished.file == self.mock_file
        assert finished.metadata == metadata

        assert len(finished.frames) == 3
        assert len(finished.frames[0].detections) == 3
        assert len(finished.frames[1].detections) == 2
        assert len(finished.frames[2].detections) == 2

    def test_finish_drop_discarded(self) -> None:
        metadata = {"test": 42}
        chunk = TrackedChunk(
            file=self.mock_file,
            metadata=metadata,
            is_last_chunk=True,
            frames=self.frames,
            frame_group_id=1,
        )

        finished = chunk.finish(
            self._is_last,
            discarded_tracks={1},
            keep_discarded=False,
        )

        assert finished.file == self.mock_file
        assert finished.metadata == metadata

        assert len(finished.frames) == 3
        assert len(finished.frames[0].detections) == 2
        assert len(finished.frames[1].detections) == 2
        assert len(finished.frames[2].detections) == 2


class TestFinishedChunk(TestTrackedChunk):

    def _tracked_chunk(self, is_last_chunk: bool) -> TrackedChunk:
        return TrackedChunk(
            file=self.mock_file,
            metadata={"test": 42},
            is_last_chunk=is_last_chunk,
            frames=self.frames,
            frame_group_id=1,
        )

    def _finished_chunk(
        self, is_last_chunk: bool, keep_discarded: bool
    ) -> FinishedChunk:
        return self._tracked_chunk(is_last_chunk).finish(
            is_last=self._is_last,
            discarded_tracks={1},
            keep_discarded=keep_discarded,
        )

    def _mock_expected_dict(self, frame: int, det: FinishedDetection) -> dict:
        return {
            **det.to_dict(),
            FRAME: frame,
            OCCURRENCE: self._mock_occurrence(frame).timestamp(),
            INPUT_FILE_PATH: self.mock_file.as_posix(),
        }

    def _expected_dicts_drop_discarded(self, last_chunk: bool) -> list[dict]:
        return [
            self._mock_expected_dict(
                frame=1,
                det=self._mock_detection(2).finish(is_last=True, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=1,
                det=self._mock_detection(3).finish(is_last=False, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=2,
                det=self._mock_detection(3).finish(is_last=False, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=2,
                det=self._mock_detection(4).finish(is_last=False, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=3,
                det=self._mock_detection(3).finish(last_chunk, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=3,
                det=self._mock_detection(4).finish(last_chunk, is_discarded=False),
            ),
        ]

    def _expected_dicts_keep_discarded(self, last_chunk: bool) -> list[dict]:
        return [
            self._mock_expected_dict(
                frame=1,
                det=self._mock_detection(1).finish(
                    is_last=last_chunk, is_discarded=True
                ),
            )
        ] + self._expected_dicts_drop_discarded(last_chunk)

    def test_to_detection_dicts_keep_discarded(self) -> None:
        for is_last_chunk in [True, False]:
            with self.subTest(is_last_chunk=is_last_chunk):
                result = self._finished_chunk(
                    is_last_chunk, keep_discarded=True
                ).to_detection_dicts()
                expected = self._expected_dicts_keep_discarded(last_chunk=True)

                assert expected == result

    def test_to_detection_dicts_drop_discarded(self) -> None:
        for is_last_chunk in [True, False]:
            with self.subTest(is_last_chunk=is_last_chunk):
                result = self._finished_chunk(
                    is_last_chunk, keep_discarded=False
                ).to_detection_dicts()
                expected = self._expected_dicts_drop_discarded(last_chunk=True)

                assert expected == result
