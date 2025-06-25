import unittest
from datetime import datetime, timedelta

from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    FINISHED,
    FIRST,
    FRAME,
    INTERPOLATED_DETECTION,
    OCCURRENCE,
    TRACK_ID,
    H,
    W,
    X,
    Y,
)
from OTVision.domain.detection import (
    Detection,
    FinishedDetection,
    TrackedDetection,
    TrackId,
)
from OTVision.domain.frame import DetectedFrame, TrackedFrame
from tests.track.helper.data_builder import DEFAULT_START_DATE, DataBuilder


class TestDetection:

    def data_builder(self) -> DataBuilder:
        return DataBuilder().batch_append_classified_frames(number_of_classifications=5)

    def detections(self) -> list[Detection]:
        return list(self.data_builder().build_objects()[1].detections)

    def test_of_track(self) -> None:
        detections = self.detections()
        print(f"test_of_track: {len(detections)}")

        id: TrackId = 0
        is_first: bool = True
        for det in detections:
            id += 1
            is_first = not is_first

            tracked_det = det.of_track(id, is_first)

            assert det.label == tracked_det.label
            assert det.conf == tracked_det.conf
            assert det.x == tracked_det.x
            assert det.y == tracked_det.y
            assert det.w == tracked_det.w
            assert det.h == tracked_det.h
            assert is_first == tracked_det.is_first
            assert id == tracked_det.track_id


class TestTrackedDetection(TestDetection):

    def tracked_detections(self) -> list[TrackedDetection]:
        tracked_dets: list[TrackedDetection] = []

        id: TrackId = 0
        is_first: bool = True
        for det in super().detections():
            id += 1
            is_first = not is_first
            tracked_dets.append(det.of_track(id, is_first))
        return tracked_dets

    def assert_correctness(
        self,
        is_last: bool,
        is_discarded: bool,
        det: TrackedDetection,
        finished_det: FinishedDetection,
    ) -> None:
        assert det.label == finished_det.label
        assert det.conf == finished_det.conf
        assert det.x == finished_det.x
        assert det.y == finished_det.y
        assert det.w == finished_det.w
        assert det.h == finished_det.h
        assert det.is_first == finished_det.is_first
        assert det.track_id == finished_det.track_id
        assert is_last == finished_det.is_last
        assert is_discarded == finished_det.is_discarded

    def test_finish(self) -> None:
        tracked_dets = self.tracked_detections()

        is_last: bool = True
        is_discarded: bool = True
        for det in tracked_dets:
            is_last = not is_last
            is_discarded = is_last != is_discarded

            finished_det = det.finish(is_last, is_discarded)
            self.assert_correctness(is_last, is_discarded, det, finished_det)

    def test_as_last_detection(self) -> None:
        tracked_dets = self.tracked_detections()

        is_discarded: bool = True
        for det in tracked_dets:
            is_discarded = not is_discarded

            finished_det = det.as_last_detection(is_discarded)
            self.assert_correctness(True, is_discarded, det, finished_det)

    def test_as_intermediate_detection(self) -> None:
        tracked_dets = self.tracked_detections()

        is_discarded: bool = True
        for det in tracked_dets:
            is_discarded = not is_discarded

            finished_det = det.as_intermediate_detection(is_discarded)
            self.assert_correctness(False, is_discarded, det, finished_det)


class TestFinishedDetection(TestTrackedDetection):

    def finished_detections(self) -> list[FinishedDetection]:
        finished_dets: list[FinishedDetection] = []

        is_last: bool = True
        is_discarded: bool = True
        for det in super().tracked_detections():
            is_last = not is_last
            is_discarded = is_last != is_discarded

            finished_dets.append(det.finish(is_last, is_discarded))
        return finished_dets

    def test_to_dict(self) -> None:
        dets = self.finished_detections()

        for det in dets:
            dict = det.to_dict()

            assert dict[CLASS] == det.label
            assert dict[CONFIDENCE] == det.conf
            assert dict[X] == det.x
            assert dict[Y] == det.y
            assert dict[W] == det.w
            assert dict[H] == det.h
            assert not dict[INTERPOLATED_DETECTION]
            assert dict[FIRST] == det.is_first
            assert dict[FINISHED] == det.is_last
            assert dict[TRACK_ID] == det.track_id

    def test_from_tracked_detection(self) -> None:
        tracked_dets = super().tracked_detections()
        finished_dets = self.finished_detections()

        for expected, tracked in zip(finished_dets, tracked_dets):
            finished = FinishedDetection.from_tracked_detection(
                tracked, expected.is_last, expected.is_discarded
            )
            assert expected == finished


class TestTrackedFrame:

    def data_builder(self) -> DataBuilder:
        return DataBuilder().batch_append_classified_frames(
            number_of_frames=4, number_of_classifications=5
        )

    def frames(self) -> list[DetectedFrame]:
        return list(self.data_builder().objects.values())

    def create_tracked_frames(self) -> list[tuple[TrackedFrame, list[TrackId]]]:
        tracked_frames: list[tuple[TrackedFrame, list[TrackId]]] = []

        id_count: TrackId = 0
        is_first = True
        for frame in self.frames():
            tracked_dets = []
            ids: list[TrackId] = []

            for det in frame.detections:
                id_count += 1
                is_first = not is_first
                ids.append(id_count)
                tracked_dets.append(det.of_track(id_count, is_first))

            max_id = max(ids)
            finished_ids = {max_id + 1, ids[-1]}
            discarded_ids = {max_id + 2, ids[-2]}

            tracked_frame = TrackedFrame(
                no=frame.no,
                occurrence=frame.occurrence,
                source=frame.source,
                output=frame.output,
                detections=tuple(tracked_dets),
                image=None,
                finished_tracks=finished_ids,
                discarded_tracks=discarded_ids,
            )

            tracked_frames.append((tracked_frame, ids))

        return tracked_frames

    def test_init_tracked_frame(self) -> None:
        for frame, ids in self.create_tracked_frames():
            assert frame.observed_tracks == set(ids)
            assert frame.unfinished_tracks == set(ids[:-2])

    def test_finish_keep_discarded(self) -> None:
        frames_and_ids: list[tuple[TrackedFrame, list[TrackId]]] = []
        discarded_ids: set[TrackId] = set()
        finished_ids: set[TrackId] = set()

        for frame, ids in self.create_tracked_frames():
            frames_and_ids.append((frame, ids))
            discarded_ids.update(frame.discarded_tracks)
            finished_ids.update(frame.finished_tracks)

        def is_last(no: int, id: int) -> bool:
            return id in finished_ids

        for frame, ids in frames_and_ids:
            finished = frame.finish(is_last, discarded_ids, True)

            assert len(finished.detections) == 5

            for det in finished.detections:
                assert det.is_last == (det.track_id in finished_ids)
                assert det.is_discarded == (det.track_id in discarded_ids)

    def test_finish_drop_discarded(self) -> None:
        frames_and_ids: list[tuple[TrackedFrame, list[TrackId]]] = []
        discarded_ids: set[TrackId] = set()
        finished_ids: set[TrackId] = set()

        for frame, ids in self.create_tracked_frames():
            frames_and_ids.append((frame, ids))
            discarded_ids.update(frame.discarded_tracks)
            finished_ids.update(frame.finished_tracks)

        def is_last(no: int, id: int) -> bool:
            return id in finished_ids

        for frame, ids in frames_and_ids:
            finished = frame.finish(is_last, discarded_ids, False)

            assert len(finished.detections) == len(
                frame.observed_tracks.difference(discarded_ids)
            )

            for det in finished.detections:
                assert det.is_last == (det.track_id in finished_ids)
                assert not det.is_discarded


class TestFinishedFrame(unittest.TestCase):

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
            source=self.mock_file,
            output=self.mock_file,
            detections=[self._mock_detection(track_id=i) for i in observed],
            image=None,
            finished_tracks=finished,
            discarded_tracks=discarded,
        )

    def _is_last(self, frame_no: int, track_id: int) -> bool:
        return (track_id, frame_no) in [(1, 1), (2, 1), (3, 3), (4, 3)]

    def setUp(self) -> None:
        self.mock_file = "/mock/path"

        mock_tracked_frame_1 = self._mock_frame(
            no=1,
            observed={1, 2, 3},
            finished=set(),
            discarded=set(),
        )

        mock_tracked_frame_2 = self._mock_frame(
            no=2,
            observed={3, 4},
            finished=set(),
            discarded={1},
        )

        mock_tracked_frame_3 = self._mock_frame(
            no=3,
            observed={3, 4},
            finished={2},
            discarded=set(),
        )

        frames = [
            mock_tracked_frame_1,
            mock_tracked_frame_2,
            mock_tracked_frame_3,
        ]

        self.finished_frames = [
            frame.finish(self._is_last, discarded_tracks={1}, keep_discarded=False)
            for frame in frames
        ]

    def _mock_expected_dict(self, frame: int, det: FinishedDetection) -> dict:
        return {
            **det.to_dict(),
            FRAME: frame,
            OCCURRENCE: self._mock_occurrence(frame).timestamp(),
            # INPUT_FILE_PATH: self.mock_file,
        }

    def expected_dicts_1(self) -> list[dict]:
        return [
            self._mock_expected_dict(
                frame=1,
                det=self._mock_detection(2).finish(is_last=True, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=1,
                det=self._mock_detection(3).finish(is_last=False, is_discarded=False),
            ),
        ]

    def expected_dicts_2(self) -> list[dict]:
        return [
            self._mock_expected_dict(
                frame=2,
                det=self._mock_detection(3).finish(is_last=False, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=2,
                det=self._mock_detection(4).finish(is_last=False, is_discarded=False),
            ),
        ]

    def expected_dicts_3(self) -> list[dict]:
        return [
            self._mock_expected_dict(
                frame=3,
                det=self._mock_detection(3).finish(is_last=True, is_discarded=False),
            ),
            self._mock_expected_dict(
                frame=3,
                det=self._mock_detection(4).finish(is_last=True, is_discarded=False),
            ),
        ]

    def test_to_detection_dicts(self) -> None:
        result1 = self.finished_frames[0].to_detection_dicts()
        assert self.expected_dicts_1() == result1

        result2 = self.finished_frames[1].to_detection_dicts()
        assert self.expected_dicts_2() == result2

        result3 = self.finished_frames[2].to_detection_dicts()
        assert self.expected_dicts_3() == result3
