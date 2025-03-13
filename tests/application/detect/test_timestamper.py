from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from OTVision.application.frame_count_provider import FrameCountProvider
from OTVision.detect.timestamper import VideoTimestamper, parse_start_time_from
from OTVision.domain.frame import Frame, FrameKeys

SOURCE = "path/to/Test-Cars_FR20_2022-02-03_04-00-00.mp4"
EXPECTED_DURATION = timedelta(seconds=3)
ACTUAL_DURATION = timedelta(seconds=4)
NUMBER_OF_FRAMES = 60
START_TIME = parse_start_time_from(Path(SOURCE), None)


class TestParseStartTimeFromFilename:
    @pytest.mark.parametrize(
        "file_name, start_date",
        [
            (
                "prefix_FR20_2022-01-01_00-00-00.mp4",
                datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            ),
            (
                "Test-Cars_FR20_2022-02-03_04-05-06.mp4",
                datetime(2022, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
            (
                "Test_Cars_FR20_2022-02-03_04-05-06.mp4",
                datetime(2022, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
            (
                "Test_Cars_2022-02-03_04-05-06.mp4",
                datetime(2022, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
            (
                "2022-02-03_04-05-06.mp4",
                datetime(2022, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
            (
                "2022-02-03_04-05-06-suffix.mp4",
                datetime(2022, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
        ],
    )
    def test_get_start_time_from(self, file_name: str, start_date: datetime) -> None:
        parsed_date = parse_start_time_from(Path(file_name), None)

        assert parsed_date == start_date


class TestVideoTimestamper:

    @pytest.mark.parametrize("expected_duration", [EXPECTED_DURATION, None])
    @patch("OTVision.detect.timestamper.get_duration", return_value=ACTUAL_DURATION)
    def test_stamp_frame(
        self, mock_get_duration: Mock, expected_duration: timedelta | None
    ) -> None:
        first_frame = create_frame_without_occurrence(1)
        second_frame = create_frame_without_occurrence(2)
        third_frame = create_frame_without_occurrence(3)
        given_frame_count_provider = create_frame_count_provider()

        if expected_duration is not None:
            time_per_frame = EXPECTED_DURATION / NUMBER_OF_FRAMES
        else:
            time_per_frame = ACTUAL_DURATION / NUMBER_OF_FRAMES

        target = VideoTimestamper(
            video_file=Path(SOURCE),
            expected_duration=expected_duration,
            frame_count_provider=given_frame_count_provider,
            start_time=None,
        )
        actual_first_frame = target.stamp(first_frame)
        actual_second_frame = target.stamp(second_frame)
        actual_third_frame = target.stamp(third_frame)

        assert actual_first_frame == create_expected_frame(first_frame, START_TIME)
        assert actual_second_frame == create_expected_frame(
            second_frame, START_TIME + time_per_frame
        )
        assert actual_third_frame == create_expected_frame(
            third_frame, START_TIME + 2 * time_per_frame
        )
        given_frame_count_provider.provide.assert_called_once_with(Path(SOURCE))

        if expected_duration is None:
            mock_get_duration.assert_called_once_with(Path(SOURCE))
        else:
            mock_get_duration.assert_not_called()


def create_frame_without_occurrence(frame_number: int) -> dict:
    return {
        FrameKeys.data: Mock(),
        FrameKeys.frame: frame_number,
        FrameKeys.source: SOURCE,
    }


def create_frame_count_provider() -> Mock:
    mock = Mock(spec=FrameCountProvider)
    mock.provide.return_value = NUMBER_OF_FRAMES
    return mock


def create_expected_frame(raw_data: dict, occurrence: datetime) -> Frame:
    return Frame(
        data=raw_data[FrameKeys.data],
        frame=raw_data[FrameKeys.frame],
        source=raw_data[FrameKeys.source],
        occurrence=occurrence,
    )
