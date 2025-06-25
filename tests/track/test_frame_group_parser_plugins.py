import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from OTVision import version
from OTVision.application.config import Config
from OTVision.dataformat import (
    EXPECTED_DURATION,
    FILENAME,
    FIRST_TRACKED_VIDEO_START,
    LAST_TRACKED_VIDEO_END,
    LENGTH,
    METADATA,
    OTTRACK_VERSION,
    OTVISION_VERSION,
    RECORDED_START_DATE,
    TRACKER,
    TRACKING,
    VIDEO,
)
from OTVision.helpers.files import InproperFormattedFilename
from OTVision.track.model.filebased.frame_group import FrameGroup
from OTVision.track.parser.frame_group_parser_plugins import (
    MISSING_START_DATE,
    TimeThresholdFrameGroupParser,
    tracker_metadata,
)
from tests.track.helper.data_builder import (
    DEFAULT_HOSTNAME,
    DEFAULT_START_DATE,
    DataBuilder,
)

DEFAULT_CONFIG = Config()
THRESHOLD = timedelta(minutes=1)
EXPECTED_TRACK_METADATA = tracker_metadata(
    sigma_l=DEFAULT_CONFIG.track.sigma_l,
    sigma_h=DEFAULT_CONFIG.track.sigma_h,
    sigma_iou=DEFAULT_CONFIG.track.sigma_iou,
    t_min=DEFAULT_CONFIG.track.t_min,
    t_miss_max=DEFAULT_CONFIG.track.t_miss_max,
)


class TestTimeThresholdFrameGroupParser:

    def test_convert(self) -> None:
        given = create_given()
        parser = create_target(given)

        order_key = "order-key"
        file_path = Path(f"{order_key}/{DEFAULT_HOSTNAME}_2022-05-04_12-00-01.otdet")
        start_date = datetime(2022, 5, 4, 12, 0, 1, tzinfo=timezone.utc)
        end_date = start_date + timedelta(seconds=1)
        builder = DataBuilder(
            input_file_path=file_path,
            start_date=start_date,
        )
        builder.append_classified_frame()
        otdet = builder.build_ot_det()
        metadata = otdet[METADATA]
        result = parser.convert(file_path, metadata)

        expected = FrameGroup(
            id=1,
            start_date=start_date,
            end_date=end_date,
            hostname=DEFAULT_HOSTNAME,
            files=[file_path],
            metadata_by_file={file_path: metadata},
        )

        assert expected == result

    def test_merge_empty(self) -> None:
        parser = create_target(create_given())
        assert [] == parser.merge([])

    def dummy_frame_groups(
        self,
        time_diff: timedelta,
        hostname_a: str = DEFAULT_HOSTNAME,
        hostname_b: str = DEFAULT_HOSTNAME,
    ) -> tuple[FrameGroup, FrameGroup]:
        start_date_a = datetime(2022, 5, 4, 12, 0, 1, tzinfo=timezone.utc)
        end_date_a = start_date_a + timedelta(minutes=15)
        start_date_b = end_date_a + time_diff
        end_date_b = start_date_b + timedelta(minutes=20)

        file_a = Path("file/a.otdet")
        file_b = Path("file/b.otdet")

        metadata_a = {file_a: {"test": 1, OTTRACK_VERSION: "V_XY"}}
        metadata_b = {file_b: {"test": 2, OTTRACK_VERSION: "V_ZA"}}

        frame_group_a = FrameGroup(
            id=1,
            start_date=start_date_a,
            end_date=end_date_a,
            hostname=hostname_a,
            files=[file_a],
            metadata_by_file=metadata_a,
        )

        frame_group_b = FrameGroup(
            id=2,
            start_date=start_date_b,
            end_date=end_date_b,
            hostname=hostname_b,
            files=[file_b],
            metadata_by_file=metadata_b,
        )
        return frame_group_a, frame_group_b

    def test_merge_to_single_group(self) -> None:
        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(seconds=5)
        )

        expected = FrameGroup(
            id=1,
            start_date=frame_group_a.start_date,
            end_date=frame_group_b.end_date,
            hostname=DEFAULT_HOSTNAME,
            files=frame_group_a.files + frame_group_b.files,
            metadata_by_file={
                **frame_group_a.metadata_by_file,
                **frame_group_b.metadata_by_file,
            },
        )

        parser = create_target(create_given())
        result_1 = parser.merge([frame_group_a, frame_group_b])
        # check merge applies correct ordering
        result_2 = parser.merge([frame_group_b, frame_group_a])

        assert [expected] == result_1
        assert [expected] == result_2

    def test_merge_threshold_exceeded(self) -> None:
        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=THRESHOLD + timedelta(microseconds=1)
        )

        parser = create_target(create_given())
        result_1 = parser.merge([frame_group_a, frame_group_b])
        # check merge applies correct ordering
        result_2 = parser.merge([frame_group_b, frame_group_a])

        expected = [frame_group_a, frame_group_b]
        assert expected == result_1
        assert expected == result_2

    def test_merge_host_changes(self) -> None:
        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(microseconds=5),
            hostname_a="hostA",
            hostname_b="hostB",
        )

        parser = create_target(create_given())
        result_1 = parser.merge([frame_group_a, frame_group_b])
        # check merge applies correct ordering
        result_2 = parser.merge([frame_group_b, frame_group_a])

        expected = [frame_group_a, frame_group_b]
        assert expected == result_1
        assert expected == result_2

    def test_update_metadata(self) -> None:
        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(microseconds=5)
        )
        parser = create_target(create_given())
        merged = parser.merge([frame_group_a, frame_group_b])

        expected_metadata = self.expected_metadata_merged(frame_group_a, frame_group_b)

        result = parser.update_metadata(merged[0])
        assert expected_metadata == result

    def test_get_hostname(self) -> None:
        hostname = "HOSTXYZ"
        metadata = {
            VIDEO: {FILENAME: f"order-key/{hostname}_2022-05-04_13-00-01.otdet"}
        }

        parser = create_target(create_given())
        result = parser.get_hostname(metadata)

        assert hostname == result

    def test_get_hostname_error(self) -> None:
        invalid_metadata = {VIDEO: {FILENAME: "somevideofile.mp4"}}
        parser = create_target(create_given())

        with pytest.raises(InproperFormattedFilename):
            parser.get_hostname(invalid_metadata)

    def expected_metadata_merged(
        self, frame_group_a: FrameGroup, frame_group_b: FrameGroup
    ) -> dict[Path, dict]:
        file_a = Path("file/a.otdet")
        file_b = Path("file/b.otdet")

        expected_metadata = {
            file_a: {
                "test": 1,
                OTTRACK_VERSION: version.ottrack_version(),
                TRACKING: {
                    OTVISION_VERSION: version.otvision_version(),
                    FIRST_TRACKED_VIDEO_START: frame_group_a.start_date.timestamp(),
                    LAST_TRACKED_VIDEO_END: frame_group_b.end_date.timestamp(),
                    TRACKER: EXPECTED_TRACK_METADATA,
                },
            },
            file_b: {
                "test": 2,
                OTTRACK_VERSION: version.ottrack_version(),
                TRACKING: {
                    OTVISION_VERSION: version.otvision_version(),
                    FIRST_TRACKED_VIDEO_START: frame_group_a.start_date.timestamp(),
                    LAST_TRACKED_VIDEO_END: frame_group_b.end_date.timestamp(),
                    TRACKER: EXPECTED_TRACK_METADATA,
                },
            },
        }

        return expected_metadata

    def test_extract_start_date_from(self) -> None:
        date = DEFAULT_START_DATE
        metadata = {
            VIDEO: {
                RECORDED_START_DATE: date.timestamp(),
            }
        }

        parser = create_target(create_given())
        assert date == parser.extract_start_date_from(metadata)

    def test_extract_start_date_from_missing(self) -> None:
        metadata: dict = {VIDEO: {}}

        parser = create_target(create_given())
        assert MISSING_START_DATE == parser.extract_start_date_from(metadata)

    def test_extract_expected_duration_from(self) -> None:
        seconds = 42
        time = timedelta(seconds=seconds)
        metadata = {
            VIDEO: {
                EXPECTED_DURATION: seconds,
            }
        }

        parser = create_target(create_given())
        assert time == parser.extract_expected_duration_from(metadata)

    def test_extract_expected_duration_from_missing(self) -> None:
        seconds = 42
        time = f"00:00:{seconds}"
        expected_time = timedelta(seconds=seconds)
        metadata: dict = {VIDEO: {LENGTH: time, EXPECTED_DURATION: None}}

        parser = create_target(create_given())
        assert expected_time == parser.extract_expected_duration_from(metadata)

    def test_updated_metadata_copy(self) -> None:
        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(seconds=5)
        )
        parser = create_target(create_given())
        merged = parser.merge([frame_group_a, frame_group_b])[0]

        expected = FrameGroup(
            id=1,
            start_date=frame_group_a.start_date,
            end_date=frame_group_b.end_date,
            hostname=DEFAULT_HOSTNAME,
            files=frame_group_a.files + frame_group_b.files,
            metadata_by_file=self.expected_metadata_merged(
                frame_group_a, frame_group_b
            ),
        )

        result = parser.updated_metadata_copy(merged)
        assert expected == result

    @patch.object(TimeThresholdFrameGroupParser, "parse")
    def test_process_all_merged(self, mock_parse: Any) -> None:
        file_a = Path("file/a.otdet")
        file_b = Path("file/b.otdet")

        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(seconds=5)
        )

        mock_parse.side_effect = [frame_group_a, frame_group_b]

        instance = TimeThresholdFrameGroupParser(create_given(), THRESHOLD)

        result = instance.process_all([file_a, file_b])

        mock_parse.assert_has_calls(
            [unittest.mock.call(file_a), unittest.mock.call(file_b)]
        )  # Ensure correct calls
        assert len(result) == 1

        expected = FrameGroup(
            id=0,
            start_date=frame_group_a.start_date,
            end_date=frame_group_b.end_date,
            hostname=DEFAULT_HOSTNAME,
            files=frame_group_a.files + frame_group_b.files,
            metadata_by_file=self.expected_metadata_merged(
                frame_group_a, frame_group_b
            ),
        )
        assert expected == result[0]

    def expected_metadata_of(self, frame_group: FrameGroup, test: int) -> dict:
        return {
            "test": test,
            OTTRACK_VERSION: version.ottrack_version(),
            TRACKING: {
                OTVISION_VERSION: version.otvision_version(),
                FIRST_TRACKED_VIDEO_START: frame_group.start_date.timestamp(),
                LAST_TRACKED_VIDEO_END: frame_group.end_date.timestamp(),
                TRACKER: EXPECTED_TRACK_METADATA,
            },
        }

    @patch.object(TimeThresholdFrameGroupParser, "parse")
    def test_process_all_separate(self, mock_parse: Any) -> None:
        file_a = Path("file/a.otdet")
        file_b = Path("file/b.otdet")

        frame_group_a, frame_group_b = self.dummy_frame_groups(
            time_diff=timedelta(minutes=5)
        )
        mock_parse.side_effect = [frame_group_a, frame_group_b]
        instance = TimeThresholdFrameGroupParser(create_given(), THRESHOLD)

        result = instance.process_all([file_a, file_b])

        mock_parse.assert_has_calls(
            [unittest.mock.call(file_a), unittest.mock.call(file_b)]
        )  # Ensure correct calls
        assert len(result) == 2

        expected = [
            FrameGroup(
                id=0,
                start_date=frame_group_a.start_date,
                end_date=frame_group_a.end_date,
                hostname=DEFAULT_HOSTNAME,
                files=[file_a],
                metadata_by_file={file_a: self.expected_metadata_of(frame_group_a, 1)},
            ),
            FrameGroup(
                id=1,
                start_date=frame_group_b.start_date,
                end_date=frame_group_b.end_date,
                hostname=DEFAULT_HOSTNAME,
                files=[file_b],
                metadata_by_file={file_b: self.expected_metadata_of(frame_group_b, 2)},
            ),
        ]
        assert expected == result


def create_given() -> Mock:
    given = Mock()
    given.get.return_value = DEFAULT_CONFIG
    return given


def create_target(given: Mock) -> TimeThresholdFrameGroupParser:
    return TimeThresholdFrameGroupParser(given)
