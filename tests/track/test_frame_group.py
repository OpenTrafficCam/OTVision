from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from OTVision.dataformat import EXPECTED_DURATION, FILENAME, RECORDED_START_DATE, VIDEO
from OTVision.track.model.filebased.frame_group import FrameGroup
from tests.track.helper.data_builder import (
    DEFAULT_HOSTNAME,
    DEFAULT_INPUT_FILE_PATH,
    DEFAULT_START_DATE,
)


class TestFrameGroup:

    def test_start_date(self) -> None:
        start_date = datetime(2022, 5, 4, 12, 0, 0)
        group = self._create_frame_group(start_date=start_date)

        assert group.start_date == start_date

    def test_end_date(self) -> None:
        end_date = datetime(2022, 5, 4, 12, 0, 1)
        group = self._create_frame_group(end_date=end_date)

        assert group.end_date == end_date

    def _dummy_frame_groups(self) -> tuple[FrameGroup, FrameGroup]:
        first_start = datetime(2022, 5, 4, 12, 0, 0)
        first_end = first_start + timedelta(seconds=1)
        second_start = first_end + timedelta(seconds=1)
        second_end = second_start + timedelta(seconds=1)
        first_group = self._create_frame_group(
            start_date=first_start,
            end_date=first_end,
            input_file_path=Path("file/a.otdet"),
        )
        second_group = self._create_frame_group(
            start_date=second_end,
            end_date=second_end,
            input_file_path=Path("file/b.otdet"),
        )
        return (first_group, second_group)

    def test_merge(self) -> None:
        first_group, second_group = self._dummy_frame_groups()

        merge_first_second: FrameGroup = first_group.merge(second_group)
        merge_second_first: FrameGroup = second_group.merge(first_group)

        assert merge_first_second.start_date == first_group.start_date
        assert merge_first_second.end_date == second_group.end_date
        assert merge_first_second.hostname == DEFAULT_HOSTNAME
        assert len(merge_first_second.files) == 2

        assert merge_first_second.files == merge_second_first.files
        assert merge_first_second.start_date == merge_second_first.start_date
        assert merge_first_second.end_date == merge_second_first.end_date
        assert merge_first_second.hostname == merge_second_first.hostname
        assert len(merge_first_second.metadata_by_file) == 2
        assert len(merge_second_first.metadata_by_file) == 2

    def test_with_id(self) -> None:
        frame_group = self._create_frame_group()

        expected_id = 42
        result = frame_group.with_id(expected_id)
        assert expected_id == result.id

    def test_get_output_files(self) -> None:
        first_group, second_group = self._dummy_frame_groups()
        merged: FrameGroup = first_group.merge(second_group)

        result = merged.get_output_files(with_suffix=".exmpl")
        expected = [Path("file/a.exmpl"), Path("file/b.exmpl")]

        assert expected == result

    def _mock_file(self, is_file: bool) -> MagicMock:
        mock_file = MagicMock(spec=Path)
        mock_file.is_file.return_value = is_file
        return mock_file

    @patch.object(FrameGroup, "get_output_files")
    def test_get_existing_output_files(self, mock_get_output_files: Any) -> None:

        mock_file1 = self._mock_file(True)
        mock_file2 = self._mock_file(False)
        mock_file3 = self._mock_file(True)
        mock_get_output_files.return_value = [mock_file1, mock_file2, mock_file3]

        instance = FrameGroup(
            id=1,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_START_DATE,
            hostname=DEFAULT_HOSTNAME,
            files=[],
            metadata_by_file=dict(),
        )

        result = instance.get_existing_output_files(".exmpl")
        expected = [mock_file1, mock_file3]

        assert expected == result

    @patch.object(FrameGroup, "get_output_files")
    def test_check_any_output_file_exists_true(
        self, mock_get_output_files: Any
    ) -> None:
        mock_file1 = self._mock_file(False)
        mock_file2 = self._mock_file(True)
        mock_file3 = self._mock_file(False)
        mock_get_output_files.return_value = [mock_file1, mock_file2, mock_file3]

        instance = FrameGroup(
            id=1,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_START_DATE,
            hostname=DEFAULT_HOSTNAME,
            files=[],
            metadata_by_file=dict(),
        )

        assert instance.check_any_output_file_exists(".exmpl")

    @patch.object(FrameGroup, "get_output_files")
    def test_check_any_output_file_exists_false(
        self, mock_get_output_files: Any
    ) -> None:
        mock_file1 = self._mock_file(False)
        mock_file2 = self._mock_file(False)
        mock_file3 = self._mock_file(False)
        mock_get_output_files.return_value = [mock_file1, mock_file2, mock_file3]

        instance = FrameGroup(
            id=1,
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_START_DATE,
            hostname=DEFAULT_HOSTNAME,
            files=[],
            metadata_by_file=dict(),
        )

        assert not instance.check_any_output_file_exists(".exmpl")

    def _create_frame_group(
        self,
        start_date: datetime = DEFAULT_START_DATE,
        end_date: datetime = DEFAULT_START_DATE,
        hostname: str = DEFAULT_HOSTNAME,
        input_file_path: Path = DEFAULT_INPUT_FILE_PATH,
    ) -> FrameGroup:
        return FrameGroup(
            id=1,
            start_date=start_date,
            end_date=end_date,
            hostname=hostname,
            files=[input_file_path],
            metadata_by_file={
                input_file_path: {
                    VIDEO: {
                        FILENAME: input_file_path.as_posix(),
                        RECORDED_START_DATE: start_date.timestamp(),
                        EXPECTED_DURATION: 1,
                    }
                }
            },
        )
