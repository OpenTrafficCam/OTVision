import copy
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from OTVision.detect.detect import Timestamper
from OTVision.track.preprocess import DATE_FORMAT


@pytest.fixture
def paths_with_legal_fileformats() -> list[Path]:
    return [
        Path("vid_a.mov"),
        Path("vid_b.mkv"),
        Path("vid_c.avi"),
        Path("vid_d.mpg"),
        Path("vid_e.mpeg"),
        Path("vid_f.m4v"),
        Path("vid_g.wmv"),
        Path("img_h.jpeg"),
        Path("img_i.jpg"),
        Path("img_j.png"),
    ]


@pytest.fixture
def paths_with_illegal_fileformats() -> list[Path]:
    return [Path("err_a.video"), Path("err_b.image")]


class TestTimestamper:
    @pytest.mark.parametrize(
        "file_name, start_date",
        [
            ("prefix_FR20_2022-01-01_00-00-00.mp4", datetime(2022, 1, 1, 0, 0, 0)),
            ("Test-Cars_FR20_2022-02-03_04-05-06.mp4", datetime(2022, 2, 3, 4, 5, 6)),
            ("Test_Cars_FR20_2022-02-03_04-05-06.mp4", datetime(2022, 2, 3, 4, 5, 6)),
        ],
    )
    def test_get_start_time_from(self, file_name: str, start_date: datetime) -> None:
        parsed_date = Timestamper()._get_start_time_from(Path(file_name))

        assert parsed_date == start_date

    def test_stamp_frames(self) -> None:
        start_date = datetime(2022, 1, 2, 3, 4, 5)
        time_per_frame = timedelta(microseconds=10000)
        detections: dict[str, dict[str, dict]] = {
            "metadata": {},
            "data": {
                "1": {"classified": []},
                "2": {"classified": [{"class": "car"}]},
                "3": {"classified": []},
            },
        }

        second_frame = start_date + time_per_frame
        third_frame = second_frame + time_per_frame
        expected_dict = copy.deepcopy(detections)
        expected_dict["data"]["1"]["occurrence"] = start_date.strftime(DATE_FORMAT)
        expected_dict["data"]["2"]["occurrence"] = second_frame.strftime(DATE_FORMAT)
        expected_dict["data"]["3"]["occurrence"] = third_frame.strftime(DATE_FORMAT)
        stamped_dict = Timestamper()._stamp(detections, start_date, time_per_frame)

        assert expected_dict == stamped_dict
