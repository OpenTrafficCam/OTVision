import bz2
import copy
import json
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jsonschema import validate

import OTVision.config as config
from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DETECTION,
    DETECTIONS,
    METADATA,
    MODEL,
    OCCURRENCE,
    OTDET_VERSION,
    OTVISION_VERSION,
    WEIGHTS,
    H,
    W,
    X,
    Y,
)
from OTVision.detect.detect import (
    DATETIME_FORMAT,
    OTVisionDetect,
    Timestamper,
    derive_filename,
    parse_start_time_from,
)
from OTVision.detect.otdet import OtdetBuilder
from OTVision.detect.yolo import Yolov8, create_model
from tests.conftest import YieldFixture

CYCLIST_VIDEO_LENGTH = timedelta(seconds=3)
DEVIATION = 0.22
BICYCLE_UPPER_LIMIT = int(60 * (1 + DEVIATION))
PERSON_UPPER_LIMIT = int(120 * (1 + DEVIATION))
CAR_UPPER_LIMIT = int(120 * (1 + DEVIATION))
BICYCLE_LOWER_LIMIT = int(60 * (1 - DEVIATION))
PERSON_LOWER_LIMIT = int(120 * (1 - DEVIATION))
CAR_LOWER_LIMIT = int(120 * (1 - DEVIATION))
MODEL_WEIGHTS = (
    "tests/data/yolov8m.mlpackage" if platform.system() == "Darwin" else "yolov8m"
)
EXPECTED_DURATION = timedelta(seconds=3)


CAR = "car"
TRUCK = "truck"
PERSON = "person"
BICYCLE = "bicycle"

otdet_schema = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "vid": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "filetype": {"type": "string"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "fps": {"type": "number"},
                        "frames": {"type": "number"},
                    },
                },
                "det": {
                    "type": "object",
                    "properties": {
                        "detector": {"type": "string"},
                        "weights": {"type": "string"},
                        "conf": {"type": "number"},
                        "iou": {"type": "number"},
                        "size": {"type": "number"},
                        "chunksize": {"type": "number"},
                        "normalized": {"type": "boolean"},
                    },
                },
            },
        }
    },
    "data": {
        "type": "object",
        "properties": {
            "propertyNames": {"pattern": "[1-9][0-9]*"},
            "properties": {
                "classified": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "class": "string",
                            "conf": "number",
                            "x": "number",
                            "y": "number",
                            "w": "number",
                            "h": "number",
                        },
                    },
                }
            },
        },
    },
}


@dataclass
class Detection:
    det_class: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(d[CLASS], d[CONFIDENCE], d[X], d[Y], d[W], d[H])

    def is_normalized(self) -> bool:
        return (
            (self.w >= 0 and self.w < 1)
            and (self.y >= 0 and self.y < 1)
            and (self.w >= 0 and self.w < 1)
            and (self.h >= 0 and self.h < 1)
        )


@dataclass
class Frame:
    number: int
    detections: list[Detection]

    @staticmethod
    def from_dict(frame_number: str, d: dict) -> "Frame":
        detections = [Detection.from_dict(detection) for detection in d[DETECTIONS]]
        return Frame(int(frame_number), detections)


def read_bz2_otdet(otdet: Path) -> dict:
    with bz2.open(otdet, "r") as file:
        result_otdet_json = json.load(file)
    return result_otdet_json


def remove_ignored_metadata(data: dict) -> dict:
    data[OTDET_VERSION] = "ignored"
    data[DETECTION][OTVISION_VERSION] = "ignored"
    return data


def count_classes(frames: list[Frame]) -> dict:
    class_counts: dict[str, int] = {}
    for frame in frames:
        for det in frame.detections:
            if det.det_class in class_counts.keys():
                class_counts[det.det_class] += 1
            else:
                class_counts[det.det_class] = 0
    return class_counts


@pytest.fixture(scope="module")
def detect_test_data_dir(test_data_dir: Path) -> Path:
    return test_data_dir / "detect"


@pytest.fixture(scope="module")
def detect_test_tmp_dir(test_data_tmp_dir: Path) -> YieldFixture[Path]:
    detect_tmp_dir = test_data_tmp_dir / "detect"
    detect_tmp_dir.mkdir(exist_ok=False)
    yield detect_tmp_dir
    shutil.rmtree(detect_tmp_dir)


@pytest.fixture(scope="module")
def cyclist_mp4(detect_test_data_dir: Path, detect_test_tmp_dir: Path) -> Path:
    file_name = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    src = detect_test_data_dir / file_name
    dest = detect_test_tmp_dir / file_name
    shutil.copy2(src, dest)
    return dest


@pytest.fixture(scope="module")
def rotated_cyclist_mp4(detect_test_data_dir: Path, detect_test_tmp_dir: Path) -> Path:
    file_name = "rotated-Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
    src = detect_test_data_dir / file_name
    dest = detect_test_tmp_dir / file_name
    shutil.copy2(src, dest)
    return dest


@pytest.fixture(scope="module")
def truck_mp4(detect_test_data_dir: Path, detect_test_tmp_dir: Path) -> Path:
    file_name = "Testvideo_Cars-Truck_FR20_2020-01-01_00-00-00.mp4"
    src = detect_test_data_dir / file_name
    dest = detect_test_tmp_dir / file_name
    shutil.copy2(src, dest)
    return dest


@pytest.fixture(scope="module")
def default_cyclist_otdet(detect_test_data_dir: Path) -> Path:
    file_name = "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.otdet"
    return detect_test_data_dir / "default" / file_name


@pytest.fixture(scope="session")
def yolov8m() -> Yolov8:
    model_name = (
        "tests/data/yolov8m.mlpackage" if platform.system() == "Darwin" else "yolov8m"
    )
    return create_model(
        weights=model_name,
        confidence=0.25,
        iou=0.45,
        img_size=640,
        half_precision=False,
        normalized=False,
    )


@pytest.mark.parametrize(
    "video_file, detection_file, detect_start, detect_end",
    [
        ("video.mp4", "video.otdet", None, None),
        ("video.mp4", "video_end_20.otdet", None, 20),
        ("video.mp4", "video_start_10.otdet", 10, None),
        ("video.mp4", "video_start_10_end_20.otdet", 10, 20),
    ],
)
def test_derive_filename(
    video_file: str,
    detection_file: str,
    detect_start: int | None,
    detect_end: int | None,
) -> None:
    actual = derive_filename(
        video_file=Path(video_file),
        detect_start=detect_start,
        detect_end=detect_end,
        detect_suffix=".otdet",
    )

    assert actual == Path(detection_file)


class TestDetect:
    conf: float = 0.25
    filetypes: list[str] = config.CONFIG[config.FILETYPES][config.VID]

    @pytest.fixture(scope="class")
    def result_cyclist_otdet(
        self, yolov8m: Yolov8, cyclist_mp4: Path, detect_test_tmp_dir: Path
    ) -> Path:
        target = create_otvision_detect(
            create_config_from(
                paths=[cyclist_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
            )
        )
        target.start()

        return detect_test_tmp_dir / f"{cyclist_mp4.stem}.otdet"

    def test_detect_emptyDirAsParam(self, detect_test_tmp_dir: Path) -> None:
        empty_dir = detect_test_tmp_dir / "empty"
        empty_dir.mkdir()

        target = create_otvision_detect(
            create_config_from(
                paths=[empty_dir],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
            )
        )
        target.start()

        assert os.listdir(empty_dir) == []

    def test_detect_create_otdet(self, result_cyclist_otdet: Path) -> None:
        assert result_cyclist_otdet.exists()

    def test_detect_otdet_valid_json(self, result_cyclist_otdet: Path) -> None:
        read_bz2_otdet(result_cyclist_otdet)

    def test_detect_otdet_matches_schema(self, result_cyclist_otdet: Path) -> None:
        assert result_cyclist_otdet.exists()

        result_cyclist_otdet_json = read_bz2_otdet(result_cyclist_otdet)
        assert result_cyclist_otdet
        validate(result_cyclist_otdet_json, otdet_schema)

    def test_detect_metadata_matches(
        self, result_cyclist_otdet: Path, default_cyclist_otdet: Path
    ) -> None:
        result_cyclist_metadata = remove_ignored_metadata(
            read_bz2_otdet(result_cyclist_otdet)[METADATA]
        )
        expected_cyclist_metadata = remove_ignored_metadata(
            read_bz2_otdet(default_cyclist_otdet)[METADATA]
        )
        result_cyclist_metadata = self.__verify_and_ignore_model_file_name(
            expected_cyclist_metadata, result_cyclist_metadata
        )
        assert result_cyclist_metadata == expected_cyclist_metadata

    def __verify_and_ignore_model_file_name(
        self,
        expected_cyclist_metadata: dict,
        actual_cyclist_metadata: dict,
    ) -> dict:
        actual_model = Path(actual_cyclist_metadata[DETECTION][MODEL][WEIGHTS]).stem
        expected_model = expected_cyclist_metadata[DETECTION][MODEL][WEIGHTS]
        assert actual_model == expected_model
        actual_cyclist_metadata[DETECTION][MODEL][WEIGHTS] = expected_model
        return actual_cyclist_metadata

    def test_detect_error_raised_on_wrong_filetype(
        self, detect_test_tmp_dir: Path
    ) -> None:
        video_file_name = "video.vid"
        detect_error_wrong_filetype_dir = detect_test_tmp_dir / "wrong_filetype"
        detect_error_wrong_filetype_dir.mkdir()
        video_path = detect_error_wrong_filetype_dir / video_file_name
        video_path.touch()
        _config = create_config_from(
            paths=[video_path],
            weights=MODEL_WEIGHTS,
            expected_duration=EXPECTED_DURATION,
        )
        target = create_otvision_detect(_config)
        target.start()

        assert os.listdir(detect_error_wrong_filetype_dir) == [video_file_name]

    def test_detect_bboxes_normalized(self, truck_mp4: Path) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        _config = create_config_from(
            paths=[truck_mp4],
            weights=MODEL_WEIGHTS,
            confidence=0.25,
            normalized=True,
            expected_duration=EXPECTED_DURATION,
        )
        target = create_otvision_detect(_config)
        target.start()

        otdet_dict = read_bz2_otdet(otdet_file)

        detections = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        for det in detections:
            for bbox in det.detections:
                assert bbox.is_normalized()
                assert bbox.conf >= self.conf
        otdet_file.unlink()

    def test_detect_bboxes_denormalized(self, truck_mp4: Path) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        _config = create_config_from(
            paths=[truck_mp4],
            weights=MODEL_WEIGHTS,
            expected_duration=EXPECTED_DURATION,
            normalized=False,
        )
        target = create_otvision_detect(_config)
        target.start()
        otdet_dict = read_bz2_otdet(otdet_file)

        frames = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        denormalized_bbox_found = False
        for frame in frames:
            for det in frame.detections:
                denormalized_bbox_found = (
                    denormalized_bbox_found or not det.is_normalized()
                )
                assert det.conf >= self.conf
        assert denormalized_bbox_found
        otdet_file.unlink()

    @pytest.mark.parametrize("conf", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_detect_conf_bbox_above_thresh(
        self, yolov8m: Yolov8, truck_mp4: Path, conf: float
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        _config = create_config_from(
            paths=[truck_mp4],
            weights=MODEL_WEIGHTS,
            expected_duration=EXPECTED_DURATION,
            confidence=conf,
        )
        target = create_otvision_detect(_config)
        target.start()
        otdet_dict = read_bz2_otdet(otdet_file)

        detections = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        for det in detections:
            for bbox in det.detections:
                assert bbox.conf >= conf
        otdet_file.unlink()

    @pytest.mark.parametrize("overwrite", [True, False])
    def test_detect_overwrite(
        self, yolov8m: Yolov8, truck_mp4: Path, overwrite: bool
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)

        target = create_otvision_detect(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                overwrite=True,
            )
        )
        target.start()

        first_mtime = otdet_file.stat().st_mtime_ns
        target = create_otvision_detect(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                overwrite=overwrite,
            )
        )
        target.start()
        second_mtime = otdet_file.stat().st_mtime_ns

        if overwrite:
            assert first_mtime != second_mtime
        else:
            assert first_mtime == second_mtime
        otdet_file.unlink()

    def test_detect_fulfill_minimum_detection_requirements(
        self, yolov8m: Yolov8, cyclist_mp4: Path
    ) -> None:
        class_counts = self._get_detection_counts_for(cyclist_mp4, CYCLIST_VIDEO_LENGTH)

        assert class_counts[CAR] >= CAR_LOWER_LIMIT
        assert class_counts[PERSON] >= PERSON_LOWER_LIMIT
        assert class_counts[BICYCLE] >= BICYCLE_LOWER_LIMIT
        assert class_counts[CAR] <= CAR_UPPER_LIMIT
        assert class_counts[PERSON] <= PERSON_UPPER_LIMIT
        assert class_counts[BICYCLE] <= BICYCLE_UPPER_LIMIT

    def test_detection_in_rotated_video(
        self,
        yolov8m: Yolov8,
        cyclist_mp4: Path,
        rotated_cyclist_mp4: Path,
        test_data_dir: Path,
        test_data_tmp_dir: Path,
    ) -> None:
        rotated_counts = self._get_detection_counts_for(
            rotated_cyclist_mp4, CYCLIST_VIDEO_LENGTH
        )

        assert rotated_counts[CAR] >= CAR_LOWER_LIMIT
        assert rotated_counts[PERSON] >= PERSON_LOWER_LIMIT
        assert rotated_counts[BICYCLE] >= BICYCLE_LOWER_LIMIT
        assert rotated_counts[CAR] <= CAR_UPPER_LIMIT
        assert rotated_counts[PERSON] <= PERSON_UPPER_LIMIT
        assert rotated_counts[BICYCLE] <= BICYCLE_UPPER_LIMIT

    def _get_detection_counts_for(
        self,
        converted_video: Path,
        expected_duration: timedelta = EXPECTED_DURATION,
    ) -> dict[str, float]:
        target = create_otvision_detect(
            create_config_from(
                paths=[converted_video],
                weights=MODEL_WEIGHTS,
                expected_duration=expected_duration,
                confidence=0.5,
            )
        )
        target.start()
        result_otdet = converted_video.parent / converted_video.with_suffix(".otdet")
        otdet_dict = read_bz2_otdet(result_otdet)
        frames = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        class_counts = count_classes(frames)
        return class_counts

    @patch("OTVision.detect.detect.log")
    @patch("OTVision.detect.detect.create_model")
    def test_detect_(
        self, mock_create_model: Mock, mock_log: Mock, test_data_tmp_dir: Path
    ) -> None:
        model = Mock()
        mock_create_model.return_value = model

        video_file_without_date = test_data_tmp_dir / "video_without_date.mp4"
        video_file_without_date.touch()

        target = create_otvision_detect(
            create_config_from(
                paths=[video_file_without_date],
                weights=MODEL_WEIGHTS,
                expected_duration=timedelta(seconds=3),
            )
        )
        target.start()
        detect_config = target.config.detect
        mock_create_model.assert_called_once_with(
            weights=MODEL_WEIGHTS,
            confidence=detect_config.yolo_config.conf,
            iou=detect_config.yolo_config.iou,
            img_size=detect_config.yolo_config.img_size,
            half_precision=detect_config.half_precision,
            normalized=detect_config.yolo_config.normalized,
        )
        model.detect.assert_not_called()
        mock_log.warning.assert_called_once_with(
            f"Video file name of '{video_file_without_date}' "
            f"must include date and time in format: {DATETIME_FORMAT}"
        )


class TestTimestamper:
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
        parsed_date = parse_start_time_from(Path(file_name))

        assert parsed_date == start_date

    def test_stamp_frames(self) -> None:
        start_date = datetime(2022, 1, 2, 3, 4, 5)
        time_per_frame = timedelta(microseconds=10000)
        detections: dict[str, dict[str, dict]] = {
            METADATA: {},
            DATA: {
                "1": {DETECTIONS: []},
                "2": {DETECTIONS: [{CLASS: "car"}]},
                "3": {DETECTIONS: []},
            },
        }

        second_frame = start_date + time_per_frame
        third_frame = second_frame + time_per_frame
        expected_dict = copy.deepcopy(detections)
        expected_dict[DATA]["1"][OCCURRENCE] = start_date.timestamp()
        expected_dict[DATA]["2"][OCCURRENCE] = second_frame.timestamp()
        expected_dict[DATA]["3"][OCCURRENCE] = third_frame.timestamp()
        stamped_dict = Timestamper()._stamp(detections, start_date, time_per_frame)

        assert expected_dict == stamped_dict


@pytest.fixture
def paths_with_illegal_fileformats() -> list[Path]:
    return [Path("err_a.video"), Path("err_b.image")]


def create_config_from(
    paths: list[Path],
    weights: str,
    expected_duration: timedelta,
    confidence: float | None = None,
    normalized: bool | None = None,
    overwrite: bool | None = None,
) -> config.Config:
    temp_config = config.Config().to_dict()
    temp_config[config.DETECT][config.PATHS] = [str(path) for path in paths]
    temp_config[config.DETECT][config.YOLO][config.WEIGHTS] = weights
    temp_config[config.DETECT][config.EXPECTED_DURATION] = int(
        expected_duration.total_seconds()
    )
    if confidence is not None:
        temp_config[config.DETECT][config.YOLO][config.CONF] = confidence
    if normalized is not None:
        temp_config[config.DETECT][config.YOLO][config.NORMALIZED] = normalized
    if overwrite is not None:
        temp_config[config.DETECT][config.OVERWRITE] = overwrite

    return config.Config.from_dict(temp_config)


def create_otvision_detect(otvision_config: config.Config) -> OTVisionDetect:
    detect = OTVisionDetect(OtdetBuilder())
    detect.update_config(otvision_config)
    return detect
