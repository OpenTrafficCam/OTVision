import bz2
import json
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from jsonschema import validate

import OTVision.config as config
from OTVision.application.update_current_config import UpdateCurrentConfig
from OTVision.dataformat import (
    CLASS,
    CONFIDENCE,
    DATA,
    DETECTION,
    DETECTIONS,
    METADATA,
    MODEL,
    OTDET_VERSION,
    OTVISION_VERSION,
    WEIGHTS,
    H,
    W,
    X,
    Y,
)
from OTVision.detect.builder import DetectBuilder
from OTVision.detect.detect import OTVisionVideoDetect
from tests.conftest import YieldFixture

CONF = 0.25
IOU = 0.45
IMG_SIZE = 640
HALF_PRECISION = False
NORMALIZED = False
OVERWRITE = True


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


class TestDetect:
    conf: float = 0.25
    filetypes: list[str] = config.CONFIG[config.FILETYPES][config.VID]

    @pytest.fixture(scope="class")
    def detect_builder(self) -> DetectBuilder:
        return DetectBuilder()

    @pytest.fixture(scope="class")
    def update_current_config(
        self, detect_builder: DetectBuilder
    ) -> UpdateCurrentConfig:
        return detect_builder.update_current_config

    @pytest.fixture(scope="class")
    def otvision_detect(self, detect_builder: DetectBuilder) -> OTVisionVideoDetect:
        return detect_builder.build()

    @pytest.fixture(scope="class")
    def result_cyclist_otdet(
        self,
        otvision_detect: OTVisionVideoDetect,
        cyclist_mp4: Path,
        detect_test_tmp_dir: Path,
        update_current_config: UpdateCurrentConfig,
    ) -> Path:
        update_current_config.update(
            create_config_from(
                paths=[cyclist_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
            )
        )
        otvision_detect.start()

        return detect_test_tmp_dir / f"{cyclist_mp4.stem}.otdet"

    def test_detect_emptyDirAsParam(
        self,
        otvision_detect: OTVisionVideoDetect,
        detect_test_tmp_dir: Path,
        update_current_config: UpdateCurrentConfig,
    ) -> None:
        empty_dir = detect_test_tmp_dir / "empty"
        empty_dir.mkdir()
        update_current_config.update(
            create_config_from(
                paths=[empty_dir],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
            )
        )
        otvision_detect.start()

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
        self,
        otvision_detect: OTVisionVideoDetect,
        detect_test_tmp_dir: Path,
        update_current_config: UpdateCurrentConfig,
    ) -> None:
        video_file_name = "video.vid"
        detect_error_wrong_filetype_dir = detect_test_tmp_dir / "wrong_filetype"
        detect_error_wrong_filetype_dir.mkdir()
        video_path = detect_error_wrong_filetype_dir / video_file_name
        video_path.touch()
        update_current_config.update(
            create_config_from(
                paths=[video_path],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
            )
        )
        otvision_detect.start()

        assert os.listdir(detect_error_wrong_filetype_dir) == [video_file_name]

    def test_detect_bboxes_normalized(
        self,
        otvision_detect: OTVisionVideoDetect,
        truck_mp4: Path,
        update_current_config: UpdateCurrentConfig,
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        update_current_config.update(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                confidence=0.25,
                normalized=True,
                expected_duration=EXPECTED_DURATION,
            )
        )
        otvision_detect.start()

        otdet_dict = read_bz2_otdet(otdet_file)

        detections = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        for det in detections:
            for bbox in det.detections:
                assert bbox.is_normalized()
                assert bbox.conf >= self.conf
        otdet_file.unlink()

    def test_detect_bboxes_denormalized(
        self,
        otvision_detect: OTVisionVideoDetect,
        truck_mp4: Path,
        update_current_config: UpdateCurrentConfig,
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        update_current_config.update(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                normalized=False,
            )
        )
        otvision_detect.start()
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
        self,
        otvision_detect: OTVisionVideoDetect,
        update_current_config: UpdateCurrentConfig,
        truck_mp4: Path,
        conf: float,
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        update_current_config.update(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                confidence=conf,
            )
        )
        otvision_detect.start()
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
        self,
        otvision_detect: OTVisionVideoDetect,
        update_current_config: UpdateCurrentConfig,
        truck_mp4: Path,
        overwrite: bool,
    ) -> None:
        otdet_file = truck_mp4.parent / truck_mp4.with_suffix(".otdet")
        otdet_file.unlink(missing_ok=True)
        update_current_config.update(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                overwrite=True,
            )
        )
        otvision_detect.start()

        first_mtime = otdet_file.stat().st_mtime_ns
        update_current_config.update(
            create_config_from(
                paths=[truck_mp4],
                weights=MODEL_WEIGHTS,
                expected_duration=EXPECTED_DURATION,
                overwrite=overwrite,
            )
        )
        otvision_detect.start()
        second_mtime = otdet_file.stat().st_mtime_ns

        if overwrite:
            assert first_mtime != second_mtime
        else:
            assert first_mtime == second_mtime
        otdet_file.unlink()

    def test_detect_fulfill_minimum_detection_requirements(
        self,
        otvision_detect: OTVisionVideoDetect,
        update_current_config: UpdateCurrentConfig,
        cyclist_mp4: Path,
    ) -> None:
        class_counts = self._get_detection_counts_for(
            otvision_detect, update_current_config, cyclist_mp4, CYCLIST_VIDEO_LENGTH
        )

        assert class_counts[CAR] >= CAR_LOWER_LIMIT
        assert class_counts[PERSON] >= PERSON_LOWER_LIMIT
        assert class_counts[BICYCLE] >= BICYCLE_LOWER_LIMIT
        assert class_counts[CAR] <= CAR_UPPER_LIMIT
        assert class_counts[PERSON] <= PERSON_UPPER_LIMIT
        assert class_counts[BICYCLE] <= BICYCLE_UPPER_LIMIT

    def test_detection_in_rotated_video(
        self,
        otvision_detect: OTVisionVideoDetect,
        update_current_config: UpdateCurrentConfig,
        cyclist_mp4: Path,
        rotated_cyclist_mp4: Path,
        test_data_dir: Path,
        test_data_tmp_dir: Path,
    ) -> None:
        rotated_counts = self._get_detection_counts_for(
            otvision_detect,
            update_current_config,
            rotated_cyclist_mp4,
            CYCLIST_VIDEO_LENGTH,
        )

        assert rotated_counts[CAR] >= CAR_LOWER_LIMIT
        assert rotated_counts[PERSON] >= PERSON_LOWER_LIMIT
        assert rotated_counts[BICYCLE] >= BICYCLE_LOWER_LIMIT
        assert rotated_counts[CAR] <= CAR_UPPER_LIMIT
        assert rotated_counts[PERSON] <= PERSON_UPPER_LIMIT
        assert rotated_counts[BICYCLE] <= BICYCLE_UPPER_LIMIT

    def _get_detection_counts_for(
        self,
        otvision_detect: OTVisionVideoDetect,
        update_current_config: UpdateCurrentConfig,
        converted_video: Path,
        expected_duration: timedelta = EXPECTED_DURATION,
    ) -> dict[str, float]:
        update_current_config.update(
            create_config_from(
                paths=[converted_video],
                weights=MODEL_WEIGHTS,
                expected_duration=expected_duration,
                confidence=0.5,
                overwrite=True,
            )
        )
        otvision_detect.start()
        result_otdet = converted_video.parent / converted_video.with_suffix(".otdet")
        otdet_dict = read_bz2_otdet(result_otdet)
        frames = [
            Frame.from_dict(number, det) for number, det in otdet_dict[DATA].items()
        ]
        class_counts = count_classes(frames)
        return class_counts


@pytest.fixture
def paths_with_illegal_fileformats() -> list[Path]:
    return [Path("err_a.video"), Path("err_b.image")]


def create_config_from(
    paths: list[Path],
    weights: str,
    expected_duration: timedelta,
    confidence: float = CONF,
    normalized: bool = NORMALIZED,
    overwrite: bool = OVERWRITE,
) -> config.Config:
    temp_config = config.Config().to_dict()
    temp_config[config.DETECT][config.PATHS] = [str(path) for path in paths]
    temp_config[config.DETECT][config.YOLO][config.WEIGHTS] = weights
    temp_config[config.DETECT][config.EXPECTED_DURATION] = int(
        expected_duration.total_seconds()
    )
    temp_config[config.DETECT][config.YOLO][config.CONF] = confidence
    temp_config[config.DETECT][config.YOLO][config.NORMALIZED] = normalized
    temp_config[config.DETECT][config.OVERWRITE] = overwrite

    return config.Config.from_dict(temp_config)
