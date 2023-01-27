import shutil
from pathlib import Path

import pytest

from OTVision.helpers.files import unzip
from OTVision.pre_annotation import (
    _pre_annotate,
    _write_bbox,
    _write_class_labels,
    _zip_annotated_dir,
    main,
)
from tests.conftest import YieldFixture


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def test_resources_dir() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture
def example_image(test_data_dir: Path) -> Path:
    return Path(test_data_dir, "Testvideo_CamView_Cars-Cyclist.png")


@pytest.fixture
def class_labels() -> dict[int, str]:
    return {0: "car", 1: "person", 2: "cat", 3: "truck", 4: "bicycle"}


@pytest.fixture
def detections() -> list[list[list[float]]]:
    return [
        [
            [0.1, 0.2, 0.512, 0.1234, 0.42, 10.0],
            [0.45, 0.4, 0.612, 0.56, 0.43, 10.0],
        ]
    ]


@pytest.fixture
def cvat_yolo_example_dataset_zipped(
    test_resources_dir: Path, example_image: Path
) -> YieldFixture[Path]:
    # Set up
    pre_annotation_dir = Path(test_resources_dir, "pre_annotation")
    example_dataset = Path(pre_annotation_dir, "example_dataset")
    example_dataset.mkdir(parents=True, exist_ok=True)

    obj_train_data = example_dataset / "obj_train_data"
    obj_train_data.mkdir(parents=True, exist_ok=True)

    # Copy images to obj_train_data folder
    dest_img = obj_train_data / Path(example_image).name

    shutil.copy(src=example_image, dst=dest_img)

    # Create empty annotation text files
    ann = dest_img.with_suffix(".txt")

    ann.touch()

    # Create obj.names, obj.data, train.txt
    obj_names = example_dataset / "obj.names"
    obj_data = example_dataset / "obj.data"
    train_txt = example_dataset / "train.txt"

    obj_names.touch()
    obj_data.touch()
    train_txt.touch()

    # zip folder
    zip_file = example_dataset.with_name(f"{example_dataset.name}.zip")
    shutil.make_archive(str(example_dataset), "zip", root_dir=example_dataset)

    # Remove dataset
    shutil.rmtree(example_dataset)

    yield zip_file

    shutil.rmtree(pre_annotation_dir)


def test_pre_annotate_validDirPassedAsParam_returnsCorrectAnnotationZipFile(
    cvat_yolo_example_dataset_zipped: Path,
) -> None:
    cvat_dir_zipped = _pre_annotate(
        cvat_yolo_zip=cvat_yolo_example_dataset_zipped,
        model_weights="yolov5s",
        chunk_size=100,
        img_type="png",
        filter_classes=None,
    )

    cvat_dir_unzipped = unzip(cvat_dir_zipped)
    obj_train_data = cvat_dir_unzipped / "obj_train_data"
    obj_names = cvat_dir_unzipped / "obj.names"
    files = list(obj_train_data.iterdir())

    assert len(files) == 1
    assert "Testvideo_CamView_Cars-Cyclist" in [f.stem for f in files]
    assert Path(files[0]).suffix == ".txt"
    assert obj_names.stat().st_size > 0


def test_write_class_labels_cvatYoloZipAsParam_writeLabels(
    cvat_yolo_example_dataset_zipped: Path, class_labels: dict[int, str]
) -> None:
    cvat_dir_unzipped = unzip(cvat_yolo_example_dataset_zipped)
    obj_names_file_path = Path(cvat_dir_unzipped, "obj.names")

    _write_class_labels(cvat_yolo_dir=cvat_dir_unzipped, class_labels=class_labels)
    with open(obj_names_file_path) as f:
        obj_names_content = {
            cls_id: line.rstrip() for cls_id, line in enumerate(f.readlines())
        }

    assert len(obj_names_content) == len(class_labels)

    for obj_names_id, cls_id in zip(obj_names_content, class_labels):
        assert obj_names_id == cls_id
        assert obj_names_content[obj_names_id] == class_labels[cls_id]


def test_write_bbox_cvatYoloZipAs1stParam_validDetectionsAs2ndParam_writeBboxes(
    cvat_yolo_example_dataset_zipped: Path, detections: list, class_labels: dict
) -> None:
    cvat_dir_unzipped = unzip(cvat_yolo_example_dataset_zipped)
    _write_bbox(cvat_dir_unzipped, "png", detections, class_labels, None)

    obj_train_data = Path(cvat_dir_unzipped, "obj_train_data")
    cvat_anns = [f for f in obj_train_data.iterdir() if f.suffix == ".txt"]

    assert len(cvat_anns) == len(detections)

    for cvat_ann, bboxes in zip(cvat_anns, detections):
        assert cvat_ann.suffix == ".txt"

        with open(cvat_ann) as f:
            cvat_bboxes = [line.rstrip() for line in f.readlines()]
            for cvat_bbox, bbox in zip(cvat_bboxes, bboxes):
                x, y, w, h, _, _cls = bbox
                cvat_cls, cvat_x, cvat_y, cvat_w, cvat_h = cvat_bbox.split(" ")
                assert x == float(cvat_x)
                assert y == float(cvat_y)
                assert w == float(cvat_w)
                assert h == float(cvat_h)
                assert int(_cls) == int(cvat_cls)


def test_zip_annotated_dir(cvat_yolo_example_dataset_zipped: Path) -> None:
    cvat_dir_unzipped = unzip(cvat_yolo_example_dataset_zipped)
    annotated_zip = _zip_annotated_dir(cvat_dir_unzipped, "png")
    annotated_unzipped = unzip(annotated_zip)
    assert (
        annotated_zip.is_file()
        and f"{cvat_dir_unzipped.name}_annotated" == annotated_zip.stem
        and annotated_zip.suffix == ".zip"
    )

    annotated_obj_train_data = annotated_unzipped / "obj_train_data"
    annotated_obj_train_data_content = [
        f for f in annotated_obj_train_data.iterdir() if f.stem == "png"
    ]

    assert Path(annotated_unzipped, "obj_train_data").is_dir()
    assert Path(annotated_unzipped, "obj.names").exists()
    assert Path(annotated_unzipped, "obj.data").exists()
    assert Path(annotated_unzipped, "train.txt").exists()
    assert not annotated_obj_train_data_content, "Method should remove all images"


def test_main_notExistingPathAsParam_raiseOSError(test_data_dir: Path) -> None:
    path = Path(test_data_dir, "file_not_exists.png")

    with pytest.raises(OSError, match=r"Path at: '.*' does not exist!"):
        main(path, "yolov5s")
