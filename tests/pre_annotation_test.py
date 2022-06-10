import shutil
from pathlib import Path

import pytest

from OTVision.helpers.files import unzip
from OTVision.pre_annotation import (
    _pre_annotate,
    _write_bbox,
    _write_class_labels,
    _zip_annotated_dir,
)


@pytest.fixture
def true_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def test_resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def example_images(true_data_dir):
    img_1 = Path(true_data_dir, "Testvideo_CamView_Cars-Cyclist.png")
    img_2 = Path(true_data_dir, "Testvideo_CamView_Cars-Truck.png")
    return [img_1, img_2]


@pytest.fixture
def class_labels():
    return ["car", "person", "cat", "truck", "bicycle"]


@pytest.fixture
def detections():
    return [
        [
            [0.5, 0.4, 0.2, 0.2, 0.8, 1.0],
            [0.6, 0.5, 0.3, 0.4, 0.7, 2.0],
            [0.5, 0.5, 0.2, 0.2, 0.6, 3.0],
        ],
        [
            [0.1, 0.2, 0.512, 0.1234, 0.42, 10.0],
            [0.45, 0.4, 0.612, 0.56, 0.43, 10.0],
        ],
    ]


@pytest.fixture
def cvat_yolo_example_dataset_zipped(test_resources_dir, example_images):
    # Set up
    pre_annotation_dir = Path(test_resources_dir, "pre_annotation")
    example_dataset = Path(pre_annotation_dir, "example_dataset")
    example_dataset.mkdir(parents=True, exist_ok=True)

    obj_train_data = example_dataset / "obj_train_data"
    obj_train_data.mkdir(parents=True, exist_ok=True)

    # Copy images to obj_train_data folder
    dest_img_0 = obj_train_data / Path(example_images[0]).name
    dest_img_1 = obj_train_data / Path(example_images[1]).name

    shutil.copy(src=example_images[0], dst=dest_img_0)
    shutil.copy(src=example_images[1], dst=dest_img_1)

    # Create empty annotation text files
    ann_0 = dest_img_0.with_suffix(".txt")
    ann_1 = dest_img_1.with_suffix(".txt")

    ann_0.touch()
    ann_1.touch()

    # Create obj.names, obj.data, train.txt
    obj_names = example_dataset / "obj.names"
    obj_data = example_dataset / "obj.data"
    train_txt = example_dataset / "train.txt"

    obj_names.touch()
    obj_data.touch()
    train_txt.touch()

    # zip folder
    zip_file = example_dataset.with_name(f"{example_dataset.name}.zip")
    shutil.make_archive(example_dataset, "zip", root_dir=example_dataset)

    # Remove dataset
    shutil.rmtree(example_dataset)

    yield zip_file

    shutil.rmtree(pre_annotation_dir)


def test_pre_annotate_validDirPassedAsParam_returnsCorrectAnnotationZipFile(
    cvat_yolo_example_dataset_zipped,
):
    cvat_dir_zipped = _pre_annotate(
        cvat_yolo_zip=cvat_yolo_example_dataset_zipped,
        model_weights="yolov5s",
        chunk_size=100,
        img_type="png",
    )

    cvat_dir_unzipped = unzip(cvat_dir_zipped)
    obj_train_data = cvat_dir_unzipped / "obj_train_data"
    obj_names = cvat_dir_unzipped / "obj.names"
    files = [f for f in obj_train_data.iterdir()]

    assert len(files) == 2
    assert "Testvideo_CamView_Cars-Cyclist" in [f.stem for f in files]
    assert "Testvideo_CamView_Cars-Truck" in [f.stem for f in files]
    assert Path(files[0]).suffix == ".txt" and Path(files[1]).suffix == ".txt"
    assert obj_names.stat().st_size > 0


def test_write_class_labels_cvatYoloZipAsParam_writeLabels(
    cvat_yolo_example_dataset_zipped, class_labels
):
    cvat_dir_unzipped = unzip(cvat_yolo_example_dataset_zipped)
    obj_names_file_path = Path(cvat_dir_unzipped, "obj.names")

    _write_class_labels(cvat_yolo_dir=cvat_dir_unzipped, class_labels=class_labels)
    obj_names_content = []
    with open(obj_names_file_path) as f:
        obj_names_content = [line.rstrip() for line in f.readlines()]

    assert len(obj_names_content) == len(class_labels)

    for obj_names_label, class_label in zip(obj_names_content, class_labels):
        assert obj_names_label == class_label


def test_write_bbox_cvatYoloZipAs1stParam_validDetectionsAs2ndParam_writeBboxes(
    cvat_yolo_example_dataset_zipped, detections
):
    cvat_dir_unzipped = unzip(cvat_yolo_example_dataset_zipped)
    _write_bbox(cvat_dir_unzipped, "png", detections)

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


def test_zip_annotated_dir(cvat_yolo_example_dataset_zipped):
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
    assert Path(annotated_unzipped, "train.txt").exists()
    assert len(annotated_obj_train_data_content) == 0, "Method should remove all images"
