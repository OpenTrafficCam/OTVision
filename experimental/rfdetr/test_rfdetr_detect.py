from __future__ import annotations

import argparse
import bz2
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import rfdetr_detect as detect


@dataclass
class FakeDetections:
    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray


def detections(*rows: tuple[tuple[float, float, float, float], int, float]):
    return FakeDetections(
        xyxy=np.array([box for box, _, _ in rows], dtype=np.float32),
        class_id=np.array([class_id for _, class_id, _ in rows]),
        confidence=np.array([score for _, _, score in rows], dtype=np.float32),
    )


def test_same_box_keeps_only_highest_class() -> None:
    box = (10.0, 20.0, 50.0, 60.0)
    rows = detect.detection_rows(detections((box, 2, 0.3), (box, 14, 0.6)), 0.25)

    assert [row["class"] for row in rows] == ["delivery_van"]
    assert rows[0]["confidence"] == pytest.approx(0.6)


def test_distinct_boxes_are_all_kept() -> None:
    rows = detect.detection_rows(
        detections(
            ((10.0, 20.0, 50.0, 60.0), 2, 0.5),
            ((10.0, 20.0, 50.0, 60.5), 2, 0.4),
        ),
        0.25,
    )

    assert len(rows) == 2


def test_threshold_applies_to_best_class_only() -> None:
    box = (0.0, 0.0, 4.0, 4.0)
    below = detect.detection_rows(detections((box, 2, 0.2), (box, 14, 0.1)), 0.25)
    above = detect.detection_rows(detections((box, 2, 0.2), (box, 14, 0.3)), 0.25)

    assert below == []
    assert [row["class"] for row in above] == ["delivery_van"]


def test_rows_use_top_left_xywh_like_otvision() -> None:
    (row,) = detect.detection_rows(detections(((10.0, 20.0, 50.0, 80.0), 0, 0.9)), 0)

    assert (row["x"], row["y"], row["w"], row["h"]) == (10.0, 20.0, 40.0, 60.0)


def test_start_time_from_file_name() -> None:
    video = Path("cam_FR20_2026-06-16_16-43-40.mp4")

    assert detect.start_time(video, None) == datetime(2026, 6, 16, 16, 43, 40, tzinfo=UTC)


def test_start_time_override_wins() -> None:
    video = Path("cam_FR20_2026-06-16_16-43-40.mp4")

    assert detect.start_time(video, "2020-01-02_03-04-05") == datetime(
        2020, 1, 2, 3, 4, 5, tzinfo=UTC
    )


def test_missing_start_time_is_an_error() -> None:
    with pytest.raises(ValueError, match="--start-time"):
        detect.start_time(Path("Video_eine Stunde KFZ.mp4"), None)


def test_otdet_is_bzip2_json_with_frame_occurrences(tmp_path: Path) -> None:
    metadata = {
        "filename": "video",
        "filetype": ".mp4",
        "width": 640,
        "height": 360,
        "recorded_fps": 15.0,
        "actual_fps": 15.0,
        "number_of_frames": 2,
        "recorded_start_date": 1000.0,
        "length": "0:00:00",
    }
    args = argparse.Namespace(
        model_name="RF-DETR-Large",
        weights=Path("weights.pth"),
        resolution=800,
        threshold=0.25,
        dtype="float16",
        batch_size=8,
    )
    row = {"class": "car", "confidence": 0.9, "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    output = tmp_path / "video.otdet"

    detect.atomic_write_otdet(output, detect.otdet_payload(metadata, [[row], []], args))

    with bz2.open(output, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    assert payload["metadata"]["otdet_version"] == "1.4"
    assert payload["data"]["1"] == {"detections": [row], "occurrence": 1000.0}
    assert payload["data"]["2"]["occurrence"] == pytest.approx(1000.0 + 1 / 15)
    assert list(tmp_path.iterdir()) == [output]
