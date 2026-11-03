#!/usr/bin/env python3
"""Experimental RF-DETR detection that writes OTVision .otdet files.

Lean port of the Horizon site runner (horizon_multicam
scripts/site_detection/site_detect.py): RF-DETR's TorchScript path, frame
batching and atomic .otdet 1.4 output. RF-DETR ranks every query/class pair,
so one query can emit the same box under several classes; only the best class
per box is kept.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

OTC_CLASSES = {
    0: "pedestrian",
    1: "bicyclist",
    2: "car",
    3: "motorcyclist",
    4: "private_van",
    5: "bus",
    6: "train",
    7: "truck",
    8: "scooter_driver",
    9: "cargo_bike_driver",
    10: "bicyclist_with_trailer",
    11: "car_with_trailer",
    12: "private_van_with_trailer",
    13: "truck_with_trailer",
    14: "delivery_van",
    15: "delivery_van_with_trailer",
    16: "truck_with_semitrailer",
    17: "other",
}
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
START_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
FPS_RE = re.compile(r"_FR(\d+)_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_otdet(output: Path, payload: dict[str, object]) -> None:
    """Write through a sibling .part file and publish only on success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(f"{output.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with bz2.open(part, "wt", encoding="utf-8") as destination:
            json.dump(payload, destination, separators=(",", ":"))
        with part.open("r+b") as source:
            os.fsync(source.fileno())
        os.replace(part, output)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def start_time(video: Path, override: str | None) -> datetime:
    """Recording start from --start-time or the file name, both in UTC."""
    if override is not None:
        text = override
    else:
        match = START_RE.search(video.stem)
        if match is None:
            raise ValueError(
                f"No start time in {video.name}; pass --start-time "
                f"in {DATETIME_FORMAT} format"
            )
        text = match.group(0)
    return datetime.strptime(text, DATETIME_FORMAT).replace(tzinfo=UTC)


def video_metadata(
    video: Path, capture: cv2.VideoCapture, start: datetime
) -> dict[str, object]:
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if actual_fps <= 0:
        raise ValueError(f"Invalid frame rate for {video}: {actual_fps}")
    fps_match = FPS_RE.search(video.stem)
    recorded_fps = float(fps_match.group(1)) if fps_match else actual_fps
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "filename": video.stem,
        "filetype": video.suffix,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "recorded_fps": recorded_fps,
        "actual_fps": actual_fps,
        "number_of_frames": frames,
        "recorded_start_date": start.timestamp(),
        "length": format_length(frames, recorded_fps),
    }


def format_length(number_of_frames: int, fps: float) -> str:
    seconds = round(number_of_frames / fps) if fps > 0 else 0
    return str(timedelta(seconds=seconds))


def otdet_payload(
    metadata: dict[str, object],
    frame_rows: list[list[dict[str, object]]],
    args: argparse.Namespace,
) -> dict[str, object]:
    fps = float(metadata["actual_fps"])  # type: ignore[arg-type]
    start = float(metadata["recorded_start_date"])  # type: ignore[arg-type]
    return {
        "metadata": {
            "otdet_version": "1.4",
            "video": {
                **metadata,
                "number_of_frames": len(frame_rows),
                "length": format_length(
                    len(frame_rows),
                    float(metadata["recorded_fps"]),  # type: ignore[arg-type]
                ),
            },
            "detection": {
                "otvision_version": "0.0",
                "model": {
                    "name": args.model_name,
                    "weights": str(args.weights),
                    "iou_threshold": None,
                    "image_size": args.resolution,
                    "max_confidence": args.threshold,
                    "half_precision": args.dtype == "float16",
                    "classes": {str(key): value for key, value in OTC_CLASSES.items()},
                },
                "chunksize": args.batch_size,
                "normalized_bbox": False,
                "detect_start": None,
                "detect_end": None,
            },
        },
        "data": {
            str(index): {
                "detections": rows,
                "occurrence": start + (index - 1) / fps,
            }
            for index, rows in enumerate(frame_rows, start=1)
        },
    }


def detection_rows(detections: object, threshold: float) -> list[dict[str, object]]:
    """Convert one frame's detections, keeping the best class per box.

    Boxes are compared exactly: class duplicates of one query share the same
    coordinates, while distinct queries practically never do.
    """
    best: dict[tuple[float, ...], tuple[float, int]] = {}
    for box, class_id, confidence in zip(
        detections.xyxy,  # type: ignore[attr-defined]
        detections.class_id,  # type: ignore[attr-defined]
        detections.confidence,  # type: ignore[attr-defined]
    ):
        key = tuple(float(value) for value in box.tolist())
        score = float(confidence)
        if key not in best or score > best[key][0]:
            best[key] = (score, int(class_id))
    rows: list[dict[str, object]] = []
    for (x1, y1, x2, y2), (score, class_id) in best.items():
        if score <= threshold:
            continue
        rows.append(
            {
                "class": OTC_CLASSES.get(class_id, f"class_{class_id}"),
                "confidence": score,
                "x": x1,
                "y": y1,
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
            }
        )
    return rows


def decode_batch(capture: cv2.VideoCapture, batch_size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for _ in range(batch_size):
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return frames


def predict_batch(
    model: object, images: list[np.ndarray], batch_size: int, threshold: float
) -> list[object]:
    """Predict a batch, padding to the fixed batch size the JIT trace expects."""
    padded = list(images)
    while len(padded) < batch_size:
        padded.append(images[-1])
    result = model.predict(  # type: ignore[attr-defined]
        padded, threshold=threshold, include_source_image=False
    )
    results = result if isinstance(result, list) else [result]
    if len(results) != len(padded):
        raise ValueError(f"Expected {len(padded)} detection results")
    return results[: len(images)]


def load_model(args: argparse.Namespace) -> object:
    from rfdetr import RFDETRLarge

    model = RFDETRLarge(
        pretrain_weights=str(args.weights),
        resolution=args.resolution,
        num_classes=len(OTC_CLASSES),
    )
    model.optimize_for_inference(
        compile=True,
        batch_size=args.batch_size,
        dtype=torch.float16 if args.dtype == "float16" else torch.float32,
    )
    return model


def detect_video(
    model: object, video: Path, output: Path, args: argparse.Namespace
) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    frame_rows: list[list[dict[str, object]]] = []
    try:
        metadata = video_metadata(video, capture, start_time(video, args.start_time))
        while frames := decode_batch(capture, args.batch_size):
            for result in predict_batch(model, frames, args.batch_size, args.threshold):
                frame_rows.append(detection_rows(result, args.threshold))
    finally:
        capture.release()
    expected = int(metadata["number_of_frames"])  # type: ignore[arg-type]
    if len(frame_rows) != expected:
        raise ValueError(
            f"{video.name}: expected {expected} frames, decoded {len(frame_rows)}; "
            "refusing to write a short .otdet"
        )
    atomic_write_otdet(output, otdet_payload(metadata, frame_rows, args))
    return len(frame_rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("-p", "--paths", type=Path, nargs="+", required=True)
    result.add_argument("-w", "--weights", type=Path, required=True)
    result.add_argument("--weights-sha256", required=True)
    result.add_argument("--model-name", default="RF-DETR-Large")
    result.add_argument("--resolution", type=int, default=800)
    result.add_argument("--threshold", type=float, default=0.25)
    result.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument(
        "--start-time",
        help=f"recording start in UTC, {DATETIME_FORMAT}; default: from file name",
    )
    result.add_argument(
        "-o", "--output", type=Path, help="output .otdet path (single video only)"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.output is not None and len(args.paths) != 1:
        raise SystemExit("--output needs exactly one video")
    actual_sha = sha256_file(args.weights)
    if actual_sha != args.weights_sha256:
        raise SystemExit(f"Unexpected weights SHA-256: {actual_sha}")
    model = load_model(args)
    for video in args.paths:
        output = args.output or video.with_suffix(".otdet")
        if output.exists():
            print(f"skip {video.name}: {output} exists", flush=True)
            continue
        started = time.perf_counter()
        frames = detect_video(model, video, output, args)
        elapsed = time.perf_counter() - started
        print(
            f"{video.name}: {frames} frames in {elapsed:.0f}s "
            f"({frames / elapsed:.1f} frames/s) -> {output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
