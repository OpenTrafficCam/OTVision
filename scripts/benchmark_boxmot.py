#!/usr/bin/env python3
"""Performance benchmark for BOXMOT tracker integration.

This script provides lightweight performance validation for BOXMOT tracking:
- Frames per second (FPS) comparison: IOU vs ByteTrack
- Memory usage over multiple frames
- Track ID mapping growth analysis

Run with: uv run scripts/benchmark_boxmot.py
"""

import time
from typing import Iterator

import numpy as np

from OTVision.domain.detection import Detection, TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo
from OTVision.track.tracker.tracker_plugin_iou import IouTracker

try:
    from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter

    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("⚠️  BOXMOT not installed. Install with: uv pip install -e .[tracking_boxmot]")
    print("Benchmarking will only run for IOU tracker.\n")


def create_synthetic_frame(frame_no: int, num_detections: int = 10) -> DetectedFrame:
    """Create a synthetic detected frame for benchmarking."""
    detections = []
    for i in range(num_detections):
        # Simulate moving objects
        x = 100.0 + (i * 50) + (frame_no * 2)
        y = 150.0 + (i * 60) + (frame_no * 1.5)

        detections.append(
            Detection(
                label="car" if i % 2 == 0 else "pedestrian",
                conf=0.85 + (i % 10) * 0.01,
                x=x,
                y=y,
                w=50.0,
                h=80.0,
            )
        )

    return DetectedFrame(
        no=FrameNo(frame_no),
        occurrence=frame_no * 0.033,  # ~30 FPS
        source="benchmark_video.mp4",
        output="benchmark.otdet",
        detections=detections,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def id_generator_factory() -> Iterator[TrackId]:
    """Create a new ID generator."""
    track_id = 0
    while True:
        yield track_id
        track_id += 1


def benchmark_tracker(
    tracker_name: str, tracker: object, num_frames: int = 100, num_detections: int = 10
) -> dict:
    """Benchmark a tracker and return performance metrics."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {tracker_name}")
    print(f"{'=' * 60}")
    print(f"Frames: {num_frames}, Detections/frame: {num_detections}")

    id_gen = id_generator_factory()
    tracked_frames = []

    start_time = time.time()

    for i in range(num_frames):
        frame = create_synthetic_frame(i, num_detections)
        tracked_frame = tracker.track_frame(frame, id_gen)
        tracked_frames.append(tracked_frame)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate metrics
    fps = num_frames / elapsed_time
    avg_detections_per_frame = sum(len(f.detections) for f in tracked_frames) / len(
        tracked_frames
    )
    total_finished_tracks = sum(len(f.finished_tracks) for f in tracked_frames)
    total_discarded_tracks = sum(len(f.discarded_tracks) for f in tracked_frames)

    # Memory metrics (track ID mapping size if available)
    mapping_size = 0
    if hasattr(tracker, "_track_id_mapping"):
        mapping_size = len(tracker._track_id_mapping)

    # Results
    results = {
        "tracker": tracker_name,
        "num_frames": num_frames,
        "num_detections": num_detections,
        "elapsed_time": elapsed_time,
        "fps": fps,
        "avg_detections_per_frame": avg_detections_per_frame,
        "total_finished_tracks": total_finished_tracks,
        "total_discarded_tracks": total_discarded_tracks,
        "mapping_size": mapping_size,
    }

    # Print results
    print(f"\n{'Results:':20} ")
    print(f"  {'Elapsed time:':<25} {elapsed_time:.2f}s")
    print(f"  {'FPS:':<25} {fps:.1f} frames/sec")
    print(f"  {'Avg detections/frame:':<25} {avg_detections_per_frame:.1f}")
    print(f"  {'Total finished tracks:':<25} {total_finished_tracks}")
    print(f"  {'Total discarded tracks:':<25} {total_discarded_tracks}")
    if mapping_size > 0:
        print(f"  {'Track ID mapping size:':<25} {mapping_size}")

    return results


def main() -> None:
    """Run performance benchmarks."""
    print("\n" + "=" * 60)
    print(" BOXMOT Integration Performance Benchmark")
    print("=" * 60)

    results = []

    # Benchmark IOU tracker
    print("\n[1/2] Benchmarking IOU Tracker...")
    try:
        from OTVision.application.config import Config
        from OTVision.application.get_current_config import GetCurrentConfig
        from OTVision.domain.current_config import CurrentConfig

        current_config = CurrentConfig(Config())
        get_current_config = GetCurrentConfig(current_config)

        iou_tracker = IouTracker(get_current_config=get_current_config)
        iou_results = benchmark_tracker(
            "IOU Tracker", iou_tracker, num_frames=100, num_detections=10
        )
        results.append(iou_results)
    except Exception as e:
        print(f"❌ Failed to benchmark IOU tracker: {e}")

    # Benchmark BOXMOT ByteTrack
    if BOXMOT_AVAILABLE:
        print("\n[2/2] Benchmarking BOXMOT ByteTrack...")
        try:
            boxmot_tracker = BoxmotTrackerAdapter(
                tracker_type="bytetrack", device="cpu", half=False
            )
            boxmot_results = benchmark_tracker(
                "BOXMOT ByteTrack", boxmot_tracker, num_frames=100, num_detections=10
            )
            results.append(boxmot_results)
        except Exception as e:
            print(f"❌ Failed to benchmark BOXMOT tracker: {e}")
    else:
        print("\n[2/2] Skipping BOXMOT benchmark (not installed)")

    # Summary comparison
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print(" Performance Comparison")
        print("=" * 60)

        iou_fps = results[0]["fps"]
        boxmot_fps = results[1]["fps"]
        speedup = boxmot_fps / iou_fps if iou_fps > 0 else 0

        print(f"\n  IOU Tracker:      {iou_fps:.1f} FPS")
        print(f"  BOXMOT ByteTrack: {boxmot_fps:.1f} FPS")
        print(f"  Speedup:          {speedup:.2f}x")

        if speedup > 1:
            print(f"\n✅ BOXMOT is {speedup:.1f}x faster than IOU tracker")
        else:
            print(f"\n⚠️  BOXMOT is slower than IOU tracker (may need optimization)")

    print("\n" + "=" * 60)
    print(" Benchmark Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
