"""Tests for video frame provider module."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from OTVision.dataformat import FILENAME, FILETYPE, VIDEO
from OTVision.track.video_frame_provider import (
    PyAvVideoFrameProvider,
    SequentialVideoFrameProvider,
    resolve_video_path_from_otdet,
)


class TestResolveVideoPathFromOtdet:
    """Tests for resolve_video_path_from_otdet function."""

    def test_resolve_video_path_basic(self) -> None:
        """Test basic video path resolution from OTDET metadata."""
        otdet_file = Path("/path/to/detections/video_2020-01-01.otdet")
        metadata = {
            VIDEO: {
                FILENAME: "video_2020-01-01",
                FILETYPE: ".mp4",
            }
        }

        result = resolve_video_path_from_otdet(otdet_file, metadata)

        expected = Path("/path/to/detections/video_2020-01-01.mp4")
        assert result == expected

    def test_resolve_video_path_h264(self) -> None:
        """Test video path resolution with h264 extension."""
        otdet_file = Path("/data/Testvideo_FR20_2020-01-01_00-00-00.otdet")
        metadata = {
            VIDEO: {
                FILENAME: "Testvideo_FR20_2020-01-01_00-00-00",
                FILETYPE: ".h264",
            }
        }

        result = resolve_video_path_from_otdet(otdet_file, metadata)

        expected = Path("/data/Testvideo_FR20_2020-01-01_00-00-00.h264")
        assert result == expected

    def test_resolve_video_path_missing_video_key_raises(self) -> None:
        """Test that missing VIDEO key raises KeyError."""
        otdet_file = Path("/path/to/video.otdet")
        metadata: dict[str, Any] = {"other_key": {}}

        with pytest.raises(KeyError):
            resolve_video_path_from_otdet(otdet_file, metadata)

    def test_resolve_video_path_missing_filename_raises(self) -> None:
        """Test that missing FILENAME key raises KeyError."""
        otdet_file = Path("/path/to/video.otdet")
        metadata = {
            VIDEO: {
                FILETYPE: ".mp4",
            }
        }

        with pytest.raises(KeyError):
            resolve_video_path_from_otdet(otdet_file, metadata)


class TestPyAvVideoFrameProvider:
    """Tests for PyAvVideoFrameProvider class."""

    def test_missing_video_file_raises_file_not_found(self) -> None:
        """Test that missing video file raises FileNotFoundError."""
        non_existent_path = Path("/nonexistent/video.mp4")

        with pytest.raises(FileNotFoundError) as exc_info:
            PyAvVideoFrameProvider(non_existent_path)

        assert "Video file not found" in str(exc_info.value)
        assert "Appearance-based trackers require" in str(exc_info.value)

    def test_get_frame_from_real_video(self, test_data_dir: Path) -> None:
        """Test getting a frame from a real video file."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = PyAvVideoFrameProvider(video_path)

        try:
            frame = provider.get_frame(1)

            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 3  # Height, Width, Channels
            assert frame.shape[2] == 3  # BGR format
        finally:
            provider.close()

    def test_get_frame_invalid_frame_number_returns_none(
        self, test_data_dir: Path
    ) -> None:
        """Test that invalid frame numbers return None."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = PyAvVideoFrameProvider(video_path)

        try:
            # Frame 0 is invalid (1-indexed)
            frame = provider.get_frame(0)
            assert frame is None

            # Negative frame number
            frame = provider.get_frame(-1)
            assert frame is None
        finally:
            provider.close()


class TestSequentialVideoFrameProvider:
    """Tests for SequentialVideoFrameProvider class."""

    def test_missing_video_file_raises_file_not_found(self) -> None:
        """Test that missing video file raises FileNotFoundError."""
        non_existent_path = Path("/nonexistent/video.mp4")

        with pytest.raises(FileNotFoundError) as exc_info:
            SequentialVideoFrameProvider(non_existent_path)

        assert "Video file not found" in str(exc_info.value)

    def test_sequential_frame_access(self, test_data_dir: Path) -> None:
        """Test sequential frame access."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = SequentialVideoFrameProvider(video_path)

        try:
            # Get first few frames sequentially
            frame1 = provider.get_frame(1)
            frame2 = provider.get_frame(2)
            frame3 = provider.get_frame(3)

            assert frame1 is not None
            assert frame2 is not None
            assert frame3 is not None

            # All frames should be valid numpy arrays
            for frame in [frame1, frame2, frame3]:
                assert isinstance(frame, np.ndarray)
                assert len(frame.shape) == 3
                assert frame.shape[2] == 3  # BGR
        finally:
            provider.close()

    def test_backward_seek_restarts_video(self, test_data_dir: Path) -> None:
        """Test that requesting an earlier frame restarts the video."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = SequentialVideoFrameProvider(video_path)

        try:
            # Get frame 5 first
            frame5 = provider.get_frame(5)
            assert frame5 is not None

            # Request frame 2 (before current position)
            # This should restart the video
            frame2 = provider.get_frame(2)
            assert frame2 is not None

            # Frames should be different
            assert not np.array_equal(frame5, frame2)
        finally:
            provider.close()

    def test_get_frame_invalid_frame_number_returns_none(
        self, test_data_dir: Path
    ) -> None:
        """Test that invalid frame numbers return None."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = SequentialVideoFrameProvider(video_path)

        try:
            # Frame 0 is invalid (1-indexed)
            frame = provider.get_frame(0)
            assert frame is None
        finally:
            provider.close()


class TestVideoFrameProviderProtocol:
    """Tests verifying protocol compliance."""

    def test_pyav_provider_implements_protocol(self, test_data_dir: Path) -> None:
        """Test that PyAvVideoFrameProvider implements VideoFrameProvider protocol."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = PyAvVideoFrameProvider(video_path)

        # Check protocol methods exist
        assert hasattr(provider, "get_frame")
        assert hasattr(provider, "close")
        assert callable(provider.get_frame)
        assert callable(provider.close)

        provider.close()

    def test_sequential_provider_implements_protocol(self, test_data_dir: Path) -> None:
        """Test that SequentialVideoFrameProvider implements VideoFrameProvider."""
        video_path = (
            test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"
        )

        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        provider = SequentialVideoFrameProvider(video_path)

        # Check protocol methods exist
        assert hasattr(provider, "get_frame")
        assert hasattr(provider, "close")
        assert callable(provider.get_frame)
        assert callable(provider.close)

        provider.close()
