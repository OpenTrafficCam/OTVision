"""
Ultralytics-compatible detection result stub used by ``BotsortTracker``.
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest

from OTVision import dataformat
from OTVision.track.tracker.tracker_plugin_botsort import (
    UltralyticsResultsLite,
    extract_frame_rate_from_metadata,
)


def test_ultralytics_results_lite_len_and_slice() -> None:
    """Slicing preserves conf/xywh/cls shapes for filtered detections."""
    conf = np.array([0.9, 0.4, 0.2], dtype=np.float32)
    xywh = np.array(
        [[10.0, 20.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    cls = np.array([0, 1, 2], dtype=np.int32)
    results = UltralyticsResultsLite(conf=conf, xywh=xywh, cls=cls)

    assert len(results) == 3
    high = conf >= 0.15
    filt = results[high]
    assert len(filt) == 3
    assert filt.conf.shape == (3,)
    assert filt.xywh.shape == (3, 4)
    assert filt.cls.shape == (3,)


def test_ultralytics_results_lite_xyxy_matches_center_box() -> None:
    """Center (10, 20), w=4, h=6 converts to x1=8, y1=17, x2=12, y2=23."""
    conf = np.array([0.5], dtype=np.float32)
    xywh = np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)
    results = UltralyticsResultsLite(conf=conf, xywh=xywh, cls=cls)

    xyxy = results.xyxy
    assert xyxy.shape == (1, 4)
    assert xyxy[0, 0] == pytest.approx(8.0)
    assert xyxy[0, 1] == pytest.approx(17.0)
    assert xyxy[0, 2] == pytest.approx(12.0)
    assert xyxy[0, 3] == pytest.approx(23.0)


def test_ultralytics_results_lite_empty_xyxy() -> None:
    """Empty results expose a zero-row xyxy array."""
    results = UltralyticsResultsLite(
        conf=np.zeros((0,), dtype=np.float32),
        xywh=np.zeros((0, 4), dtype=np.float32),
        cls=np.zeros((0,), dtype=np.int32),
    )
    assert len(results) == 0
    assert results.xyxy.shape == (0, 4)


def test_extract_frame_rate_prefers_actual_over_recorded() -> None:
    """Prefer actual_fps when both actual and recorded values are present."""
    metadata = {
        dataformat.VIDEO: {
            dataformat.RECORDED_FPS: 30.0,
            dataformat.ACTUAL_FPS: 29.97,
        }
    }
    assert extract_frame_rate_from_metadata(metadata) == pytest.approx(29.97)


def test_extract_frame_rate_falls_back_to_recorded() -> None:
    """Fall back to recorded_fps when actual_fps is missing/zero."""
    metadata = {
        dataformat.VIDEO: {
            dataformat.RECORDED_FPS: 25.0,
            dataformat.ACTUAL_FPS: 0.0,
        }
    }
    assert extract_frame_rate_from_metadata(metadata) == pytest.approx(25.0)


def test_extract_frame_rate_handles_string_encoded_fps() -> None:
    """Real .otdet files may store FPS as strings (e.g. '20.0')."""
    metadata = {
        dataformat.VIDEO: {
            dataformat.RECORDED_FPS: "20.0",
            dataformat.ACTUAL_FPS: "0.0",
        }
    }
    assert extract_frame_rate_from_metadata(metadata) == pytest.approx(20.0)
