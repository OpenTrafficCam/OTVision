"""Ultralytics-compatible detection result stub used by ``BotsortTracker``."""

import numpy as np
import pytest

from OTVision.track.tracker.tracker_plugin_botsort import UltralyticsResultsLite


def test_ultralytics_results_lite_len_and_slice() -> None:
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
    # center (10, 20), w=4, h=6 -> x1=8, y1=17, x2=12, y2=23
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
    results = UltralyticsResultsLite(
        conf=np.zeros((0,), dtype=np.float32),
        xywh=np.zeros((0, 4), dtype=np.float32),
        cls=np.zeros((0,), dtype=np.int32),
    )
    assert len(results) == 0
    assert results.xyxy.shape == (0, 4)
