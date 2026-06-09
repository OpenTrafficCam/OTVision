from datetime import date
from pathlib import Path

from flatten_camera import flatten_camera


def _f(d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"x")


def test_scoped_flatten_leaves_unselected_day_untouched(tmp_path):
    cam = tmp_path / "OTCamera07"
    _f(cam / "2026-06-03", "OTCamera07_FR20_2026-06-03_00-00-00.otdet")
    _f(cam / "2026-06-03", "._OTCamera07_FR20_2026-06-03_00-00-00.otdet")
    _f(cam / "2026-06-07", "OTCamera07_FR20_2026-06-07_00-00-00.otdet")
    _f(cam / "2026-06-07", "._OTCamera07_FR20_2026-06-07_00-00-00.otdet")
    res = flatten_camera(cam, date_filter=lambda d: d == date(2026, 6, 3), log=lambda m: None)
    assert res.ok and res.moved == 1
    assert (cam / "OTCamera07_FR20_2026-06-03_00-00-00.otdet").exists()
    assert not (cam / "2026-06-03").exists()
    assert (cam / "2026-06-07" / "OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
    assert (cam / "2026-06-07" / "._OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
