import bz2
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from otc_state import get_tracked_through
from watch_cameras import (
    WatchConfig,
    discover_cameras,
    now_utc,
    process_camera,
    safe_parallelism,
    verify_outputs,
)


def _make_day(cam, day):
    d = cam / f"{day:%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    for h in range(24):
        for m in (0, 15, 30, 45):
            f = d / f"OTCamera07_FR20_{day:%Y-%m-%d}_{h:02d}-{m:02d}-00.otdet"
            f.write_bytes(b"x")
            os.utime(f, (1_000_000, 1_000_000))


def _cfg(tmp_path):
    return WatchConfig(
        config=Path("config.continuous.botsort.yaml"),
        log_dir=tmp_path / "logs",
        block_days=4,
        idle_minutes=5,
        stable_minutes=0,
        slots_per_day=96,
    )


def test_fires_flattens_tracks_verifies_marks(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    seen = {}

    def fake_flatten(camera, date_filter=None, log=None, **k):
        for f in list(camera.rglob("*.otdet")):
            from otc_coverage import parse_otc_filename

            if date_filter(parse_otc_filename(f.name)[1].date()):
                f.rename(camera / f.name)
        from flatten_camera import FlattenResult

        return FlattenResult(camera=camera)

    def fake_track(paths, log=None):
        seen["n"] = len(paths)
        for p in paths:
            with bz2.open(p.with_suffix(".ottrk"), "wt") as fh:
                fh.write("{}")
        return True

    out = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
        cfg=_cfg(tmp_path),
        flatten_fn=fake_flatten,
        track_fn=fake_track,
        log=lambda m: None,
    )
    assert out.status == "tracked" and seen["n"] == 96 * 4
    assert get_tracked_through(cam) == date(2026, 6, 6)


def test_no_mark_when_ottrk_missing(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))

    def flat(camera, date_filter=None, log=None, **k):
        for f in list(camera.rglob("*.otdet")):
            from otc_coverage import parse_otc_filename

            if date_filter(parse_otc_filename(f.name)[1].date()):
                f.rename(camera / f.name)
        from flatten_camera import FlattenResult

        return FlattenResult(camera=camera)

    out = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
        cfg=_cfg(tmp_path),
        flatten_fn=flat,
        track_fn=lambda paths, log=None: True,
        log=lambda m: None,
    )
    assert out.status == "failed" and get_tracked_through(cam) is None


def test_discover(tmp_path):
    (tmp_path / "OTCamera07").mkdir()
    (tmp_path / "otcamera23").mkdir()
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert {p.name for p in discover_cameras(tmp_path)} == {"OTCamera07", "otcamera23"}


def test_safe_parallelism_caps_to_cores():
    assert safe_parallelism(1000, reserve=2) <= (os.cpu_count() or 4)
    assert safe_parallelism(1, reserve=2) == 1


REAL = Path("/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OTV_LIVE_VOLUME") != "1" or not REAL.exists(),
    reason="set OTV_LIVE_VOLUME=1 with the volume mounted",
)
def test_end_to_end_small(tmp_path):
    src = REAL / "2026-06-03"
    samples = sorted(src.glob("OTCamera07_FR20_2026-06-03_01-3*-00.otdet"))[:2]
    if len(samples) < 2:
        pytest.skip("samples unavailable")
    cam = tmp_path / "OTCamera07" / "2026-06-03"
    cam.mkdir(parents=True)
    for s in samples:
        shutil.copy2(s, cam / s.name)
    camera = tmp_path / "OTCamera07"
    cfg = WatchConfig(
        config=Path("config.continuous.botsort.yaml"),
        log_dir=tmp_path / "logs",
        block_days=1,
        idle_minutes=0,
        stable_minutes=0,
        slots_per_day=2,
    )
    out = process_camera(camera, now=now_utc(), cfg=cfg)
    assert out.status == "tracked"
    assert len(list(camera.glob("*.ottrk"))) == 2
    assert verify_outputs([camera / s.name for s in samples]) == []
