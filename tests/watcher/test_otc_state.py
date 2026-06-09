from datetime import date, datetime, timedelta
from pathlib import Path

from otc_state import check_stable, get_tracked_through, set_tracked_through


def test_marker_roundtrip(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    assert get_tracked_through(cam) is None
    set_tracked_through(cam, date(2026, 6, 6), days=4, files=384, at="t1")
    assert get_tracked_through(cam) == date(2026, 6, 6)


def test_stability_requires_unchanged_for_window(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    f = cam / "a.otdet"
    f.write_bytes(b"x")
    t0 = datetime(2026, 6, 9, 12, 0)
    assert check_stable(cam, "blk", [f], now=t0, stable_minutes=5) is False
    assert (
        check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=2), stable_minutes=5)
        is False
    )


def test_stability_uses_mtime_ns_for_same_size_rewrite(tmp_path, monkeypatch):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    f = cam / "a.otdet"
    f.write_bytes(b"x")
    t0 = datetime(2026, 6, 9, 12, 0)
    assert check_stable(cam, "blk", [f], now=t0, stable_minutes=5) is False

    real_stat = Path.stat

    class FakeStat:
        st_size = f.stat().st_size
        st_mtime = f.stat().st_mtime
        st_mtime_ns = f.stat().st_mtime_ns + 1

    def fake_stat(self, *args, **kwargs):
        if self == f:
            return FakeStat()
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    assert (
        check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=6), stable_minutes=5)
        is False
    )
    assert (
        check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=12), stable_minutes=5)
        is True
    )
    f.write_bytes(b"xx")
    assert (
        check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=7), stable_minutes=5)
        is False
    )


def test_stability_zero_is_immediate(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    f = cam / "a.otdet"
    f.write_bytes(b"x")
    assert (
        check_stable(
            cam, "blk", [f], now=datetime(2026, 6, 9, 12, 0), stable_minutes=0
        )
        is True
    )


def test_marker_read_tolerates_partial_json_and_cleans_tmp(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    (cam / ".otc_watch_state.json").write_text("{")
    (cam / ".otc_watch_state.json.123.tmp").write_text("{}")
    assert get_tracked_through(cam) is None
    set_tracked_through(cam, date(2026, 6, 6), days=4, files=384, at="t1")
    assert get_tracked_through(cam) == date(2026, 6, 6)
    assert not list(cam.glob(".otc_watch_state.json.*.tmp"))
