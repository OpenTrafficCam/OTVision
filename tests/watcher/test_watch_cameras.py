import bz2
import json
import os
import shutil
import time
from multiprocessing import Process, Queue
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from otc_state import get_tracked_through
from watch_cameras import (
    WatchConfig,
    acquire_track_slot,
    discover_cameras,
    main,
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


def test_corrupt_marker_does_not_flatten_or_track(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    (cam / ".otc_watch_state.json").write_text("{")
    called = {"flatten": False, "track": False}

    def flatten(*args, **kwargs):
        called["flatten"] = True

    def track(*args, **kwargs):
        called["track"] = True

    out = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
        cfg=_cfg(tmp_path),
        flatten_fn=flatten,
        track_fn=track,
        log=lambda m: None,
    )
    assert out.status == "failed" and "state marker unreadable" in out.detail
    assert called == {"flatten": False, "track": False}


def test_repeated_track_failures_backoff_and_quarantine(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    cfg = _cfg(tmp_path)
    cfg.max_failures = 2

    def flatten(camera, date_filter=None, log=None, **k):
        from flatten_camera import FlattenResult

        return FlattenResult(camera=camera)

    out1 = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
        cfg=cfg,
        flatten_fn=flatten,
        track_fn=lambda paths, log=None: False,
        log=lambda m: None,
    )
    out2 = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 1, tzinfo=timezone.utc),
        cfg=cfg,
        flatten_fn=flatten,
        track_fn=lambda paths, log=None: False,
        log=lambda m: None,
    )
    out3 = process_camera(
        cam,
        now=datetime(2026, 6, 9, 12, 2, tzinfo=timezone.utc),
        cfg=cfg,
        flatten_fn=flatten,
        track_fn=lambda paths, log=None: True,
        log=lambda m: None,
    )
    assert out1.status == "failed"
    assert out2.status == "skipped" and "backoff" in out2.detail
    assert out3.status == "skipped" and "backoff" in out3.detail


def test_verify_outputs_rejects_truncated_or_non_json_bz2(tmp_path):
    good = tmp_path / "good.otdet"
    truncated = tmp_path / "truncated.otdet"
    non_json = tmp_path / "non-json.otdet"
    for p in (good, truncated, non_json):
        p.write_bytes(b"x")
    with bz2.open(good.with_suffix(".ottrk"), "wt") as fh:
        json.dump({"ok": True}, fh)
    payload = json.dumps({"items": list(range(2000))}).encode()
    truncated.with_suffix(".ottrk").write_bytes(bz2.compress(payload)[:-10])
    with bz2.open(non_json.with_suffix(".ottrk"), "wt") as fh:
        fh.write("not json")

    bad = verify_outputs([good, truncated, non_json])
    assert good.with_suffix(".ottrk") not in bad
    assert truncated.with_suffix(".ottrk") in bad
    assert non_json.with_suffix(".ottrk") in bad


def test_discover(tmp_path):
    (tmp_path / "OTCamera07").mkdir()
    (tmp_path / "otcamera23").mkdir()
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert {p.name for p in discover_cameras(tmp_path)} == {"OTCamera07", "otcamera23"}


def test_safe_parallelism_caps_to_cores():
    assert safe_parallelism(1000, reserve=2) <= (os.cpu_count() or 4)
    assert safe_parallelism(1, reserve=2) == 1


def test_cli_rejects_invalid_knobs(tmp_path, capsys):
    for args in (
        ["--once", "--block-days", "0", str(tmp_path)],
        ["--once", "--slots-per-day", "0", str(tmp_path)],
        ["--once", "--slots-per-day", "7", str(tmp_path)],
        ["--once", "--idle-minutes", "-1", str(tmp_path)],
        ["--once", "--stable-minutes", "-1", str(tmp_path)],
        ["--once", "--reserve-cores", "-1", str(tmp_path)],
        ["--once", "--max-parallel", "0", str(tmp_path)],
        ["--once", "--cores-per-track", "0", str(tmp_path)],
        ["--once", "--max-failures", "0", str(tmp_path)],
    ):
        assert main(args) == 2
    assert "fatal" in capsys.readouterr().err


def _hold_slot(lock_dir, q):
    import track_continuous as tc
    from watch_cameras import acquire_track_slot

    tc.LOCK_DIR = Path(lock_dir)
    with acquire_track_slot(1) as slot:
        q.put(slot is not None)
        time.sleep(0.4)


def test_track_slot_budget_spans_processes(tmp_path):
    q = Queue()
    p = Process(target=_hold_slot, args=(tmp_path / "locks", q))
    p.start()
    assert q.get(timeout=2) is True
    import track_continuous as tc

    tc.LOCK_DIR = tmp_path / "locks"
    with acquire_track_slot(1) as slot:
        assert slot is None
    p.join(timeout=2)
    assert p.exitcode == 0


def test_run_track_uses_start_new_session_without_preexec(tmp_path, monkeypatch):
    from watch_cameras import _run_track_in_slot
    import watch_cameras as wc

    seen = {}

    class FakeProc:
        pid = 123456

        def wait(self, timeout=None):
            return 0

    def fake_popen(*args, **kwargs):
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(wc.subprocess, "Popen", fake_popen)
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    assert _run_track_in_slot([cam / "a.otdet"], _cfg(tmp_path), cam, lambda m: None)
    assert seen["start_new_session"] is True
    assert "preexec_fn" not in seen


def test_active_child_registry_kills_worker_thread_process_group(monkeypatch):
    import watch_cameras as wc

    calls = []
    proc = type("Proc", (), {"pid": 12345})()

    monkeypatch.setattr(wc.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    wc._register_child(proc)
    wc._terminate_active_children(wc.signal.SIGTERM)
    wc._unregister_child(proc)

    assert calls == [(12345, wc.signal.SIGTERM)]


REAL = Path("/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OTV_LIVE_VOLUME") != "1" or not REAL.exists(),
    reason="set OTV_LIVE_VOLUME=1 with the volume mounted",
)
def test_end_to_end_small(tmp_path):
    src = REAL / "2026-06-03"
    samples = [
        src / "OTCamera07_FR20_2026-06-03_00-00-00.otdet",
        src / "OTCamera07_FR20_2026-06-03_12-00-00.otdet",
    ]
    samples = [s for s in samples if s.exists()]
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
