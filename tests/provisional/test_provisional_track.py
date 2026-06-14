import bz2
import json
from contextlib import nullcontext
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import provisional_track as pt
from provisional_track import (
    EXCLUDE_MANDATORY,
    ProvConfig,
    build_detect_cmd,
    build_track_cmd,
    decide,
    in_scope_otdet,
    in_scope_video,
    main,
    process_camera,
    write_manifest,
)

NOW = datetime(2026, 6, 14, tzinfo=timezone.utc)


def _otdet(cam: Path, day: str, time: str, host: str | None = None) -> Path:
    host = host or cam.name
    sub = cam / day
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / f"{host}_FR20_{day}_{time}.otdet"
    f.write_bytes(b"x")
    return f


def _cfg(tmp_path: Path) -> ProvConfig:
    return ProvConfig(
        root=tmp_path,
        cutoff=date(2026, 6, 3),
        exclude=set(("OTCamera07", "OTCamera18", "OTCamera20")),
        worktree=tmp_path / "wt",
        venv_python=tmp_path / "py",
        config=tmp_path / "c.yaml",
        log_dir=tmp_path / "logs",
        manifest_dir=tmp_path / "manifest",
    )


# --- in_scope_otdet ---------------------------------------------------------


def test_in_scope_otdet_keeps_le_cutoff_sorted(tmp_path):
    cam = tmp_path / "OTCamera05"
    early = _otdet(cam, "2026-06-03", "23-45-00")
    first = _otdet(cam, "2026-06-03", "00-00-00")
    after = _otdet(cam, "2026-06-04", "00-00-00")  # excluded by cutoff
    out = in_scope_otdet(cam, date(2026, 6, 3))
    assert out == [first, early]  # sorted by timestamp, 06-04 dropped
    assert after not in out


def test_in_scope_otdet_ignores_appledouble_and_foreign_host(tmp_path):
    cam = tmp_path / "OTCamera05"
    good = _otdet(cam, "2026-06-03", "00-00-00")
    foreign = _otdet(cam, "2026-06-03", "00-15-00", host="OTCamera09")
    junk = cam / "2026-06-03" / "._OTCamera05_FR20_2026-06-03_00-30-00.otdet"
    junk.write_bytes(b"x")
    out = in_scope_otdet(cam, date(2026, 6, 3))
    assert out == [good]
    assert foreign not in out


# --- decide / in_scope_video ------------------------------------------------


def test_decide_skips_excluded(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
    assert action == "skip"
    assert "excluded" in reason


def test_decide_excludes_case_insensitive(tmp_path):
    cam = tmp_path / "otcamera21"  # lowercase dir
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)
    cfg.exclude = {"OTCamera21"}  # excluded name in different case
    action, reason = decide(cam, cfg, tracked_through=None)
    assert action == "skip"
    assert "excluded" in reason


def test_decide_skips_when_watcher_marked(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=date(2026, 6, 6))
    assert action == "skip"
    assert "tracked_through" in reason


def test_decide_skips_no_inscope_otdet(tmp_path):
    cam = tmp_path / "OTCamera03"
    _otdet(cam, "2026-06-04", "00-00-00")  # after cutoff
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
    assert action == "skip"
    assert "no .otdet" in reason


def test_decide_skips_when_all_already_tracked(tmp_path):
    cam = tmp_path / "OTCamera05"
    f = _otdet(cam, "2026-06-03", "00-00-00")
    with bz2.open(f.with_suffix(".ottrk"), "wt") as fh:
        fh.write("{}")  # valid bz2 JSON -> counts as done
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
    assert action == "skip"
    assert "already tracked" in reason


def test_decide_retracks_when_existing_ottrk_is_corrupt(tmp_path):
    cam = tmp_path / "OTCamera05"
    f = _otdet(cam, "2026-06-03", "00-00-00")
    f.with_suffix(".ottrk").write_bytes(b"not-bz2")  # corrupt -> NOT done
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
    assert action == "track"


def test_decide_tracks_when_pending(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    _otdet(cam, "2026-06-03", "00-15-00")
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
    assert action == "track"
    assert "2" in reason


def test_in_scope_video_le_cutoff(tmp_path):
    cam = tmp_path / "OTCamera02"
    sub = cam / "2026-06-03"
    sub.mkdir(parents=True)
    v_in = sub / "OTCamera02_FR20_2026-06-03_00-00-00.mp4"
    v_in.write_bytes(b"x")
    sub2 = cam / "2026-06-04"
    sub2.mkdir(parents=True)
    v_out = sub2 / "OTCamera02_FR20_2026-06-04_00-00-00.mp4"
    v_out.write_bytes(b"x")
    assert in_scope_video(cam, date(2026, 6, 3)) == [v_in]


# --- command builders -------------------------------------------------------


def test_build_track_cmd(tmp_path):
    cfg = _cfg(tmp_path)
    paths = [tmp_path / "a.otdet", tmp_path / "b.otdet"]
    logfile = tmp_path / "x.otvision.log"
    cmd = build_track_cmd(paths, cfg, logfile)
    assert cmd[0] == str(cfg.venv_python)
    assert cmd[1] == str(cfg.worktree / "track.py")
    assert "--tracker" in cmd and cmd[cmd.index("--tracker") + 1] == "botsort"
    assert "--overwrite" in cmd
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == str(cfg.config)
    assert str(paths[0]) in cmd and str(paths[1]) in cmd
    assert cmd[cmd.index("--logfile") + 1] == str(logfile)
    assert "--logfile-overwrite" in cmd


def test_build_detect_cmd(tmp_path):
    cfg = _cfg(tmp_path)
    vids = [tmp_path / "a.mp4"]
    logfile = tmp_path / "d.otvision.log"
    cmd = build_detect_cmd(vids, cfg, logfile)
    assert cmd[1] == str(cfg.worktree / "detect.py")
    assert "--overwrite" in cmd
    assert str(vids[0]) in cmd


# --- process_camera (slot injected so tests never touch real .locks/slots) --


def test_process_camera_tracks_and_records(tmp_path):
    cam = tmp_path / "OTCamera05"
    f1 = _otdet(cam, "2026-06-03", "00-00-00")
    f2 = _otdet(cam, "2026-06-03", "00-15-00")
    cfg = _cfg(tmp_path)
    seen = {}

    def fake_run(cmd, c, console):
        seen["cmd"] = cmd
        for p in cmd:
            pp = Path(p)
            if pp.suffix == ".otdet":
                with bz2.open(pp.with_suffix(".ottrk"), "wt") as fh:
                    fh.write("{}")
        return 0

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=fake_run, slot_factory=lambda: nullcontext(0)
    )
    assert entry["decision"] == "track"
    assert entry["status"] == "ok"
    assert entry["n_otdet"] == 2
    assert sorted(entry["otdet"]) == sorted([str(f1), str(f2)])
    assert all(Path(p).exists() for p in entry["ottrk"])
    assert entry["date_range"] == ["2026-06-03", "2026-06-03"]


def test_process_camera_skips_excluded_without_running(tmp_path):
    cam = tmp_path / "OTCamera07"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)
    called = {"n": 0}

    def fake_run(cmd, c, console):
        called["n"] += 1
        return 0

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=fake_run, slot_factory=lambda: nullcontext(0)
    )
    assert entry["decision"] == "skip"
    assert called["n"] == 0


def test_process_camera_reports_failure(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)

    def fake_run(cmd, c, console):
        return 1  # non-zero exit

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=fake_run, slot_factory=lambda: nullcontext(0)
    )
    assert entry["status"] == "FAILED"


def test_process_camera_failed_when_exit0_but_ottrk_missing(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)

    def fake_run(cmd, c, console):
        return 0  # success exit, but writes NO .ottrk

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=fake_run, slot_factory=lambda: nullcontext(0)
    )
    assert entry["status"] == "FAILED"
    assert "missing/invalid" in entry["detail"]


def test_process_camera_errors_on_malformed_marker(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    (cam / ".otc_watch_state.json").write_text('{"tracked_through": "not-a-date"}')
    cfg = _cfg(tmp_path)

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=lambda *a: 0, slot_factory=lambda: nullcontext(0)
    )
    assert entry["status"] == "ERROR"  # fail closed on a bad marker


def test_process_camera_skipped_when_no_slot(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)
    called = {"n": 0}

    def fake_run(cmd, c, console):
        called["n"] += 1
        return 0

    entry = process_camera(
        cam, cfg, now=NOW, run_fn=fake_run, slot_factory=lambda: nullcontext(None)
    )
    assert entry["status"] == "skipped"
    assert "no host-wide track slot" in entry["detail"]
    assert called["n"] == 0  # never ran track without a slot


# --- manifest ---------------------------------------------------------------


def test_write_manifest_json_and_md(tmp_path):
    cfg = _cfg(tmp_path)
    entries = [
        {
            "camera": "OTCamera05",
            "decision": "track",
            "reason": "2 .otdet",
            "status": "ok",
            "n_otdet": 2,
            "otdet": ["/x/a.otdet", "/x/b.otdet"],
            "ottrk": ["/x/a.ottrk", "/x/b.ottrk"],
            "date_range": ["2026-06-03", "2026-06-03"],
            "detail": "",
        },
        {
            "camera": "OTCamera07",
            "decision": "skip",
            "reason": "excluded",
            "status": "skipped",
            "n_otdet": 0,
            "otdet": [],
            "ottrk": [],
            "date_range": None,
            "detail": "",
        },
    ]
    meta = {"stamp": "20260614-101010", "commit": "abc123"}
    json_path, md_path = write_manifest(cfg, entries, meta)
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text())
    assert data["run"]["commit"] == "abc123"
    assert len(data["cameras"]) == 2
    assert "OTCamera05" in md_path.read_text()
    assert data["cameras"][0]["otdet"] == ["/x/a.otdet", "/x/b.otdet"]


# --- main (dry-run) ---------------------------------------------------------


def test_main_dry_run_lists_decisions(tmp_path, capsys):
    c5 = tmp_path / "OTCamera05"
    _otdet(c5, "2026-06-03", "00-00-00")
    c7 = tmp_path / "OTCamera07"
    _otdet(c7, "2026-06-03", "00-00-00")
    rc = main(
        [
            str(tmp_path),
            "--cutoff",
            "2026-06-03",
            "--worktree",
            str(tmp_path / "wt"),
            "--venv-python",
            str(tmp_path / "py"),
            "--config",
            str(tmp_path / "c.yaml"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "OTCamera05" in out and "track" in out
    assert "OTCamera07" in out and "excluded" in out
    assert not (tmp_path / "manifest").exists() or not list(
        (tmp_path / "manifest").glob("*.json")
    )


# --- safety guarantees ------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "OTCamera07",
        "OTCamera18",
        "OTCamera20",
        "otcamera21",
        "otcamera23",
        "otcamera26",
    ],
)
def test_all_mandatory_excludes_are_never_tracked(tmp_path, name):
    cam = tmp_path / name
    _otdet(cam, "2026-06-03", "00-00-00", host=name)
    cfg = _cfg(tmp_path)
    cfg.exclude = set(EXCLUDE_MANDATORY)  # as main() builds it
    action, reason = decide(cam, cfg, tracked_through=None)
    assert action == "skip"
    assert "excluded" in reason


def test_lock_and_slot_are_shared_with_the_watcher():
    # Same objects as the live watcher uses -> same lock files / same slot pool.
    import track_continuous as tc
    import watch_cameras as wc

    assert pt.camera_lock is tc.camera_lock
    assert pt.LOCK_DIR == tc.LOCK_DIR
    assert pt.acquire_track_slot is wc.acquire_track_slot
    assert pt.track_slot_budget is wc.track_slot_budget


def test_in_scope_otdet_skips_regex_match_with_invalid_date(tmp_path):
    cam = tmp_path / "OTCamera05"
    good = _otdet(cam, "2026-06-03", "00-00-00")
    # matches the OTC filename regex but month 13 / hour 25 are not a real date
    bad = cam / "2026-06-03" / "OTCamera05_FR20_2026-13-45_25-99-99.otdet"
    bad.write_bytes(b"x")
    out = in_scope_otdet(cam, date(2026, 6, 3))
    assert out == [good]  # bad filename skipped, not raised
