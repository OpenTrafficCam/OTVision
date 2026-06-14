# Provisional BoT-SORT Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable, idempotent launcher that produces *intentionally partial* BoT-SORT `.ottrk` (capped at end of 2026-06-03) for the not-yet-tracked OTC015 cameras, so OTAnalytics implementers have data to work with now — and writes a manifest of exactly what was tracked.

**Architecture:** A thin orchestration script (`provisional_track.py`) in the main repo. It reuses the existing watcher helpers (`parse_otc_filename`, `get_tracked_through`, `camera_lock`, `discover_cameras`), selects each camera's `.otdet` with timestamp ≤ cutoff, and runs ONE continuous `track.py` per camera. The tracker code comes from the `feature/botsort-reid-filemode` branch via an **isolated git worktree**, invoked as a subprocess with `PYTHONPATH=<worktree>` so `import OTVision` resolves to that branch (the main dir — which the live cron uses — is never modified). Source files are never moved (rsync depends on the date-foldered layout); we only *add* `.ottrk` beside each `.otdet`.

**Tech Stack:** Python 3.12, pytest, existing OTVision CLI (`track.py`, `detect.py`), `fcntl` advisory locks, BoT-SORT (Ultralytics) tracker.

---

## Context the engineer needs

- **Data root:** `/Volumes/platomo data/Projekte/OTC015_Team-Red/videos` — 15 camera dirs `OTCameraNN`.
- **The live cron watcher** runs `watch_cameras.py --once` every 2 minutes on this same root, tracking complete, stable 4-day blocks with `config.continuous.botsort.yaml` and marking `tracked_through` in `<camera>/.otc_watch_state.json`. It IS the eventual full re-track. **Do not stop it.** Our job is the stop-gap for cameras that have no full block yet.
- **Coexistence:** `watch_cameras.py` and `track_continuous.py` both serialize a camera with `track_continuous.camera_lock`, which `flock`s `<repo>/.locks/<camera>-<sha1(resolved path)[:12]>.lock`. Our launcher MUST import that exact function (not reimplement it) so it shares the same lock files and never runs a camera concurrently with the watcher.
- **Cutoff:** `2026-06-03` (inclusive, end of day). `.otdet` currently exist only from `2026-06-03` onward, so for most cameras this is a single day. That is expected — the run is provisional.
- **Camera classification (verified 2026-06-14):**
  - Excluded / fully tracked (have `tracked_through=2026-06-06`): `OTCamera07`, `OTCamera18`, `OTCamera20`. (Listed `21/23/26` are absent from this dir.)
  - This-run targets (have ≤June-3 `.otdet`, no marker): `OTCamera05, 09, 10, 12, 15, 16, 17, 19`.
  - Need detection first (June-3 video, no `.otdet`): `OTCamera02, 06, 11` — handled by `--detect` mode, NOT run in this pass.
  - No data ≤ June 3 (video starts 06-04): `OTCamera03` — naturally skipped.
- **ReID is OFF.** We use the branch's shipped yaml (`WITH_REID: false`). `VideoBackedTracker` is only wired when `WITH_REID: true`, so no video decode happens.
- **`import OTVision` is pinned** to `/home/Sebastian-Gerken/OTVision` by a plain `_otvision.pth`. `PYTHONPATH=<worktree>` lands at `sys.path[1]`, ahead of that, so it wins (verified). The smoke test in Task 1 re-confirms against the real worktree.

### Files

- **Create:** `provisional_track.py` — the launcher (orchestration + pure helpers).
- **Create:** `config.provisional.botsort.yaml` — copied verbatim from `origin/feature/botsort-reid-filemode:user_config.otvision.yaml` (so it is parseable by the worktree's config parser). May be edited to `GMC_METHOD: none` if the smoke test fails.
- **Create:** `tests/provisional/__init__.py`, `tests/provisional/test_provisional_track.py`.
- **Worktree (not committed):** reid-filemode branch checked out at `../OTVision-reid-filemode`.
- **Output (not committed):** `logs_track/provisional/manifest_<stamp>.json` + `.md`, plus per-camera `*.console.log` / `*.otvision.log`.

---

## Task 1: Set up the reid-filemode worktree, run-config, and verify import shadowing

**Files:**
- Create: `config.provisional.botsort.yaml`
- Worktree: `../OTVision-reid-filemode`

- [ ] **Step 1: Create the isolated worktree for the tracker code**

```bash
cd /home/Sebastian-Gerken/OTVision
git fetch origin feature/botsort-reid-filemode
git worktree add ../OTVision-reid-filemode origin/feature/botsort-reid-filemode
```

Expected: `Preparing worktree ... HEAD is now at 5cbd74d ...`. The worktree is placed OUTSIDE the repo so `pytest` from the repo root never collects the branch's own tests.

- [ ] **Step 2: Copy the branch's config as our run-config**

```bash
cd /home/Sebastian-Gerken/OTVision
git show origin/feature/botsort-reid-filemode:user_config.otvision.yaml > config.provisional.botsort.yaml
grep -E 'WITH_REID|GMC_METHOD|TRACKER_TYPE' config.provisional.botsort.yaml
```

Expected: shows `WITH_REID: false`, `GMC_METHOD: sparseOptFlow`. Confirm `WITH_REID: false`.

- [ ] **Step 3: Verify `import OTVision` resolves to the worktree (run from the worktree cwd — this is the real mechanic)**

```bash
cd /home/Sebastian-Gerken/OTVision-reid-filemode
PYTHONPATH=. /home/Sebastian-Gerken/OTVision/.venv/bin/python \
  -c "import OTVision, sys; print(OTVision.__file__); print(sys.path[:2])"
```

Expected: prints a path under `.../OTVision-reid-filemode/OTVision/__init__.py` — **NOT** `/home/Sebastian-Gerken/OTVision/OTVision/...`, and `sys.path[0]` is the worktree. This proves the `cwd=worktree` mechanic the launcher relies on (Codex review #6); running the same check from the *main* repo cwd would wrongly print the main package.

- [ ] **Step 4: Verify the worktree's track.py CLI accepts the flags we use (from the worktree cwd)**

```bash
cd /home/Sebastian-Gerken/OTVision-reid-filemode
VENV=/home/Sebastian-Gerken/OTVision/.venv/bin/python
PYTHONPATH=. $VENV track.py --help 2>&1 | grep -E '\-\-tracker|\-\-overwrite|\-\-logfile|\-p'
PYTHONPATH=. $VENV detect.py --help 2>&1 | grep -E '\-\-overwrite|\-\-logfile|\-p'
```

Expected: `track.py` shows `--tracker {iou,botsort}`, `--overwrite/--no-overwrite`, `--logfile`, `--logfile-overwrite`, `-p`. `detect.py` shows `-p`, `--overwrite/--no-overwrite`, `--logfile`, `--logfile-overwrite`. If a flag name differs, update the command builders in Task 4 to match before proceeding.

- [ ] **Step 5: Commit the run-config**

```bash
cd /home/Sebastian-Gerken/OTVision
git add config.provisional.botsort.yaml
git commit -m "chore: add provisional run-config copied from botsort-reid-filemode"
```

---

## Task 2: `in_scope_otdet` — select same-host `.otdet` with timestamp ≤ cutoff

**Files:**
- Create: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/provisional/test_provisional_track.py
import os
from datetime import date
from pathlib import Path

import pytest

from provisional_track import in_scope_otdet


def _otdet(cam: Path, day: str, time: str, host: str | None = None) -> Path:
    host = host or cam.name
    sub = cam / day
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / f"{host}_FR20_{day}_{time}.otdet"
    f.write_bytes(b"x")
    return f


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'provisional_track'` (and missing `tests/provisional/__init__.py`).

- [ ] **Step 3: Create the package marker and the helper**

Create `tests/provisional/__init__.py` (empty file).

Create `provisional_track.py` with the header and first helper:

```python
#!/usr/bin/env python3
"""Provisional, date-capped BoT-SORT tracking for not-yet-tracked OTC cameras.

Runs the feature/botsort-reid-filemode tracker (via an isolated worktree) over
each camera's .otdet with timestamp <= cutoff, as ONE continuous track.py run.
Intentionally partial: the cron watcher remains the eventual full re-track.

Never moves source files (rsync depends on the date-foldered layout); only adds
.ottrk beside each .otdet. Never touches excluded cameras or cameras the watcher
already marked (tracked_through). Idempotent: a camera whose in-scope .otdet all
have a VALID .ottrk is skipped; otherwise it is re-tracked in full (continuity-safe).
"""
from __future__ import annotations

import argparse
import bz2
import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from otc_coverage import parse_otc_filename
from otc_state import StateUnreadable, get_tracked_through
from track_continuous import LOCK_DIR, camera_lock
from watch_cameras import acquire_track_slot, discover_cameras, track_slot_budget

SCRIPT_DIR = Path(__file__).resolve().parent
# MANDATORY excludes: always applied, never removable via --exclude (safety).
EXCLUDE_MANDATORY = frozenset(("OTCamera07", "OTCamera18", "OTCamera20"))
CUTOFF_DEFAULT = date(2026, 6, 3)
VIDEO_EXT = ".mp4"


def in_scope_otdet(camera: Path, cutoff: date) -> list[Path]:
    """Same-host .otdet whose embedded timestamp date is <= cutoff, time-sorted."""
    host = camera.name.lower()
    found: list[tuple[datetime, Path]] = []
    for f in camera.rglob("*.otdet"):
        if f.name.startswith("._"):
            continue
        parsed = parse_otc_filename(f.name)
        if not parsed:
            continue
        fhost, dt = parsed
        if fhost.lower() != host:
            continue
        if dt.date() <= cutoff:
            found.append((dt, f))
    return [f for _, f in sorted(found)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/__init__.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): select same-host .otdet <= cutoff"
```

---

## Task 3: `in_scope_video` and `decide` — camera eligibility

**Files:**
- Modify: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/provisional/test_provisional_track.py
from provisional_track import ProvConfig, decide, in_scope_video


def _cfg(tmp_path) -> ProvConfig:
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


def test_decide_skips_excluded(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    action, reason = decide(cam, _cfg(tmp_path), tracked_through=None)
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
    import bz2

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProvConfig'`.

- [ ] **Step 3: Implement `ProvConfig`, `in_scope_video`, `ottrk_done`, `decide`**

Add to `provisional_track.py`:

```python
@dataclass
class ProvConfig:
    root: Path
    cutoff: date
    exclude: set[str]
    worktree: Path
    venv_python: Path
    config: Path
    log_dir: Path
    manifest_dir: Path
    detect: bool = False
    dry_run: bool = False
    max_parallel: int = 4
    # Host-wide track-slot budget, SHARED with the watcher's slot pool in
    # <repo>/.locks/slots (Codex review #8). Defaults mirror the watcher cron
    # (--max-parallel 12 --cores-per-track 2 --reserve-cores 2) so both parties
    # size the same pool and the flock cap bounds total track.py across BOTH.
    host_max_parallel: int = 12
    cores_per_track: int = 2
    reserve_cores: int = 2


def in_scope_video(camera: Path, cutoff: date) -> list[Path]:
    """Same-host .mp4 whose embedded timestamp date is <= cutoff, time-sorted."""
    host = camera.name.lower()
    found: list[tuple[datetime, Path]] = []
    for f in camera.rglob(f"*{VIDEO_EXT}"):
        if f.name.startswith("._"):
            continue
        stem = f.name[: -len(VIDEO_EXT)] + ".otdet"
        parsed = parse_otc_filename(stem)
        if not parsed:
            continue
        fhost, dt = parsed
        if fhost.lower() != host:
            continue
        if dt.date() <= cutoff:
            found.append((dt, f))
    return [f for _, f in sorted(found)]


def _ottrk_ok(path: Path) -> bool:
    """A .ottrk counts as done only if it is non-empty, valid bz2 JSON.

    Existence alone is not enough: a partial write, an old per-file output, or
    an IOU-tracker result must NOT satisfy the skip condition (Codex review #4/#5).
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with bz2.open(path, "rt") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


def ottrk_done(otdet_paths: list[Path]) -> tuple[int, int]:
    done = sum(1 for p in otdet_paths if _ottrk_ok(p.with_suffix(".ottrk")))
    return done, len(otdet_paths)


def decide(camera: Path, cfg: ProvConfig, tracked_through) -> tuple[str, str]:
    """Return ("track"|"skip", reason). Pure given tracked_through."""
    if camera.name in cfg.exclude:
        return "skip", "excluded (fully tracked, never touch)"
    if tracked_through is not None:
        return "skip", f"watcher set tracked_through={tracked_through}"
    scope = in_scope_otdet(camera, cfg.cutoff)
    if not scope:
        if cfg.detect and in_scope_video(camera, cfg.cutoff):
            return "track", "no .otdet <= cutoff; will detect then track"
        return "skip", f"no .otdet <= {cfg.cutoff}"
    done, total = ottrk_done(scope)
    if done == total:
        return "skip", f"already tracked ({done}/{total} .ottrk present)"
    return "track", f"{total} .otdet <= {cfg.cutoff} ({done} already have .ottrk; full re-track)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): camera eligibility decision (exclude/marker/done)"
```

---

## Task 4: Command builders for track and detect

**Files:**
- Modify: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/provisional/test_provisional_track.py
from provisional_track import build_detect_cmd, build_track_cmd


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k build -v`
Expected: FAIL — `ImportError: cannot import name 'build_track_cmd'`.

- [ ] **Step 3: Implement the builders**

Add to `provisional_track.py` (adjust flag names here ONLY if Task 1 Step 4 revealed differences):

```python
def build_track_cmd(otdet_paths: list[Path], cfg: ProvConfig, logfile: Path) -> list[str]:
    return [
        str(cfg.venv_python),
        str(cfg.worktree / "track.py"),
        "-p",
        *[str(p) for p in otdet_paths],
        "-c",
        str(cfg.config),
        "--tracker",
        "botsort",
        "--overwrite",  # camera-level idempotency is handled by decide(); re-track in full
        "--logfile",
        str(logfile),
        "--logfile-overwrite",
    ]


def build_detect_cmd(video_paths: list[Path], cfg: ProvConfig, logfile: Path) -> list[str]:
    return [
        str(cfg.venv_python),
        str(cfg.worktree / "detect.py"),
        "-p",
        *[str(p) for p in video_paths],
        "-c",
        str(cfg.config),
        "--overwrite",
        "--logfile",
        str(logfile),
        "--logfile-overwrite",
    ]


def run_subprocess(cmd: list[str], cfg: ProvConfig, console: Path) -> int:
    """Run cmd so that `import OTVision` resolves to the WORKTREE, not the main dir.

    cwd=worktree makes the worktree sys.path[0], which beats the editable .pth-
    installed main package. (Just setting PYTHONPATH is NOT enough: when cwd is the
    main repo, sys.path[0]='' already contains the main OTVision/ and wins — Codex
    review #6.) PYTHONPATH is kept as a redundant safeguard. All paths in cmd are
    absolute, so cwd does not affect -p/-c/--logfile resolution.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cfg.worktree)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    with console.open("a") as fh:
        fh.write("# (cwd=%s) " % cfg.worktree + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(cfg.worktree)
        )
    return proc.returncode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k build -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): track/detect command builders + PYTHONPATH runner"
```

---

## Task 5: `process_camera` — lock, optional detect, track, manifest entry

**Files:**
- Modify: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test (injected runner; no real subprocess)**

```python
# add to tests/provisional/test_provisional_track.py
import bz2
from contextlib import nullcontext

from provisional_track import process_camera

NOW = datetime(2026, 6, 14, tzinfo=timezone.utc)
SLOT_OK = lambda: nullcontext(0)      # inject a slot so tests never touch .locks/slots
SLOT_NONE = lambda: nullcontext(None)  # emulate "no host slot free"


def test_process_camera_tracks_and_records(tmp_path):
    cam = tmp_path / "OTCamera05"
    f1 = _otdet(cam, "2026-06-03", "00-00-00")
    f2 = _otdet(cam, "2026-06-03", "00-15-00")
    cfg = _cfg(tmp_path)
    seen = {}

    def fake_run(cmd, c, console):
        seen["cmd"] = cmd
        # emulate track.py: write .ottrk beside each input .otdet
        for p in cmd:
            pp = Path(p)
            if pp.suffix == ".otdet":
                with bz2.open(pp.with_suffix(".ottrk"), "wt") as fh:
                    fh.write("{}")
        return 0

    entry = process_camera(cam, cfg, now=NOW, run_fn=fake_run, slot_factory=SLOT_OK)
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

    entry = process_camera(cam, cfg, now=NOW, run_fn=fake_run, slot_factory=SLOT_OK)
    assert entry["decision"] == "skip"
    assert called["n"] == 0


def test_process_camera_reports_failure(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)

    def fake_run(cmd, c, console):
        return 1  # non-zero exit

    entry = process_camera(cam, cfg, now=NOW, run_fn=fake_run, slot_factory=SLOT_OK)
    assert entry["status"] == "FAILED"


def test_process_camera_skipped_when_no_slot(tmp_path):
    cam = tmp_path / "OTCamera05"
    _otdet(cam, "2026-06-03", "00-00-00")
    cfg = _cfg(tmp_path)
    called = {"n": 0}

    def fake_run(cmd, c, console):
        called["n"] += 1
        return 0

    entry = process_camera(cam, cfg, now=NOW, run_fn=fake_run, slot_factory=SLOT_NONE)
    assert entry["status"] == "skipped"
    assert "no host-wide track slot" in entry["detail"]
    assert called["n"] == 0  # never ran track without a slot
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k process_camera -v`
Expected: FAIL — `ImportError: cannot import name 'process_camera'`.

- [ ] **Step 3: Implement `process_camera`**

Add to `provisional_track.py`:

```python
def _entry(camera: Path, decision: str, reason: str, **extra) -> dict:
    e = {"camera": camera.name, "decision": decision, "reason": reason,
         "status": "skipped", "n_otdet": 0, "otdet": [], "ottrk": [],
         "date_range": None, "detail": ""}
    e.update(extra)
    return e


def _default_slot(cfg: ProvConfig):
    """Acquire one slot from the watcher's SHARED host-wide pool (<repo>/.locks/slots)."""
    budget = track_slot_budget(cfg.host_max_parallel, cfg.reserve_cores, cfg.cores_per_track)
    return acquire_track_slot(budget)


def process_camera(
    camera: Path,
    cfg: ProvConfig,
    *,
    now: datetime,
    run_fn: Callable[[list[str], ProvConfig, Path], int] = run_subprocess,
    slot_factory: Callable[[], object] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    slot_factory = slot_factory or (lambda: _default_slot(cfg))
    with camera_lock(camera) as got:
        if not got:
            return _entry(camera, "skip", "locked by another run (watcher?)")
        try:
            tt = get_tracked_through(camera)
        except StateUnreadable:
            return _entry(camera, "skip", "state marker unreadable", status="ERROR")
        action, reason = decide(camera, cfg, tt)
        if action == "skip":
            log(f"{camera.name}: skip - {reason}")
            return _entry(camera, "skip", reason)

        # Host-wide CPU coordination: take a slot from the same pool the watcher uses.
        with slot_factory() as slot:
            if slot is None:
                log(f"{camera.name}: no host-wide track slot; retry next run")
                return _entry(camera, "track", reason, status="skipped",
                              detail="no host-wide track slot")

            stamp = now.strftime("%Y%m%d-%H%M%S")
            console = cfg.log_dir / f"{camera.name}_{stamp}.console.log"

            # optional detection of in-scope videos missing their .otdet
            if cfg.detect:
                scope_otdet_names = {p.name for p in in_scope_otdet(camera, cfg.cutoff)}
                missing = [
                    v for v in in_scope_video(camera, cfg.cutoff)
                    if (v.with_suffix(".otdet").name) not in scope_otdet_names
                ]
                if missing:
                    log(f"{camera.name}: detecting {len(missing)} video(s) <= {cfg.cutoff}")
                    dlog = cfg.log_dir / f"{camera.name}_{stamp}.detect.otvision.log"
                    if run_fn(build_detect_cmd(missing, cfg, dlog), cfg, console) != 0:
                        return _entry(camera, "track", reason, status="FAILED",
                                      detail="detect failed")

            otdet = in_scope_otdet(camera, cfg.cutoff)
            if not otdet:
                return _entry(camera, "track", reason, status="FAILED",
                              detail="no .otdet after detect step")
            dr = [parse_otc_filename(otdet[0].name)[1].date().isoformat(),
                  parse_otc_filename(otdet[-1].name)[1].date().isoformat()]

            tlog = cfg.log_dir / f"{camera.name}_{stamp}.otvision.log"
            rc = run_fn(build_track_cmd(otdet, cfg, tlog), cfg, console)
            status = "ok" if rc == 0 else "FAILED"
            ottrk = [str(p.with_suffix(".ottrk")) for p in otdet
                     if _ottrk_ok(p.with_suffix(".ottrk"))]
            return _entry(
                camera, "track", reason, status=status,
                n_otdet=len(otdet), otdet=[str(p) for p in otdet], ottrk=ottrk,
                date_range=dr, console_log=str(console),
                detail="" if rc == 0 else f"track exit {rc}",
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k process_camera -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): per-camera lock+detect+track+entry"
```

---

## Task 6: Manifest writer

**Files:**
- Modify: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/provisional/test_provisional_track.py
from provisional_track import write_manifest


def test_write_manifest_json_and_md(tmp_path):
    cfg = _cfg(tmp_path)
    entries = [
        {"camera": "OTCamera05", "decision": "track", "reason": "2 .otdet",
         "status": "ok", "n_otdet": 2, "otdet": ["/x/a.otdet", "/x/b.otdet"],
         "ottrk": ["/x/a.ottrk", "/x/b.ottrk"], "date_range": ["2026-06-03", "2026-06-03"],
         "detail": ""},
        {"camera": "OTCamera07", "decision": "skip", "reason": "excluded",
         "status": "skipped", "n_otdet": 0, "otdet": [], "ottrk": [],
         "date_range": None, "detail": ""},
    ]
    meta = {"stamp": "20260614-101010", "commit": "abc123"}
    json_path, md_path = write_manifest(cfg, entries, meta)
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text())
    assert data["run"]["commit"] == "abc123"
    assert len(data["cameras"]) == 2
    assert "OTCamera05" in md_path.read_text()
    # full otdet path list is preserved for reconstruction
    assert data["cameras"][0]["otdet"] == ["/x/a.otdet", "/x/b.otdet"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k manifest -v`
Expected: FAIL — `ImportError: cannot import name 'write_manifest'`.

- [ ] **Step 3: Implement `write_manifest`**

Add to `provisional_track.py`:

```python
def write_manifest(cfg: ProvConfig, entries: list[dict], meta: dict) -> tuple[Path, Path]:
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "stamp": meta.get("stamp"),
        "started": meta.get("started"),
        "branch": "feature/botsort-reid-filemode",
        "commit": meta.get("commit"),
        "config": str(cfg.config),
        "with_reid": False,
        "cutoff": cfg.cutoff.isoformat(),
        "root": str(cfg.root),
        "exclude": sorted(cfg.exclude),
        "detect": cfg.detect,
        "worktree": str(cfg.worktree),
    }
    doc = {"run": run, "cameras": entries}
    json_path = cfg.manifest_dir / f"manifest_{meta.get('stamp')}.json"
    json_path.write_text(json.dumps(doc, indent=2))

    lines = [f"# Provisional tracking manifest {meta.get('stamp')}", "",
             f"- cutoff: {run['cutoff']} (inclusive)",
             f"- config: {run['config']} (WITH_REID=false)",
             f"- branch/commit: {run['branch']} @ {run['commit']}",
             f"- root: {run['root']}", "",
             "| camera | decision | status | n_otdet | date_range | reason |",
             "|---|---|---|---|---|---|"]
    for e in entries:
        rng = "-".join(e["date_range"]) if e.get("date_range") else ""
        lines.append(f"| {e['camera']} | {e['decision']} | {e['status']} | "
                     f"{e['n_otdet']} | {rng} | {e['reason']} |")
    md_path = cfg.manifest_dir / f"manifest_{meta.get('stamp')}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k manifest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): write JSON + markdown manifest"
```

---

## Task 7: `main()` — CLI, discovery, parallelism, dry-run

**Files:**
- Modify: `provisional_track.py`
- Test: `tests/provisional/test_provisional_track.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/provisional/test_provisional_track.py
from provisional_track import main


def test_main_dry_run_lists_decisions(tmp_path, capsys):
    # one trackable, one excluded, one already-marked
    c5 = tmp_path / "OTCamera05"
    _otdet(c5, "2026-06-03", "00-00-00")
    c7 = tmp_path / "OTCamera07"
    _otdet(c7, "2026-06-03", "00-00-00")
    rc = main([
        str(tmp_path),
        "--cutoff", "2026-06-03",
        "--worktree", str(tmp_path / "wt"),
        "--venv-python", str(tmp_path / "py"),
        "--config", str(tmp_path / "c.yaml"),
        "--log-dir", str(tmp_path / "logs"),
        "--manifest-dir", str(tmp_path / "manifest"),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OTCamera05" in out and "track" in out
    assert "OTCamera07" in out and "excluded" in out
    # dry-run writes no manifest
    assert not (tmp_path / "manifest").exists() or not list((tmp_path / "manifest").glob("*.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -k main -v`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement `now_utc`, dry-run path, and `main`**

Add to `provisional_track.py`:

```python
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _git_commit(worktree: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def assert_with_reid_off(config: Path) -> None:
    """Fail loudly unless the run-config really parses WITH_REID: false.

    ReID needs sibling-video decode; this provisional pass must stay otdet-only.
    Guards against an edited/wrong config silently enabling the VideoBackedTracker.
    """
    data = yaml.safe_load(config.read_text())
    with_reid = data.get("TRACK", {}).get("BOT_SORT", {}).get("WITH_REID", None)
    if with_reid is not False:
        raise SystemExit(
            f"[fatal] {config}: TRACK.BOT_SORT.WITH_REID must be false, got {with_reid!r}"
        )


@contextmanager
def launcher_lock():
    """Single-instance guard: refuse to run two provisional launchers at once."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handle = (LOCK_DIR / "provisional_launcher.lock").open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("[fatal] another provisional_track.py run holds the launcher lock",
                  file=sys.stderr)
            raise SystemExit(2)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--cutoff", default=CUTOFF_DEFAULT.isoformat(),
                   help="Inclusive end date YYYY-MM-DD (default 2026-06-03).")
    p.add_argument("--exclude", default="",
                   help="EXTRA camera names to never touch (comma-separated). "
                        "OTCamera07/18/20 are ALWAYS excluded regardless of this.")
    p.add_argument("--worktree", required=True,
                   help="Path to the feature/botsort-reid-filemode git worktree.")
    p.add_argument("--venv-python", default=str(SCRIPT_DIR / ".venv" / "bin" / "python"))
    p.add_argument("--config", default=str(SCRIPT_DIR / "config.provisional.botsort.yaml"))
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track" / "provisional"))
    p.add_argument("--manifest-dir", default=str(SCRIPT_DIR / "logs_track" / "provisional"))
    p.add_argument("--detect", action="store_true",
                   help="Detect in-scope videos that lack .otdet, then track.")
    p.add_argument("--max-parallel", type=int, default=4,
                   help="Cameras WE attempt concurrently (each still needs a host slot).")
    p.add_argument("--host-max-parallel", type=int, default=12,
                   help="Host-wide track-slot pool size; MUST match the watcher cron (12).")
    p.add_argument("--cores-per-track", type=int, default=2,
                   help="Must match the watcher cron (2) so the shared pool sizes alike.")
    p.add_argument("--reserve-cores", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    root = Path(a.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
        return 2
    cfg = ProvConfig(
        root=root,
        cutoff=date.fromisoformat(a.cutoff),
        exclude=set(EXCLUDE_MANDATORY) | set(s for s in a.exclude.split(",") if s),
        worktree=Path(a.worktree),
        venv_python=Path(a.venv_python),
        config=Path(a.config),
        log_dir=Path(a.log_dir),
        manifest_dir=Path(a.manifest_dir),
        detect=a.detect,
        dry_run=a.dry_run,
        max_parallel=a.max_parallel,
        host_max_parallel=a.host_max_parallel,
        cores_per_track=a.cores_per_track,
        reserve_cores=a.reserve_cores,
    )
    cameras = discover_cameras(root)

    if cfg.dry_run:
        for cam in cameras:
            try:
                tt = get_tracked_through(cam)
            except StateUnreadable:
                print(f"{cam.name}: skip - state marker unreadable")
                continue
            action, reason = decide(cam, cfg, tt)
            print(f"{cam.name}: {action} - {reason}")
        return 0

    if not cfg.worktree.is_dir():
        print(f"[fatal] worktree not found: {cfg.worktree}", file=sys.stderr)
        return 2
    if not cfg.config.is_file():
        print(f"[fatal] config not found: {cfg.config}", file=sys.stderr)
        return 2
    assert_with_reid_off(cfg.config)  # preflight: stay otdet-only

    now = now_utc()
    meta = {"stamp": now.strftime("%Y%m%d-%H%M%S"), "started": now.isoformat(),
            "commit": _git_commit(cfg.worktree)}
    entries: list[dict] = []
    workers = max(1, min(cfg.max_parallel, len(cameras) or 1))
    with launcher_lock():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process_camera, cam, cfg, now=now): cam for cam in cameras}
            for fut in as_completed(futs):
                cam = futs[fut]
                try:
                    entries.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    entries.append(_entry(cam, "skip", f"ERROR {e}", status="ERROR"))
    entries.sort(key=lambda e: e["camera"])
    json_path, md_path = write_manifest(cfg, entries, meta)
    by: dict[str, int] = {}
    for e in entries:
        by[e["status"]] = by.get(e["status"], 0) + 1
    print(f"[{meta['started']}] {len(entries)} cameras | "
          + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print(f"manifest: {json_path}")
    failed = sum(1 for e in entries if e["status"] in ("FAILED", "ERROR"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes; run the whole suite**

Run: `cd /home/Sebastian-Gerken/OTVision && .venv/bin/python -m pytest tests/provisional/test_provisional_track.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd /home/Sebastian-Gerken/OTVision
git add provisional_track.py tests/provisional/test_provisional_track.py
git commit -m "feat(provisional): main CLI with discovery, dry-run, manifest"
```

---

## Task 8: Smoke test on ONE real camera (validates tracker + GMC)

**Files:** none (operational validation)

- [ ] **Step 1: Dry-run against the live root to confirm classification**

```bash
cd /home/Sebastian-Gerken/OTVision
.venv/bin/python provisional_track.py \
  "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
  --worktree ../OTVision-reid-filemode --dry-run
```

Expected: `OTCamera07/18/20: skip - excluded ...`; `OTCamera03: skip - no .otdet <= 2026-06-03`; `OTCamera02/06/11: skip - no .otdet <= 2026-06-03`; `OTCamera05/09/10/12/15/16/17/19: track - N .otdet <= 2026-06-03 ...`. If a target is misclassified, stop and fix `decide`.

- [ ] **Step 2: Track a SINGLE camera directly via the worktree (smallest target, e.g. OTCamera12)**

```bash
ROOT="/Volumes/platomo data/Projekte/OTC015_Team-Red/videos"
CFG=/home/Sebastian-Gerken/OTVision/config.provisional.botsort.yaml
LOG=/home/Sebastian-Gerken/OTVision/logs_track/provisional/smoke_OTCamera12.otvision.log
mkdir -p "$(dirname "$LOG")"
mapfile -t F < <(find "$ROOT/OTCamera12" -name 'OTCamera12_*2026-06-03_*.otdet' ! -name '._*' | sort)
echo "tracking ${#F[@]} files"
cd /home/Sebastian-Gerken/OTVision-reid-filemode   # cwd=worktree so import OTVision = branch code
PYTHONPATH=. /home/Sebastian-Gerken/OTVision/.venv/bin/python track.py \
  -p "${F[@]}" -c "$CFG" --tracker botsort --overwrite \
  --logfile "$LOG" --logfile-overwrite
```

Expected: exit 0. If it fails with a GMC/optical-flow error (`sparseOptFlow` needs images we don't provide), edit `config.provisional.botsort.yaml` and set `GMC_METHOD: none` under `BOT_SORT`, then re-run this step. Commit the config change if made:

```bash
git add config.provisional.botsort.yaml && git commit -m "fix(provisional): GMC none (no images in .otdet mode)"
```

- [ ] **Step 3: Verify the .ottrk landed beside the .otdet and is valid bz2 JSON**

```bash
cd /home/Sebastian-Gerken/OTVision
ROOT="/Volumes/platomo data/Projekte/OTC015_Team-Red/videos"
T=$(find "$ROOT/OTCamera12" -name '*2026-06-03_00-00-00.ottrk' ! -name '._*' | head -1)
.venv/bin/python -c "import bz2,json,sys; json.load(bz2.open(sys.argv[1],'rt')); print('valid:', sys.argv[1])" "$T"
```

Expected: `valid: .../OTCamera12_..._2026-06-03_00-00-00.ottrk`. Confirms the file is next to the otdet (date folder, layout untouched) and well-formed.

---

## Task 9: Execute the provisional run (this pass) + capture manifest

**Files:** none (operational; produces `.ottrk` + manifest)

- [ ] **Step 1: Run the launcher over the live root (ReID off, detect off, watcher left running)**

```bash
cd /home/Sebastian-Gerken/OTVision
.venv/bin/python provisional_track.py \
  "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
  --worktree ../OTVision-reid-filemode \
  --config config.provisional.botsort.yaml \
  --max-parallel 4
```

Expected: a summary line like `... cameras | ok=8, skipped=7` (8 targets tracked; 07/18/20 excluded, 02/03/06/11 skipped), and `manifest: logs_track/provisional/manifest_<stamp>.json`. The per-camera `flock` means any camera the watcher is mid-tracking is reported `skip - locked by another run`; re-running picks it up (or it gets a `tracked_through` and is skipped as done).

- [ ] **Step 2: Re-run to prove idempotency (no double work)**

```bash
cd /home/Sebastian-Gerken/OTVision
.venv/bin/python provisional_track.py \
  "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
  --worktree ../OTVision-reid-filemode --dry-run
```

Expected: previously-tracked targets now report `skip - already tracked (N/N .ottrk present)`. Confirms re-runs don't redo finished cameras.

- [ ] **Step 3: Inspect the manifest**

```bash
cd /home/Sebastian-Gerken/OTVision
ls -t logs_track/provisional/manifest_*.md | head -1 | xargs cat
```

Expected: table of all cameras with decision/status/date_range; the JSON sibling holds the full per-camera `.otdet`/`.ottrk` path lists for reconstruction.

---

## Notes / future extension (NOT executed in this pass)

- **`--detect` mode** (for `OTCamera02/06/11` once you decide to fill them): adds `--detect` to the Task 9 command. It runs `detect.py` on in-scope videos lacking `.otdet`, which **writes new `.otdet` into the date-foldered source tree** (rsync sees added files, same as added `.ottrk`). Detection uses the run-config `DETECT` block (yolov8s, conf 0.25) — switch to the production model first if these tracks must match the campaign's detection model. First detect run downloads YOLO weights.
- **Cleanup:** `git worktree remove ../OTVision-reid-filemode` when the provisional effort is done. The `config.provisional.botsort.yaml`, launcher, and manifests stay in the repo.

---

## Codex review outcome (2026-06-14)

**Applied to the plan:**
- **#6 (critical) — wrong-tracker import.** `run_subprocess` now sets `cwd=worktree`; smoke tests run from the worktree cwd. Without this, `sys.path[0]=''`=main repo would import the *main* tracker, defeating the whole point.
- **#7 — exclude safety.** `OTCamera07/18/20` are now `EXCLUDE_MANDATORY` (always unioned, no `--exclude` opt-out).
- **#4/#5 — idempotency.** `ottrk_done` now requires each `.ottrk` to be non-empty **valid bz2 JSON**, not merely present (rejects partial/foreign/IOU outputs).
- **#8 — host-wide CPU coordination (decided: reuse slots).** `process_camera` acquires a slot from the watcher's shared `<repo>/.locks/slots` pool via `acquire_track_slot`/`track_slot_budget`; `host_max_parallel`/`cores_per_track`/`reserve_cores` default to the watcher cron's values so both size the same pool. No oversubscription.
- **gaps — preflight + singleton.** `assert_with_reid_off` fails the run unless the config truly parses `WITH_REID: false`; `launcher_lock` prevents two concurrent launcher instances.

**Decided, no code change:**
- **#3 — non-atomic `.ottrk` writes (decided: accept).** Identical to the existing watcher, which production accepts; a partial copy fails `_ottrk_ok` bz2 validation and is re-tracked on the next run. Provisional `.ottrk` are best-effort; the manifest records every output.

**Dismissed:**
- **#2** — `bonn_track_botsort.py` is NOT used by this plan (unrelated stray script); recommend deleting/gitignoring it so nobody runs it by accident.
- **#1** — watcher flatten *moves* files: pre-existing watcher behavior you chose to keep; our launcher never moves files.

---

## Self-review

- **Spec coverage:** cutoff ≤ June 3 (Task 2, `decide`), never-touch excludes + marker skip (Task 3), no flatten / `.ottrk` beside `.otdet` (Tasks 4–5, verified Task 8.3), shared lock with watcher (Task 5 via `camera_lock`), reid-filemode tracker+yaml WITH_REID off (Tasks 1,4), manifest of tracked files (Task 6, captured Task 9.3), repeatable/idempotent (Task 3 done-check + Task 9.2), expandable to detection (Tasks 4–5 `--detect`). ✅
- **Type consistency:** `ProvConfig` fields (incl. `host_max_parallel`/`cores_per_track`/`reserve_cores`), `in_scope_otdet`/`in_scope_video`/`_ottrk_ok`/`ottrk_done`/`decide`/`build_track_cmd`/`build_detect_cmd`/`run_subprocess`/`_default_slot`/`process_camera`(with `slot_factory`)/`write_manifest`/`assert_with_reid_off`/`launcher_lock`/`main` signatures are used consistently across tasks. `EXCLUDE_MANDATORY` (not `EXCLUDE_DEFAULT`) is the single exclude constant. Manifest entry dict shape (`camera, decision, reason, status, n_otdet, otdet, ottrk, date_range, detail`) is produced by `_entry`/`process_camera` and consumed unchanged by `write_manifest`. New imports: `bz2`, `fcntl`, `yaml`, `contextmanager`, `LOCK_DIR`, `acquire_track_slot`, `track_slot_budget`. ✅
- **Test isolation:** `process_camera` tests inject `slot_factory` (`nullcontext`) so they never create real `.locks/slots`; `_ottrk_ok` tests use valid/corrupt bz2 to prove validation. ✅
- **Placeholder scan:** every code/command step contains concrete content; flag-name verification is gated in Task 1.4 with an explicit "adjust here only if" instruction. ✅
