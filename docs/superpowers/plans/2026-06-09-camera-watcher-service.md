# Camera Watcher Service Implementation Plan (rev. 2.1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Before spawning subagents:** verify you have open agent/Codex thread capacity (don't exhaust the pool); reuse one running guide agent rather than spawning duplicates. This is an execution-time concern, separate from the runtime OS-concurrency guard in Task 8.

**Goal:** A polling service that watches the project `videos/` tree and, whenever a camera has a complete, **stable** 4-day block of `.otdet` not yet processed, flattens exactly those days and tracks them as **one continuous BoT-SORT run** (continuous IDs within the block), then verifies the `.ottrk` outputs before marking the block done.

**Architecture:** Three repo-root modules beside the existing helpers. `otc_coverage.py` = pure assessment (filenames + mtime, no metadata reads): exact 96-slot complete-day detection, 4-day block selection, coarse settledness. `otc_state.py` = per-camera persistence: the `tracked_through` marker **and** the stability snapshot, both written atomically. `watch_cameras.py` = the poll loop: per camera, acquire the per-camera lock (reused from `track_continuous.camera_lock`) → assess → require snapshot stability → scoped flatten → one continuous track → verify `.ottrk` → advance marker. Poll, not inotify (CIFS/SMB doesn't deliver other-host writes). Deploy via cron `--once` (lock makes overlap safe) or `--interval`.

**Tech Stack:** Python 3.12 stdlib (argparse, dataclasses, pathlib, json, re, datetime, subprocess, bz2, fcntl, hashlib, os). pytest. Reuses `flatten_camera.flatten_camera`, `track_continuous.camera_lock`, `track.py`, `config.continuous.botsort.yaml`.

---

## Policy & Definitions (the contract; all thresholds are CLI knobs)

**Filename grammar.** `<HOST>_FR<fps>_<YYYY-MM-DD>_<HH-MM-SS>.otdet` (e.g. `OTCamera07_FR20_2026-06-03_00-15-00.otdet`).

**Complete day (exact, not just a count).** A date is complete iff its `.otdet` slot-times equal **exactly** the expected cadence set `{00:00, 00:15, …, 23:45}` (96 slots for 15-min cadence; `--slots-per-day`). Off-cadence times, missing slots, or duplicate basenames ⇒ not complete. Single host per camera; foreign-host files are ignored (and logged).

**Coverage domain ("including the previous moving").** Scan `.otdet` in the camera **root (already-flattened/tracked) and date subfolders (new arrivals)** together.

**Block policy — 4-day continuous blocks (your decision).** Complete days form consecutive-date runs. Process in **non-overlapping blocks of exactly `--block-days` (default 4)** consecutive complete days lying beyond `tracked_through`. Each block is tracked by **one** `track.py` invocation over **all** the block's `.otdet`, so BoT-SORT IDs are continuous across the whole block (track.py still splits internally only at real >1-min recording gaps). `tracked_through` advances to the block's last day. **Leftover complete days fewer than `--block-days` wait** until they accumulate to a full block (documented limitation; set `--block-days 1` to flush a tail manually).

**Settledness / transfer-complete — INPUT `.otdet` (two layers).**
1. *Coarse age:* no block `.otdet` modified within `--idle-minutes` (default 5).
2. *Stability snapshot (primary):* the block's `(name, size, mtime)` signature must be **unchanged for ≥ `--stable-minutes` (default 5)**. Persisted in `<camera>/.otc_watch_scan.json`: first sighting records the signature + timestamp; a later poll fires only if the signature is identical and old enough; any change resets the timer. This is robust on SMB where a single mtime check is weak.
3. *Temp policy (scoped + age):* fire is blocked only by a `.<…>.otdet.XXXX` atomic-write temp that is **inside the block's date folders** AND **newer than `--idle-minutes`** (same coarse-age knob used for the file-idle check below). Stale temps elsewhere (e.g. the existing `.WQoSZM`) do **not** block — this unbreaks OTCamera07.

**Transfer-complete — OUTPUT `.ottrk` (your question).** `track.py` writes `.ottrk` **non-atomically** (`stream_ottrk_file_writer.py` → `helpers/files.py` `bz2.open(final_path,"wt")`), so a partial file can briefly exist at the real path. Therefore: after `track.py` exits 0, **verify** every block `.otdet` has a sibling `.ottrk` that is non-empty and bz2-readable; only then advance the marker. **Downstream consumes based on the marker (`tracked_through`), never raw `.ottrk` mtime.**

**Scoped flatten (fixes Task-6 leak).** With `date_filter`, the flatten restricts **all** side effects to the selected days' subfolders: moved sources, AppleDouble cleanup, empty-folder removal, and temp reporting. Subfolders of unselected (in-progress) days are never touched.

**Locking & marker safety.** `process_camera` acquires the per-camera `fcntl` lock (reused from `track_continuous.camera_lock`) **before** reading the marker, then holds it across assess→stability→flatten→track→verify→**marker write**. This closes the read-then-lock race: two cron runs can never both act on the same pending block. Cron `--once` overlap is therefore safe. The marker temp file name includes the PID so two writers never collide even without the lock.

**OS concurrency guard (your 32-thread safety).** Cap concurrent camera processing at `min(--max-parallel, cpu_count − --reserve-cores)`; before launching a track, if 1-min load average exceeds `cpu_count − reserve`, defer that camera to the next poll. Each `track.py` is one process; this prevents oversubscription even though we don't expect to hit 32.

**Failure handling.** Any failure (lock contention, flatten conflict, track non-zero exit, `.ottrk` verification miss, crash) ⇒ marker is **not** advanced ⇒ retried next poll. All stages idempotent (atomic moves; `--overwrite` re-derivation; atomic marker). Cameras isolated; one failure never blocks others.

**ID-continuity note (reviewer point 6).** Encoded choice = **4-day continuous blocks**: continuous IDs within each block, reset only between blocks. Not rolling-daily (would reset daily) and not full-run re-track (would re-do all history each fire).

---

## File Structure

- `otc_coverage.py` (create) — `parse_otc_filename`, `expected_slots`, `complete_dates` (exact), `consecutive_runs`, `next_block`, `_scan`, `assess_camera`. Read-only.
- `otc_state.py` (create) — `get_tracked_through` / `set_tracked_through` (marker) + `check_stable` (snapshot). Atomic, PID-unique temp.
- `flatten_camera.py` (modify) — `date_filter` that scopes moves **and** cleanup/removal/temp-report to selected days.
- `watch_cameras.py` (create) — `verify_outputs`, `discover_cameras`, `safe_parallelism`, `process_camera` (lock + stability + scoped flatten + track + verify + mark), poll loop, CLI.
- `tests/watcher/conftest.py` + `tests/watcher/test_*.py` (create).
- `pytest.ini` (modify/create) — register the `integration` marker.

---

### Task 0: Test bootstrap + integration marker

**Files:** Create `tests/watcher/conftest.py`; Modify/Create `pytest.ini`.

- [ ] **Step 1: conftest exposes repo root**

```python
# tests/watcher/conftest.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _isolate_locks(tmp_path_factory, monkeypatch):
    """Keep per-camera lock files out of the repo during tests."""
    import track_continuous as tc
    monkeypatch.setattr(tc, "LOCK_DIR", tmp_path_factory.mktemp("locks"))
```

- [ ] **Step 2: register the integration marker** (append to `pytest.ini`, or create it)

```ini
[pytest]
markers =
    integration: tests that need the live mounted videos volume (set OTV_LIVE_VOLUME=1)
```

- [ ] **Step 3: verify collection** — Run: `python -m pytest tests/watcher -q` → Expected: `no tests ran`.
- [ ] **Step 4: Commit**

```bash
git add tests/watcher/conftest.py pytest.ini
git commit -m "test(watcher): conftest + integration marker"
```

---

### Task 1: Filename parsing

**Files:** Create `otc_coverage.py`; Test `tests/watcher/test_otc_coverage.py`.

- [ ] **Step 1: Failing test**

```python
# tests/watcher/test_otc_coverage.py
from datetime import datetime
from otc_coverage import parse_otc_filename


def test_parse_valid():
    assert parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet") == (
        "OTCamera07", datetime(2026, 6, 3, 0, 15, 0))


def test_parse_rejects_junk():
    assert parse_otc_filename("._OTCamera07_FR20_2026-06-03_00-15-00.otdet") is None
    assert parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet.TMP9") is None
    assert parse_otc_filename("notes.txt") is None
```

- [ ] **Step 2: Run** → FAIL `ModuleNotFoundError: otc_coverage`.
- [ ] **Step 3: Implement**

```python
# otc_coverage.py
"""Pure coverage assessment for the camera watcher (filenames + mtime only)."""
from __future__ import annotations

import re
from datetime import datetime

_OTC_RE = re.compile(
    r"^(?P<host>[A-Za-z0-9]+)_FR\d+_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.otdet$"
)


def parse_otc_filename(name: str) -> tuple[str, datetime] | None:
    m = _OTC_RE.match(name)
    if not m:
        return None
    return m["host"], datetime.strptime(f"{m['date']}_{m['time']}", "%Y-%m-%d_%H-%M-%S")
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(watcher): parse OTCamera .otdet filenames`.

---

### Task 2: Exact complete-day validation + runs

**Files:** Modify `otc_coverage.py`; Test same.

- [ ] **Step 1: Failing test**

```python
# append
from datetime import date, timedelta
from otc_coverage import expected_slots, complete_dates, consecutive_runs


def _slots(day, times):  # times: list[(h, m)]
    base = datetime(day.year, day.month, day.day)
    return [base.replace(hour=h, minute=m) for h, m in times]


def test_expected_slots_15min():
    s = expected_slots(96)
    assert (0, 0) in s and (23, 45) in s and len(s) == 96 and (0, 7) not in s


def test_complete_requires_exact_set():
    full = _slots(date(2026, 6, 3), [(h, m) for h in range(24) for m in (0, 15, 30, 45)])
    missing_one = full[:-1]
    off_cadence = full[:-1] + _slots(date(2026, 6, 3), [(23, 47)])
    assert complete_dates(full, 96) == {date(2026, 6, 3)}
    assert complete_dates(missing_one, 96) == set()        # 95 slots -> incomplete
    assert complete_dates(off_cadence, 96) == set()        # off-cadence -> incomplete


def test_runs():
    ds = {date(2026, 6, 3), date(2026, 6, 4), date(2026, 6, 6)}
    assert consecutive_runs(ds) == [[date(2026, 6, 3), date(2026, 6, 4)], [date(2026, 6, 6)]]
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement** (append)

```python
from datetime import date, timedelta


def expected_slots(slots_per_day: int) -> set[tuple[int, int]]:
    step = 24 * 60 // slots_per_day
    return {(m // 60, m % 60) for m in range(0, 24 * 60, step)}


def complete_dates(slot_datetimes, slots_per_day: int = 96) -> set[date]:
    want = expected_slots(slots_per_day)
    by_day: dict[date, set] = {}
    for dt in slot_datetimes:
        by_day.setdefault(dt.date(), set()).add((dt.hour, dt.minute))
    return {d for d, got in by_day.items() if got == want}


def consecutive_runs(days: set[date]) -> list[list[date]]:
    runs: list[list[date]] = []
    for d in sorted(days):
        if runs and d - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(watcher): exact 96-slot complete-day validation`.

---

### Task 3: 4-day block selection (`next_block`)

**Files:** Modify `otc_coverage.py`; Test same.

- [ ] **Step 1: Failing test**

```python
# append
from otc_coverage import next_block

D = lambda n: date(2026, 6, n)  # noqa: E731


def test_block_needs_full_block_days():
    assert next_block({D(3), D(4), D(5)}, None, 4) == []
    assert next_block({D(3), D(4), D(5), D(6)}, None, 4) == [D(3), D(4), D(5), D(6)]


def test_non_overlapping_next_block():
    run8 = {D(n) for n in range(3, 11)}
    assert next_block(run8, tracked_through=D(6), block_days=4) == [D(7), D(8), D(9), D(10)]


def test_leftover_smaller_than_block_waits():
    run6 = {D(n) for n in range(3, 9)}
    assert next_block(run6, tracked_through=D(6), block_days=4) == []  # only 7,8 pending
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement** (append)

```python
def next_block(complete: set[date], tracked_through: date | None,
               block_days: int = 4) -> list[date]:
    """Next non-overlapping block of exactly `block_days` consecutive complete
    days beyond `tracked_through`. [] if no full block is available yet."""
    for run in consecutive_runs(complete):
        pend = [d for d in run if tracked_through is None or d > tracked_through]
        if len(pend) >= block_days:
            return pend[:block_days]
    return []
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(watcher): non-overlapping 4-day block selection`.

---

### Task 4: Scan + coarse assess (`assess_camera`)

**Files:** Modify `otc_coverage.py`; Test same.

- [ ] **Step 1: Failing test**

```python
# append
import os
from pathlib import Path
from otc_coverage import assess_camera


def _make_day(cam: Path, day: date, complete=True, old=True):
    d = cam / f"{day:%Y-%m-%d}"; d.mkdir(parents=True, exist_ok=True)
    times = [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    if not complete:
        times = times[:40]
    for h, m in times:
        f = d / f"OTCamera07_FR20_{day:%Y-%m-%d}_{h:02d}-{m:02d}-00.otdet"
        f.write_bytes(b"x")
        if old:
            os.utime(f, (1_000_000, 1_000_000))


def test_fires_on_complete_settled_block(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    rep = assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                        block_days=4, slots_per_day=96, idle_minutes=5)
    assert rep.fire and rep.tracked_through_after == date(2026, 6, 6)
    assert len(rep.otdet_paths) == 96 * 4


def test_scoped_temp_in_block_blocks_but_stale_outside_does_not(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    # stale temp OUTSIDE the block (an in-progress day 7) must NOT block
    d7 = cam / "2026-06-07"; d7.mkdir()
    (d7 / ".OTCamera07_FR20_2026-06-07_10-00-00.otdet.OLD").write_bytes(b"")
    os.utime(d7 / ".OTCamera07_FR20_2026-06-07_10-00-00.otdet.OLD", (1_000_000, 1_000_000))
    assert assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                         block_days=4, slots_per_day=96, idle_minutes=5).fire is True
    # fresh temp INSIDE the block DOES block
    (cam / "2026-06-06" / ".OTCamera07_FR20_2026-06-06_10-30-00.otdet.NEW").write_bytes(b"")
    rep = assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                        block_days=4, slots_per_day=96, idle_minutes=5)
    assert rep.fire is False and "temp" in rep.reason.lower()


def test_foreign_host_ignored(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    bad = cam / "2026-06-03" / "OTCamera09_FR20_2026-06-03_00-00-00.otdet"
    bad.write_bytes(b"x"); os.utime(bad, (1_000_000, 1_000_000))
    rep = assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                        block_days=4, slots_per_day=96, idle_minutes=5)
    assert rep.fire is True                              # 96 OTCamera07 slots/day still complete
    assert all("OTCamera09" not in p.name for p in rep.otdet_paths)


def test_duplicate_basename_blocks(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    dup = "OTCamera07_FR20_2026-06-03_00-00-00.otdet"   # already in 2026-06-03/
    (cam / dup).write_bytes(b"x"); os.utime(cam / dup, (1_000_000, 1_000_000))  # also in root
    rep = assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                        block_days=4, slots_per_day=96, idle_minutes=5)
    assert rep.fire is False and "duplicate" in rep.reason.lower()
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement** (append)

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoverageReport:
    fire: bool
    reason: str
    days: list[date] = field(default_factory=list)
    otdet_paths: list[Path] = field(default_factory=list)
    tracked_through_after: date | None = None


def _scan(camera: Path):
    """Same-host valid .otdet (root + subdirs), temps, duplicate basenames, and a
    foreign-host count. The camera's host = its directory name; foreign-host files
    are ignored (so a stray OTCamera09 file can't complete/track OTCamera07)."""
    host = camera.name.lower()
    files, temps, seen, dups, foreign = [], [], set(), set(), 0
    for f in camera.rglob("*"):
        if not f.is_file() or f.name.startswith("._"):
            continue
        if f.name.startswith(".") and ".otdet" in f.name:
            inner = parse_otc_filename(f.name.lstrip(".").rsplit(".otdet", 1)[0] + ".otdet")
            temps.append((inner[1].date() if inner else None, f))
            continue
        parsed = parse_otc_filename(f.name)
        if not parsed:
            continue
        fhost, dt = parsed
        if fhost.lower() != host:
            foreign += 1
            continue
        if f.name in seen:
            dups.add(f.name)
        seen.add(f.name)
        files.append((dt, f))
    return files, temps, sorted(dups), foreign


def assess_camera(camera: Path, *, now, tracked_through, block_days=4,
                  slots_per_day=96, idle_minutes=5) -> CoverageReport:
    files, temps, dups, foreign = _scan(camera)
    note = f"; {foreign} foreign-host file(s) ignored" if foreign else ""
    if not files:
        return CoverageReport(False, "no same-host .otdet found" + note)
    complete = complete_dates([dt for dt, _ in files], slots_per_day)
    days = next_block(complete, tracked_through, block_days)
    if not days:
        return CoverageReport(False, f"no full {block_days}-day block pending" + note)
    dayset = set(days)
    window = [(dt, p) for dt, p in files if dt.date() in dayset]
    if any(p.name in dups for _, p in window):     # half-flatten / mixed source
        return CoverageReport(False, "duplicate .otdet basenames in block")
    fresh = now.timestamp() - idle_minutes * 60
    for d, p in temps:               # scoped + age: a fresh temp inside the block blocks
        if d in dayset and p.stat().st_mtime > fresh:
            return CoverageReport(False, f"fresh temp write in block ({p.name})")
    if any(p.stat().st_mtime > fresh for _, p in window):
        return CoverageReport(False, f"block not idle (<{idle_minutes}m)")
    paths = [p for _, p in sorted(window)]
    return CoverageReport(True, "fire", days, paths, max(days))
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(watcher): assess_camera with scoped+age temp policy`.

---

### Task 5: Marker + stability snapshot (`otc_state.py`)

**Files:** Create `otc_state.py`; Test `tests/watcher/test_otc_state.py`.

- [ ] **Step 1: Failing test**

```python
# tests/watcher/test_otc_state.py
from datetime import date, datetime, timedelta
from pathlib import Path
from otc_state import get_tracked_through, set_tracked_through, check_stable


def test_marker_roundtrip(tmp_path):
    cam = tmp_path / "OTCamera07"; cam.mkdir()
    assert get_tracked_through(cam) is None
    set_tracked_through(cam, date(2026, 6, 6), days=4, files=384, at="t1")
    assert get_tracked_through(cam) == date(2026, 6, 6)


def test_stability_requires_unchanged_for_window(tmp_path):
    cam = tmp_path / "OTCamera07"; cam.mkdir()
    f = cam / "a.otdet"; f.write_bytes(b"x")
    t0 = datetime(2026, 6, 9, 12, 0)
    assert check_stable(cam, "blk", [f], now=t0, stable_minutes=5) is False        # first sighting
    assert check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=2), stable_minutes=5) is False
    assert check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=6), stable_minutes=5) is True
    f.write_bytes(b"xx")                                                            # changed -> reset
    assert check_stable(cam, "blk", [f], now=t0 + timedelta(minutes=7), stable_minutes=5) is False


def test_stability_zero_is_immediate(tmp_path):
    cam = tmp_path / "OTCamera07"; cam.mkdir()
    f = cam / "a.otdet"; f.write_bytes(b"x")
    assert check_stable(cam, "blk", [f], now=datetime(2026, 6, 9, 12, 0), stable_minutes=0) is True
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement**

```python
# otc_state.py
"""Per-camera persistence: tracked-through marker + stability snapshot (atomic)."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path

MARKER = ".otc_watch_state.json"
SCAN = ".otc_watch_scan.json"


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # PID-unique: concurrency-safe
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def get_tracked_through(camera: Path) -> date | None:
    p = camera / MARKER
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("tracked_through")
    return date.fromisoformat(v) if v else None


def set_tracked_through(camera: Path, through: date, *, days: int, files: int, at: str) -> None:
    p = camera / MARKER
    state = json.loads(p.read_text()) if p.exists() else {"history": []}
    state.update(camera=camera.name, tracked_through=through.isoformat(), updated=at)
    state.setdefault("history", []).append(
        {"through": through.isoformat(), "days": days, "files": files, "at": at})
    _atomic_write(p, state)


def _signature(files: list[Path]) -> str:
    h = hashlib.sha1()
    for f in sorted(files):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def check_stable(camera: Path, block_key: str, files: list[Path], *,
                 now: datetime, stable_minutes: int) -> bool:
    """True iff the block's (name,size,mtime) signature has been unchanged for
    >= stable_minutes. Persists first-sighting in .otc_watch_scan.json.
    stable_minutes <= 0 means 'no stability gate' (immediate) -- used by tests."""
    if stable_minutes <= 0:
        return True
    p = camera / SCAN
    data = json.loads(p.read_text()) if p.exists() else {}
    sig = _signature(files)
    entry = data.get(block_key)
    if entry and entry["sig"] == sig:
        first = datetime.fromisoformat(entry["first_seen"])
        return (now - first).total_seconds() >= stable_minutes * 60
    data[block_key] = {"sig": sig, "first_seen": now.isoformat()}
    _atomic_write(p, data)
    return False
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(watcher): atomic marker + stability snapshot`.

---

### Task 6: Scoped flatten — cleanup must respect `date_filter`

**Files:** Modify `flatten_camera.py` (`find_sources`, `flatten_camera`); Test `tests/watcher/test_flatten_date_filter.py`.

- [ ] **Step 1: Failing test** (cleanup must NOT touch unselected days)

```python
# tests/watcher/test_flatten_date_filter.py
from datetime import date
from pathlib import Path
from flatten_camera import flatten_camera


def _f(d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True); (d / name).write_bytes(b"x")


def test_scoped_flatten_leaves_unselected_day_untouched(tmp_path):
    cam = tmp_path / "OTCamera07"
    _f(cam / "2026-06-03", "OTCamera07_FR20_2026-06-03_00-00-00.otdet")
    _f(cam / "2026-06-03", "._OTCamera07_FR20_2026-06-03_00-00-00.otdet")  # junk in selected day
    _f(cam / "2026-06-07", "OTCamera07_FR20_2026-06-07_00-00-00.otdet")    # in-progress day
    _f(cam / "2026-06-07", "._OTCamera07_FR20_2026-06-07_00-00-00.otdet")  # junk in unselected day
    res = flatten_camera(cam, date_filter=lambda d: d == date(2026, 6, 3), log=lambda m: None)
    assert res.ok and res.moved == 1
    assert (cam / "OTCamera07_FR20_2026-06-03_00-00-00.otdet").exists()      # moved
    assert not (cam / "2026-06-03").exists()                                 # selected day removed
    # unselected day fully intact, including its ._ junk:
    assert (cam / "2026-06-07" / "OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
    assert (cam / "2026-06-07" / "._OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
```

- [ ] **Step 2: Run** → FAIL (`date_filter` kwarg unknown, or cleanup deletes the unselected `._`).
- [ ] **Step 3: Implement.** Refactor `find_sources` to be subfolder-scoped and add `date_filter` to `flatten_camera`. Replace `find_sources` with:

```python
from datetime import date as _date


def _subdir_date(name: str) -> _date | None:
    try:
        return _date.fromisoformat(name)
    except ValueError:
        return None


def find_sources(camera: Path, types: tuple[str, ...], date_filter=None):
    """Data files / junk / temps in scoped SUBfolders (date_filter on subdir name)."""
    root = camera.resolve()
    sources, appledouble, temp_dotfiles = [], [], []
    for sub in sorted(p for p in camera.iterdir() if p.is_dir()):
        if date_filter is not None:
            d = _subdir_date(sub.name)
            if d is None or not date_filter(d):
                continue                      # unselected day: do not scan/clean/remove
        for f in sub.rglob("*"):
            if not f.is_file():
                continue
            if f.name.startswith(APPLEDOUBLE_PREFIX):
                appledouble.append(f)
            elif f.name.startswith("."):
                temp_dotfiles.append(f)
            elif f.suffix.lower() in types:
                sources.append(f)
    return sorted(sources), sorted(appledouble), sorted(temp_dotfiles)
```

Add `date_filter=None` to the `flatten_camera` signature and pass it through:

```python
    sources, appledouble, temp_dotfiles = find_sources(camera, types, date_filter)
```

(Removing the old root-vs-subdir rglob means cleanup/removal now only ever see scoped subfolders. The empty-folder-removal loop already iterates `{s.parent for s in sources}`, which are now only selected days.)

- [ ] **Step 4: Run** → PASS. **Step 5: Regression** — Run: `python flatten_camera.py --dry-run "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07"` → Expected: still reports files/junk/temps, no traceback (no `date_filter` ⇒ all subdirs).
- [ ] **Step 6: Commit** `fix(flatten): scope date_filter to cleanup, removal, and temp reporting`.

---

### Task 7: `process_camera` — lock + stability + scoped flatten + track + verify + mark

**Files:** Create `watch_cameras.py`; Test `tests/watcher/test_watch_cameras.py`.

- [ ] **Step 1: Failing test** (with injected fakes; covers verify-gate and no-mark-on-failure)

```python
# tests/watcher/test_watch_cameras.py
import bz2, os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from otc_state import get_tracked_through
from watch_cameras import process_camera, WatchConfig, verify_outputs


def _make_day(cam, day, n=96):
    d = cam / f"{day:%Y-%m-%d}"; d.mkdir(parents=True, exist_ok=True)
    for h in range(24):
        for m in (0, 15, 30, 45):
            f = d / f"OTCamera07_FR20_{day:%Y-%m-%d}_{h:02d}-{m:02d}-00.otdet"
            f.write_bytes(b"x"); os.utime(f, (1_000_000, 1_000_000))


def _cfg(tmp_path):
    return WatchConfig(config=Path("config.continuous.botsort.yaml"),
                       log_dir=tmp_path / "logs", block_days=4, idle_minutes=5,
                       stable_minutes=0, slots_per_day=96)  # stable_minutes=0 -> immediate


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

    def fake_track(paths, log=None):     # simulate track writing valid .ottrk
        seen["n"] = len(paths)
        for p in paths:
            with bz2.open(p.with_suffix(".ottrk"), "wt") as fh:
                fh.write("{}")
        return True

    out = process_camera(cam, now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
                         cfg=_cfg(tmp_path), flatten_fn=fake_flatten, track_fn=fake_track,
                         log=lambda m: None)
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
    out = process_camera(cam, now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc),
                         cfg=_cfg(tmp_path), flatten_fn=flat,
                         track_fn=lambda paths, log=None: True,  # exits 0 but writes NO .ottrk
                         log=lambda m: None)
    assert out.status == "failed" and get_tracked_through(cam) is None
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement**

```python
# watch_cameras.py
"""Polling watcher: flatten+track each camera's next complete, stable 4-day block.

Poll (not inotify): the videos volume is CIFS/SMB; inotify misses other hosts'
writes. Run via cron --once (lock makes overlap safe) or --interval.
"""
from __future__ import annotations

import argparse
import bz2
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from flatten_camera import flatten_camera
from otc_coverage import assess_camera
from otc_state import check_stable, get_tracked_through, set_tracked_through
from track_continuous import camera_lock          # reuse the per-camera flock

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
TRACK_SCRIPT = SCRIPT_DIR / "track.py"
DEFAULT_CONFIG = SCRIPT_DIR / "config.continuous.botsort.yaml"
CAMERA_GLOBS = ("OTCamera*", "otcamera*")


@dataclass
class WatchConfig:
    config: Path
    log_dir: Path
    block_days: int = 4
    idle_minutes: int = 5
    stable_minutes: int = 5
    slots_per_day: int = 96
    reserve_cores: int = 2


@dataclass
class Outcome:
    camera: Path
    status: str           # "idle" | "stabilizing" | "tracked" | "failed" | "skipped"
    detail: str = ""


def _bz2_ok(path: Path) -> bool:
    try:
        with bz2.open(path, "rb") as fh:
            fh.read(64)
        return True
    except Exception:       # noqa: BLE001
        return False


def verify_outputs(otdet_paths: list[Path]) -> list[Path]:
    """.ottrk siblings that are missing / empty / not bz2-readable."""
    bad = []
    for p in otdet_paths:
        t = p.with_suffix(".ottrk")
        if not t.exists() or t.stat().st_size == 0 or not _bz2_ok(t):
            bad.append(t)
    return bad


def process_camera(camera: Path, *, now: datetime, cfg: WatchConfig,
                   flatten_fn: Callable = flatten_camera, track_fn: Callable | None = None,
                   log: Callable[[str], None] = print) -> Outcome:
    if track_fn is None:
        track_fn = lambda paths, log=log: _run_track(paths, cfg, camera, log)
    # Lock FIRST, then read marker / assess / stability / flatten / track / verify /
    # mark -- all under the lock, so two cron runs can never act on a stale block.
    with camera_lock(camera) as got:
        if not got:
            return Outcome(camera, "skipped", "locked by another run")
        tt = get_tracked_through(camera)
        rep = assess_camera(camera, now=now, tracked_through=tt, block_days=cfg.block_days,
                            slots_per_day=cfg.slots_per_day, idle_minutes=cfg.idle_minutes)
        if not rep.fire:
            return Outcome(camera, "idle", rep.reason)
        block_key = f"{rep.days[0]}_{rep.days[-1]}"
        if not check_stable(camera, block_key, rep.otdet_paths, now=now,
                            stable_minutes=cfg.stable_minutes):
            return Outcome(camera, "stabilizing", f"block {block_key} not stable yet")
        wanted = set(rep.days)
        fres = flatten_fn(camera, date_filter=lambda d: d in wanted, log=log)
        if not getattr(fres, "ok", True):
            return Outcome(camera, "failed", "flatten conflict")
        flat = [camera / p.name for p in rep.otdet_paths]
        if not track_fn(flat, log=log):
            return Outcome(camera, "failed", "track failed; retry next poll")
        missing = verify_outputs(flat)
        if missing:
            log(f"{camera.name}: {len(missing)} .ottrk missing/invalid; not marking")
            return Outcome(camera, "failed", f"{len(missing)} .ottrk incomplete")
        set_tracked_through(camera, rep.tracked_through_after, days=len(rep.days),
                            files=len(flat), at=now.isoformat())
        return Outcome(camera, "tracked", f"through {rep.tracked_through_after}")


def _run_track(paths: list[Path], cfg: WatchConfig, camera: Path,
               log: Callable[[str], None]) -> bool:
    stamp = now_utc().strftime("%Y%m%d-%H%M%S")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.log_dir / f"{camera.name}_{stamp}.otvision.log"
    console = cfg.log_dir / f"{camera.name}_{stamp}.console.log"
    cmd = [str(PYTHON), str(TRACK_SCRIPT), "-p", *[str(p) for p in paths],
           "-c", str(cfg.config), "--tracker", "botsort", "--overwrite",
           "--logfile", str(logfile), "--logfile-overwrite"]
    with console.open("w") as fh:
        fh.write(f"# track {len(paths)} files\n"); fh.flush()
        try:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e:
            log(f"{camera.name}: track exit {e.returncode} (see {console.name})")
            return False


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
```

- [ ] **Step 4: Run** → PASS (2). **Step 5: Commit** `feat(watcher): process_camera under lock with .ottrk verify gate`.

---

### Task 8: Discovery, OS concurrency guard, poll loop, CLI

**Files:** Modify `watch_cameras.py`; Test same.

- [ ] **Step 1: Failing test**

```python
# append
from watch_cameras import discover_cameras, safe_parallelism


def test_discover(tmp_path):
    (tmp_path / "OTCamera07").mkdir(); (tmp_path / "otcamera23").mkdir()
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert {p.name for p in discover_cameras(tmp_path)} == {"OTCamera07", "otcamera23"}


def test_safe_parallelism_caps_to_cores():
    assert safe_parallelism(1000, reserve=2) <= (os.cpu_count() or 4)
    assert safe_parallelism(1, reserve=2) == 1
```

- [ ] **Step 2: Run** → FAIL (import).
- [ ] **Step 3: Implement** (append)

```python
def discover_cameras(root: Path) -> list[Path]:
    found = []
    for pattern in CAMERA_GLOBS:
        found.extend(d for d in root.glob(pattern) if d.is_dir())
    seen, out = set(), []
    for c in sorted(found):
        if c.resolve() not in seen:
            seen.add(c.resolve()); out.append(c)
    return out


def safe_parallelism(requested: int, reserve: int = 2) -> int:
    return max(1, min(requested, (os.cpu_count() or 4) - reserve))


def _overloaded(reserve: int) -> bool:
    try:
        return os.getloadavg()[0] > (os.cpu_count() or 4) - reserve
    except OSError:
        return False


def poll_once(root: Path, cfg: WatchConfig, now: datetime,
              log: Callable[[str], None] = print) -> list[Outcome]:
    outcomes = []
    for cam in discover_cameras(root):
        if _overloaded(cfg.reserve_cores):
            log(f"{cam.name}: deferring (load > cpu-{cfg.reserve_cores})")
            outcomes.append(Outcome(cam, "skipped", "system load high"))
            continue
        try:
            outcomes.append(process_camera(cam, now=now, cfg=cfg, log=log))
        except Exception as e:   # noqa: BLE001 - one camera must not kill the loop
            log(f"{cam.name}: ERROR {e}")
            outcomes.append(Outcome(cam, "failed", str(e)))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Watch cameras; track complete 4-day blocks.")
    p.add_argument("root")
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--block-days", type=int, default=4)
    p.add_argument("--idle-minutes", type=int, default=5)
    p.add_argument("--stable-minutes", type=int, default=5)
    p.add_argument("--slots-per-day", type=int, default=96)
    p.add_argument("--reserve-cores", type=int, default=2)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track"))
    a = p.parse_args(argv)
    root = Path(a.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr); return 2
    cfg = WatchConfig(config=Path(a.config), log_dir=Path(a.log_dir),
                      block_days=a.block_days, idle_minutes=a.idle_minutes,
                      stable_minutes=a.stable_minutes, slots_per_day=a.slots_per_day,
                      reserve_cores=a.reserve_cores)

    def run():
        now = now_utc()
        outs = poll_once(root, cfg, now)
        by = {}
        for o in outs:
            by[o.status] = by.get(o.status, 0) + 1
        print(f"[{now.isoformat()}] {len(outs)} cameras | " +
              ", ".join(f"{k}={v}" for k, v in sorted(by.items())))

    if a.once:
        run(); return 0
    while True:
        run(); time.sleep(max(60, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run** → PASS (4). **Step 5: Commit** `feat(watcher): discovery, OS-load guard, poll loop, CLI`.

---

### Task 9: End-to-end integration (env-gated, real track.py)

**Files:** Test `tests/watcher/test_watch_cameras.py`.

- [ ] **Step 1: Failing/skipped test**

```python
# append
import shutil
import pytest

REAL = Path("/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("OTV_LIVE_VOLUME") != "1" or not REAL.exists(),
                    reason="set OTV_LIVE_VOLUME=1 with the volume mounted")
def test_end_to_end_small(tmp_path):
    src = REAL / "2026-06-03"
    samples = sorted(src.glob("OTCamera07_FR20_2026-06-03_01-3*-00.otdet"))[:2]
    if len(samples) < 2:
        pytest.skip("samples unavailable")
    cam = tmp_path / "OTCamera07" / "2026-06-03"; cam.mkdir(parents=True)
    for s in samples:
        shutil.copy2(s, cam / s.name)
    camera = tmp_path / "OTCamera07"
    cfg = WatchConfig(config=Path("config.continuous.botsort.yaml"),
                      log_dir=tmp_path / "logs", block_days=1, idle_minutes=0,
                      stable_minutes=0, slots_per_day=2)   # tiny block for speed
    out = process_camera(camera, now=now_utc(), cfg=cfg)
    assert out.status == "tracked"
    assert len(list(camera.glob("*.ottrk"))) == 2
    assert verify_outputs([camera / s.name for s in samples]) == []
```

- [ ] **Step 2: Run** — Run: `OTV_LIVE_VOLUME=1 python -m pytest tests/watcher/test_watch_cameras.py::test_end_to_end_small -q` → PASS (or skipped without the volume).
- [ ] **Step 3: Commit** `test(watcher): env-gated end-to-end integration`.

---

### Task 10: Deployment doc + full suite

**Files:** Create `docs/watcher-deploy.md`.

- [ ] **Step 1: Write deployment note**

```markdown
# Camera watcher — deployment

Cron, single pass per run (lock makes overlapping runs safe):

    */10 * * * * cd /home/Sebastian-Gerken/OTVision && \
      .venv/bin/python watch_cameras.py --once \
      "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
      >> logs_track/watch.log 2>&1

Knobs: --block-days 4, --idle-minutes 5, --stable-minutes 5, --slots-per-day 96,
--reserve-cores 2.

Per-camera state (in the camera dir):
- .otc_watch_state.json  -> tracked_through (the downstream "ready" signal)
- .otc_watch_scan.json   -> stability snapshot (first-sighting timestamps)
- lock: .locks/<camera>-<hash>.lock under the repo (flock; auto-released on exit)

Recovery: a crash never advances tracked_through, so the block is retried next
poll. To reprocess a camera, delete its .otc_watch_state.json. flock is released
automatically when a process dies; there are no stale lock files to clear.

Downstream MUST gate on tracked_through, not on raw .ottrk mtime (track writes
.ottrk non-atomically). Leftover complete days < block-days wait for a full block;
to flush a tail, run once with --block-days 1.
```

- [ ] **Step 2: Full suite** — Run: `python -m pytest tests/watcher -q` → all pass (integration skipped without the volume).
- [ ] **Step 3: Commit** `docs(watcher): cron deployment + recovery guide`.

---

## Self-Review

**Reviewer points folded in:** (1) locking — Task 7 holds `camera_lock` across flatten/track/**marker**; (2) scoped cleanup — Task 6 scopes moves+`._`+dir-removal+temp-report to selected days; (3) temp policy — Task 4 scoped+age (unblocks stale `.WQoSZM`); (4) transfer-complete — input stability snapshot (Task 5/7) + output `.ottrk` verify (Task 7); (5) exact 96-slot validation — Task 2; (6) ID continuity — **4-day continuous blocks** (Task 3 + one track per block); (7) marker concurrency — PID-unique temp + lock (Task 5/7); (8) integration env-gated — Task 9; (9) OS-load/thread guard — Task 8 `safe_parallelism`+`_overloaded`, plus execution-time subagent-capacity note in the header.

**Your concerns:** `.ottrk` completeness → `verify_outputs` gate + marker-as-contract (Task 7, policy). 5-min unchanged check → `check_stable` snapshot, default 5 (Task 5). Thread safety → `safe_parallelism`/`_overloaded` (Task 8).

**Rev.2 review patches:** (P1) lock acquired **before** marker read/assess — `process_camera` is fully lock-first (Task 7), so no read-then-mark race; tests isolate `LOCK_DIR` via an autouse fixture (Task 0). (P1) `check_stable` short-circuits `True` when `stable_minutes <= 0` (Task 5) — fixes the `stable_minutes=0` immediate-mode used by Task 7/9 tests. (P2) duplicate `.otdet` basenames in a block block firing — `_scan` detects, `assess_camera` rejects, with a test (Task 4). (P2) foreign-host files are ignored by host match in `_scan` (camera dir name, case-insensitive), with a test (Task 4). (P3) temp-freshness now consistently uses `--idle-minutes` in both policy and `assess_camera`.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `_scan` returns `(files, temps, dups, foreign)` and is consumed only inside `assess_camera` (Task 4). `CoverageReport(fire, reason, days, otdet_paths, tracked_through_after)` (Task 4) consumed unchanged in Task 7. `WatchConfig` fields (block_days, idle_minutes, stable_minutes, slots_per_day, reserve_cores) consistent across Tasks 7/8. `flatten_camera(..., date_filter=)` (Task 6) matches the Task 7 call. `check_stable(camera, block_key, files, *, now, stable_minutes)` and `set_tracked_through(camera, through, *, days, files, at)` match their callers.

**Known limitation (documented):** a trailing run shorter than `--block-days` is never auto-processed; flush with `--block-days 1`.
