# Camera Watcher Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A polling service that watches the project `videos/` tree and, whenever a camera has a complete, settled ≥4-day window of `.otdet` not yet processed, automatically flattens those days and continuously tracks them with BoT-SORT.

**Architecture:** Three small, pure-where-possible modules in the repo root next to the existing helpers. `otc_coverage.py` decides *whether and what* to process from filenames + mtimes (no heavy metadata reads). `otc_state.py` persists a per-camera "tracked-through" marker so already-done days are never reprocessed. `watch_cameras.py` is the poll loop: for each camera it assesses coverage, and on a fire it calls the existing `flatten_camera` (scoped to the window's date folders) then runs `track.py` over exactly the window's files. Filesystem-event watching is deliberately avoided — the volume is a CIFS/SMB mount where inotify does not see other hosts' writes — so we poll (cron `--once`, or `--interval`).

**Tech Stack:** Python 3.12 stdlib (argparse, dataclasses, pathlib, json, re, datetime, subprocess, concurrent.futures), pytest. Reuses repo modules `flatten_camera.py` and `track.py` + `config.continuous.botsort.yaml`.

---

## Policy & Definitions (the spec this plan implements)

These are the decisions crystallised from the design discussion. They are the contract the tasks below must satisfy. Knobs are CLI flags with the stated defaults.

- **OTCamera filename grammar.** `<HOST>_FR<fps>_<YYYY-MM-DD>_<HH-MM-SS>.otdet`, e.g. `OTCamera07_FR20_2026-06-03_00-00-00.otdet`. `HOST` is the camera id; date+time give the recording start.
- **Slot / cadence.** Recordings are 15-minute segments → **96 slots/day** (`--slots-per-day`, default 96).
- **Complete day.** A calendar date whose `.otdet` cover **all** expected slots (count of distinct slot-times == slots-per-day). An in-progress day is `< 96` ⇒ automatically *not* complete ⇒ never tracked early.
- **Coverage scan domain ("including the previous moving").** Coverage is computed over `.otdet` found in **both** the camera root (already-flattened/already-tracked days) **and** the date subfolders (new arrivals), so a camera that was flattened in a previous cycle is assessed correctly and new days extend the window.
- **Pending window + fire rule (default `--min-days 4`).** Let `complete` = complete days (root + subdirs); group into maximal consecutive-date runs. `pending` = complete days `> tracked_through` (or all, if no marker). **Fire** on the earliest run that (a) has length ≥ `min_days` and (b) contains pending days; the days to process are that run's pending days. Effect: the first fire needs ≥4 consecutive complete days; once a run is ≥4 and partly tracked, each newly-completed day in that run fires (rolling). A brand-new run after a gap must itself reach ≥4 before firing.
- **Settled (race guard vs upstream detector).** Fire only if the window's files are settled: **no** `.<name>.otdet.XXXX` atomic-write temp files anywhere under the camera, **and** no window `.otdet` modified within the last `--idle-minutes` (default 30). The in-progress day is already excluded by completeness; this is belt-and-suspenders.
- **Scoped flatten.** On fire, flatten **only** the window's date folders (move their files to the camera root via atomic rename). Folders of incomplete/in-progress days are left untouched so the detector can keep writing.
- **Track exactly the window.** Run one `track.py` over exactly the window's `.otdet` (now in the root), `--tracker botsort -c config.continuous.botsort.yaml --overwrite`. IDs reset at the boundary between previously-tracked and new days (accepted trade-off of the rolling policy).
- **Marker (state).** `<camera>/.otc_watch_state.json`, written atomically (temp + `os.replace`):
  ```json
  {"camera": "OTCamera07", "tracked_through": "2026-06-06",
   "updated": "2026-06-09T15:00:00+00:00",
   "history": [{"through": "2026-06-06", "days": 4, "files": 384, "at": "..."}]}
  ```
  `tracked_through` advances to `max(window days)` **only after** flatten+track both succeed.
- **Failure handling.** Any failure (flatten conflict, track non-zero exit, crash) ⇒ marker is **not** advanced ⇒ the window is reassessed and retried on the next poll. All stages are idempotent (atomic moves; `--overwrite` re-derives `.ottrk`; marker is the single source of "done"). Cameras are independent; one failing camera never blocks others.
- **Run model.** `--once` does a single pass (intended for `cron`, the recommended deployment — no long-lived daemon to babysit). `--interval N` loops every N seconds for ad-hoc use.

---

## File Structure

- `otc_coverage.py` (create) — pure coverage logic: filename parsing, complete-day detection, run grouping, fire decision, settledness, `assess_camera`. No tracking, no moving.
- `otc_state.py` (create) — per-camera marker read/write (atomic), `get_tracked_through` / `set_tracked_through`.
- `flatten_camera.py` (modify) — add an optional `date_filter` so a flatten can be scoped to specific days.
- `watch_cameras.py` (create) — discovery, `process_camera` (DI-friendly: `flatten_fn`/`track_fn` injectable), `run_track` subprocess wrapper, poll loop + CLI.
- `tests/watcher/conftest.py` (create) — put repo root on `sys.path` so the root-level modules import under pytest.
- `tests/watcher/test_otc_coverage.py` (create)
- `tests/watcher/test_otc_state.py` (create)
- `tests/watcher/test_flatten_date_filter.py` (create)
- `tests/watcher/test_watch_cameras.py` (create)

---

### Task 0: Test bootstrap

**Files:**
- Create: `tests/watcher/conftest.py`

- [ ] **Step 1: Create conftest so root modules import under pytest**

```python
# tests/watcher/conftest.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 2: Verify pytest collects the (empty) dir**

Run: `python -m pytest tests/watcher -q`
Expected: `no tests ran` (exit 5) — confirms collection works, no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/watcher/conftest.py
git commit -m "test(watcher): add conftest to expose repo root modules"
```

---

### Task 1: Filename parsing (`otc_coverage.parse_otc_filename`)

**Files:**
- Create: `otc_coverage.py`
- Test: `tests/watcher/test_otc_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watcher/test_otc_coverage.py
from datetime import datetime
from otc_coverage import parse_otc_filename


def test_parse_valid_filename():
    host, dt = parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet")
    assert host == "OTCamera07"
    assert dt == datetime(2026, 6, 3, 0, 15, 0)


def test_parse_rejects_dotfile_and_garbage():
    assert parse_otc_filename("._OTCamera07_FR20_2026-06-03_00-15-00.otdet") is None
    assert parse_otc_filename("notes.txt") is None
    assert parse_otc_filename("OTCamera07_FR20_2026-06-03_00-15-00.otdet.TMP9") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'otc_coverage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# otc_coverage.py
"""Coverage logic for the camera watcher: decide whether/what to process.

Pure, filename- and mtime-based (no heavy .otdet metadata reads) so it is cheap
to run on every poll. See docs/superpowers/plans for the policy it implements.
"""
from __future__ import annotations

import re
from datetime import datetime

_OTC_RE = re.compile(
    r"^(?P<host>[A-Za-z0-9]+)_FR\d+_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.otdet$"
)


def parse_otc_filename(name: str) -> tuple[str, datetime] | None:
    """Return (host, start_datetime) for an OTCamera .otdet name, else None."""
    m = _OTC_RE.match(name)
    if not m:
        return None
    dt = datetime.strptime(f"{m['date']}_{m['time']}", "%Y-%m-%d_%H-%M-%S")
    return m["host"], dt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add otc_coverage.py tests/watcher/test_otc_coverage.py
git commit -m "feat(watcher): parse OTCamera .otdet filenames"
```

---

### Task 2: Complete days + consecutive runs

**Files:**
- Modify: `otc_coverage.py`
- Test: `tests/watcher/test_otc_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/watcher/test_otc_coverage.py
from datetime import date, timedelta
from otc_coverage import complete_dates, consecutive_runs


def _day_slots(day: date, n: int):
    """n distinct 15-min slot datetimes on `day`."""
    base = datetime(day.year, day.month, day.day)
    return [base + timedelta(minutes=15 * i) for i in range(n)]


def test_complete_dates_requires_all_slots():
    full = _day_slots(date(2026, 6, 3), 96)       # complete
    partial = _day_slots(date(2026, 6, 4), 40)    # incomplete
    dts = full + partial
    assert complete_dates(dts, slots_per_day=96) == {date(2026, 6, 3)}


def test_consecutive_runs_groups_by_adjacency():
    ds = {date(2026, 6, 3), date(2026, 6, 4), date(2026, 6, 6)}
    assert consecutive_runs(ds) == [
        [date(2026, 6, 3), date(2026, 6, 4)],
        [date(2026, 6, 6)],
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'complete_dates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to otc_coverage.py
from collections import Counter
from datetime import date, timedelta


def complete_dates(slot_datetimes, slots_per_day: int = 96) -> set[date]:
    """Dates having `slots_per_day` distinct slot times among the given datetimes."""
    by_day: dict[date, set] = {}
    for dt in slot_datetimes:
        by_day.setdefault(dt.date(), set()).add(dt.timetuple()[3:5])  # (H, M)
    return {d for d, slots in by_day.items() if len(slots) >= slots_per_day}


def consecutive_runs(days: set[date]) -> list[list[date]]:
    """Group dates into maximal runs of calendar-adjacent days, sorted."""
    runs: list[list[date]] = []
    for d in sorted(days):
        if runs and d - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add otc_coverage.py tests/watcher/test_otc_coverage.py
git commit -m "feat(watcher): detect complete days and consecutive runs"
```

---

### Task 3: Fire decision (`pending_window`)

**Files:**
- Modify: `otc_coverage.py`
- Test: `tests/watcher/test_otc_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/watcher/test_otc_coverage.py
from otc_coverage import pending_window

D = lambda n: date(2026, 6, n)  # noqa: E731


def test_first_fire_needs_min_days():
    three = {D(3), D(4), D(5)}
    four = {D(3), D(4), D(5), D(6)}
    assert pending_window(three, tracked_through=None, min_days=4) == []
    assert pending_window(four, tracked_through=None, min_days=4) == [D(3), D(4), D(5), D(6)]


def test_rolling_fire_after_marker():
    comp = {D(3), D(4), D(5), D(6), D(7)}   # run already >=4, tracked through 6
    assert pending_window(comp, tracked_through=D(6), min_days=4) == [D(7)]


def test_new_short_run_after_gap_waits():
    comp = {D(3), D(4), D(5), D(6),  D(9), D(10)}  # gap; new run len 2
    assert pending_window(comp, tracked_through=D(6), min_days=4) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'pending_window'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to otc_coverage.py
def pending_window(
    complete: set[date], tracked_through: date | None, min_days: int = 4
) -> list[date]:
    """Days to process now: the earliest >=min_days run that has untracked days."""
    for run in consecutive_runs(complete):
        if len(run) < min_days:
            continue
        pend = [d for d in run if tracked_through is None or d > tracked_through]
        if pend:
            return pend
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add otc_coverage.py tests/watcher/test_otc_coverage.py
git commit -m "feat(watcher): rolling >=N-day fire decision"
```

---

### Task 4: Scan + assess a camera on disk (`assess_camera`)

**Files:**
- Modify: `otc_coverage.py`
- Test: `tests/watcher/test_otc_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/watcher/test_otc_coverage.py
import os
from pathlib import Path
from otc_coverage import assess_camera


def _make_otdet(dirpath: Path, day: date, n_slots: int, old: bool = True):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(n_slots):
        t = datetime(day.year, day.month, day.day) + timedelta(minutes=15 * i)
        f = dirpath / f"OTCamera07_FR20_{t:%Y-%m-%d}_{t:%H-%M-%S}.otdet"
        f.write_bytes(b"x")
        if old:
            past = 10_000_000  # ~115 days; safely older than idle window
            os.utime(f, (past, past))


def test_assess_fires_on_four_complete_settled_days(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_otdet(cam / f"2026-06-0{n}", date(2026, 6, n), 96)
    now = datetime(2026, 6, 9, 12, 0, 0)
    rep = assess_camera(cam, now=now, tracked_through=None,
                        min_days=4, slots_per_day=96, idle_minutes=30)
    assert rep.fire is True
    assert rep.tracked_through_after == date(2026, 6, 6)
    assert len(rep.otdet_paths) == 96 * 4


def test_assess_holds_when_temp_present(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_otdet(cam / f"2026-06-0{n}", date(2026, 6, n), 96)
    (cam / "2026-06-06" / ".OTCamera07_FR20_2026-06-06_10-30-00.otdet.TMP").write_bytes(b"")
    rep = assess_camera(cam, now=datetime(2026, 6, 9, 12, 0), tracked_through=None,
                        min_days=4, slots_per_day=96, idle_minutes=30)
    assert rep.fire is False
    assert "temp" in rep.reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'assess_camera'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to otc_coverage.py
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
    """(parsed otdet files, has_temp_dotfile) under camera, root + subdirs."""
    files, has_temp = [], False
    for f in camera.rglob("*"):
        if not f.is_file():
            continue
        if f.name.startswith("._"):
            continue
        if f.name.startswith("."):           # .NAME.otdet.XXXX atomic-write temp
            if ".otdet" in f.name:
                has_temp = True
            continue
        parsed = parse_otc_filename(f.name)
        if parsed:
            files.append((parsed[1], f))       # (start_dt, path)
    return files, has_temp


def assess_camera(
    camera: Path, *, now: datetime, tracked_through: date | None,
    min_days: int = 4, slots_per_day: int = 96, idle_minutes: int = 30,
) -> CoverageReport:
    files, has_temp = _scan(camera)
    if not files:
        return CoverageReport(False, "no .otdet found")
    complete = complete_dates([dt for dt, _ in files], slots_per_day)
    days = pending_window(complete, tracked_through, min_days)
    if not days:
        return CoverageReport(False, f"no complete >= {min_days}-day window pending")
    if has_temp:
        return CoverageReport(False, "temp/partial write present (detector active)")
    dayset = set(days)
    window = [(dt, p) for dt, p in files if dt.date() in dayset]
    idle_cutoff = now.timestamp() - idle_minutes * 60
    if any(p.stat().st_mtime > idle_cutoff for _, p in window):
        return CoverageReport(False, f"window not settled (<{idle_minutes}m old)")
    paths = [p for _, p in sorted(window)]
    return CoverageReport(True, "fire", days, paths, max(days))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_otc_coverage.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add otc_coverage.py tests/watcher/test_otc_coverage.py
git commit -m "feat(watcher): assess_camera coverage report from disk"
```

---

### Task 5: Per-camera state marker (`otc_state.py`)

**Files:**
- Create: `otc_state.py`
- Test: `tests/watcher/test_otc_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watcher/test_otc_state.py
from datetime import date
from otc_state import get_tracked_through, set_tracked_through


def test_roundtrip_and_default(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    assert get_tracked_through(cam) is None
    set_tracked_through(cam, date(2026, 6, 6), days=4, files=384, at="2026-06-09T15:00:00+00:00")
    assert get_tracked_through(cam) == date(2026, 6, 6)


def test_history_appends(tmp_path):
    cam = tmp_path / "OTCamera07"
    cam.mkdir()
    set_tracked_through(cam, date(2026, 6, 6), days=4, files=384, at="t1")
    set_tracked_through(cam, date(2026, 6, 7), days=1, files=96, at="t2")
    import json
    state = json.loads((cam / ".otc_watch_state.json").read_text())
    assert state["tracked_through"] == "2026-06-07"
    assert len(state["history"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_otc_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'otc_state'`.

- [ ] **Step 3: Write minimal implementation**

```python
# otc_state.py
"""Per-camera 'tracked-through' marker, written atomically."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

MARKER = ".otc_watch_state.json"


def _path(camera: Path) -> Path:
    return camera / MARKER


def get_tracked_through(camera: Path) -> date | None:
    p = _path(camera)
    if not p.exists():
        return None
    val = json.loads(p.read_text()).get("tracked_through")
    return date.fromisoformat(val) if val else None


def set_tracked_through(camera: Path, through: date, *, days: int, files: int, at: str) -> None:
    p = _path(camera)
    state = json.loads(p.read_text()) if p.exists() else {"history": []}
    state["camera"] = camera.name
    state["tracked_through"] = through.isoformat()
    state["updated"] = at
    state.setdefault("history", []).append(
        {"through": through.isoformat(), "days": days, "files": files, "at": at}
    )
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, p)  # atomic
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_otc_state.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add otc_state.py tests/watcher/test_otc_state.py
git commit -m "feat(watcher): atomic per-camera tracked-through marker"
```

---

### Task 6: Scoped flatten (`flatten_camera` `date_filter`)

**Files:**
- Modify: `flatten_camera.py` (function `find_sources` and `flatten_camera`)
- Test: `tests/watcher/test_flatten_date_filter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watcher/test_flatten_date_filter.py
from datetime import date
from pathlib import Path
from flatten_camera import flatten_camera


def _f(d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"x")


def test_date_filter_moves_only_selected_days(tmp_path):
    cam = tmp_path / "OTCamera07"
    _f(cam / "2026-06-03", "OTCamera07_FR20_2026-06-03_00-00-00.otdet")
    _f(cam / "2026-06-07", "OTCamera07_FR20_2026-06-07_00-00-00.otdet")  # in-progress day
    res = flatten_camera(cam, date_filter=lambda d: d == date(2026, 6, 3),
                         clean_appledouble=True, log=lambda m: None)
    assert res.ok and res.moved == 1
    assert (cam / "OTCamera07_FR20_2026-06-03_00-00-00.otdet").exists()
    # the in-progress day was left untouched in its folder:
    assert (cam / "2026-06-07" / "OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
    assert not (cam / "OTCamera07_FR20_2026-06-07_00-00-00.otdet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_flatten_date_filter.py -q`
Expected: FAIL — `TypeError: flatten_camera() got an unexpected keyword argument 'date_filter'`.

- [ ] **Step 3: Write minimal implementation**

In `flatten_camera.py`, add the import and a filter param. At the top with the other imports add:

```python
from otc_coverage import parse_otc_filename
```

Change the `flatten_camera` signature to accept `date_filter`:

```python
def flatten_camera(
    camera: Path,
    types: tuple[str, ...] = DEFAULT_TYPES,
    clean_appledouble: bool = True,
    dry_run: bool = False,
    date_filter=None,                      # Callable[[date], bool] | None
    log: Callable[[str], None] = print,
) -> FlattenResult:
```

Immediately after `sources, appledouble, temp_dotfiles = find_sources(camera, types)` insert:

```python
    if date_filter is not None:
        kept = []
        for s in sources:
            parsed = parse_otc_filename(s.name)
            if parsed and date_filter(parsed[1].date()):
                kept.append(s)
        sources = kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_flatten_date_filter.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the existing flatten path to confirm no regression**

Run: `python flatten_camera.py --dry-run "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07"`
Expected: prints the file/junk/temp summary as before (no traceback).

- [ ] **Step 6: Commit**

```bash
git add flatten_camera.py tests/watcher/test_flatten_date_filter.py
git commit -m "feat(flatten): optional date_filter to scope a flatten to chosen days"
```

---

### Task 7: `process_camera` with injected flatten/track (unit)

**Files:**
- Create: `watch_cameras.py`
- Test: `tests/watcher/test_watch_cameras.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/watcher/test_watch_cameras.py
from datetime import date, datetime, timedelta
import os
from pathlib import Path
from otc_state import get_tracked_through
from watch_cameras import process_camera, WatchConfig


def _make_day(cam: Path, day: date, n: int = 96):
    d = cam / f"{day:%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = datetime(day.year, day.month, day.day) + timedelta(minutes=15 * i)
        f = d / f"OTCamera07_FR20_{t:%Y-%m-%d}_{t:%H-%M-%S}.otdet"
        f.write_bytes(b"x")
        os.utime(f, (10_000_000, 10_000_000))


def test_process_camera_fires_flattens_tracks_marks(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    calls = {"flatten_dates": None, "tracked": None}

    def fake_flatten(camera, date_filter=None, log=None, **kw):
        calls["flatten_dates"] = [date(2026, 6, n) for n in (3, 4, 5, 6)
                                  if date_filter(date(2026, 6, n))]
        # simulate the move into root
        for f in list(camera.rglob("*.otdet")):
            if date_filter(parse(f)):
                f.rename(camera / f.name)
        from flatten_camera import FlattenResult
        return FlattenResult(camera=camera, moved=len(calls["flatten_dates"]) * 96)

    def fake_track(paths, log=None):
        calls["tracked"] = len(paths)
        return True

    def parse(p):
        from otc_coverage import parse_otc_filename
        return parse_otc_filename(p.name)[1].date()

    cfg = WatchConfig(config=Path("config.continuous.botsort.yaml"),
                      log_dir=tmp_path / "logs", min_days=4, idle_minutes=30,
                      slots_per_day=96)
    outcome = process_camera(cam, now=datetime(2026, 6, 9, 12, 0), cfg=cfg,
                             flatten_fn=fake_flatten, track_fn=fake_track,
                             log=lambda m: None)
    assert outcome.status == "tracked"
    assert calls["flatten_dates"] == [date(2026, 6, n) for n in (3, 4, 5, 6)]
    assert calls["tracked"] == 96 * 4
    assert get_tracked_through(cam) == date(2026, 6, 6)


def test_process_camera_does_not_mark_on_track_failure(tmp_path):
    cam = tmp_path / "OTCamera07"
    for n in (3, 4, 5, 6):
        _make_day(cam, date(2026, 6, n))
    cfg = WatchConfig(config=Path("config.continuous.botsort.yaml"),
                      log_dir=tmp_path / "logs", min_days=4, idle_minutes=30,
                      slots_per_day=96)
    out = process_camera(cam, now=datetime(2026, 6, 9, 12, 0), cfg=cfg,
                         flatten_fn=lambda *a, **k: __import__("flatten_camera").FlattenResult(camera=cam),
                         track_fn=lambda paths, log=None: False, log=lambda m: None)
    assert out.status == "failed"
    assert get_tracked_through(cam) is None     # NOT advanced on failure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_watch_cameras.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'watch_cameras'`.

- [ ] **Step 3: Write minimal implementation**

```python
# watch_cameras.py
"""Polling watcher: flatten+track each camera once a complete, settled window arrives.

Poll (not inotify): the videos volume is a CIFS/SMB mount, where inotify does not
observe other hosts' writes. Run via cron with --once (recommended) or --interval.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from flatten_camera import flatten_camera
from otc_coverage import assess_camera
from otc_state import get_tracked_through, set_tracked_through

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
TRACK_SCRIPT = SCRIPT_DIR / "track.py"
DEFAULT_CONFIG = SCRIPT_DIR / "config.continuous.botsort.yaml"
CAMERA_GLOBS = ("OTCamera*", "otcamera*")


@dataclass
class WatchConfig:
    config: Path
    log_dir: Path
    min_days: int = 4
    idle_minutes: int = 30
    slots_per_day: int = 96


@dataclass
class Outcome:
    camera: Path
    status: str            # "idle" | "tracked" | "failed"
    detail: str = ""


def process_camera(
    camera: Path, *, now: datetime, cfg: WatchConfig,
    flatten_fn: Callable = flatten_camera, track_fn: Callable | None = None,
    log: Callable[[str], None] = print,
) -> Outcome:
    if track_fn is None:
        track_fn = lambda paths, log=log: run_track(paths, cfg, camera, log)
    tt = get_tracked_through(camera)
    rep = assess_camera(camera, now=now, tracked_through=tt,
                        min_days=cfg.min_days, slots_per_day=cfg.slots_per_day,
                        idle_minutes=cfg.idle_minutes)
    if not rep.fire:
        log(f"{camera.name}: idle ({rep.reason})")
        return Outcome(camera, "idle", rep.reason)

    wanted = set(rep.days)
    log(f"{camera.name}: firing for {rep.days[0]}..{rep.days[-1]} "
        f"({len(rep.otdet_paths)} files)")
    fres = flatten_fn(camera, date_filter=lambda d: d in wanted, log=log)
    if not getattr(fres, "ok", True):
        return Outcome(camera, "failed", "flatten conflict")

    flat_paths = [camera / p.name for p in rep.otdet_paths]
    if track_fn(flat_paths, log=log):
        set_tracked_through(camera, rep.tracked_through_after,
                            days=len(rep.days), files=len(flat_paths),
                            at=now.isoformat())
        return Outcome(camera, "tracked", f"through {rep.tracked_through_after}")
    return Outcome(camera, "failed", "track failed; will retry next poll")


def run_track(paths: list[Path], cfg: WatchConfig, camera: Path,
              log: Callable[[str], None]) -> bool:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.log_dir / f"{camera.name}_{stamp}.otvision.log"
    cmd = [str(PYTHON), str(TRACK_SCRIPT),
           "-p", *[str(p) for p in paths],
           "-c", str(cfg.config), "--tracker", "botsort", "--overwrite",
           "--logfile", str(logfile), "--logfile-overwrite"]
    console = cfg.log_dir / f"{camera.name}_{stamp}.console.log"
    with console.open("w") as fh:
        fh.write("# " + " ".join(cmd[:6]) + f" ... ({len(paths)} paths)\n")
        fh.flush()
        try:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e:
            log(f"{camera.name}: track exit {e.returncode} (see {console.name})")
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_watch_cameras.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add watch_cameras.py tests/watcher/test_watch_cameras.py
git commit -m "feat(watcher): process_camera assess->flatten->track->mark"
```

---

### Task 8: Discovery, poll loop, CLI

**Files:**
- Modify: `watch_cameras.py`
- Test: `tests/watcher/test_watch_cameras.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/watcher/test_watch_cameras.py
from watch_cameras import discover_cameras


def test_discover_cameras_matches_globs(tmp_path):
    (tmp_path / "OTCamera07").mkdir()
    (tmp_path / "otcamera23").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "notes.txt").write_bytes(b"x")
    found = {p.name for p in discover_cameras(tmp_path)}
    assert found == {"OTCamera07", "otcamera23"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/watcher/test_watch_cameras.py::test_discover_cameras_matches_globs -q`
Expected: FAIL — `ImportError: cannot import name 'discover_cameras'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to watch_cameras.py
def discover_cameras(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in CAMERA_GLOBS:
        found.extend(d for d in root.glob(pattern) if d.is_dir())
    seen, unique = set(), []
    for c in sorted(found):
        if c.resolve() not in seen:
            seen.add(c.resolve()); unique.append(c)
    return unique


def poll_once(root: Path, cfg: WatchConfig, now: datetime,
              log: Callable[[str], None] = print) -> list[Outcome]:
    outcomes = []
    for cam in discover_cameras(root):
        try:
            outcomes.append(process_camera(cam, now=now, cfg=cfg, log=log))
        except Exception as e:  # noqa: BLE001 - one camera must not kill the loop
            log(f"{cam.name}: ERROR {e}")
            outcomes.append(Outcome(cam, "failed", str(e)))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    import time

    p = argparse.ArgumentParser(description="Watch cameras; flatten+track complete windows.")
    p.add_argument("root", help="Project videos directory containing camera folders.")
    p.add_argument("--once", action="store_true", help="Single pass then exit (use with cron).")
    p.add_argument("--interval", type=int, default=600, help="Seconds between passes (loop mode).")
    p.add_argument("--min-days", type=int, default=4)
    p.add_argument("--idle-minutes", type=int, default=30)
    p.add_argument("--slots-per-day", type=int, default=96)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track"))
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
        return 2
    cfg = WatchConfig(config=Path(args.config), log_dir=Path(args.log_dir),
                      min_days=args.min_days, idle_minutes=args.idle_minutes,
                      slots_per_day=args.slots_per_day)

    def run() -> None:
        now = datetime.now(timezone.utc)
        outs = poll_once(root, cfg, now)
        fired = [o for o in outs if o.status == "tracked"]
        failed = [o for o in outs if o.status == "failed"]
        print(f"[{now.isoformat()}] {len(outs)} cameras | "
              f"{len(fired)} tracked, {len(failed)} failed, "
              f"{len(outs) - len(fired) - len(failed)} idle")

    if args.once:
        run(); return 0
    while True:
        run()
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/watcher/test_watch_cameras.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add watch_cameras.py tests/watcher/test_watch_cameras.py
git commit -m "feat(watcher): discovery, poll loop, and CLI"
```

---

### Task 9: End-to-end integration on a throwaway tree (real track.py)

**Files:**
- Test: `tests/watcher/test_watch_cameras.py`

- [ ] **Step 1: Write the failing test (uses 2 real .otdet, slots-per-day=2 to stay fast)**

```python
# append to tests/watcher/test_watch_cameras.py
import shutil
import pytest

REAL = Path("/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07")


@pytest.mark.integration
def test_end_to_end_small(tmp_path):
    src = REAL / "2026-06-03"
    samples = sorted(src.glob("OTCamera07_FR20_2026-06-03_01-3*-00.otdet"))[:2]
    if len(samples) < 2:
        pytest.skip("real sample .otdet not available")
    cam = tmp_path / "OTCamera07" / "2026-06-03"
    cam.mkdir(parents=True)
    for s in samples:
        shutil.copy2(s, cam / s.name)
    camera = tmp_path / "OTCamera07"
    cfg = WatchConfig(config=Path("config.continuous.botsort.yaml"),
                      log_dir=tmp_path / "logs", min_days=1,
                      idle_minutes=0, slots_per_day=2)
    out = process_camera(camera, now=datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc), cfg=cfg)
    assert out.status == "tracked"
    ottrk = sorted(camera.glob("*.ottrk"))
    assert len(ottrk) == 2                     # flat .ottrk produced in camera root
    assert get_tracked_through(camera) == date(2026, 6, 3)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/watcher/test_watch_cameras.py::test_end_to_end_small -q`
Expected: PASS (or `skipped` if the volume isn't mounted on the runner).

- [ ] **Step 3: Commit**

```bash
git add tests/watcher/test_watch_cameras.py
git commit -m "test(watcher): end-to-end flatten+track integration"
```

---

### Task 10: Deployment doc (cron) + full suite green

**Files:**
- Create: `docs/watcher-deploy.md`

- [ ] **Step 1: Write the deployment note**

```markdown
# Camera watcher — deployment

Recommended: cron, single pass each run (no long-lived daemon).

    */10 * * * * cd /home/Sebastian-Gerken/OTVision && \
      .venv/bin/python watch_cameras.py --once \
      "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
      >> logs_track/watch.log 2>&1

Ad-hoc loop instead of cron:

    nohup .venv/bin/python watch_cameras.py --interval 600 \
      "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos" \
      >> logs_track/watch.log 2>&1 &

Knobs: --min-days (default 4), --idle-minutes (30), --slots-per-day (96).
State per camera: <camera>/.otc_watch_state.json (tracked_through). Delete it to
reprocess a camera from scratch. Failures never advance the marker → auto-retried.
```

- [ ] **Step 2: Run the whole watcher suite**

Run: `python -m pytest tests/watcher -q`
Expected: all pass (integration test may show `skipped` off the volume).

- [ ] **Step 3: Commit**

```bash
git add docs/watcher-deploy.md
git commit -m "docs(watcher): cron deployment guide"
```

---

## Self-Review

**Spec coverage:**
- Poll (not inotify) on CIFS → Task 8 `poll_once`/CLI + Task 10 cron. ✓
- Coverage over root + subdirs ("including previous moving") → Task 4 `_scan` walks `rglob`. ✓
- Complete day = all slots → Task 2 `complete_dates`. ✓
- Rolling ≥4-day fire, only not-yet-done range → Task 3 `pending_window` + Task 4. ✓
- Settled / detector-race guard (temp files + idle mtime) → Task 4. ✓
- Scoped flatten (leave in-progress day) → Task 6 `date_filter` + Task 7 wiring. ✓
- Track exactly the window via `track.py -p <files>` → Task 7 `run_track`. ✓
- Marker advanced only on success; failures retried → Task 5 + Task 7 (`test_..._not_mark_on_track_failure`). ✓
- Idempotent / crash-safe → atomic moves (existing), `--overwrite`, atomic marker (Task 5). ✓
- Per-camera isolation → Task 8 `poll_once` try/except. ✓

**Placeholder scan:** none — every code step is complete and runnable.

**Type consistency:** `CoverageReport(fire, reason, days, otdet_paths, tracked_through_after)` used identically in Tasks 4/7. `WatchConfig`/`Outcome` fields consistent across Tasks 7/8. `flatten_camera(..., date_filter=...)` signature (Task 6) matches the call in Task 7. `set_tracked_through(camera, through, *, days, files, at)` matches caller in Task 7.

**Open policy knob for your review:** the fire rule fires for *each new complete day* once a run is ≥`min_days` (rolling). If you'd rather fire only on whole new ≥4-day blocks, that's a one-line change in `pending_window` (require `len(pend) >= min_days`). Flagged here rather than assumed.
