"""Pure coverage assessment for the camera watcher (filenames + mtime only)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

_OTC_RE = re.compile(
    r"^(?P<host>[A-Za-z0-9]+)_FR\d+_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.otdet$"
)


def parse_otc_filename(name: str) -> tuple[str, datetime] | None:
    m = _OTC_RE.match(name)
    if not m:
        return None
    return m["host"], datetime.strptime(
        f"{m['date']}_{m['time']}", "%Y-%m-%d_%H-%M-%S"
    )


def expected_slots(slots_per_day: int) -> set[tuple[int, int]]:
    if 24 * 60 % slots_per_day != 0:
        raise ValueError("slots_per_day must divide 1440 minutes")
    step = 24 * 60 // slots_per_day
    return {(m // 60, m % 60) for m in range(0, 24 * 60, step)}


def complete_dates(slot_datetimes, slots_per_day: int = 96) -> set[date]:
    """Dates that cover every time-window of the day.

    A day is complete when each of ``slots_per_day`` equal-width windows
    (``1440 / slots_per_day`` minutes wide) holds at least one recording.
    Membership is by window, ignoring the exact second and sub-window phase:
    recorder clock drift (``..._06-00-01``) or a fixed phase offset
    (``..._00-08-06``) still counts toward its window. A genuinely missing
    window (no file at all) keeps the day incomplete.

    Note: windows derive from naive local filename timestamps over a fixed
    1440-minute day, so DST-transition days (23h/25h civil time) are not
    modelled and may never read complete -- an accepted limitation.
    """
    if slots_per_day <= 0 or 24 * 60 % slots_per_day != 0:
        raise ValueError("slots_per_day must divide 1440 minutes")
    step = 24 * 60 // slots_per_day
    full = set(range(slots_per_day))
    by_day: dict[date, set] = {}
    for dt in slot_datetimes:
        window = (dt.hour * 60 + dt.minute) // step
        by_day.setdefault(dt.date(), set()).add(window)
    return {d for d, got in by_day.items() if got == full}


def consecutive_runs(days: set[date]) -> list[list[date]]:
    runs: list[list[date]] = []
    for d in sorted(days):
        if runs and d - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


def next_block(
    complete: set[date], tracked_through: date | None, block_days: int = 4
) -> list[date]:
    """Next non-overlapping exact-size pending block."""
    for run in consecutive_runs(complete):
        pend = [d for d in run if tracked_through is None or d > tracked_through]
        if len(pend) >= block_days:
            return pend[:block_days]
    return []


@dataclass
class CoverageReport:
    fire: bool
    reason: str
    days: list[date] = field(default_factory=list)
    otdet_paths: list[Path] = field(default_factory=list)
    tracked_through_after: date | None = None


def _scan(camera: Path):
    host = camera.name.lower()
    files, temps, seen, dups, foreign = [], [], set(), set(), 0
    for f in camera.rglob("*"):
        try:
            is_file = f.is_file()
        except FileNotFoundError:
            continue
        if not is_file or f.name.startswith("._"):
            continue
        if f.name.startswith(".") and ".otdet" in f.name:
            inner = parse_otc_filename(
                f.name.lstrip(".").rsplit(".otdet", 1)[0] + ".otdet"
            )
            if inner and inner[0].lower() != host:
                foreign += 1
                continue
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


def assess_camera(
    camera: Path,
    *,
    now,
    tracked_through,
    block_days=4,
    slots_per_day=96,
    idle_minutes=5,
) -> CoverageReport:
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
    if any(p.name in dups for _, p in window):
        return CoverageReport(False, "duplicate .otdet basenames in block")
    fresh = now.timestamp() - idle_minutes * 60
    for d, p in temps:
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if d in dayset and mtime > fresh:
            return CoverageReport(False, f"fresh temp write in block ({p.name})")
    for _, p in window:
        try:
            if p.stat().st_mtime > fresh:
                return CoverageReport(False, f"block not idle (<{idle_minutes}m)")
        except FileNotFoundError:
            return CoverageReport(False, "block changed during scan")
    paths = [p for _, p in sorted(window)]
    return CoverageReport(True, "fire", days, paths, max(days))
