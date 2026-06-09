"""Pure coverage assessment for the camera watcher (filenames + mtime only)."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

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


def next_block(
    complete: set[date], tracked_through: date | None, block_days: int = 4
) -> list[date]:
    """Next non-overlapping exact-size pending block."""
    for run in consecutive_runs(complete):
        pend = [d for d in run if tracked_through is None or d > tracked_through]
        if len(pend) >= block_days:
            return pend[:block_days]
    return []
