"""Per-camera persistence: tracked-through marker + stability snapshot."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

MARKER = ".otc_watch_state.json"
SCAN = ".otc_watch_scan.json"
MAX_HISTORY = 50
MAX_SCAN_BLOCKS = 50


class StateUnreadable(RuntimeError):
    pass


@dataclass
class FailureState:
    count: int
    next_retry: datetime
    quarantined: bool


def _atomic_write(path: Path, data: dict) -> None:
    for stale in path.parent.glob(f"{path.name}.*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def get_tracked_through(camera: Path) -> date | None:
    p = camera / MARKER
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get("tracked_through")
    except (json.JSONDecodeError, OSError) as e:
        raise StateUnreadable(str(p)) from e
    return date.fromisoformat(v) if v else None


def set_tracked_through(
    camera: Path, through: date, *, days: int, files: int, at: str
) -> None:
    p = camera / MARKER
    try:
        state = json.loads(p.read_text()) if p.exists() else {"history": []}
    except (json.JSONDecodeError, OSError) as e:
        raise StateUnreadable(str(p)) from e
    state.update(camera=camera.name, tracked_through=through.isoformat(), updated=at)
    state.setdefault("history", []).append(
        {"through": through.isoformat(), "days": days, "files": files, "at": at}
    )
    state["history"] = state["history"][-MAX_HISTORY:]
    _atomic_write(p, state)


def _read_state(camera: Path) -> dict:
    p = camera / MARKER
    if not p.exists():
        return {"history": []}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise StateUnreadable(str(p)) from e


def block_failure(camera: Path, block_key: str, now: datetime) -> FailureState | None:
    data = _read_state(camera)
    entry = data.get("failures", {}).get(block_key)
    if not entry:
        return None
    next_retry = datetime.fromisoformat(entry["next_retry"])
    return FailureState(
        count=entry.get("count", 0),
        next_retry=next_retry,
        quarantined=entry.get("quarantined", False),
    )


def record_failure(
    camera: Path,
    block_key: str,
    reason: str,
    *,
    now: datetime,
    max_failures: int = 3,
) -> FailureState:
    data = _read_state(camera)
    failures = data.setdefault("failures", {})
    entry = failures.get(block_key, {})
    count = int(entry.get("count", 0)) + 1
    quarantined = count >= max_failures
    delay_minutes = min(60, 2 ** max(0, count - 1) * 10)
    next_retry = now if quarantined else now + timedelta(minutes=delay_minutes)
    failures[block_key] = {
        "count": count,
        "reason": reason,
        "last_at": now.isoformat(),
        "next_retry": next_retry.isoformat(),
        "quarantined": quarantined,
    }
    _atomic_write(camera / MARKER, data)
    return FailureState(count=count, next_retry=next_retry, quarantined=quarantined)


def clear_failure(camera: Path, block_key: str) -> None:
    data = _read_state(camera)
    failures = data.get("failures", {})
    if block_key in failures:
        failures.pop(block_key, None)
        _atomic_write(camera / MARKER, data)


def _signature(files: list[Path], base: Path) -> str:
    h = hashlib.sha1()
    for f in sorted(files):
        try:
            name = str(f.relative_to(base))
        except ValueError:
            name = f.name
        try:
            st = f.stat()
        except FileNotFoundError:
            h.update(f"{name}:missing\n".encode())
            continue
        h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def check_stable(
    camera: Path,
    block_key: str,
    files: list[Path],
    *,
    now: datetime,
    stable_minutes: int,
) -> bool:
    if stable_minutes <= 0:
        return True
    p = camera / SCAN
    data = json.loads(p.read_text()) if p.exists() else {}
    sig = _signature(files, camera)
    entry = data.get(block_key)
    if entry and entry["sig"] == sig:
        first = datetime.fromisoformat(entry["first_seen"])
        return (now - first).total_seconds() >= stable_minutes * 60
    data[block_key] = {"sig": sig, "first_seen": now.isoformat()}
    if len(data) > MAX_SCAN_BLOCKS:
        data = dict(
            sorted(data.items(), key=lambda item: item[1].get("first_seen", ""))[
                -MAX_SCAN_BLOCKS:
            ]
        )
    _atomic_write(p, data)
    return False
