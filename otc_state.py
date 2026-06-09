"""Per-camera persistence: tracked-through marker + stability snapshot."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path

MARKER = ".otc_watch_state.json"
SCAN = ".otc_watch_scan.json"


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
    except (json.JSONDecodeError, OSError):
        return None
    return date.fromisoformat(v) if v else None


def set_tracked_through(
    camera: Path, through: date, *, days: int, files: int, at: str
) -> None:
    p = camera / MARKER
    try:
        state = json.loads(p.read_text()) if p.exists() else {"history": []}
    except (json.JSONDecodeError, OSError):
        state = {"history": []}
    state.update(camera=camera.name, tracked_through=through.isoformat(), updated=at)
    state.setdefault("history", []).append(
        {"through": through.isoformat(), "days": days, "files": files, "at": at}
    )
    _atomic_write(p, state)


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
    _atomic_write(p, data)
    return False
