"""Poll camera dirs and track complete, stable blocks."""
from __future__ import annotations

import argparse
import bz2
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from flatten_camera import flatten_camera
from otc_coverage import assess_camera
from otc_state import (
    StateUnreadable,
    block_failure,
    check_stable,
    clear_failure,
    get_tracked_through,
    record_failure,
    set_tracked_through,
)
from track_continuous import camera_lock

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
TRACK_SCRIPT = SCRIPT_DIR / "track.py"
DEFAULT_CONFIG = SCRIPT_DIR / "config.continuous.botsort.yaml"
CAMERA_GLOBS = ("OTCamera*", "otcamera*")
_ACTIVE_CHILDREN = set()
_ACTIVE_CHILDREN_LOCK = threading.Lock()
_SIGNAL_HANDLERS_INSTALLED = False
_PREVIOUS_SIGNAL_HANDLERS = {}


@dataclass
class WatchConfig:
    config: Path
    log_dir: Path
    block_days: int = 4
    idle_minutes: int = 5
    stable_minutes: int = 5
    slots_per_day: int = 96
    reserve_cores: int = 2
    max_parallel: int = 1
    cores_per_track: int = 4
    max_failures: int = 3


@dataclass
class Outcome:
    camera: Path
    status: str
    detail: str = ""


def _bz2_ok(path: Path) -> bool:
    try:
        with bz2.open(path, "rt") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


def verify_outputs(otdet_paths: list[Path]) -> list[Path]:
    bad = []
    for p in otdet_paths:
        t = p.with_suffix(".ottrk")
        if not t.exists() or t.stat().st_size == 0 or not _bz2_ok(t):
            bad.append(t)
    return bad


def process_camera(
    camera: Path,
    *,
    now: datetime,
    cfg: WatchConfig,
    flatten_fn: Callable = flatten_camera,
    track_fn: Callable | None = None,
    log: Callable[[str], None] = print,
) -> Outcome:
    if track_fn is None:
        track_fn = lambda paths, log=log: _run_track(paths, cfg, camera, log)
    with camera_lock(camera) as got:
        if not got:
            return Outcome(camera, "skipped", "locked by another run")
        try:
            tt = get_tracked_through(camera)
        except StateUnreadable:
            return Outcome(camera, "failed", "state marker unreadable")
        rep = assess_camera(
            camera,
            now=now,
            tracked_through=tt,
            block_days=cfg.block_days,
            slots_per_day=cfg.slots_per_day,
            idle_minutes=cfg.idle_minutes,
        )
        if not rep.fire:
            return Outcome(camera, "idle", rep.reason)
        block_key = f"{rep.days[0]}_{rep.days[-1]}"
        try:
            failure = block_failure(camera, block_key, now)
        except StateUnreadable:
            return Outcome(camera, "failed", "state marker unreadable")
        if failure:
            if failure.quarantined:
                return Outcome(camera, "skipped", f"quarantined after {failure.count} failures")
            if now < failure.next_retry:
                return Outcome(camera, "skipped", f"backoff until {failure.next_retry.isoformat()}")
        if not check_stable(
            camera, block_key, rep.otdet_paths, now=now, stable_minutes=cfg.stable_minutes
        ):
            return Outcome(camera, "stabilizing", f"block {block_key} not stable yet")
        wanted = set(rep.days)
        fres = flatten_fn(camera, date_filter=lambda d: d in wanted, log=log)
        if not getattr(fres, "ok", True):
            record_failure(
                camera,
                block_key,
                "flatten conflict",
                now=now,
                max_failures=cfg.max_failures,
            )
            return Outcome(camera, "failed", "flatten conflict")
        flat = [camera / p.name for p in rep.otdet_paths]
        track_result = track_fn(flat, log=log)
        if track_result == "no_slot":
            return Outcome(camera, "skipped", "no host-wide track slot")
        if not track_result:
            record_failure(
                camera,
                block_key,
                "track failed",
                now=now,
                max_failures=cfg.max_failures,
            )
            return Outcome(camera, "failed", "track failed; retry next poll")
        missing = verify_outputs(flat)
        if missing:
            log(f"{camera.name}: {len(missing)} .ottrk missing/invalid; not marking")
            record_failure(
                camera,
                block_key,
                f"{len(missing)} .ottrk incomplete",
                now=now,
                max_failures=cfg.max_failures,
            )
            return Outcome(camera, "failed", f"{len(missing)} .ottrk incomplete")
        clear_failure(camera, block_key)
        set_tracked_through(
            camera,
            rep.tracked_through_after,
            days=len(rep.days),
            files=len(flat),
            at=now.isoformat(),
        )
        return Outcome(camera, "tracked", f"through {rep.tracked_through_after}")


def _run_track(
    paths: list[Path], cfg: WatchConfig, camera: Path, log: Callable[[str], None]
) -> bool | str:
    slots = track_slot_budget(cfg.max_parallel, cfg.reserve_cores, cfg.cores_per_track)
    with acquire_track_slot(slots) as slot:
        if slot is None:
            log(f"{camera.name}: no host-wide track slot available")
            return "no_slot"
        return _run_track_in_slot(paths, cfg, camera, log)


def _run_track_in_slot(
    paths: list[Path], cfg: WatchConfig, camera: Path, log: Callable[[str], None]
) -> bool:
    stamp = now_utc().strftime("%Y%m%d-%H%M%S")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.log_dir / f"{camera.name}_{stamp}.otvision.log"
    console = cfg.log_dir / f"{camera.name}_{stamp}.console.log"
    cmd = [
        str(PYTHON),
        str(TRACK_SCRIPT),
        "-p",
        *[str(p) for p in paths],
        "-c",
        str(cfg.config),
        "--tracker",
        "botsort",
        "--overwrite",
        "--logfile",
        str(logfile),
        "--logfile-overwrite",
    ]
    with console.open("w") as fh:
        fh.write(f"# track {len(paths)} files\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _install_signal_handlers()
        _register_child(proc)
        try:
            rc = proc.wait()
        finally:
            _unregister_child(proc)
        if rc == 0:
            return True
        log(f"{camera.name}: track exit {rc} (see {console.name})")
        return False


def _register_child(proc) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.add(proc)


def _unregister_child(proc) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.discard(proc)


def _terminate_active_children(signum: int) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN)
    for proc in children:
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass


def _handle_shutdown(signum, frame):
    _terminate_active_children(signum)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with _ACTIVE_CHILDREN_LOCK:
            if not _ACTIVE_CHILDREN:
                break
        time.sleep(0.05)
    _terminate_active_children(signal.SIGKILL)
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        _PREVIOUS_SIGNAL_HANDLERS[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_shutdown)
    _SIGNAL_HANDLERS_INSTALLED = True


def track_slot_budget(max_parallel: int, reserve: int, cores_per_track: int = 4) -> int:
    cores = max(1, (os.cpu_count() or 4) - reserve)
    return max(1, min(max_parallel, cores // max(1, cores_per_track)))


def _slot_root() -> Path:
    import track_continuous as tc

    return tc.LOCK_DIR / "slots"


@contextmanager
def acquire_track_slot(budget: int):
    root = _slot_root()
    root.mkdir(parents=True, exist_ok=True)
    handle = None
    slot = None
    try:
        for i in range(max(1, budget)):
            candidate = (root / f"slot-{i}.lock").open("w")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = candidate
                slot = i
                break
            except OSError:
                candidate.close()
        yield slot
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def discover_cameras(root: Path) -> list[Path]:
    found = []
    for pattern in CAMERA_GLOBS:
        found.extend(d for d in root.glob(pattern) if d.is_dir())
    seen, out = set(), []
    for c in sorted(found):
        if c.resolve() not in seen:
            seen.add(c.resolve())
            out.append(c)
    return out


def safe_parallelism(requested: int, reserve: int = 2) -> int:
    return max(1, min(requested, (os.cpu_count() or 4) - reserve))


def _overloaded(reserve: int) -> bool:
    try:
        return os.getloadavg()[0] > (os.cpu_count() or 4) - reserve
    except OSError:
        return False


def poll_once(
    root: Path, cfg: WatchConfig, now: datetime, log: Callable[[str], None] = print
) -> list[Outcome]:
    _install_signal_handlers()
    outcomes = []
    futures = {}
    workers = safe_parallelism(cfg.max_parallel, cfg.reserve_cores)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for cam in discover_cameras(root):
            if _overloaded(cfg.reserve_cores):
                log(f"{cam.name}: deferring (load > cpu-{cfg.reserve_cores})")
                outcomes.append(Outcome(cam, "skipped", "system load high"))
                continue
            futures[pool.submit(process_camera, cam, now=now, cfg=cfg, log=log)] = cam
        for fut in as_completed(futures):
            cam = futures[fut]
            try:
                outcomes.append(fut.result())
            except Exception as e:
                log(f"{cam.name}: ERROR {e}")
                outcomes.append(Outcome(cam, "failed", str(e)))
    finally:
        pool.shutdown(wait=True)
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
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--cores-per-track", type=int, default=4)
    p.add_argument("--max-failures", type=int, default=3)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track"))
    a = p.parse_args(argv)
    root = Path(a.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
        return 2
    error = _validate_args(a)
    if error:
        print(f"[fatal] {error}", file=sys.stderr)
        return 2
    cfg = WatchConfig(
        config=Path(a.config),
        log_dir=Path(a.log_dir),
        block_days=a.block_days,
        idle_minutes=a.idle_minutes,
        stable_minutes=a.stable_minutes,
        slots_per_day=a.slots_per_day,
        reserve_cores=a.reserve_cores,
        max_parallel=a.max_parallel,
        cores_per_track=a.cores_per_track,
        max_failures=a.max_failures,
    )

    def run():
        now = now_utc()
        outs = poll_once(root, cfg, now)
        by = {}
        for o in outs:
            by[o.status] = by.get(o.status, 0) + 1
        print(
            f"[{now.isoformat()}] {len(outs)} cameras | "
            + ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
        )

    if a.once:
        run()
        return 0
    while True:
        started = time.monotonic()
        run()
        time.sleep(max(0, max(60, a.interval) - (time.monotonic() - started)))


def _validate_args(a) -> str | None:
    if a.block_days < 1:
        return "--block-days must be >= 1"
    if a.slots_per_day < 1:
        return "--slots-per-day must be >= 1"
    if 24 * 60 % a.slots_per_day != 0:
        return "--slots-per-day must divide 1440"
    if a.idle_minutes < 0:
        return "--idle-minutes must be >= 0"
    if a.stable_minutes < 0:
        return "--stable-minutes must be >= 0"
    if a.reserve_cores < 0:
        return "--reserve-cores must be >= 0"
    if a.max_parallel < 1:
        return "--max-parallel must be >= 1"
    if a.cores_per_track < 1:
        return "--cores-per-track must be >= 1"
    if a.max_failures < 1:
        return "--max-failures must be >= 1"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
