"""Poll camera dirs and track complete, stable blocks."""
from __future__ import annotations

import argparse
import bz2
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from flatten_camera import flatten_camera
from otc_coverage import assess_camera
from otc_state import check_stable, get_tracked_through, set_tracked_through
from track_continuous import camera_lock

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
    max_parallel: int = 1


@dataclass
class Outcome:
    camera: Path
    status: str
    detail: str = ""


def _bz2_ok(path: Path) -> bool:
    try:
        with bz2.open(path, "rb") as fh:
            fh.read(64)
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
        tt = get_tracked_through(camera)
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
        if not check_stable(
            camera, block_key, rep.otdet_paths, now=now, stable_minutes=cfg.stable_minutes
        ):
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
        try:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e:
            log(f"{camera.name}: track exit {e.returncode} (see {console.name})")
            return False


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
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track"))
    a = p.parse_args(argv)
    root = Path(a.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
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
        run()
        time.sleep(max(60, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())
