#!/usr/bin/env python3
"""Continuous BoT-SORT tracking orchestrator for OTVision.

Runs ONE ``track.py`` process per camera directory. Each process tracks every
``*.otdet`` found under that directory (recursively) as a SINGLE continuous run,
so BoT-SORT track IDs persist across consecutive videos of the same camera.

With ``--flatten`` it first flattens each camera (atomic move of date-foldered
files up into the camera root, via flatten_camera) and only then tracks it -- one
reliable, idempotent command from raw date folders to flat .ottrk. A camera is
NOT tracked if its flatten reports a conflict, and a duplicate-.otdet guard
refuses to track a half-flattened folder (which would double-count frames).

Why per-camera, not per-file: continuity is per-camera. ``track.py`` groups
consecutive videos of the same camera (gap <= 1 min) into one FrameGroup with
persistent IDs; IDs reset at gaps > 1 min and NEVER cross cameras (different
hostname = new group). So one process per camera keeps full within-camera
continuity, and running several cameras at once is free parallelism.

Examples
--------
    # ONE workflow: flatten (move) then continuously track the OTCamera07 run:
    python track_continuous.py --flatten "/Volumes/.../videos/OTCamera07"

    # Track an already-flat camera (no flatten):
    python track_continuous.py "/Volumes/.../videos/OTCamera07"

    # Many sites at once, flatten+track each, in parallel (shell glob):
    python track_continuous.py --flatten "/Volumes/.../videos"/OTCamera*

    # Auto-discover camera subdirs under a project videos root:
    python track_continuous.py --flatten --discover-under "/Volumes/.../videos"

Background launch (fire-and-forget, survives logout):
    mkdir -p logs_track
    nohup .venv/bin/python track_continuous.py --flatten \\
        "/Volumes/platomo data/Projekte/OTC015_Team-Red/videos/OTCamera07" \\
        > logs_track/pipeline.log 2>&1 &
    # then watch:  tail -f logs_track/OTCamera07_*.otvision.log
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flatten_camera import flatten_camera

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
TRACK_SCRIPT = SCRIPT_DIR / "track.py"
DEFAULT_CONFIG = SCRIPT_DIR / "config.continuous.botsort.yaml"
CAMERA_GLOBS = ("OTCamera*", "otcamera*")
LOCK_DIR = SCRIPT_DIR / ".locks"


@dataclass
class Result:
    camera: Path
    status: str  # "ok" | "FAILED" | "ERROR" | "skipped"
    n_otdet: int
    detail: str
    seconds: float
    console_log: Path | None


def discover_cameras(paths: list[str], discover_under: str | None) -> list[Path]:
    """Resolve camera directories from explicit paths and/or a parent to scan."""
    cameras: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            cameras.append(path)
        else:
            print(f"[skip] not a directory: {path}", file=sys.stderr)
    if discover_under:
        parent = Path(discover_under)
        for pattern in CAMERA_GLOBS:
            cameras.extend(sorted(d for d in parent.glob(pattern) if d.is_dir()))
    # de-duplicate by resolved path, preserve first-seen order
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in cameras:
        rc = c.resolve()
        if rc not in seen:
            seen.add(rc)
            unique.append(c)
    return unique


def count_otdet(camera_dir: Path) -> int:
    """Count real .otdet files, ignoring macOS AppleDouble sidecars (._*.otdet)."""
    return sum(1 for f in camera_dir.rglob("*.otdet") if not f.name.startswith("._"))


def duplicate_otdet(camera_dir: Path) -> list[str]:
    """.otdet basenames seen more than once (sign of an incomplete flatten)."""
    from collections import Counter

    names = [f.name for f in camera_dir.rglob("*.otdet") if not f.name.startswith("._")]
    return sorted(n for n, c in Counter(names).items() if c > 1)


@contextmanager
def camera_lock(camera_dir: Path):
    """Host-local advisory lock: never let two runs process one camera at once.

    flock on a local lockfile keyed by the camera's resolved path. Released
    automatically when the process exits (even on crash), so there are no stale
    locks to clean up. Yields True if acquired, False if another run holds it.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(camera_dir.resolve()).encode()).hexdigest()[:12]
    handle = (LOCK_DIR / f"{camera_dir.name}-{key}.lock").open("w")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def track_camera(
    camera_dir: Path,
    config: Path,
    log_dir: Path,
    overwrite: bool,
    stamp: str,
    flatten: bool = False,
    clean_appledouble: bool = True,
) -> Result:
    base = log_dir / f"{camera_dir.name}_{stamp}"
    console_log = base.with_suffix(".console.log")  # this script's per-camera log
    otvision_log = base.with_suffix(".otvision.log")  # track.py DEBUG trace
    start = time.monotonic()

    with console_log.open("w") as fh:

        def w(msg: str) -> None:
            fh.write(msg + "\n")
            fh.flush()

        # --- per-camera lock: refuse to run a camera another run already holds ---
        with camera_lock(camera_dir) as acquired:
            if not acquired:
                w("[lock] another run already holds this camera; skipping.")
                return Result(camera_dir, "skipped", count_otdet(camera_dir),
                              "locked by another run",
                              time.monotonic() - start, console_log)

            # --- stage 1: flatten by atomic move (optional) ---
            if flatten:
                fres = flatten_camera(camera_dir, clean_appledouble=clean_appledouble, log=w)
                if not fres.ok:
                    return Result(camera_dir, "FAILED", count_otdet(camera_dir),
                                  "flatten conflict; not tracked",
                                  time.monotonic() - start, console_log)

            # --- guard: a half-flattened folder would double-count frames ---
            dups = duplicate_otdet(camera_dir)
            if dups:
                w(f"[fatal] duplicate .otdet basenames (incomplete flatten): {dups[:5]}")
                return Result(camera_dir, "FAILED", count_otdet(camera_dir),
                              "duplicate .otdet; not tracked",
                              time.monotonic() - start, console_log)

            n = count_otdet(camera_dir)
            if n == 0:
                return Result(camera_dir, "skipped", 0, "no .otdet files",
                              time.monotonic() - start, console_log)

            # --- stage 2: continuous BoT-SORT track ---
            cmd = [
                str(PYTHON), str(TRACK_SCRIPT),
                "-p", str(camera_dir),
                "-c", str(config),
                "--tracker", "botsort",
                "--overwrite" if overwrite else "--no-overwrite",
                "--logfile", str(otvision_log),
                "--logfile-overwrite",
            ]
            w("# " + " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
                status, detail = "ok", ""
            except subprocess.CalledProcessError as e:
                status, detail = "FAILED", f"track exit code {e.returncode}"
            except Exception as e:  # noqa: BLE001
                status, detail = "ERROR", str(e)
    return Result(camera_dir, status, n, detail, time.monotonic() - start, console_log)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuous BoT-SORT tracking, one run per camera directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "cameras",
        nargs="*",
        help="Camera directories (each tracked as one continuous run).",
    )
    p.add_argument(
        "--discover-under",
        metavar="PARENT",
        help="Add every OTCamera*/otcamera* subdirectory of PARENT.",
    )
    p.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Tracker config YAML (default: {DEFAULT_CONFIG.name}).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Max cameras tracked concurrently (default: 8).",
    )
    p.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "logs_track"),
        help="Directory for per-camera logs (default: ./logs_track).",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip cameras whose .ottrk already exist (default: overwrite).",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten each camera (atomic move of date-foldered files up to the "
             "camera root) before tracking it.",
    )
    p.add_argument(
        "--keep-appledouble",
        action="store_true",
        help="When flattening, keep macOS ._ junk files instead of deleting them.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run (incl. flatten plan), then exit.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not PYTHON.exists():
        print(f"[fatal] venv python not found: {PYTHON}", file=sys.stderr)
        return 2
    if not TRACK_SCRIPT.exists():
        print(f"[fatal] track.py not found: {TRACK_SCRIPT}", file=sys.stderr)
        return 2
    config = Path(args.config)
    if not config.exists():
        print(f"[fatal] config not found: {config}", file=sys.stderr)
        return 2

    cameras = discover_cameras(args.cameras, args.discover_under)
    if not cameras:
        print("[fatal] no camera directories given/found.", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    overwrite = not args.no_overwrite

    print(f"Continuous BoT-SORT tracking | config={config.name} | stamp={stamp}")
    print(f"Cameras ({len(cameras)}), max {args.max_parallel} in parallel, "
          f"flatten={args.flatten}, overwrite={overwrite}:")
    for c in cameras:
        print(f"  - {c}  ({count_otdet(c)} .otdet)")
    print(f"Logs: {log_dir}")

    if args.dry_run:
        if args.flatten:
            print("\n[dry-run] flatten plan per camera:")
            for c in cameras:
                flatten_camera(c, clean_appledouble=not args.keep_appledouble,
                               dry_run=True, log=lambda m: print("   " + m))
        print("\n[dry-run] exiting without flattening or tracking.")
        return 0

    results: list[Result] = []
    workers = max(1, min(args.max_parallel, len(cameras)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(track_camera, c, config, log_dir, overwrite, stamp,
                        args.flatten, not args.keep_appledouble): c
            for c in cameras
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            mins = r.seconds / 60.0
            print(f"[{r.status:>7}] {r.camera.name}: {r.n_otdet} files, "
                  f"{mins:.1f} min {('- ' + r.detail) if r.detail else ''}")

    print("\n=== Summary ===")
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status in ("FAILED", "ERROR")]
    skipped = [r for r in results if r.status == "skipped"]
    for r in sorted(results, key=lambda x: x.camera.name):
        line = f"  {r.status:>7}  {r.camera.name}  ({r.n_otdet} .otdet)"
        if r.detail:
            line += f"  [{r.detail}]"
        if r.console_log:
            line += f"  -> {r.console_log.name}"
        print(line)
    print(f"\n{len(ok)} ok, {len(bad)} failed, {len(skipped)} skipped.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
