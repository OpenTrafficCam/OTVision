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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
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
# Matched case-insensitively (dirs are mixed-case: OTCamera07 vs otcamera21).
# 07/18/20 + 21/23/26 are the operator's "fully tracked, never touch" set; the
# 21/23/26 dirs also carry a tracked_through marker, so this is defense-in-depth.
EXCLUDE_MANDATORY = frozenset(
    (
        "OTCamera07",
        "OTCamera18",
        "OTCamera20",
        "OTCamera21",
        "OTCamera23",
        "OTCamera26",
    )
)
CUTOFF_DEFAULT = date(2026, 6, 3)
VIDEO_EXT = ".mp4"


def _safe_parse(name: str) -> tuple[str, datetime] | None:
    """parse_otc_filename, but never raises: a regex-match with an invalid date
    (e.g. month 13) raises ValueError upstream; treat it as unparseable -> skip."""
    try:
        return parse_otc_filename(name)
    except ValueError:
        return None


def in_scope_otdet(camera: Path, cutoff: date) -> list[Path]:
    """Same-host .otdet whose embedded timestamp date is <= cutoff, time-sorted."""
    host = camera.name.lower()
    found: list[tuple[datetime, Path]] = []
    for f in camera.rglob("*.otdet"):
        if f.name.startswith("._"):
            continue
        parsed = _safe_parse(f.name)
        if not parsed:
            continue
        fhost, dt = parsed
        if fhost.lower() != host:
            continue
        if dt.date() <= cutoff:
            found.append((dt, f))
    return [f for _, f in sorted(found)]


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
    # <repo>/.locks/slots. Defaults mirror the watcher cron (--max-parallel 12
    # --cores-per-track 2 --reserve-cores 2) so both parties size the same pool
    # and the flock cap bounds total track.py across BOTH.
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
        parsed = _safe_parse(stem)
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

    Existence alone is not enough: a half-written / truncated / empty file must
    NOT satisfy the skip-or-success condition (it would fail to decode). This
    does NOT distinguish tracker type or schema -- any well-formed bz2 JSON
    passes; for this provisional pass targets start with zero .ottrk, so that is
    sufficient.
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


def decide(
    camera: Path, cfg: ProvConfig, tracked_through: date | None
) -> tuple[str, str]:
    """Return ("track"|"skip", reason). Pure given tracked_through."""
    if camera.name.lower() in {e.lower() for e in cfg.exclude}:
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
    return (
        "track",
        f"{total} .otdet <= {cfg.cutoff} ({done} already have .ottrk; full re-track)",
    )


def build_track_cmd(
    otdet_paths: list[Path], cfg: ProvConfig, logfile: Path
) -> list[str]:
    return [
        str(cfg.venv_python),
        str(cfg.worktree / "track.py"),
        "-p",
        *[str(p) for p in otdet_paths],
        "-c",
        str(cfg.config),
        "--tracker",
        "botsort",
        "--overwrite",  # camera-level idempotency handled by decide(); re-track in full
        "--logfile",
        str(logfile),
        "--logfile-overwrite",
    ]


def build_detect_cmd(
    video_paths: list[Path], cfg: ProvConfig, logfile: Path
) -> list[str]:
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
    main repo, sys.path[0]='' already contains the main OTVision/ and wins.)
    PYTHONPATH is kept as a redundant safeguard. All paths in cmd are absolute, so
    cwd does not affect -p/-c/--logfile resolution.
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


def _entry(camera: Path, decision: str, reason: str, **extra: object) -> dict:
    e = {
        "camera": camera.name,
        "decision": decision,
        "reason": reason,
        "status": "skipped",
        "n_otdet": 0,
        "otdet": [],
        "ottrk": [],
        "date_range": None,
        "detail": "",
    }
    e.update(extra)
    return e


def _default_slot(cfg: ProvConfig) -> AbstractContextManager[object]:
    """Acquire a slot from the watcher's SHARED host pool (<repo>/.locks/slots)."""
    budget = track_slot_budget(
        cfg.host_max_parallel, cfg.reserve_cores, cfg.cores_per_track
    )
    return acquire_track_slot(budget)


def process_camera(
    camera: Path,
    cfg: ProvConfig,
    *,
    now: datetime,
    run_fn: Callable[[list[str], ProvConfig, Path], int] = run_subprocess,
    slot_factory: Callable[[], AbstractContextManager[object]] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    slot_factory = slot_factory or (lambda: _default_slot(cfg))
    with camera_lock(camera) as got:
        if not got:
            return _entry(camera, "skip", "locked by another run (watcher?)")
        try:
            # date.fromisoformat on a malformed marker raises ValueError, not
            # StateUnreadable -- catch both so a bad marker fails CLOSED.
            tt = get_tracked_through(camera)
        except (StateUnreadable, ValueError):
            return _entry(camera, "skip", "state marker unreadable", status="ERROR")
        action, reason = decide(camera, cfg, tt)
        if action == "skip":
            log(f"{camera.name}: skip - {reason}")
            return _entry(camera, "skip", reason)

        stamp = now.strftime("%Y%m%d-%H%M%S")
        console = cfg.log_dir / f"{camera.name}_{stamp}.console.log"

        # optional detection of in-scope videos missing their .otdet. A separate
        # slot per heavy step (not one held across both) so a long detect never
        # starves the watcher of the shared host-wide pool.
        if cfg.detect:
            scope_otdet_names = {p.name for p in in_scope_otdet(camera, cfg.cutoff)}
            missing = [
                v
                for v in in_scope_video(camera, cfg.cutoff)
                if (v.with_suffix(".otdet").name) not in scope_otdet_names
            ]
            if missing:
                with slot_factory() as slot:
                    if slot is None:
                        return _entry(
                            camera,
                            "track",
                            reason,
                            status="skipped",
                            detail="no host-wide track slot (detect)",
                        )
                    log(
                        f"{camera.name}: detecting {len(missing)} "
                        f"video(s) <= {cfg.cutoff}"
                    )
                    dlog = cfg.log_dir / f"{camera.name}_{stamp}.detect.otvision.log"
                    if run_fn(build_detect_cmd(missing, cfg, dlog), cfg, console) != 0:
                        return _entry(
                            camera,
                            "track",
                            reason,
                            status="FAILED",
                            detail="detect failed",
                        )

        otdet = in_scope_otdet(camera, cfg.cutoff)
        if not otdet:
            return _entry(
                camera,
                "track",
                reason,
                status="FAILED",
                detail="no .otdet after detect step",
            )
        first = _safe_parse(otdet[0].name)
        last = _safe_parse(otdet[-1].name)
        assert first is not None and last is not None  # in_scope_otdet parsed them
        dr = [first[1].date().isoformat(), last[1].date().isoformat()]

        with slot_factory() as slot:
            if slot is None:
                log(f"{camera.name}: no host-wide track slot; retry next run")
                return _entry(
                    camera,
                    "track",
                    reason,
                    status="skipped",
                    detail="no host-wide track slot",
                )
            tlog = cfg.log_dir / f"{camera.name}_{stamp}.otvision.log"
            rc = run_fn(build_track_cmd(otdet, cfg, tlog), cfg, console)
            done, total = ottrk_done(otdet)
            ottrk = [
                str(p.with_suffix(".ottrk"))
                for p in otdet
                if _ottrk_ok(p.with_suffix(".ottrk"))
            ]
            if rc != 0:
                status, detail = "FAILED", f"track exit {rc}"
            elif done != total:
                # exit 0 but not every .ottrk is valid -> NOT a success
                status, detail = (
                    "FAILED",
                    f"{total - done}/{total} .ottrk missing/invalid",
                )
            else:
                status, detail = "ok", ""
            return _entry(
                camera,
                "track",
                reason,
                status=status,
                n_otdet=total,
                otdet=[str(p) for p in otdet],
                ottrk=ottrk,
                date_range=dr,
                console_log=str(console),
                detail=detail,
            )


def write_manifest(
    cfg: ProvConfig, entries: list[dict], meta: dict
) -> tuple[Path, Path]:
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

    lines = [
        f"# Provisional tracking manifest {meta.get('stamp')}",
        "",
        f"- cutoff: {run['cutoff']} (inclusive)",
        f"- config: {run['config']} (WITH_REID=false)",
        f"- branch/commit: {run['branch']} @ {run['commit']}",
        f"- root: {run['root']}",
        "",
        "| camera | decision | status | n_otdet | date_range | reason |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        rng = "-".join(e["date_range"]) if e.get("date_range") else ""
        lines.append(
            f"| {e['camera']} | {e['decision']} | {e['status']} | "
            f"{e['n_otdet']} | {rng} | {e['reason']} |"
        )
    md_path = cfg.manifest_dir / f"manifest_{meta.get('stamp')}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _git_commit(worktree: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
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
            f"[fatal] {config}: TRACK.BOT_SORT.WITH_REID must be false, "
            f"got {with_reid!r}"
        )


@contextmanager
def launcher_lock() -> Iterator[None]:
    """Single-instance guard: refuse to run two provisional launchers at once."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handle = (LOCK_DIR / "provisional_launcher.lock").open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(
                "[fatal] another provisional_track.py run holds the launcher lock",
                file=sys.stderr,
            )
            raise SystemExit(2)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument(
        "--cutoff",
        default=CUTOFF_DEFAULT.isoformat(),
        help="Inclusive end date YYYY-MM-DD (default 2026-06-03).",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="EXTRA camera names to never touch (comma-separated). "
        "OTCamera07/18/20 are ALWAYS excluded regardless of this.",
    )
    p.add_argument(
        "--worktree",
        required=True,
        help="Path to the feature/botsort-reid-filemode git worktree.",
    )
    p.add_argument(
        "--venv-python", default=str(SCRIPT_DIR / ".venv" / "bin" / "python")
    )
    p.add_argument(
        "--config", default=str(SCRIPT_DIR / "config.provisional.botsort.yaml")
    )
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "logs_track" / "provisional"))
    p.add_argument(
        "--manifest-dir", default=str(SCRIPT_DIR / "logs_track" / "provisional")
    )
    p.add_argument(
        "--detect",
        action="store_true",
        help="Detect in-scope videos that lack .otdet, then track.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Cameras WE attempt concurrently (each still needs a host slot).",
    )
    p.add_argument(
        "--host-max-parallel",
        type=int,
        default=12,
        help="Host-wide track-slot pool size; MUST match the watcher cron (12).",
    )
    p.add_argument(
        "--cores-per-track",
        type=int,
        default=2,
        help="Must match the watcher cron (2) so the shared pool sizes alike.",
    )
    p.add_argument("--reserve-cores", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    root = Path(a.root)
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
        return 2
    # Resolve to absolute: the track/detect subprocess runs with cwd=worktree,
    # so any relative -c/--logfile/path would resolve against the wrong tree.
    cfg = ProvConfig(
        root=root.resolve(),
        cutoff=date.fromisoformat(a.cutoff),
        exclude=set(EXCLUDE_MANDATORY) | set(s for s in a.exclude.split(",") if s),
        worktree=Path(a.worktree).resolve(),
        venv_python=Path(a.venv_python).resolve(),
        config=Path(a.config).resolve(),
        log_dir=Path(a.log_dir).resolve(),
        manifest_dir=Path(a.manifest_dir).resolve(),
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
            except (StateUnreadable, ValueError):
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
    meta = {
        "stamp": now.strftime("%Y%m%d-%H%M%S"),
        "started": now.isoformat(),
        "commit": _git_commit(cfg.worktree),
    }
    entries: list[dict] = []
    workers = max(1, min(cfg.max_parallel, len(cameras) or 1))
    with launcher_lock():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(process_camera, cam, cfg, now=now): cam for cam in cameras
            }
            for fut in as_completed(futs):
                cam = futs[fut]
                try:
                    entries.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    entries.append(_entry(cam, "skip", f"ERROR {e}", status="ERROR"))
    entries.sort(key=lambda ent: ent["camera"])
    json_path, md_path = write_manifest(cfg, entries, meta)
    by: dict[str, int] = {}
    for ent in entries:
        by[ent["status"]] = by.get(ent["status"], 0) + 1
    print(
        f"[{meta['started']}] {len(entries)} cameras | "
        + ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
    )
    print(f"manifest: {json_path}")
    failed = sum(1 for e in entries if e["status"] in ("FAILED", "ERROR"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
