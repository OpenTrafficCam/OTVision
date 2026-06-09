#!/usr/bin/env python3
"""Flatten an OTCamera date-foldered recording dir into one flat folder, by MOVE.

Turns  CAMERA/<date>/<files>  into  CAMERA/<files>  (flat), as required by the
downstream pipeline.

Move, not copy: source and target are the same filesystem, so each file is moved
with an atomic rename -- instant (no data copied, even for GB of mp4) and never
leaves a half-written file. A rename either happened or it didn't, so there is
nothing to "verify".

Super reliable / re-runnable:
  * Idempotent  - re-running flattens only what's still in subfolders; once flat
                  it's a no-op. Safe to repeat as new recordings arrive.
  * Atomic      - per-file os.rename; an interrupted run leaves every file either
                  fully in its date folder or fully in the root, never both.
  * Safe        - skips dotfiles (macOS ._ junk AND live .otdet.XXXX atomic-write
                  temps); never clobbers a differing root file; removes a date
                  folder only once it is genuinely empty.

Filenames embed the full date+time, so flattening across days never collides.

    python flatten_camera.py --dry-run "/Volumes/.../OTCamera07"
    python flatten_camera.py "/Volumes/.../OTCamera07"          # do the move
    python flatten_camera.py "/Volumes/.../OTCamera07" --keep-appledouble
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Callable

DEFAULT_TYPES = (".otdet", ".mp4", ".log")
APPLEDOUBLE_PREFIX = "._"


@dataclass
class FlattenResult:
    camera: Path
    moved: int = 0
    deduped: int = 0          # redundant identical source dropped (already flat)
    cleaned: int = 0          # ._ junk removed
    removed_dirs: int = 0
    conflicts: list[str] = field(default_factory=list)
    kept_dirs: list[str] = field(default_factory=list)
    temp_files: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts


def _subdir_date(name: str) -> _date | None:
    try:
        return _date.fromisoformat(name)
    except ValueError:
        return None


def find_sources(camera: Path, types: tuple[str, ...], date_filter=None):
    """Data files / junk / temps in scoped SUBfolders."""
    sources, appledouble, temp_dotfiles = [], [], []
    for sub in sorted(p for p in camera.iterdir() if p.is_dir()):
        if date_filter is not None:
            d = _subdir_date(sub.name)
            if d is None or not date_filter(d):
                continue
        for f in sub.rglob("*"):
            if not f.is_file():
                continue
            if f.name.startswith(APPLEDOUBLE_PREFIX):
                appledouble.append(f)
            elif f.name.startswith("."):
                temp_dotfiles.append(f)  # e.g. .OTCamera..._10-30-00.otdet.WQoSZM
            elif f.suffix.lower() in types:
                sources.append(f)
    return sorted(sources), sorted(appledouble), sorted(temp_dotfiles)


def flatten_camera(
    camera: Path,
    types: tuple[str, ...] = DEFAULT_TYPES,
    clean_appledouble: bool = True,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    date_filter=None,
) -> FlattenResult:
    """Move every data file from `camera`'s subfolders into `camera` itself."""
    res = FlattenResult(camera=camera)
    sources, appledouble, temp_dotfiles = find_sources(camera, types, date_filter)
    res.temp_files = temp_dotfiles

    exts = ", ".join(sorted({s.suffix for s in sources})) or "none"
    log(f"flatten: {len(sources)} files ({exts}) to move into {camera}")
    log(f"flatten: ._ junk={len(appledouble)} | temp/other dotfiles={len(temp_dotfiles)}")
    if temp_dotfiles:
        log("flatten: [!] temp/partial dotfiles present (upstream may still be writing); "
            "they are left untouched:")
        for f in temp_dotfiles[:10]:
            log(f"           {f.relative_to(camera)}")

    # collision: same basename in two different subfolders
    by_name: dict[str, list[Path]] = {}
    for s in sources:
        by_name.setdefault(s.name, []).append(s)
    for name, group in by_name.items():
        if len(group) > 1:
            res.conflicts.append(
                f"basename in multiple subfolders: {name} -> "
                f"{[str(p.relative_to(camera)) for p in group]}"
            )
    if res.conflicts:
        for c in res.conflicts:
            log(f"flatten: [fatal] {c}")
        return res

    if not sources:
        log("flatten: already flat / nothing to move.")
        return res

    if dry_run:
        total = sum(s.stat().st_size for s in sources)
        log(f"flatten: [dry-run] would move {len(sources)} files "
            f"({total / 1e9:.2f} GB) via atomic rename; no changes made.")
        return res

    # --- move (atomic rename) ---
    for s in sources:
        dst = camera / s.name
        if dst.exists():
            # Drop the source as a duplicate ONLY if the root file is byte-identical.
            # Same size is not enough -- it could silently delete a differing file of
            # equal size (e.g. a re-detection of the same timestamp).
            try:
                same_size = dst.stat().st_size == s.stat().st_size
            except OSError:
                same_size = False
            if same_size and filecmp.cmp(str(s), str(dst), shallow=False):
                s.unlink()          # verified-identical duplicate -> drop source
                res.deduped += 1
            else:
                reason = "different size" if not same_size else "same size, different content"
                res.conflicts.append(
                    f"root file exists ({reason}), left in place: {s.name}"
                )
                log(f"flatten: [warn] {res.conflicts[-1]}")
        else:
            shutil.move(str(s), str(dst))   # os.rename on same fs: atomic, instant
            res.moved += 1

    # --- clean ._ junk + remove emptied date folders ---
    if clean_appledouble:
        for f in appledouble:
            try:
                f.unlink(); res.cleaned += 1
            except OSError:
                pass
    for d in sorted({s.parent.resolve() for s in sources}):
        try:
            remaining = list(d.iterdir())
        except FileNotFoundError:
            continue
        if not remaining:
            d.rmdir(); res.removed_dirs += 1
        else:
            non_junk = [x for x in remaining if not x.name.startswith(APPLEDOUBLE_PREFIX)]
            why = "live temp/other files" if non_junk else "._ junk (use default cleaning)"
            res.kept_dirs.append(f"{d.name}/ ({len(remaining)} item(s): {why})")

    log(f"flatten: moved {res.moved}, deduped {res.deduped}, cleaned {res.cleaned} junk, "
        f"removed {res.removed_dirs} empty folder(s).")
    for k in res.kept_dirs:
        log(f"flatten: [kept] {k}")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("camera", help="Camera directory containing date subfolders.")
    p.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES),
                   help=f"Extensions to flatten (default: {' '.join(DEFAULT_TYPES)}).")
    p.add_argument("--keep-appledouble", action="store_true",
                   help="Do not delete macOS ._ junk files.")
    p.add_argument("--dry-run", action="store_true", help="Show plan; change nothing.")
    args = p.parse_args(argv)

    camera = Path(args.camera)
    if not camera.is_dir():
        print(f"[fatal] not a directory: {camera}", file=sys.stderr)
        return 2
    types = tuple(t if t.startswith(".") else f".{t}" for t in (s.lower() for s in args.types))

    res = flatten_camera(
        camera, types=types, clean_appledouble=not args.keep_appledouble,
        dry_run=args.dry_run,
    )
    if not res.ok:
        print(f"[fatal] flatten incomplete for {camera} ({len(res.conflicts)} conflict(s)).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
