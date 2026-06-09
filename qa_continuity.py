#!/usr/bin/env python3
"""QA the continuity of .otdet files BEFORE continuous tracking.

Replicates track.py's frame-grouping decision exactly (same hostname AND
0 <= next.start - prev.end <= 1 min  ->  one continuous group) using OTVision's
own metadata extractors. Reports:

  * how many continuous FrameGroups the .otdet will actually form,
  * every gap where track IDs WILL reset (with size + location),
  * missing 15-minute slots and duplicate/overlapping files,
  * per-group span and the overall coverage.

Use it to confirm a camera's files are one unbroken stream (and thus track as a
single continuous run) before launching track_continuous.py.

    python qa_continuity.py "/Volumes/.../videos/OTCamera07"
    python qa_continuity.py --slot-minutes 15 "/Volumes/.../OTCamera07" "/Volumes/.../OTCamera18"
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

from OTVision.detect.otdet import (
    extract_expected_duration_from_otdet,
    extract_hostname_from_otdet,
    extract_start_date_from_otdet,
)
from OTVision.helpers.files import read_json_bz2_metadata

GAP_THRESHOLD = timedelta(minutes=1)  # track.py's TimeThresholdFrameGroupParser


def scan_camera(camera_dir: Path, slot_minutes: int) -> bool:
    """Return True if the camera forms a single continuous group, else False."""
    files = sorted(
        f for f in camera_dir.rglob("*.otdet") if not f.name.startswith("._")
    )
    print(f"\n{'=' * 72}\nCAMERA: {camera_dir}")
    if not files:
        print("  no .otdet files found.")
        return False

    recs = []
    errors = []
    for f in files:
        try:
            md = read_json_bz2_metadata(f)
            start = extract_start_date_from_otdet(md)
            dur = extract_expected_duration_from_otdet(md)
            host = extract_hostname_from_otdet(md)
            recs.append((start, start + dur, host, f, dur))
        except Exception as e:  # noqa: BLE001
            errors.append((f, str(e)))

    if errors:
        print(f"  [!] {len(errors)} file(s) failed to read metadata:")
        for f, e in errors[:10]:
            print(f"      {f.name}: {e}")

    if not recs:
        return False

    recs.sort(key=lambda r: r[0])
    n = len(recs)
    days = sorted({r[0].date() for r in recs})
    print(f"  files: {n} | hosts: {sorted({r[2] for r in recs})} | "
          f"days: {days[0]} .. {days[-1]} ({len(days)})")
    print(f"  span: {recs[0][0]}  ->  {recs[-1][1]}")

    # --- replicate merge(): walk sorted files, break group on host/gap ---
    groups = []  # list of [first_idx, last_idx]
    gaps = []    # (prev_rec, next_rec, delta, reason)
    cur = [0, 0]
    for i in range(1, n):
        p_start, p_end, p_host, p_file, _ = recs[i - 1]
        c_start, c_end, c_host, c_file, _ = recs[i]
        delta = c_start - p_end
        if p_host != c_host:
            groups.append(cur); cur = [i, i]
            gaps.append((recs[i - 1], recs[i], delta, "hostname change"))
        elif timedelta(0) <= delta <= GAP_THRESHOLD:
            cur[1] = i  # merge
        else:
            groups.append(cur); cur = [i, i]
            reason = "overlap/negative" if delta < timedelta(0) else f"gap {delta}"
            gaps.append((recs[i - 1], recs[i], delta, reason))
    groups.append(cur)

    # --- missing 15-min slots (by expected cadence) ---
    slot = timedelta(minutes=slot_minutes)
    missing = []
    for i in range(1, n):
        expected = recs[i - 1][0] + slot
        actual = recs[i][0]
        # only flag clean multiples short by a slot (don't double-count big gaps)
        if actual - recs[i - 1][0] > slot + timedelta(seconds=30):
            k = actual - recs[i - 1][0]
            n_missing = int(k / slot) - 1
            if 0 < n_missing <= 200:
                for m in range(1, n_missing + 1):
                    missing.append(recs[i - 1][0] + m * slot)

    # --- report ---
    print(f"\n  >> {len(groups)} continuous group(s)  "
          f"(=> track IDs reset {len(groups) - 1} time(s))")
    for gi, (a, b) in enumerate(groups, 1):
        s, e = recs[a][0], recs[b][1]
        print(f"     group {gi}: {b - a + 1:>4} files | {s} -> {e} | {e - s}")

    if gaps:
        print(f"\n  ID-reset boundaries ({len(gaps)}):")
        for prev, nxt, delta, reason in gaps[:50]:
            print(f"     {prev[3].name}  ->  {nxt[3].name}   [{reason}]")
    else:
        print("\n  no breaks: every consecutive file is within the 1-min threshold.")

    if missing:
        print(f"\n  [!] {len(missing)} missing {slot_minutes}-min slot(s) (sample):")
        for ts in missing[:20]:
            print(f"      {ts}")

    one_group = len(groups) == 1 and not errors
    verdict = "ONE continuous stream" if one_group else "MULTIPLE groups / issues"
    print(f"\n  VERDICT: {verdict}")
    return one_group


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cameras", nargs="+", help="Camera directories to QA.")
    p.add_argument("--slot-minutes", type=int, default=15,
                   help="Expected recording cadence in minutes (default: 15).")
    args = p.parse_args(argv)

    all_one = True
    for c in args.cameras:
        cam = Path(c)
        if not cam.is_dir():
            print(f"[skip] not a directory: {cam}", file=sys.stderr)
            all_one = False
            continue
        all_one &= scan_camera(cam, args.slot_minutes)

    print(f"\n{'=' * 72}\nOVERALL: "
          f"{'all cameras are single continuous streams' if all_one else 'review the groups/gaps above'}")
    return 0 if all_one else 1


if __name__ == "__main__":
    raise SystemExit(main())
