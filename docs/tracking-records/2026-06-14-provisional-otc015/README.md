# Provisional BoT-SORT tracking record — OTC015_Team-Red

**Date:** 2026-06-14
**Project:** `/Volumes/platomo data/Projekte/OTC015_Team-Red/videos`
**Tool:** `provisional_track.py` (commit `52c99b0`) using the `feature/botsort-reid-filemode` tracker via worktree
**Config:** `config.provisional.botsort.yaml` (`WITH_REID: false`)

## What this is

Intentionally **partial, provisional** tracking to unblock OTAnalytics implementers — *not* a finished track. Hard cutoff: `.otdet` with timestamp **≤ 2026-06-03** only (one full day per camera). The cron watcher remains the eventual full re-track.

## Final verified result

**11 target cameras × 96 = 1056/1056 valid `.ottrk`** (independently confirmed on disk as valid bz2 JSON):

`OTCamera05, 09, 10, 12, 15, 16, 17, 19` + `otcamera28, 29, 30` — all 96/96.

| Group | Cameras | Outcome |
|---|---|---|
| Tracked (provisional) | 05, 09, 10, 12, 15, 16, 17, 19, 28, 29, 30 | 96/96 valid `.ottrk` |
| **Excluded — never touched** | **07, 18, 20, 21, 23, 26** | fully tracked already; untouched (verified every pass) |
| Skipped — no `.otdet` ≤ 06-03 | 02, 03, 06, 11 | out of scope; need detection first |

Source files were **never moved** (rsync depends on the date-foldered layout); only `.ottrk` were added beside each `.otdet`.

## Run history (manifests)

| Manifest | Result | Note |
|---|---|---|
| `manifest_20260614-204054.json` | FAILED ×11 | first attempt — `resolve()` dereferenced the venv-python symlink → ran system python → `ModuleNotFoundError: ffmpeg` |
| `manifest_20260614-204114.json` | FAILED ×11 | same bug, before the `_abs()` fix |
| `manifest_20260614-210444.json` | **ok ×9** | successful completion pass (otcamera29/30 already done from the prior run, shown there as `skipped — already tracked`) |

> Note: otcamera29/30 reached 96/96 during an earlier run that was killed before writing a manifest, so no single manifest lists all 11 as `ok`. The **authoritative completion record is the on-disk verification (1056/1056 valid)** plus `manifest_20260614-210444.json` for the other 9.

## Reconstruction

Each `manifest_*.json` lists, per camera, the exact `otdet` and `ottrk` file paths, date range, status, config, and tracker commit — enough to reconstruct precisely what was tracked. `restart_*.log` / `run_*.log` are the orchestration summaries (pass-by-pass `ok`/`skipped` tallies).

Per-camera DEBUG traces (`*.console.log`, `*.otvision.log`, ~7 MB) remain under `logs_track/provisional/` (gitignored) and are not part of this committed record.
