# Handoff: BoT-SORT loses each track's first detection

Branch `feature/botsort-integration`, PR #530, OP#9534.
Written after the cleanup pass; the cleanup is done and pushed, **this** is what is left.

## TL;DR

Every BoT-SORT track born after frame 1 of a video group is missing its first
detection in the `.ottrk`. It is not an ultralytics bug and upgrading will not
fix it. A ~10-line subclass fixes it exactly, but must not ship until the
discard path is fixed, or it will trade one missing detection per track for
extra spurious short tracks.

Nothing here is started. Step 1 decides whether steps 2-3 are worth doing.

## Evidence

Pinned `ultralytics==8.3.159`, paths relative to `.venv/lib/python3.12/site-packages/`.

**The mechanism.** A new track is created *unconfirmed* and only emitted after a
second consecutive match:

- `ultralytics/trackers/byte_tracker.py:133` - `activate()` sets
  `is_activated = True` only `if frame_id == 1`.
- `ultralytics/trackers/byte_tracker.py:378` - `# Deal with unconfirmed tracks,
  usually tracks with only one beginning frame`. Re-match uses a **hardcoded**
  `thresh=0.7`, not configurable.
- `ultralytics/trackers/byte_tracker.py:387` - unmatched unconfirmed tracks are
  `mark_removed()`, so latency is exactly one frame: confirmed next frame, or gone.
- `ultralytics/trackers/byte_tracker.py:413` - `update()` returns only
  `x.is_activated` tracks.

This is canonical ByteTrack false-positive suppression, inherited from the
reference implementation. **Upgrading ultralytics will not change it.**

**Reproduction** (object A from frame 1, object B appears frame 3, conf 0.9):

```
frame 1: dets=1 returned_track_ids=[1]
frame 2: dets=1 returned_track_ids=[1]
frame 3: dets=2 returned_track_ids=[1]        <- B present, not returned
frame 4: dets=2 returned_track_ids=[1, 2]     <- B first surfaces here
```

Measured latency vs `STrack.start_frame`: `{1: 0, 2: 1}`. `start_frame` **is**
exposed on `botsort.tracked_stracks`, so the miss is detectable at runtime.

**The fix, already verified to work:**

```python
class EagerBOTrack(BOTrack):
    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.is_activated = True

class EagerBOTSORT(BOTSORT):
    def init_track(self, dets, scores, cls, img=None):
        if len(dets) == 0:
            return []
        return [EagerBOTrack(xyxy, s, c) for xyxy, s, c in zip(dets, scores, cls)]
```

Measured with this in place: latency `{1: 0, 2: 0}` on the same sequence.
`init_track`'s ReID branch is irrelevant here because ReID is rejected for
image-less `.otdet` input anyway.

## The blocker: t_min is computed but never applied to output

Do not ship eager activation before this. ByteTrack's confirmation step is
suppressing false positives; OTVision's `t_min` is supposed to do the same job
retroactively, **but it does not reach the file**.

- `OTVision/track/builder.py:125` - `UnfinishedChunksBuffer(..., keep_discarded=True)`.
- `OTVision/domain/frame.py:142-149` - with `keep_discarded=True`, discarded
  detections are kept and merely flagged `is_discarded`.
- `OTVision/domain/detection.py:139` - `to_dict()` emits `FINISHED: self.is_last`
  and **no discard flag**. `is_discarded` dies in the domain layer.
- `OTVision/track/model/filebased/frame_chunk.py:90` - existing `# TODO remove discarded?`

Observable today on `tests/data/track/default` with `T_MIN: 5`:

```
per-track detection counts: [3, 10, 18, 60, 60, 60, 60, 60, 60, 60]
                             ^ span 2 < t_min, in the .ottrk, unmarked
```

Detection keys in output: `class, confidence, finished, first, frame, h,
input_file_path, interpolated-detection, occurrence, track-id, w, x, y`.
Nothing downstream can filter on discard.

## Plan

**Step 1 - measure, and decide (do this first).**
Run the same footage through both trackers and check whether counts actually
move. One frame at a track's start only changes a count if the object enters
already overlapping a section: plausible for sections near the frame edge,
negligible otherwise. **This has not been measured.** If counts do not move,
the correct outcome is a documented known-difference and steps 2-3 are dropped.

Starting point: `tests/data/track/default/*.otdet`, two files, 20 fps, 60 frames
each. Current BoT-SORT output is 10 and 12 distinct track ids; IOU gives 31.

**Step 2 - fix the discard path.**
Either serialize the discard flag into `.ottrk`, or set `keep_discarded=False`
for the file-based path at `builder.py:125`. Worth doing regardless of step 1;
it is the clearer defect of the two. Check with OTAnalytics which it wants,
since dropping rows changes the file contract.

**Step 3 - add eager activation.**
Only after step 2. Put `EagerBOTSORT` in
`OTVision/track/tracker/tracker_plugin_botsort.py` next to `_ensure_botsort_initialized`,
as an explicit documented deviation. Default it to match IOU-tracker semantics
(IOU emits from frame 1 and filters retroactively, so eager is the consistent
choice). If ultralytics later changes `init_track`'s shape this fails loudly at
construction rather than degrading silently.

Rejected alternative: a one-frame reconciliation buffer using
`STrack.start_frame` to backfill. Feasible, strictly more code, same result.

Rejected alternative: writing our own tracker. BoT-SORT is Kalman + Hungarian +
two-stage association + GMC + ReID; months of work to change one line, and IOU
already covers the simple case.

## Also open, all pre-existing, none introduced by this branch

- **`t_min` bypassed at end-of-group.** `OTVision/track/model/filebased/frame_chunk.py:94-106`
  force-finishes every unfinished track in the last chunk without consulting
  lifecycle state, so a short track at EOF survives.
- **One-frame gap in cross-file numbering.** `.otdet` keys are 1-based and added
  to the offset (`OTVision/track/parser/chunk_parser_plugins.py:53-59`), while the
  next offset is `last.no + 1` (`OTVision/track/tracker/filebased_tracking.py:124`),
  so the next file starts two numbers after the previous last frame. This inflates
  `last_frame - first_frame` and can wrongly satisfy `t_min`.
- **`StreamOttrkFileWriter` has no production wiring** - tests only.
- **black version skew.** `pyproject.toml` pins 25.1.0, `.pre-commit-config.yaml`
  pins 24.8.0; they disagree on `tests/abstraction/test_defaults.py` and this
  blocks commits that touch it.

## Trap for the next session

The ultralytics files in `.venv` were **replaced mid-session** by a
`uv run --extra inference_cpu` install (mtime 16:31), and the new content
differs materially from the old while both report version 8.3.159. This caused
a wrong change (rejecting all ReID) that had to be reverted in `f0d9531`.

**Re-check any claim about ultralytics internals against the installed package
before acting on it**, e.g. `uv run --extra inference_cpu python -c "import
inspect, ultralytics.trackers.bot_sort as m; print(inspect.getsource(m.BOTSORT.__init__))"`.
Do not trust a remembered reading, this document included.

For the record, on the build installed as of `f0d9531`: ReID **is** implemented
(`bot_sort.py:201`), only `MODEL: auto` is unusable for OTVision because it
expects native detector feature tensors and calls `.cpu()` on a NumPy image;
GMC **is** silently skipped when `img is None` (`byte_tracker.py:336`).
