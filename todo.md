# TODO: BOXMOT Tracking Integration Tasks

## Summary

Features needed for the BOXMOT tracking integration:

1. **Add CLI arguments for tracker selection** - Allow `--tracker bytetrack` instead of editing YAML
2. **Fix .ottrk metadata** - Currently hardcoded to "IOU" even when BOXMOT is used
3. **Expose tracker-specific parameters** - Allow configuring FPS, track_buffer, thresholds per tracker

---

## Task 1: Add CLI Arguments

Add these CLI arguments to `track.py`:

| Argument | Description | Example |
|----------|-------------|---------|
| `--tracker` | Tracker type (iou, bytetrack, botsort, etc.) | `--tracker bytetrack` |
| `--tracker-device` | Compute device | `--tracker-device cuda:0` |
| `--tracker-half-precision` | Enable FP16 | `--tracker-half-precision` |
| `--tracker-reid-weights` | Path to ReID model | `--tracker-reid-weights weights/osnet.pt` |

**Behavior:**
- `--tracker iou` → Uses IOU tracker
- `--tracker bytetrack` → Auto-enables BOXMOT with ByteTrack

**Files to modify:**
- `OTVision/domain/cli.py` - Add dataclass fields
- `OTVision/track/cli.py` - Add argument parser entries
- `OTVision/application/track/update_track_config_with_cli_args.py` - Merge CLI args

---

## Task 2: Fix Tracker Metadata in .ottrk Files

**Problem:** `OTVision/application/track/ottrk.py` line 197 has:
```python
dataformat.NAME: "IOU",  # HARDCODED!
```

**Solution:** Pass actual tracker info through the builder chain.

**Expected metadata output:**

IOU tracker:
```json
{"name": "IOU", "sigma_l": 0.27, "sigma_h": 0.42, ...}
```

BOXMOT tracker:
```json
{"name": "bytetrack", "device": "cuda:0", "half_precision": true, "reid_weights": null}
```

**Files to modify:**
- `OTVision/application/track/ottrk.py` - Add tracker fields to `OttrkBuilderConfig`, update `create_tracker_metadata()`
- `OTVision/track/ottrk_builder.py` - Pass tracker config
- `OTVision/application/track/stream_ottrk_file_writer.py` - Get tracker info from config
- `OTVision/dataformat.py` - Add constants (TRACKER_DEVICE, TRACKER_HALF_PRECISION, TRACKER_REID_WEIGHTS)

---

## Task 3: Update Tests

- Add CLI argument tests
- Test metadata contains correct tracker info

---

## Task 4: Update Documentation

- `BOXMOT_INTEGRATION.md` - Add CLI section, remove "Future Enhancement" note
- `CLAUDE.md` - Update CLI examples

---

## Usage Examples (After Implementation)

```bash
# ByteTrack on CPU
uv run track.py -p detections.otdet --tracker bytetrack

# BotSORT on GPU with ReID
uv run track.py -p detections.otdet --tracker botsort --tracker-device cuda:0 --tracker-reid-weights weights/osnet.pt

# IOU tracker
uv run track.py -p detections.otdet --tracker iou

# Override config file
uv run track.py -c config.yaml -p detections.otdet --tracker ocsort
```

---

## Task 5: Support Appearance-Based Trackers in File-Based Tracking ✓ COMPLETED

### Implementation Summary

Video frame provider feature implemented to support appearance-based trackers (BotSort, StrongSORT, DeepOcSORT, BoostTrack, HybridSORT) in file-based tracking.

**Status**: All 130 track tests pass, mypy and flake8 pass.

### Files Created/Modified

| File | Change |
|------|--------|
| `OTVision/track/video_frame_provider.py` | **NEW** - `VideoFrameProvider` protocol, `PyAvVideoFrameProvider`, `SequentialVideoFrameProvider`, `resolve_video_path_from_otdet()` |
| `OTVision/track/parser/chunk_parser_plugins.py` | Added `video_frame_provider_factory` parameter to `JsonChunkParser` |
| `OTVision/track/builder.py` | Auto-detects appearance trackers, wires up `SequentialVideoFrameProvider` |
| `tests/track/test_video_frame_provider.py` | **NEW** - 13 unit tests |
| `tests/track/parser/test_chunk_parser_plugins.py` | 3 new tests for frame loading |

### How It Works

1. `TrackBuilder.chunk_parser` detects if BOXMOT is enabled with appearance-based tracker
2. If yes, creates `JsonChunkParser` with `video_frame_provider_factory`
3. `resolve_video_path_from_otdet()` reconstructs video path from OTDET metadata
4. `SequentialVideoFrameProvider` efficiently loads frames sequentially during tracking
5. Frames are attached to `DetectedFrame.image` before passing to tracker

### Usage

```yaml
# Appearance-based tracker with ReID (requires video file)
BOXMOT:
  ENABLED: true
  TRACKER_TYPE: "botsort"
  DEVICE: "cuda:0"
  REID_WEIGHTS: "weights/osnet_x0_25_msmt17.pt"
```

Video file must exist alongside the `.otdet` file for appearance-based trackers.

---

## Task 6: Expose BOXMOT Tracker-Specific Configuration Parameters ✓ COMPLETED

### Implementation Summary

The `tracker_params` feature allows users to configure tracker-specific parameters (FPS, track_buffer, thresholds) via YAML configuration.

**Status**: All tests pass, mypy and flake8 pass.

### Files Created/Modified

| File | Change |
|------|--------|
| `OTVision/application/config.py:379` | Added `tracker_params: dict[str, Any]` field to `_TrackBoxmotConfig` |
| `OTVision/application/config.py:388` | Added `to_dict()` serialization for `TRACKER_PARAMS` |
| `OTVision/application/config_parser.py:285-299` | Parses `TRACKER_PARAMS` from YAML, handles None |
| `OTVision/track/tracker/tracker_plugin_boxmot.py:91,141-143` | Accepts & merges `tracker_params` into BOXMOT kwargs |
| `OTVision/track/builder.py:157,185` | Passes `tracker_params` to adapter |
| `OTVision/track/builder.py:160-173` | Auto-detects `frame_rate` from OTDET metadata |
| `tests/application/test_config_parser.py:233-330` | 6 comprehensive config parser tests |
| `tests/track/tracker/test_boxmot_adapter_simple.py:45-55` | Unit test for adapter with `tracker_params` |
| `BOXMOT_INTEGRATION.md:281-345` | Full `TRACKER_PARAMS` documentation section |
| `boxmot_config.example.yaml:110-117` | Example configuration |

### How It Works

```
user_config.otvision.yaml
    │
    │   TRACKER_PARAMS:
    │     track_buffer: 60
    │     track_thresh: 0.5
    │
    ▼
ConfigParser.parse_track_boxmot_config()
    │
    ▼
_TrackBoxmotConfig.tracker_params
    │
    ▼
TrackBuilder._create_boxmot_tracker_factory()
    │   - Copies tracker_params from config
    │   - Auto-detects frame_rate if not set
    │
    ▼
BoxmotTrackerAdapter.__init__()
    │   - Merges tracker_params into kwargs
    │   - Passes to BOXMOT tracker class
    │
    ▼
BOXMOT tracker initialized with custom params
```

### YAML Configuration Example

```yaml
track:
  boxmot:
    enabled: true
    tracker_type: "bytetrack"
    device: "cuda:0"
    half_precision: false
    reid_weights: null
    tracker_params:
      frame_rate: 20        # Auto-detected from OTDET if not set
      track_buffer: 60      # Keep tracks for 3 seconds at 20 FPS
      track_thresh: 0.5     # Lower threshold for difficult conditions
      match_thresh: 0.85
```

### Available Parameters by Tracker

See `BOXMOT_INTEGRATION.md` for full parameter reference for all 7 trackers.

**Key parameters:**
- `frame_rate`: FPS for motion prediction (auto-detected from OTDET metadata)
- `track_buffer`: Frames to keep lost tracks
- `track_thresh` / `det_thresh`: Detection confidence thresholds
- `match_thresh`: IOU/appearance matching threshold
