# BOXMOT Integration for OTVision

## Overview

This document describes the integration of BOXMOT (state-of-the-art multi-object tracking) into OTVision's tracking pipeline. BOXMOT provides 7 advanced tracking algorithms that can significantly improve tracking performance compared to the default IOU tracker.

## Installation

### Quick Install

```bash
# Linux/macOS
./scripts/install_boxmot.sh

# Windows
scripts\install_boxmot.cmd
```

### Manual Install

To use BOXMOT trackers, install the optional dependency group:

```bash
uv pip install -e .[tracking_boxmot]
```

This installs:
- `boxmot` (>=11.0) - Core tracking algorithms
- `filterpy` - Kalman filtering
- `lapx` - Hungarian algorithm for assignment
- `scikit-learn` - Machine learning utilities

### Verify Installation

```bash
# Test BOXMOT import
python -c "from boxmot import TRACKERS; print(TRACKERS)"

# Test OTVision adapter
python -c "from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter"
```

### ReID Weights (Optional)

Appearance-based trackers (BotSORT, BoostTrack, etc.) require ReID weights for re-identification.

**Automatic download:**
```bash
# Linux/macOS
./scripts/install_boxmot.sh --with-reid

# Windows
scripts\install_boxmot.cmd --with-reid
```

**Manual download:**
```bash
# Create weights directory
mkdir -p weights

# Download OSNet ReID model
wget -P weights/ https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt
```

## Available Trackers

### Motion-Only Trackers (Fast, CPU-Efficient)
- **ByteTrack**: High-FPS tracker using only motion cues (~1265 FPS on MOT17)
- **OcSORT**: Improved motion-only tracking (~1483 FPS on MOT17)

### Appearance + Motion Trackers (Better Accuracy)
- **BotSORT**: Best overall performance (HOTA: 69.418)
- **BoostTrack**: High identity consistency (IDF1: 83.205)
- **StrongSORT**: Balanced performance
- **DeepOcSORT**: Enhanced OcSORT with ReID
- **HybridSORT**: Hybrid motion-appearance approach

## Configuration

### YAML Configuration

Add BOXMOT configuration to your `user_config.otvision.yaml`:

```yaml
TRACK:
  BOXMOT:
    ENABLED: true                    # Enable BOXMOT tracking
    TRACKER_TYPE: "bytetrack"        # Tracker algorithm
    DEVICE: "cpu"                    # Device: "cpu", "cuda:0", etc.
    HALF_PRECISION: false            # Use FP16 (for GPU)
    REID_WEIGHTS: null               # Path to ReID weights (for appearance trackers)
```

### Configuration Reference

| YAML Key | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLED` | bool | `false` | Enable/disable BOXMOT tracking. When `false`, falls back to IOU tracker. |
| `TRACKER_TYPE` | str | `"bytetrack"` | Tracking algorithm to use (see Available Trackers) |
| `DEVICE` | str | `"cpu"` | Compute device: `"cpu"`, `"cuda:0"`, `"cuda:1"`, etc. |
| `HALF_PRECISION` | bool | `false` | Use FP16 half-precision for faster GPU inference |
| `REID_WEIGHTS` | str/null | `null` | Path to ReID model weights (required for appearance trackers) |

### Tracker Types

| Tracker | Type | ReID Required | Speed | Best For |
|---------|------|---------------|-------|----------|
| `bytetrack` | Motion-only | No | Very Fast | CPU processing, real-time |
| `ocsort` | Motion-only | No | Very Fast | Alternative to ByteTrack |
| `botsort` | Appearance | Yes | Medium | Best accuracy |
| `boosttrack` | Appearance | Yes | Medium | Identity consistency |
| `strongsort` | Appearance | Yes | Slow | Balanced performance |
| `deepocsort` | Appearance | Yes | Medium | OcSORT + ReID |
| `hybridsort` | Appearance | Yes | Medium | Hybrid approach |

## Tracker Comparison & Selection Guide

### Performance Benchmarks (MOT17)

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | FPS | Type |
|---------|-------|-------|-------|-----|------|
| BotSort | 69.418 | 78.232 | 81.812 | 46 | Appearance |
| BoostTrack | 69.254 | 75.921 | 83.205 | 25 | Appearance |
| StrongSort | 68.05 | 76.185 | 80.763 | 17 | Appearance |
| DeepOcSort | 67.796 | 75.868 | 80.514 | 12 | Appearance |
| ByteTrack | 67.68 | 78.039 | 79.157 | 1265 | Motion |
| HybridSort | 67.39 | 74.127 | 79.105 | 25 | Appearance |
| OcSort | 66.441 | 74.548 | 77.899 | 1483 | Motion |

### Star Ratings

| Tracker | Speed | Accuracy | ID Consistency | Ease of Use |
|---------|-------|----------|----------------|-------------|
| ByteTrack | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| OcSort | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| BotSort | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| BoostTrack | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| StrongSort | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| DeepOcSort | ★☆☆☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| HybridSort | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |

**Rating Legend:**
- **Speed**: Processing throughput (★★★★★ = >1000 FPS, ★☆☆☆☆ = <20 FPS)
- **Accuracy**: HOTA/MOTA metrics (★★★★★ = HOTA >69, ★★★☆☆ = HOTA <68)
- **ID Consistency**: IDF1 score (★★★★★ = IDF1 >83, ★★★☆☆ = IDF1 <80)
- **Ease of Use**: Configuration complexity (★★★★★ = no ReID weights needed)

### Tracker Approaches

**Motion-Only Trackers:**

- **ByteTrack**: Two-stage association - first matches high-confidence detections, then recovers low-confidence ones. Pure motion-based, extremely fast (~1265 FPS). Best for real-time applications.

- **OcSort**: Observation-Centric SORT with virtual trajectories and momentum-based association. Better occlusion handling than ByteTrack (~1483 FPS). Good when objects frequently leave and re-enter frame.

**Appearance-Based Trackers:**

- **BotSort**: Combines ByteTrack with ReID features and camera motion compensation (CMC). Best overall accuracy (HOTA 69.4). Recommended when accuracy is priority.

- **BoostTrack**: Boosted association with multiple cost terms (IoU, Mahalanobis, shape). Highest identity consistency (IDF1 83.2). Best for maintaining consistent track IDs across occlusions.

- **StrongSort**: Deep association with appearance features, EMA updates, and NSA Kalman filter. Balanced performance. Good general-purpose appearance tracker.

- **DeepOcSort**: OcSort enhanced with ReID embeddings and adaptive weighting. Slowest but combines motion and appearance effectively.

- **HybridSort**: Fuses short-term motion with long-term appearance gallery. Good for re-identification after long occlusions.

### Which Tracker Should I Use?

| Use Case | Recommended | Alternative |
|----------|-------------|-------------|
| CPU-only, real-time | ByteTrack | OcSort |
| GPU available, best accuracy | BotSort | BoostTrack |
| Long-term tracking, ID consistency | BoostTrack | StrongSort |
| Simple scenes, low complexity | ByteTrack | OcSort |
| Crowded scenes, many occlusions | BotSort | BoostTrack |
| Re-identification after long gaps | HybridSort | BoostTrack |
| Balanced performance | StrongSort | BotSort |

## Tracker Parameters Reference

Each tracker accepts specific parameters that can be configured via `TRACKER_PARAMS` in your YAML configuration. If `frame_rate` is not explicitly set, it is auto-detected from OTDET metadata.

### Motion-Only Trackers

#### ByteTrack Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_conf` | float | 0.1 | Minimum confidence threshold; detections below are discarded |
| `track_thresh` | float | 0.45 | High-confidence threshold for first association |
| `match_thresh` | float | 0.8 | IoU matching threshold for first association |
| `track_buffer` | int | 25 | Base buffer size for track persistence (scaled by frame_rate) |
| `frame_rate` | int | 30 | Video frame rate for buffer scaling |

#### OcSort Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_conf` | float | 0.1 | Minimum confidence threshold |
| `delta_t` | int | 3 | Time delta for velocity estimation |
| `inertia` | float | 0.2 | Inertia weight for motion model smoothing |
| `use_byte` | bool | False | Use ByteTrack-style two-stage association |
| `Q_xy_scaling` | float | 0.01 | Process noise scaling for position in Kalman filter |
| `Q_s_scaling` | float | 0.0001 | Process noise scaling for scale in Kalman filter |

### Appearance-Based Trackers

#### BotSort Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_high_thresh` | float | 0.5 | High confidence detection threshold for first association |
| `track_low_thresh` | float | 0.1 | Low confidence detection threshold for second association |
| `new_track_thresh` | float | 0.6 | Confidence threshold for initializing new tracks |
| `track_buffer` | int | 30 | Buffer frames before removing lost tracks |
| `match_thresh` | float | 0.8 | IoU matching threshold |
| `proximity_thresh` | float | 0.5 | Spatial proximity threshold for appearance gating |
| `appearance_thresh` | float | 0.25 | Appearance similarity threshold (rejects poor matches) |
| `cmc_method` | str | "ecc" | Camera motion compensation method ("ecc", "orb", "sift") |
| `frame_rate` | int | 30 | Video frame rate |
| `fuse_first_associate` | bool | False | Fuse appearance features in first association stage |
| `with_reid` | bool | True | Enable ReID appearance features |

#### StrongSort Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_conf` | float | 0.1 | Minimum confidence threshold |
| `max_cos_dist` | float | 0.2 | Maximum cosine distance for ReID matching |
| `max_iou_dist` | float | 0.7 | Maximum IoU distance for matching |
| `n_init` | int | 3 | Frames required before track is confirmed |
| `nn_budget` | int | 100 | Maximum samples in appearance gallery per track |
| `mc_lambda` | float | 0.98 | Motion cost weight in combined cost |
| `ema_alpha` | float | 0.9 | EMA alpha for appearance feature update |

#### DeepOcSort Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta_t` | int | 3 | Time delta for velocity estimation |
| `inertia` | float | 0.2 | Inertia weight for motion model |
| `w_association_emb` | float | 0.5 | Weight for embedding-based association cost |
| `alpha_fixed_emb` | float | 0.95 | Fixed alpha for embedding EMA update |
| `aw_param` | float | 0.5 | Adaptive weighting parameter |
| `embedding_off` | bool | False | Disable embedding features (motion-only mode) |
| `cmc_off` | bool | False | Disable camera motion compensation |
| `aw_off` | bool | False | Disable adaptive weighting |
| `Q_xy_scaling` | float | 0.01 | Process noise scaling for position |
| `Q_s_scaling` | float | 0.0001 | Process noise scaling for scale |

#### BoostTrack Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ecc` | bool | True | Enable ECC for camera motion compensation |
| `min_box_area` | int | 10 | Minimum bounding box area to track |
| `aspect_ratio_thresh` | float | 1.6 | Maximum aspect ratio change threshold |
| `cmc_method` | str | "ecc" | Camera motion compensation method |
| `lambda_iou` | float | 0.5 | Weight for IoU cost in combined cost |
| `lambda_mhd` | float | 0.25 | Weight for Mahalanobis distance cost |
| `lambda_shape` | float | 0.25 | Weight for shape similarity cost |
| `use_dlo_boost` | bool | True | Enable Detection-Level Observation boosting |
| `use_duo_boost` | bool | True | Enable Detection-Update-Observation boosting |
| `dlo_boost_coef` | float | 0.65 | DLO boost coefficient |

#### HybridSort Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmc_method` | str | "ecc" | Camera motion compensation method |
| `with_reid` | bool | True | Enable ReID appearance features |
| `low_thresh` | float | 0.1 | Low confidence threshold for second association |
| `delta_t` | int | 3 | Time delta for velocity estimation |
| `inertia` | float | 0.05 | Inertia weight for motion model |
| `use_byte` | bool | True | Use ByteTrack-style two-stage association |
| `use_custom_kf` | bool | True | Use custom Kalman filter implementation |
| `longterm_bank_length` | int | 30 | Long-term appearance gallery size |
| `alpha` | float | 0.9 | Smoothing factor for appearance update |

## TRACKER_PARAMS Configuration

The `TRACKER_PARAMS` field allows you to pass tracker-specific parameters directly to the BOXMOT tracker. These override the tracker's default values.

### Basic Usage

```yaml
TRACK:
  BOXMOT:
    ENABLED: true
    TRACKER_TYPE: "bytetrack"
    DEVICE: "cpu"
    TRACKER_PARAMS:
      track_thresh: 0.5
      match_thresh: 0.85
      track_buffer: 60
```

### Frame Rate Auto-Detection

If `frame_rate` is not specified in `TRACKER_PARAMS`, it is automatically detected from the OTDET file metadata. This ensures trackers use the correct frame rate for buffer scaling and motion estimation.

```yaml
# frame_rate will be auto-detected from OTDET metadata
TRACKER_PARAMS:
  track_buffer: 30  # Will be scaled by auto-detected frame_rate
```

### Tracker-Specific Examples

**ByteTrack with Higher Confidence Thresholds:**
```yaml
TRACKER_PARAMS:
  track_thresh: 0.6        # Stricter first-pass threshold
  match_thresh: 0.9        # Higher IoU requirement
  track_buffer: 50         # Longer track persistence
```

**BotSORT with Custom Appearance Settings:**
```yaml
TRACKER_PARAMS:
  track_high_thresh: 0.6
  proximity_thresh: 0.4    # Stricter spatial gating
  appearance_thresh: 0.3   # Stricter appearance matching
  cmc_method: "orb"        # Faster CMC than ECC
```

**BoostTrack with Adjusted Cost Weights:**
```yaml
TRACKER_PARAMS:
  lambda_iou: 0.6          # Higher IoU weight
  lambda_mhd: 0.2          # Lower Mahalanobis weight
  lambda_shape: 0.2        # Lower shape weight
  dlo_boost_coef: 0.7      # Stronger DLO boosting
```

**StrongSort for Dense Scenes:**
```yaml
TRACKER_PARAMS:
  max_cos_dist: 0.15       # Stricter ReID matching
  max_iou_dist: 0.6        # Stricter IoU matching
  nn_budget: 150           # Larger appearance gallery
  n_init: 5                # More frames to confirm track
```

## Usage Examples

### Example 1: Fast CPU Tracking (ByteTrack)

Recommended for CPU-only systems or when speed is priority.

**Configuration:**
```yaml
TRACK:
  BOXMOT:
    ENABLED: true
    TRACKER_TYPE: "bytetrack"
    DEVICE: "cpu"
    HALF_PRECISION: false
    REID_WEIGHTS: null
```

**Run tracking:**
```bash
uv run track.py --paths /path/to/detections/*.otdet
```

### Example 2: GPU Tracking with FP16 (ByteTrack)

Faster processing on CUDA-capable GPUs.

**Configuration:**
```yaml
TRACK:
  BOXMOT:
    ENABLED: true
    TRACKER_TYPE: "bytetrack"
    DEVICE: "cuda:0"
    HALF_PRECISION: true
    REID_WEIGHTS: null
```

### Example 3: High-Accuracy GPU Tracking (BotSORT)

Best accuracy with appearance-based tracking.

**Configuration:**
```yaml
TRACK:
  BOXMOT:
    ENABLED: true
    TRACKER_TYPE: "botsort"
    DEVICE: "cuda:0"
    HALF_PRECISION: true
    REID_WEIGHTS: "weights/osnet_x0_25_msmt17.pt"
```

### Example 4: Best Identity Consistency (BoostTrack)

For scenarios requiring consistent track IDs across occlusions.

**Configuration:**
```yaml
TRACK:
  BOXMOT:
    ENABLED: true
    TRACKER_TYPE: "boosttrack"
    DEVICE: "cuda:0"
    HALF_PRECISION: true
    REID_WEIGHTS: "weights/osnet_x0_25_msmt17.pt"
```

### Example 5: Disable BOXMOT (Use IOU Tracker)

Fall back to the default IOU tracker.

**Configuration:**
```yaml
TRACK:
  BOXMOT:
    ENABLED: false
```

Or simply omit the `BOXMOT` section entirely.

### CLI with Config File

```bash
# Use custom config file
uv run track.py --paths /path/to/*.otdet --config my_config.otvision.yaml

# Process specific detection files
uv run track.py --paths output/video1.otdet output/video2.otdet
```

## Architecture

### Components

1. **BoxmotTrackerAdapter** (`OTVision/track/tracker/tracker_plugin_boxmot.py`)
   - Implements OTVision's `Tracker` interface
   - Bridges BOXMOT API with OTVision's data structures
   - Manages track lifecycle (finished/discarded tracks)

2. **Utility Functions** (`OTVision/track/boxmot_utils.py`)
   - Coordinate conversion (xywh center <-> xyxy corners)
   - Detection format conversion (OTVision <-> BOXMOT)
   - Class label mapping (string <-> numeric IDs)

3. **Configuration** (`OTVision/application/config.py`)
   - `_TrackBoxmotConfig` dataclass for BOXMOT settings
   - Integrated into `TrackConfig`

4. **Builder Integration** (`OTVision/track/builder.py`)
   - Conditional tracker selection (IOU vs BOXMOT)
   - Configuration-driven instantiation

### Data Flow

```
DetectedFrame (OTVision)
    |
Detections -> BOXMOT array format (xywh -> xyxy, labels -> IDs)
    |
BOXMOT Tracker.update()
    |
BOXMOT tracks -> TrackedDetections (xyxy -> xywh, IDs -> labels)
    |
TrackedFrame (OTVision)
```

## Key Differences: IOU vs BOXMOT

| Aspect | IOU Tracker | BOXMOT Trackers |
|--------|-------------|-----------------|
| Algorithm | Simple intersection-over-union | SOTA algorithms (Kalman, ReID, etc.) |
| Performance | Fast, lightweight | Varies by tracker (ByteTrack fastest) |
| Accuracy | Basic | Higher (especially with ReID) |
| Configuration | Thresholds (SIGMA_L, SIGMA_H, etc.) | Tracker-specific parameters |
| Dependencies | None | boxmot, filterpy, lapx, scikit-learn |
| GPU Support | No | Yes (optional) |

## Performance Considerations

### Motion-Only vs Appearance-Based

**Motion-Only (ByteTrack, OcSORT):**
- Very fast (>1000 FPS)
- No image data required
- Works well for simple scenarios
- Lower accuracy in crowded scenes

**Appearance-Based (BotSORT, BoostTrack, etc.):**
- Higher accuracy and identity consistency
- Better handling of occlusions
- Slower (12-46 FPS)
- Requires frame images
- Requires ReID model weights

### Recommendations

| Scenario | Recommended Tracker | Device |
|----------|---------------------|--------|
| CPU-only, fast processing | ByteTrack | cpu |
| GPU available, best accuracy | BotSORT | cuda:0 |
| Balanced GPU performance | ByteTrack | cuda:0 |
| High identity consistency | BoostTrack | cuda:0 |
| Real-time processing | ByteTrack | cpu |

## Troubleshooting

### Import Error: "BOXMOT is not installed"

Install the optional dependency group:
```bash
uv pip install -e .[tracking_boxmot]
```

### ValueError: "Unknown tracker type"

Check `TRACKER_TYPE` spelling. Valid options:
- bytetrack, ocsort, botsort, boosttrack, strongsort, deepocsort, hybridsort

### ValueError: "Tracker requires frame images but frame.image is None"

Appearance-based trackers (BotSORT, etc.) require frame images. Either:
1. Switch to a motion-only tracker (ByteTrack, OcSORT)
2. Ensure detection pipeline includes frame images

### ValueError: "Tracker requires reid_weights"

Appearance-based trackers require ReID weights:

```
ValueError: Tracker type 'botsort' is an appearance-based tracker and requires
reid_weights to be specified.
```

**Solution**: Specify REID_WEIGHTS in configuration:
```yaml
TRACK:
  BOXMOT:
    TRACKER_TYPE: "botsort"
    REID_WEIGHTS: "weights/osnet_x0_25_msmt17.pt"
```

Or switch to a motion-only tracker:
```yaml
TRACK:
  BOXMOT:
    TRACKER_TYPE: "bytetrack"  # No REID_WEIGHTS needed
```

### FileNotFoundError: ReID weights not found

Download ReID weights:
```bash
# Using installation script
./scripts/install_boxmot.sh --with-reid

# Or manually
wget -P weights/ https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt
```

Then update config:
```yaml
TRACK:
  BOXMOT:
    REID_WEIGHTS: "weights/osnet_x0_25_msmt17.pt"
```

### RuntimeError: "Failed to initialize BOXMOT tracker"

This error wraps underlying BOXMOT initialization failures:

```
RuntimeError: Failed to initialize BOXMOT tracker 'bytetrack' on device 'cuda:0':
CUDA not available
```

**Common causes:**
1. **CUDA not available**: Switch to CPU or install CUDA drivers
2. **Invalid device string**: Check device format (e.g., "cuda:0", not "cuda0")
3. **Missing ReID model file**: Verify REID_WEIGHTS path exists

## Testing

The BOXMOT integration includes comprehensive test coverage:

### Unit Tests (Utility Functions)
```bash
# Test coordinate conversions and data format transformations
uv run pytest tests/track/test_boxmot_utils.py -v
```

### Integration Tests (Tracker Adapter)
```bash
# Test BoxmotTrackerAdapter with mocked BOXMOT trackers
uv run pytest tests/track/tracker/test_boxmot_adapter_simple.py -v
```

### Performance Benchmark
```bash
# Compare IOU vs ByteTrack performance
uv run scripts/benchmark_boxmot.py
```

## Known Limitations

### Production Considerations

1. **ReID Weights Validation**: Appearance-based trackers require ReID weights. The system validates this at initialization:
   ```
   ValueError: Tracker type 'botsort' is an appearance-based tracker and requires
   reid_weights to be specified. Motion-only trackers: ['bytetrack', 'ocsort']
   ```

2. **Tracker Reset Between Videos**: When tracking multiple video files, the tracker automatically resets between files to prevent track ID overlap.

3. **Memory Management**: The adapter automatically cleans up finished track IDs to prevent unbounded memory growth.

4. **Device Validation**: Invalid CUDA devices fail at initialization with clear error messages.

### Logging

When a tracker is selected, logs are generated:

```
INFO:OTVision.track.builder:Using BOXMOT tracker: bytetrack on device: cpu
INFO:OTVision.track.tracker.tracker_plugin_boxmot:Initialized BOXMOT tracker: bytetrack on device: cpu
```

Or for IOU tracker:
```
INFO:OTVision.track.builder:Using IOU tracker
```

## References

- **BOXMOT Repository**: https://github.com/mikel-brostrom/boxmot
- **BOXMOT PyPI**: https://pypi.org/project/boxmot/
- **Example Config**: `boxmot_config.example.yaml` (in project root)
- **Installation Scripts**: `scripts/install_boxmot.sh`, `scripts/install_boxmot.cmd`

## Future Enhancements

Potential improvements for future iterations:

1. **CLI Arguments**: Add command-line arguments for BOXMOT configuration
2. **Performance Metrics**: Log tracking performance statistics
3. **Auto-Selection**: Automatically choose tracker based on hardware
4. **Custom ReID Models**: Support for custom ReID architectures
5. **Tracker Ensembles**: Combine multiple trackers for robustness
