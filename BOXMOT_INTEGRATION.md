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
