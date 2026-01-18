# BOXMOT Integration for OTVision

## Overview

This document describes the integration of BOXMOT (state-of-the-art multi-object tracking) into OTVision's tracking pipeline. BOXMOT provides 7 advanced tracking algorithms that can significantly improve tracking performance compared to the default IOU tracker.

## Installation

To use BOXMOT trackers, install the optional dependency group:

```bash
uv pip install -e .[tracking_boxmot]
```

This installs:
- `boxmot` (>=11.0) - Core tracking algorithms
- `filterpy` - Kalman filtering
- `lapx` - Hungarian algorithm for assignment
- `scikit-learn` - Machine learning utilities

## Available Trackers

### Motion-Only Trackers (Fast, CPU-Efficient)
- **ByteTrack**: High-FPS tracker using only motion cues (1265 FPS on MOT17)
- **OcSORT**: Improved motion-only tracking (1483 FPS on MOT17)

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
track:
  boxmot:
    enabled: true  # Enable BOXMOT tracking
    tracker_type: "bytetrack"  # Tracker algorithm
    device: "cpu"  # Device: "cpu", "cuda:0", etc.
    half_precision: false  # Use FP16 (for GPU)
    reid_weights: null  # Path to ReID weights (for appearance trackers)
```

### Configuration Options

#### `enabled` (bool, default: `false`)
Enable/disable BOXMOT tracking. When `false`, falls back to IOU tracker.

#### `tracker_type` (str, default: `"bytetrack"`)
Tracking algorithm to use. Options:
- `"bytetrack"` - Fast motion-only tracker (recommended for CPU)
- `"ocsort"` - Alternative fast motion-only tracker
- `"botsort"` - Best overall accuracy (requires ReID weights)
- `"boosttrack"` - High identity consistency (requires ReID weights)
- `"strongsort"` - Balanced performance (requires ReID weights)
- `"deepocsort"` - Enhanced OcSORT (requires ReID weights)
- `"hybridsort"` - Hybrid approach (requires ReID weights)

#### `device` (str, default: `"cpu"`)
Compute device for tracking:
- `"cpu"` - CPU-only processing
- `"cuda:0"` - GPU device 0
- `"cuda:1"` - GPU device 1
- etc.

#### `half_precision` (bool, default: `false`)
Use FP16 half-precision for faster GPU inference. Only effective on GPU.

#### `reid_weights` (str or null, default: `null`)
Path to ReID (re-identification) model weights for appearance-based trackers.
Required for: BotSORT, BoostTrack, StrongSORT, DeepOcSORT, HybridSORT.
Not used for: ByteTrack, OcSORT.

Example ReID weights:
- `"osnet_x0_25_msmt17.pt"` - Lightweight ReID model
- `"osnet_x1_0_msmt17.pt"` - Larger, more accurate ReID model

## Usage Examples

### Example 1: Fast CPU Tracking (ByteTrack)

```yaml
track:
  boxmot:
    enabled: true
    tracker_type: "bytetrack"
    device: "cpu"
```

```bash
uv run track.py --paths /path/to/detections/*.otdet
```

### Example 2: High-Accuracy GPU Tracking (BotSORT)

```yaml
track:
  boxmot:
    enabled: true
    tracker_type: "botsort"
    device: "cuda:0"
    half_precision: true
    reid_weights: "osnet_x0_25_msmt17.pt"
```

### Example 3: Disable BOXMOT (Use IOU Tracker)

```yaml
track:
  boxmot:
    enabled: false
```

Or simply omit the `boxmot` section entirely.

## Architecture

### Components

1. **BoxmotTrackerAdapter** (`OTVision/track/tracker/tracker_plugin_boxmot.py`)
   - Implements OTVision's `Tracker` interface
   - Bridges BOXMOT API with OTVision's data structures
   - Manages track lifecycle (finished/discarded tracks)

2. **Utility Functions** (`OTVision/track/boxmot_utils.py`)
   - Coordinate conversion (xywh center ↔ xyxy corners)
   - Detection format conversion (OTVision ↔ BOXMOT)
   - Class label mapping (string ↔ numeric IDs)

3. **Configuration** (`OTVision/application/config.py`)
   - `_TrackBoxmotConfig` dataclass for BOXMOT settings
   - Integrated into `TrackConfig`

4. **Builder Integration** (`OTVision/track/builder.py`)
   - Conditional tracker selection (IOU vs BOXMOT)
   - Configuration-driven instantiation

### Data Flow

```
DetectedFrame (OTVision)
    ↓
Detections → BOXMOT array format (xywh → xyxy, labels → IDs)
    ↓
BOXMOT Tracker.update()
    ↓
BOXMOT tracks → TrackedDetections (xyxy → xywh, IDs → labels)
    ↓
TrackedFrame (OTVision)
```

## Key Differences: IOU vs BOXMOT

| Aspect | IOU Tracker | BOXMOT Trackers |
|--------|-------------|-----------------|
| Algorithm | Simple intersection-over-union | SOTA algorithms (Kalman, ReID, etc.) |
| Performance | Fast, lightweight | Varies by tracker (ByteTrack fastest) |
| Accuracy | Basic | Higher (especially with ReID) |
| Configuration | Thresholds (sigma_l, sigma_h, etc.) | Tracker-specific parameters |
| Dependencies | None | boxmot, filterpy, lapx, scikit-learn |
| GPU Support | No | Yes (optional) |

## Performance Considerations

### Motion-Only vs Appearance-Based

**Motion-Only (ByteTrack, OcSORT):**
- ✅ Very fast (>1000 FPS)
- ✅ No image data required
- ✅ Works well for simple scenarios
- ❌ Lower accuracy in crowded scenes

**Appearance-Based (BotSORT, BoostTrack, etc.):**
- ✅ Higher accuracy and identity consistency
- ✅ Better handling of occlusions
- ❌ Slower (12-46 FPS)
- ❌ Requires frame images
- ❌ Requires ReID model weights

### Recommendations

- **CPU-only, fast processing**: Use ByteTrack
- **GPU available, best accuracy**: Use BotSORT with ReID weights
- **Balanced performance**: Use ByteTrack with GPU acceleration
- **High identity consistency**: Use BoostTrack

## Troubleshooting

### Import Error: "BOXMOT is not installed"

Install the optional dependency group:
```bash
uv pip install -e .[tracking_boxmot]
```

### ValueError: "Unknown tracker type"

Check `tracker_type` spelling. Valid options:
- bytetrack, ocsort, botsort, boosttrack, strongsort, deepocsort, hybridsort

### ValueError: "Tracker requires frame images but frame.image is None"

Appearance-based trackers (BotSORT, etc.) require frame images. Either:
1. Switch to a motion-only tracker (ByteTrack, OcSORT)
2. Ensure detection pipeline includes frame images

### ValueError: "Tracker requires reid_weights"

**New in enhanced version**: Appearance-based trackers now validate ReID weights at initialization.

```
ValueError: Tracker type 'botsort' is an appearance-based tracker and requires
reid_weights to be specified.
```

**Solution**: Specify reid_weights in configuration:
```yaml
track:
  boxmot:
    tracker_type: "botsort"
    reid_weights: "/path/to/osnet_x0_25_msmt17.pt"
```

Or switch to a motion-only tracker:
```yaml
track:
  boxmot:
    tracker_type: "bytetrack"  # No reid_weights needed
```

### FileNotFoundError: ReID weights not found

Download ReID weights or provide correct path:
```bash
# Example: Download OSNet weights
wget https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt
```

Then update config:
```yaml
track:
  boxmot:
    reid_weights: "/path/to/osnet_x0_25_msmt17.pt"
```

### RuntimeError: "Failed to initialize BOXMOT tracker"

This error wraps underlying BOXMOT initialization failures (e.g., CUDA not available, invalid device).

```
RuntimeError: Failed to initialize BOXMOT tracker 'bytetrack' on device 'cuda:0':
CUDA not available
```

**Common causes:**
1. **CUDA not available**: Switch to CPU or install CUDA drivers
2. **Invalid device string**: Check device format (e.g., "cuda:0", not "cuda0")
3. **Missing ReID model file**: Verify reid_weights path exists

**Solution**: Check the error details and adjust configuration accordingly.

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
uv run pytest tests/track/tracker/test_boxmot_adapter.py -v
```

### End-to-End Pipeline Tests
```bash
# Test complete tracking pipeline and configuration selection
uv run pytest tests/integration/test_boxmot_pipeline.py -v
```

### Run All BOXMOT Tests
```bash
uv run pytest tests/track/test_boxmot_utils.py \
             tests/track/tracker/test_boxmot_adapter.py \
             tests/integration/test_boxmot_pipeline.py -v
```

### Performance Validation
A lightweight benchmark script is available for manual performance testing:

```bash
uv run scripts/benchmark_boxmot.py
```

This compares IOU vs ByteTrack performance (FPS, memory usage) with synthetic data.

## Known Limitations

### Production Considerations

1. **ReID Weights Validation**: Appearance-based trackers (BotSORT, BoostTrack, etc.) **require** ReID weights. The system now validates this at initialization and provides clear error messages:
   ```
   ValueError: Tracker type 'botsort' is an appearance-based tracker and requires
   reid_weights to be specified. Motion-only trackers: ['bytetrack', 'ocsort']
   ```

2. **Tracker Reset Between Videos**: When tracking multiple video files in sequence, the tracker automatically resets between files. This ensures:
   - Track IDs don't overlap between different videos
   - Class label mappings start fresh for each video
   - Memory is properly cleaned up

3. **Memory Management**: The adapter now automatically cleans up finished track IDs from internal mappings to prevent unbounded memory growth during long tracking sessions.

4. **Device Validation**: While basic device string format validation is in place, invalid CUDA devices will fail at BOXMOT initialization with a wrapped error message for clarity.

### Error Messages and Debugging

The integration provides clear, actionable error messages:

**BOXMOT Not Installed:**
```
ImportError: BOXMOT is not installed. Install with: uv pip install -e .[tracking_boxmot]
```

**BOXMOT Enabled but Not Installed:**
```
ImportError: BOXMOT is enabled in configuration but not installed.
Install with: uv pip install -e .[tracking_boxmot]
```

**Invalid Tracker Type:**
```
ValueError: Unknown tracker type: invalidtracker.
Supported types: ['bytetrack', 'botsort', 'boosttrack', 'strongsort',
'ocsort', 'deepocsort', 'hybridsort']
```

**Appearance Tracker Without ReID Weights:**
```
ValueError: Tracker type 'botsort' is an appearance-based tracker and requires
reid_weights to be specified. Motion-only trackers: ['bytetrack', 'ocsort']
```

**Tracker Initialization Failure:**
```
RuntimeError: Failed to initialize BOXMOT tracker 'bytetrack' on device 'cuda:0':
CUDA not available
```

### Logging

When a tracker is selected, informational logs are generated:

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
- **Integration Analysis**: See detailed implementation plan in project documentation

## Future Enhancements

Potential improvements for future iterations:

1. **CLI Arguments**: Add command-line arguments for BOXMOT configuration
2. **Performance Metrics**: Log tracking performance statistics
3. **Auto-Selection**: Automatically choose tracker based on hardware
4. **Custom ReID Models**: Support for custom ReID architectures
5. **Tracker Ensembles**: Combine multiple trackers for robustness
