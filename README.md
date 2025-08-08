# OTVision

[![PyPI version](https://img.shields.io/pypi/v/OTVision.svg)](https://pypi.org/project/OTVision/)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml/badge.svg?tag=latest)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml?query=tag%3Alatest)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml/badge.svg)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml)

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) designed to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. The resulting trajectories can be used for traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

## Features

- **Video Conversion**: Convert video files (h264 to mp4) with options to set frame rate and rotation
- **Object Detection**: Detect road users in videos using state-of-the-art YOLO models
- **Object Tracking**: Track detected objects through video frames using IOU-based, BoT-SORT, or SMILEtrack tracking algorithms
- **Coordinate Transformation**: Transform pixel coordinates to real-world UTM coordinates

## Requirements

### System Requirements

- Python 3.12 or higher
- CUDA-capable GPU (recommended for faster processing)
- Dependencies listed in [requirements.txt](requirements.txt)

## Installation

### Installation from GitHub Releases

Pre-built releases are available on GitHub for easy installation:

1. Go to the [OTVision Releases page](https://github.com/OpenTrafficCam/OTVision/releases) on GitHub
2. Download the appropriate release for your platform:
   - **Windows**: Choose between `otvision-win.zip` (standard) or `otvision-win-cuda.zip` (with CUDA support)
   - **Linux**: Choose between `otvision-linux.zip` (standard) or `otvision-linux-cuda.zip` (with CUDA support)
   - **macOS**: Choose `OTVision-macos.zip`
3. Extract the downloaded ZIP file
4. Run the installation script:
   - On Windows: Double-click `install.cmd`
   - On Linux: Run `./install.sh`
   - On macOS: Double-click `install.command`

### Manual Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/OpenTrafficCam/OTVision.git
   cd OTVision
   ```

2. Install the package:
   - On Windows:
     ```cmd
     install.cmd
     ```
   - On Linux/macOS:
     ```bash
     ./install.sh
     ```

### Manual Development Installation

For development purposes, if you want to contribute to the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/OpenTrafficCam/OTVision.git
   cd OTVision
   ```

2. Install the development version:
   - On Windows:
     ```cmd
     install_dev.cmd
     ```
   - On Linux/macOS:
     ```bash
     ./install_dev.sh
     ```

## Usage

OTVision provides several command-line scripts for different functionalities:

### Video Conversion

Convert video files (e.g., h264 to mp4):

```bash
python convert.py --paths /path/to/videos/*.h264 --input-fps 20.0 --rotation 0
```

### Object Detection

Detect objects in videos:

```bash
python detect.py --paths /path/to/videos/*.mp4 --weights yolov8s
```

### Object Tracking

Track detected objects using IOU-based tracking (default), BoT-SORT tracking, or SMILEtrack:

#### Basic Usage

```bash
python track.py --paths /path/to/detections/*.otdet
```

#### Using BoT-SORT Tracker

To use the BoT-SORT tracker instead of the default IOU tracker:

```bash
python track.py --paths /path/to/detections/*.otdet --tracker-type botsort
```

#### BoT-SORT Parameters

You can customize BoT-SORT parameters via command line:

```bash
python track.py --paths /path/to/detections/*.otdet \
  --tracker-type botsort \
  --track-high-thresh 0.6 \
  --track-low-thresh 0.1 \
  --new-track-thresh 0.7 \
  --track-buffer 30 \
  --match-thresh 0.8 \
  --reid-model-path /path/to/reid_model.pth
```

**BoT-SORT Parameter Descriptions:**
- `--track-high-thresh`: High confidence threshold for track association (default: 0.6)
- `--track-low-thresh`: Low confidence threshold for track association (default: 0.1)
- `--new-track-thresh`: Threshold for creating new tracks (default: 0.7)
- `--track-buffer`: Number of frames to keep lost tracks in buffer (default: 30)
- `--match-thresh`: Threshold for track matching using appearance features (default: 0.8)
- `--reid-model-path`: Path to re-identification model file for enhanced tracking through occlusions (optional)

#### BoT-SORT Re-Identification (Re-ID) Functionality

BoT-SORT now supports re-identification models to improve tracking performance, especially in challenging scenarios with occlusions, crowded scenes, or long-term disappearances. When a re-ID model is provided via the `--reid-model-path` parameter, BoT-SORT will:

- Extract appearance features from detected objects using the re-ID model
- Use appearance similarity in addition to motion and IoU for track association
- Better handle identity switches and track fragmentation
- Improve tracking accuracy in crowded scenes

**Re-ID Model Compatibility:** BoT-SORT uses the same re-ID model format as SMILEtrack, allowing you to use the same model file for both trackers.

#### YAML Configuration for BoT-SORT

You can configure BoT-SORT parameters using YAML configuration files in two ways:

##### Option 1: Using OTVision Configuration File

Configure BoT-SORT parameters within your main OTVision config file:

```yaml
TRACK:
  PATHS: []
  TRACKER_TYPE: botsort  # Set to "botsort" to use BoT-SORT tracker
  IOU:
    SIGMA_H: 0.42
    SIGMA_IOU: 0.38
    SIGMA_L: 0.27
    T_MIN: 5
    T_MISS_MAX: 51
  BOTSORT:
    TRACK_HIGH_THRESH: 0.6
    TRACK_LOW_THRESH: 0.1
    NEW_TRACK_THRESH: 0.7
    TRACK_BUFFER: 30
    MATCH_THRESH: 0.8
    REID_MODEL_PATH: "/path/to/reid_model.pth"  # Optional: Path to re-ID model for enhanced tracking
  OVERWRITE: true
```

Then run with your custom config:

```bash
python track.py --config /path/to/your_config.yaml
```

##### Option 2: Using Dedicated BoT-SORT Configuration File

You can use a separate YAML file specifically for BoT-SORT parameters using the `--botsort-config` parameter:

Create a dedicated BoT-SORT config file (`botsort_config.yaml`):

```yaml
track_high_thresh: 0.6
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.8
reid_model_path: "/path/to/reid_model.pth"  # Optional: Path to re-ID model for enhanced tracking
```

Then run with the dedicated BoT-SORT config:

```bash
python track.py --paths /path/to/detections/*.otdet --botsort-config /path/to/botsort_config.yaml
```

##### Using Both Configuration Files

You can use both the main OTVision config and a dedicated BoT-SORT config file together:

```bash
python track.py --config /path/to/otvision_config.yaml --botsort-config /path/to/botsort_config.yaml
```

**Note:** Individual CLI parameters take precedence over values in configuration files. For example:

```bash
python track.py --botsort-config /path/to/botsort_config.yaml --track-high-thresh 0.9
```

In this case, `track_high_thresh` will be 0.9, overriding the value in the config file.

#### Using SMILEtrack Tracker

SMILEtrack (SiMIlarity LEarning for Occlusion-Aware Multiple Object Tracking) is an advanced tracker that provides enhanced re-identification capabilities for better tracking through occlusions and long-term disappearances.

##### Basic SMILEtrack Usage

```bash
python track.py --paths /path/to/detections/*.otdet --tracker-type smiletrack
```

##### SMILEtrack Parameters

You can customize SMILEtrack parameters via command line:

```bash
python track.py --paths /path/to/detections/*.otdet \
  --tracker-type smiletrack \
  --track-high-thresh 0.6 \
  --track-low-thresh 0.1 \
  --new-track-thresh 0.7 \
  --track-buffer 30 \
  --match-thresh 0.8 \
  --proximity-thresh 0.5 \
  --appearance-thresh 0.25 \
  --reid-model-path /path/to/reid_model.pth
```

**SMILEtrack Parameter Descriptions:**
- `--track-high-thresh`: High confidence threshold for track association (default: 0.6)
- `--track-low-thresh`: Low confidence threshold for track association (default: 0.1)
- `--new-track-thresh`: Threshold for creating new tracks (default: 0.7)
- `--track-buffer`: Number of frames to keep lost tracks in buffer (default: 30)
- `--match-thresh`: IoU threshold for spatial matching (default: 0.8)
- `--proximity-thresh`: Proximity threshold for re-identification matching (default: 0.5)
- `--appearance-thresh`: Appearance threshold for re-ID feature matching (default: 0.25)
- `--reid-model-path`: Path to trained re-ID model file (optional, uses simple features if not provided)

##### YAML Configuration for SMILEtrack

Configure SMILEtrack parameters within your main OTVision config file:

```yaml
TRACK:
  PATHS: []
  TRACKER_TYPE: smiletrack  # Set to "smiletrack" to use SMILEtrack tracker
  IOU:
    SIGMA_H: 0.42
    SIGMA_IOU: 0.38
    SIGMA_L: 0.27
    T_MIN: 5
    T_MISS_MAX: 51
  SMILETRACK:
    TRACK_HIGH_THRESH: 0.6
    TRACK_LOW_THRESH: 0.1
    NEW_TRACK_THRESH: 0.7
    TRACK_BUFFER: 30
    MATCH_THRESH: 0.8
    PROXIMITY_THRESH: 0.5
    APPEARANCE_THRESH: 0.25
    REID_MODEL_PATH: "/path/to/reid_model.pth"
  OVERWRITE: true
```

##### SMILEtrack Re-ID Features

SMILEtrack includes advanced re-identification capabilities:

1. **Automatic Feature Extraction**: Extracts appearance features from detected objects
2. **Feature Smoothing**: Uses exponential moving average for stable feature representation
3. **Similarity Matching**: Employs cosine similarity for appearance-based matching
4. **Occlusion Handling**: Can re-associate tracks after long-term occlusions
5. **Configurable Re-ID Model**: Supports custom trained re-ID models or uses built-in simple features

##### Using Custom Re-ID Models

To use a trained re-ID model with SMILEtrack:

1. Place your trained PyTorch re-ID model file in an accessible location
2. Specify the model path in your configuration or command line:

```bash
python track.py --paths /path/to/detections/*.otdet \
  --tracker-type smiletrack \
  --reid-model-path /path/to/your_reid_model.pth
```

**Note**: If no re-ID model path is provided or the model file doesn't exist, SMILEtrack will use a built-in simple feature extractor based on color histograms and texture features.

### Coordinate Transformation

Transform pixel coordinates to UTM coordinates:

```bash
python transform.py --paths /path/to/tracks/*.ottrk --refpts-file /path/to/reference_points.json
```

### Configuration

OTVision supports two types of configuration files:

#### OTVision Configuration Files

OTVision can be configured using a YAML configuration file that contains settings for all modules (detection, tracking, transformation, etc.). A default configuration is provided in `user_config.otvision.yaml`. You can specify a custom OTVision configuration file using the `--config` option:

```bash
python detect.py --config /path/to/custom_config.yaml
python track.py --config /path/to/custom_config.yaml
```

#### BoT-SORT Configuration Files

For tracking operations, you can also use a dedicated BoT-SORT configuration file that contains only BoT-SORT-specific parameters. This is useful when you want to experiment with different BoT-SORT settings without modifying your main OTVision configuration. Use the `--botsort-config` option:

```bash
python track.py --botsort-config /path/to/botsort_config.yaml
```

You can use both configuration types together:

```bash
python track.py --config /path/to/otvision_config.yaml --botsort-config /path/to/botsort_config.yaml
```

**Configuration Priority:** Individual CLI parameters > BoT-SORT config file > OTVision config file > Default values

## Documentation

For detailed documentation, visit:

- [OTVision Documentation](https://opentrafficcam.org/OTVision/)
- [Getting Started Guide](https://opentrafficcam.org/OTVision/gettingstarted/firstuse/)
- [Requirements](https://opentrafficcam.org/OTVision/gettingstarted/requirements/)
- [Installation Guide](https://opentrafficcam.org/OTVision/gettingstarted/installation/)

## Contributing

We appreciate your support in the form of both code and comments. Please have a look at the [contribute](https://opentrafficcam.org/contribute) section of the OpenTrafficCam documentation for guidelines on how to contribute to the project.

## License

This software is licensed under the [GPL-3.0 License](LICENSE).

## Contact

- GitHub: [https://github.com/OpenTrafficCam](https://github.com/OpenTrafficCam)
- Email: [team@opentrafficcam.org](mailto:team@opentrafficcam.org)
