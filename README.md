# OTVision

[![PyPI version](https://img.shields.io/pypi/v/OTVision.svg)](https://pypi.org/project/OTVision/)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml/badge.svg?tag=latest)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml?query=tag%3Alatest)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml/badge.svg)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml)

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) designed to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. The resulting trajectories can be used for traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

## Features

- **Video Conversion**: Convert video files (h264 to mp4) with options to set frame rate and rotation
- **Object Detection**: Detect road users in videos using state-of-the-art YOLO models
- **Object Tracking**: Track detected objects through video frames using IOU-based or BoT-SORT tracking algorithms
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

Track detected objects using either IOU-based tracking (default) or BoT-SORT tracking:

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
  --match-thresh 0.8
```

**BoT-SORT Parameter Descriptions:**
- `--track-high-thresh`: High confidence threshold for track association (default: 0.6)
- `--track-low-thresh`: Low confidence threshold for track association (default: 0.1)
- `--new-track-thresh`: Threshold for creating new tracks (default: 0.7)
- `--track-buffer`: Number of frames to keep lost tracks in buffer (default: 30)
- `--match-thresh`: Threshold for track matching using appearance features (default: 0.8)

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
