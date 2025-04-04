# OTVision

[![PyPI version](https://img.shields.io/pypi/v/OTVision.svg)](https://pypi.org/project/OTVision/)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml/badge.svg?tag=latest)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/test.yml?query=tag%3Alatest)
[![Tests](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml/badge.svg)](https://github.com/OpenTrafficCam/OTVision/actions/workflows/build-release.yml)

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) designed to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. The resulting trajectories can be used for traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

## Features

- **Video Conversion**: Convert video files (h264 to mp4) with options to set frame rate and rotation
- **Object Detection**: Detect road users in videos using state-of-the-art YOLO models
- **Object Tracking**: Track detected objects through video frames using IOU-based tracking algorithms
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

Track detected objects:

```bash
python track.py --paths /path/to/detections/*.otdet
```

### Coordinate Transformation

Transform pixel coordinates to UTM coordinates:

```bash
python transform.py --paths /path/to/tracks/*.ottrk --refpts-file /path/to/reference_points.json
```

### Configuration

OTVision can be configured using a YAML configuration file. A default configuration is provided in `user_config.otvision.yaml`. You can specify a custom configuration file using the `--config` option:

```bash
python detect.py --config /path/to/custom_config.yaml
```

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
