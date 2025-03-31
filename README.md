# OTVision

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) designed to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. The resulting trajectories can be used for traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

## Features

- **Video Conversion**: Convert video files (h264 to mp4) with options to set frame rate and rotation
- **Object Detection**: Detect road users in videos using state-of-the-art YOLO models
- **Object Tracking**: Track detected objects through video frames using IOU-based tracking algorithms
- **Coordinate Transformation**: Transform pixel coordinates to real-world UTM coordinates for accurate measurements

## Requirements

### System Requirements

- Python 3.12 or higher
- CUDA-capable GPU (recommended for faster processing)

### Dependencies

OTVision depends on several Python packages:

- **Video Processing**:
  - av (PyAV)
  - moviepy
  - opencv-python

- **Machine Learning**:
  - torch and torchvision (PyTorch)
  - ultralytics (YOLOv8)

- **Data Processing**:
  - numpy
  - pandas
  - geopandas

- **Other Utilities**:
  - PyYAML
  - tqdm
  - fire

For a complete list of dependencies, see the [requirements.txt](requirements.txt) file.

## Installation

### Basic Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OpenTrafficCam/OTVision.git
   cd OTVision
   ```

2. Install the package:
   - On Windows:
     ```bash
     install.cmd
     ```
   - On Linux/macOS:
     ```bash
     ./install.sh
     ```

### Installation with CUDA Support (for GPU acceleration)

1. Clone the repository:
   ```bash
   git clone https://github.com/OpenTrafficCam/OTVision.git
   cd OTVision
   ```

2. Install the package with CUDA support:
   - On Windows:
     ```bash
     install.cmd
     ```
     Select the CUDA option when prompted.
   - On Linux:
     ```bash
     ./install.sh
     ```
     Select the CUDA option when prompted.

### Development Installation

For development purposes, use:
- On Windows:
  ```bash
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
- Email: team@opentrafficcam.org
