# OTVision

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. On the resulting trajectories, one can perform traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

Check out the [documentation](https://docs.opentrafficcam.org/otvision) for detailed instructions on how to install and use OTVision.

We appreciate your support in the form of both code and comments. First, please have a look at the [contribute](https://docs.opentrafficcam.org/contribute) section of the OpenTrafficCam documentation.

## Prequesites

1. Python 3.9.x (add to PATH while installation)
2. CUDA (if detection should run on GPU)
3. Microsoft Visual C++ 14.0 or greater (Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Installation

1. Clone this repository
2. Klick on .\OTVision\Install.bat (a venv will be created and packages from requirements.txt will be installed including PyTorch versions published on [pytorch.org](https://pytorch.org/get-started/locally/))
3. Klick on .\OTVision\OTVision.bat (venv will be activated and OTVision gui will be started)

## License

This software is licensed under the [GPL-3.0 License](LICENSE)
