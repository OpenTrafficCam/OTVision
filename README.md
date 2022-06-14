# OTVision

OTVision is a core module of the [OpenTrafficCam framework](https://github.com/OpenTrafficCam) to detect and track objects (road users) in videos recorded by [OTCamera](https://github.com/OpenTrafficCam/OTCamera) or other camera systems. On the resulting trajectories, one can perform traffic analysis using [OTAnalytics](https://github.com/OpenTrafficCam/OTAnalytics).

Check out the [documentation](https://opentrafficcam.org/OTVision/) for detailed instructions on how to install and use OTVision.

We appreciate your support in the form of both code and comments. First, please have a look at the [contribute](https://opentrafficcam.org/contribute) section of the OpenTrafficCam documentation.

## Prequesites

1. Python 3.9.x (add to PATH while installation)
2. CUDA (if detection should run on GPU)
3. Microsoft Visual C++ 14.0 or greater (Get it with "Microsoft C++ Build Tools": <https://visualstudio.microsoft.com/visual-cpp-build-tools/>

## Installation

### Windows

1. Clone this repository
2. Klick on .\OTVision\Install.bat (a venv will be created and packages from requirements.txt will be installed including PyTorch versions published on [pytorch.org](https://pytorch.org/get-started/locally/))
3. Klick on .\OTVision\OTVision.bat (venv will be activated and OTVision gui will be started)

### Apple M1 macOS

To install the dependencies required to run OTVision on Apple M1 the following requirements need to be satisfied:

- brew ([Installation Guide](https://brew.sh))
- [Make](https://www.gnu.org/software/make/) (Install with brew via `brew install make`)

Use the following `make` command to start the OTVision GUI assuming the command is executed in the OTVision directory.
This command will automatically install all needed project dependencies if needed. 
The default python version is set to 3.9.

```bash
make run 
```

To only install the OTVision's project dependencies with, run the following command in the OTVision directory:

```bash
make install
```

It is also possible to install the project dependencies with a different python version by passing in the version as an argument to the make command:

```bash
make PY_VERSION=3.10 run
```

```bash
make PY_VERSION=3.10 install 
```

## Development

For development please install also the ```requirements_dev.txt``` (and use flake8 for linting and black for autoformatting with line length 88).
We suggest to use VS Code for editing the code, as we also ship a settings.json.

## License

This software is licensed under the [GPL-3.0 License](LICENSE)
