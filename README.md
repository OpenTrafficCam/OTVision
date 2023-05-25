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
2. Click on .\OTVision\Install.bat (a venv will be created and packages from requirements.txt will be installed including PyTorch versions published on [pytorch.org](https://pytorch.org/get-started/locally/))
3. Click on .\OTVision\OTVision.bat (venv will be activated and OTVision gui will be started)

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

## Docker and container

In the long run it is planned to ship modules of OTVision as dedicated containers. Allowing users to configure the container input (mount data storage and config file) and run OTVision in a container environment. The container contains all necessary dependencies. There is no need to install anything more than the container engine. Therefore, two docker files are prepared.

The container were tested on a slurm cluster. Unfortunately, the container for detection grow extremly in size (about 5GB). Therefore, the approach was set on hold. For the next steps, the base images of the containers should be checked. nvidia, pytorch and yolo provide base containers to use for ml tasks. One of them could be a good start.

### Dockerfile.detect

`Dockerfile.detect` runs OTVisions detection. The following bash-Script shows the usage of the container. The name of the container and its version are parameters of the script together with the config for OTVision to run detection.

```bash
#!/bin/sh

if [ 3 -gt $# ]; then
    echo "Please provide a container name and a version and a config name!"
    exit 1
fi

host_input_dir="/scratch/ws/1/labr704e-p_trafficcam_t_30"
base_container="/OpenTrafficCam/OTVision"
home_dir=$(realpath ~)
container_folder=/projects/p_trafficcam/container
container_name=$1
container_version=$2
config_name=$3
execution_path=${container_folder}/"${container_name}"-"${container_version}".sif

singularity run --nv --contain \
-B "${home_dir}" \
--bind "${host_input_dir}"/config/"${config_name}":"${base_container}"/config/user_config.otvision.yaml:ro \
--bind "${host_input_dir}"/data:"${base_container}"/data \
--bind "${host_input_dir}"/models:"${base_container}"/models \
--pwd ${base_container}/ \
--env PYTHONPATH=${base_container} \
"${execution_path}"
```

### Dockerfile.track

`Dockerfile.track` runs OTVisions tracking. The following bash-Script shows the usage of the container.

```bash
#!/bin/sh

otvision_dir=../OTVision
host_input_dir=$(realpath ${otvision_dir})
base_container="/OpenTrafficCam/OTVision"
config_name="user_config.otvision.miovision.yaml"

docker run -it \
--mount type=bind,source="${host_input_dir}"/models,target=${base_container}/models \
--mount type=bind,source="${host_input_dir}"/config/${config_name},target=${base_container}/config/user_config.otvision.yaml,readonly \
--mount type=bind,source="${host_input_dir}"/data,target=${base_container}/data \
 otvision:0.1.3-track
 ```

## Development

For development please install also the ```requirements_dev.txt``` (and use flake8 for linting and black for autoformatting with line length 88).
We suggest to use VS Code for editing the code, as we also ship a settings.json.

## License

This software is licensed under the [GPL-3.0 License](LICENSE)
