import platform

OS = platform.system().replace("Darwin", "Mac")
"""OS OTVision is currently running on"""

ON_WINDOWS = OS == "Windows"
"""Wether OS is Windows or not"""

ON_LINUX = OS == "Linux"
"""Wether OS is Linux or not"""

ON_MAC = OS == "Mac"
"""Wether OS is MacOS or not"""

OS_RELEASE = platform.release()
"""Release of the OS OTVision is currently running on"""

OS_VERSION = platform.version()
"""Specific version of the OS OTVision is currently running on"""

PY_VERSION = 1
"""Full Python version number"""

PY_MAJOR_VERSION = int(platform.python_version_tuple()[0])
"""Python major version digit (e.g. 3 for 3.9.5) OTVision is currently running with"""

PY_MINOR_VERSION = int(platform.python_version_tuple()[1])
"""Python minor version digit (e.g. 9 for 3.9.5) OTVision is currently running with"""

PY_PATCH_VERSION = int(platform.python_version_tuple()[2])
"""Python patch version digit (e.g. 5 for 3.9.5) OTVision is currently running with"""

def _has_cuda():
    """Returns True if CUDA is installed on machine

    Returns:
        Bool: If CUDA is installed on machine or not
    """
    import torch
    return torch.cuda.is_available()
