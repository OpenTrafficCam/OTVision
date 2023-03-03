import unittest.mock as mock
from pathlib import Path
from unittest.mock import patch

# import pytest
# import sys
# from tests.conftest import YieldFixture

# @pytest.fixture()
# def add_root_dir_to_path() -> YieldFixture:
#     """Add the root dir of the repository to pythonpath for the time of the test"""

#     # Get the absolute path to the root folder of the repository
#     root_dir = Path(__file__).resolve().parents[3]

#     # Add the root folder to the PYTHONPATH
#     sys.path.insert(0, str(root_dir))

#     yield

#     # Remove
#     sys.path.remove(root_dir)


# @pytest.mark.usefixtures("add_root_dir_to_path")

test_data: dict = {"paths": {}, "input_fps": {}}

test_data["paths"]["passed"] = [r"/usr/local/bin", r"C:/Program Files/Python/Scripts"]
test_data["paths"]["expected"] = [
    Path("/usr/local/bin"),
    Path("C:/Program Files/Python/Scripts"),
]

test_data["input_fps"]


def test_convert_cli() -> None:
    # Import main functions from cli and from subpackage
    from convert import main as convert_cli  # main from the cli script in the root dir
    from OTVision import convert  # main from OTVision.convert.convert.py

    # Create a mock object based
    convert = mock.create_autospec(convert)

    # As long as the patch context manager lives, convert is replaced by mock_convert
    with patch("OTVision.convert") as mock_convert:
        # Define commands passed to convert_cli´s main
        module = "convert.py"
        paths_tag = "-p"
        paths = [r"/usr/local/bin", r"C:/Program Files/Python/Scripts"]
        input_fps_tag = "--input_fps"
        input_fps = "20"
        fps_from_filename = "--fps_from_filename"
        overwrite = "--overwrite"
        delete_input = "--no-delete_input"
        cmd = [
            module,
            paths_tag,
            *paths,
            input_fps_tag,
            input_fps,
            fps_from_filename,
            overwrite,
            delete_input,
        ]
        # Call convert_cli´s main with this command
        convert_cli(argv=cmd)

        # Define expected arguments
        expected_paths = [
            Path("/usr/local/bin"),
            Path("C:/Program Files/Python/Scripts"),
        ]
        expected_input_fps = 20
        expected_fps_from_filename = True
        expected_overwrite = True
        expected_delete_input = False

        # Asserting that the target function was called once with the expected arguments
        mock_convert.assert_called_once_with(
            paths=expected_paths,
            input_fps=expected_input_fps,
            fps_from_filename=expected_fps_from_filename,
            overwrite=expected_overwrite,
            delete_input=expected_delete_input,
        )
