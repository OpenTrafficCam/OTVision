from pathlib import Path
from unittest.mock import Mock, patch

from OTVision.application.get_config import DEFAULT_USER_CONFIG, GetConfig
from OTVision.config import Config

config_file_path = Path("path/to/my.otvision.config.yaml")


class TestGetConfig:
    def test_get_with_custom_user_config_file(self) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7184
        """  # noqa
        expected_config = Mock()
        given_parser = create_config_file_parser(expected_config)
        given_cli_args = create_cli_args(has_config=True)

        target = GetConfig(given_parser)
        actual = target.get(given_cli_args)
        assert actual == expected_config
        given_parser.parse.assert_called_once_with(config_file_path)
        given_cli_args.get_config_file.assert_called_once()

    def test_get_with_default_user_config_in_cwd(self) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7184
        """  # noqa
        expected_config = Mock()
        given_parser = create_config_file_parser(expected_config)
        given_cli_args = create_cli_args(has_config=False)

        target = GetConfig(given_parser)
        actual = target.get(given_cli_args)
        assert actual == expected_config
        given_parser.parse.assert_called_once_with(Path.cwd() / DEFAULT_USER_CONFIG)
        given_cli_args.get_config_file.assert_called_once()

    @patch("OTVision.application.get_config.Path.cwd")
    def test_get_with_no_config_supplied(self, mock_cwd: Mock) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7184
        """  # noqa
        not_existing_config_path = Path("path/does/not/exist.yaml")
        mock_cwd.return_value = not_existing_config_path
        given_parser = create_config_file_parser(Mock())
        given_cli_args = create_cli_args(has_config=False)

        target = GetConfig(given_parser)
        actual = target.get(given_cli_args)
        assert actual == Config()
        given_parser.parse.assert_not_called()
        given_cli_args.get_config_file.assert_called_once()
        mock_cwd.assert_called_once()


def create_config_file_parser(expected_config: Mock) -> Mock:
    config_parser = Mock()
    config_parser.parse.return_value = expected_config
    return config_parser


def create_cli_args(has_config: bool) -> Mock:
    cli_args = Mock()
    if has_config:
        cli_args.get_config_file.return_value = config_file_path
    else:
        cli_args.get_config_file.return_value = None
    return cli_args
