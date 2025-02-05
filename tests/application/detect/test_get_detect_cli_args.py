from unittest.mock import Mock

from OTVision.application.detect.get_detect_cli_args import GetDetectCliArgs


class TestGetDetectCliArgs:

    def test_get(self) -> None:
        """
        #Requirement https://openproject.platomo.de/projects/001-opentrafficcam-live/work_packages/7184
        """  # noqa
        expected_cli_args = Mock()
        given = create_cli_parser(expected_cli_args)
        target = GetDetectCliArgs(given)
        actual = target.get()
        assert actual == expected_cli_args
        given.parse.assert_called_once()


def create_cli_parser(cli_args: Mock) -> Mock:
    parser = Mock()
    parser.parse.return_value = cli_args
    return parser
