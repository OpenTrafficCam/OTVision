from unittest.mock import Mock

from OTVision.application.track.get_track_cli_args import GetTrackCliArgs


class TestGetTrackCliArgs:
    def test_get(self) -> None:
        expected = Mock()
        given_parser = Mock()
        given_parser.parse.return_value = expected
        target = GetTrackCliArgs(given_parser)

        actual = target.get()

        assert actual == expected
        given_parser.parse.assert_called_once()
