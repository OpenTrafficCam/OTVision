"""Unit tests for track CLI argument parsing.

This module tests the ArgparseTrackCliParser class, specifically the parsing
of tracker-related CLI arguments.
"""

from argparse import ArgumentParser

import pytest

from OTVision.domain.cli import CliParseError
from OTVision.track.cli import ArgparseTrackCliParser


class TestTrackerParamsParsing:
    """Tests for tracker CLI argument parsing."""

    def test_parse_tracker_params_valid_int_float_string(self) -> None:
        """Test parsing tracker params with int, float, and string values."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser,
            argv=[
                "-p",
                "./",
                "--tracker-params",
                "track_buffer=60",
                "match_thresh=0.8",
                "name=test",
            ],
        )

        result = cli_parser.parse()

        assert result.tracker_params == {
            "track_buffer": 60,
            "match_thresh": 0.8,
            "name": "test",
        }

    def test_parse_tracker_params_invalid_format(self) -> None:
        """Test that invalid tracker param format raises CliParseError."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser,
            argv=["-p", "./", "--tracker-params", "invalid_param"],
        )

        with pytest.raises(CliParseError, match="Invalid tracker parameter format"):
            cli_parser.parse()

    def test_parse_tracker_params_none_when_not_provided(self) -> None:
        """Test that tracker_params is None when not provided."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(parser, argv=["-p", "./"])

        result = cli_parser.parse()

        assert result.tracker_params is None

    def test_parse_all_tracker_args(self) -> None:
        """Test parsing all tracker-related CLI arguments."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser,
            argv=[
                "-p",
                "./",
                "--tracker",
                "bytetrack",
                "--tracker-device",
                "cuda:0",
                "--tracker-half-precision",
                "--tracker-reid-weights",
                "/path/to/reid.pt",
                "--tracker-params",
                "track_buffer=60",
            ],
        )

        result = cli_parser.parse()

        assert result.tracker == "bytetrack"
        assert result.tracker_device == "cuda:0"
        assert result.tracker_half_precision is True
        assert result.tracker_reid_weights == "/path/to/reid.pt"
        assert result.tracker_params == {"track_buffer": 60}

    def test_parse_tracker_iou(self) -> None:
        """Test parsing --tracker iou."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser, argv=["-p", "./", "--tracker", "iou"]
        )

        result = cli_parser.parse()

        assert result.tracker == "iou"

    def test_parse_no_tracker_half_precision(self) -> None:
        """Test parsing --no-tracker-half-precision."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser, argv=["-p", "./", "--no-tracker-half-precision"]
        )

        result = cli_parser.parse()

        assert result.tracker_half_precision is False

    def test_parse_tracker_params_empty_list(self) -> None:
        """Test that --tracker-params with no values returns None."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser,
            argv=["-p", "./", "--tracker-params"],
        )

        result = cli_parser.parse()

        assert result.tracker_params is None

    def test_parse_tracker_params_value_with_equals(self) -> None:
        """Test parsing tracker param with value containing equals sign."""
        parser = ArgumentParser()
        cli_parser = ArgparseTrackCliParser(
            parser,
            argv=["-p", "./", "--tracker-params", "equation=a=b"],
        )

        result = cli_parser.parse()

        assert result.tracker_params == {"equation": "a=b"}
