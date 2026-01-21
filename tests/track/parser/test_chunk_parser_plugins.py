from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np

from OTVision.dataformat import DETECTIONS
from OTVision.domain.detection import Detection
from OTVision.track.parser.chunk_parser_plugins import DetectionParser, JsonChunkParser
from OTVision.track.video_frame_provider import VideoFrameProvider
from tests.track.helper.data_builder import (
    DEFAULT_INPUT_FILE_PATH,
    DEFAULT_START_DATE,
    DataBuilder,
)


class TestDetectionParser:

    def test_convert(self) -> None:
        data_builder = DataBuilder().append_classified_frame(
            number_of_classifications=10
        )
        dict_input: list[dict] = data_builder.build()[1][DETECTIONS]

        parser = DetectionParser()
        result: list[Detection] = parser.convert(dict_input)

        expected = data_builder.build_objects()[1].detections
        assert expected == result


class TestJsonChunkParser:

    def input_data_builder(self) -> DataBuilder:
        data_builder = DataBuilder()
        data_builder.append_classified_frame()
        data_builder.append_non_classified_frame()
        data_builder.append_classified_frame()
        return data_builder

    def test_convert(self) -> None:
        data_builder = self.input_data_builder()
        det_input = data_builder.build()
        expected = list(data_builder.build_objects().values())

        parser = JsonChunkParser()
        result = parser.convert(DEFAULT_INPUT_FILE_PATH, 0, det_input)

        assert expected == result

    def test_frame_offset(self) -> None:
        expected_data_builder = DataBuilder()
        expected_data_builder.current_key = 5
        expected_data_builder.start_date = DEFAULT_START_DATE + timedelta(
            microseconds=1
        )
        expected_data_builder.append_classified_frame()
        expected_data_builder.start_date = DEFAULT_START_DATE + timedelta(
            microseconds=2
        )
        expected_data_builder.append_non_classified_frame()
        expected_data_builder.start_date = DEFAULT_START_DATE + timedelta(
            microseconds=3
        )
        expected_data_builder.append_classified_frame()

        det_input = self.input_data_builder().build()
        expected = list(expected_data_builder.build_objects().values())

        parser = JsonChunkParser()
        result = parser.convert(DEFAULT_INPUT_FILE_PATH, 5, det_input)
        assert expected == result

    def test_convert_without_video_provider_has_none_images(self) -> None:
        """Test that frames have image=None when no video provider is given."""
        data_builder = self.input_data_builder()
        det_input = data_builder.build()

        parser = JsonChunkParser()  # No video provider
        result = parser.convert(DEFAULT_INPUT_FILE_PATH, 0, det_input)

        # All frames should have image=None
        for frame in result:
            assert frame.image is None

    def test_convert_with_video_provider_loads_images(self) -> None:
        """Test that frames have images loaded when video provider is given."""
        data_builder = self.input_data_builder()
        det_input = data_builder.build()

        # Create mock video provider
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_provider = MagicMock(spec=VideoFrameProvider)
        mock_provider.get_frame.return_value = mock_image

        parser = JsonChunkParser()
        result = parser.convert(DEFAULT_INPUT_FILE_PATH, 0, det_input, mock_provider)

        # All frames should have the mock image
        for frame in result:
            assert frame.image is not None
            assert np.array_equal(frame.image, mock_image)

        # Video provider should have been called for each frame
        assert mock_provider.get_frame.call_count == len(det_input)

    def test_convert_with_video_provider_returning_none(self) -> None:
        """Test that frames have image=None when provider returns None."""
        data_builder = self.input_data_builder()
        det_input = data_builder.build()

        # Create mock video provider that returns None
        mock_provider = MagicMock(spec=VideoFrameProvider)
        mock_provider.get_frame.return_value = None

        parser = JsonChunkParser()
        result = parser.convert(DEFAULT_INPUT_FILE_PATH, 0, det_input, mock_provider)

        # All frames should have image=None
        for frame in result:
            assert frame.image is None
