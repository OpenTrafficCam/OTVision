from typing import Generator
from unittest.mock import Mock

import pytest

from OTVision.abstraction.observer import Observer, Subject
from OTVision.application.buffer import Buffer
from OTVision.detect.detected_frame_buffer import DetectedFrameBuffer

IntBuffer = Buffer[int, list[int], None]


class MockBuffer(IntBuffer):
    def _notify_observers(self, elements: list[int], _: None) -> None:
        self._subject.notify(elements)


class TestDetectedFrameBuffer:
    @pytest.fixture
    def subject_mock(self) -> Mock:
        return Mock(spec=Subject)

    @pytest.fixture
    def target(self, subject_mock: Mock) -> IntBuffer:
        return MockBuffer(subject=subject_mock)

    def test_buffer_element(self, target: IntBuffer) -> None:
        given_element = 1

        target.buffer(given_element)

        assert target._get_buffered_elements() == [given_element]

    def test_filter_yields_and_buffers_frames(self, target: IntBuffer) -> None:
        expected = [1, 2, 3]

        def element_generator() -> Generator[int, None, None]:
            for element in expected:
                yield element

        actual = list(target.filter(element_generator()))

        assert actual == expected
        assert target._get_buffered_elements() == expected

    def test_on_flush_notifies_subject_and_resets_buffer(
        self, target: IntBuffer, subject_mock: Mock
    ) -> None:
        elements = [1, 2, 3]
        for element in elements:
            target.buffer(element)

        target.on_flush(None)

        subject_mock.notify.assert_called_once_with(elements)
        assert target._get_buffered_elements() == []

    def test_reset_buffer_clears_elements(self, target: IntBuffer) -> None:
        elements = [1, 2, 3]
        for element in elements:
            target.buffer(element)

        assert target._get_buffered_elements() == elements
        target._reset_buffer()
        assert target._get_buffered_elements() == []

    def test_register_delegates_to_subject(
        self, target: DetectedFrameBuffer, subject_mock: Mock
    ) -> None:
        observer: Mock = Mock(spec=Observer)

        target.register(observer)

        subject_mock.register.assert_called_once_with(observer)

    def test_filter_empty_generator(self, target: IntBuffer) -> None:
        def empty_generator() -> Generator[int, None, None]:
            yield from ()

        result = list(target.filter(empty_generator()))

        assert result == []
        assert target._get_buffered_elements() == []
