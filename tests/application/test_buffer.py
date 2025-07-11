from typing import Generator

import pytest

from OTVision.application.buffer import Buffer

IntBuffer = Buffer[int, list[int]]


class MockBuffer(IntBuffer):
    def on_flush(self, event: list[int]) -> None:
        pass


class TestBuffer:
    @pytest.fixture
    def target(self) -> IntBuffer:
        return MockBuffer()

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

    def test_reset_buffer_clears_elements(self, target: IntBuffer) -> None:
        elements = [1, 2, 3]
        for element in elements:
            target.buffer(element)

        assert target._get_buffered_elements() == elements
        target._reset_buffer()
        assert target._get_buffered_elements() == []

    def test_filter_empty_generator(self, target: IntBuffer) -> None:
        def empty_generator() -> Generator[int, None, None]:
            yield from ()

        result = list(target.filter(empty_generator()))

        assert result == []
        assert target._get_buffered_elements() == []
