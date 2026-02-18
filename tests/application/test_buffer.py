from typing import AsyncIterator

import pytest

from OTVision.application.buffer import Buffer

IntBuffer = Buffer[int, list[int]]


class MockBuffer(IntBuffer):
    async def on_flush(self, event: list[int]) -> None:
        pass


class TestBuffer:
    @pytest.fixture
    def target(self) -> IntBuffer:
        return MockBuffer()

    @pytest.mark.asyncio
    async def test_buffer_element(self, target: IntBuffer) -> None:
        given_element = 1

        await target.buffer(given_element)

        assert target._get_buffered_elements() == [given_element]

    @pytest.mark.asyncio
    async def test_filter_yields_and_buffers_frames(self, target: IntBuffer) -> None:
        expected = [1, 2, 3]

        async def element_generator() -> AsyncIterator[int]:
            for element in expected:
                yield element

        actual = []
        async for item in target.filter(element_generator()):
            actual.append(item)

        assert actual == expected
        assert target._get_buffered_elements() == expected

    @pytest.mark.asyncio
    async def test_reset_buffer_clears_elements(self, target: IntBuffer) -> None:
        elements = [1, 2, 3]
        for element in elements:
            await target.buffer(element)

        assert target._get_buffered_elements() == elements
        target._reset_buffer()
        assert target._get_buffered_elements() == []

    @pytest.mark.asyncio
    async def test_filter_empty_generator(self, target: IntBuffer) -> None:
        async def empty_generator() -> AsyncIterator[int]:
            empty: list[int] = []
            for elem in empty:
                yield elem

        result = []
        async for item in target.filter(empty_generator()):
            result.append(item)

        assert result == []
        assert target._get_buffered_elements() == []
