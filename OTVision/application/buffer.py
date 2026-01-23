from abc import abstractmethod
from typing import AsyncIterator

from OTVision.abstraction.pipes_and_filter import Filter


class Buffer[T, OBSERVING_TYPE](Filter[T, T]):
    def __init__(self) -> None:
        self._buffer: list[T] = []

    async def filter(self, pipe: AsyncIterator[T]) -> AsyncIterator[T]:
        async for element in pipe:
            self.buffer(element)
            yield element

    def buffer(self, to_buffer: T) -> None:
        self._buffer.append(to_buffer)

    def _get_buffered_elements(self) -> list[T]:
        return self._buffer

    def _reset_buffer(self) -> None:
        del self._buffer
        self._buffer = list()

    @abstractmethod
    def on_flush(self, event: OBSERVING_TYPE) -> None:
        raise NotImplementedError
