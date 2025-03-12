from abc import abstractmethod
from typing import Generator

from OTVision.abstraction.observer import Observable, Subject
from OTVision.abstraction.pipes_and_filter import Filter


class Buffer[T, SUBJECT_TYPE, OBSERVING_TYPE](Observable[SUBJECT_TYPE], Filter[T, T]):
    def __init__(self, subject: Subject[SUBJECT_TYPE]) -> None:
        super().__init__(subject)
        self._buffer: list[T] = []

    def filter(self, pipe: Generator[T, None, None]) -> Generator[T, None, None]:
        for element in pipe:
            self.buffer(element)
            yield element

    def buffer(self, to_buffer: T) -> None:
        self._buffer.append(to_buffer)

    def _get_buffered_elements(self) -> list[T]:
        return self._buffer

    def _reset_buffer(self) -> None:
        self._buffer = []

    def on_flush(self, event: OBSERVING_TYPE) -> None:
        buffered_elements = self._get_buffered_elements()
        self._notify_observers(buffered_elements, event)
        self._reset_buffer()

    @abstractmethod
    def _notify_observers(self, elements: list[T], event: OBSERVING_TYPE) -> None:
        raise NotImplementedError
