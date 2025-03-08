from abc import ABC, abstractmethod
from typing import Generator


class Filter[IN, OUT](ABC):
    @abstractmethod
    def filter(self, pipe: Generator[IN, None, None]) -> Generator[OUT, None, None]:
        raise NotImplementedError
