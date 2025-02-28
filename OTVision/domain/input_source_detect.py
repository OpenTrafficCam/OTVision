from abc import ABC, abstractmethod
from typing import Generator

from numpy import ndarray

Image = ndarray


class InputSourceDetect(ABC):

    @abstractmethod
    def produce(self) -> Generator[Image, None, None]:
        raise NotImplementedError
