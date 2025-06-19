from abc import ABC, abstractmethod

from numpy import ndarray


class VideoWriter(ABC):
    @abstractmethod
    def write(self, image: ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def open(self, output: str, width: int, height: int, fps: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
