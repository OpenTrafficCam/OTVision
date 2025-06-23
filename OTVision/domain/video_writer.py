from abc import ABC, abstractmethod

from numpy import ndarray

from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.application.event.new_video_start import NewVideoStartEvent
from OTVision.detect.detected_frame_buffer import FlushEvent
from OTVision.domain.frame import Frame


class VideoWriter(Filter[Frame, Frame], ABC):
    @abstractmethod
    def write(self, image: ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def open(self, output: str, width: int, height: int, fps: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def notify_on_flush_event(self, event: FlushEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def notify_on_new_video_start(self, event: NewVideoStartEvent) -> None:
        raise NotImplementedError
