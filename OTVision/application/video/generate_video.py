from OTVision.abstraction.pipes_and_filter import Filter
from OTVision.domain.frame import Frame
from OTVision.domain.input_source_detect import InputSourceDetect


class GenerateVideo:
    def __init__(
        self, input_source: InputSourceDetect, video_writer: Filter[Frame, Frame]
    ) -> None:
        self._input_source = input_source
        self._video_writer = video_writer

    def generate(self) -> None:
        for frame in self._video_writer.filter(self._input_source.produce()):
            pass
