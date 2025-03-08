from pathlib import Path

import av

from OTVision.application.frame_count_provider import FrameCountProvider


class PyAVFrameCountProvider(FrameCountProvider):
    def provide(self, video_file: Path) -> int:
        counter = 0
        with av.open(str(video_file.absolute())) as container:
            container.streams.video[0].thread_type = "AUTO"
            for _ in container.decode(video=0):
                counter += 1
        return counter
