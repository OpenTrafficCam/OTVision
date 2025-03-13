"""
OTVision main module to detect objects in single or multiple images or videos.
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from OTVision.domain.detect_producer_consumer import (
    DetectedFrameConsumer,
    DetectedFrameProducer,
)


class OTVisionVideoDetect(DetectedFrameConsumer):
    def __init__(self, producer: DetectedFrameProducer) -> None:
        self._producer = producer

    def start(self) -> None:
        """Starts the detection of objects in multiple videos and/or images."""
        self.consume()

    def consume(self) -> None:
        for _ in self._producer.produce():
            pass
