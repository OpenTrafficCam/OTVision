from pathlib import Path

from OTVision.application.config import DEFAULT_FILETYPE, TRACK
from OTVision.config import CONFIG
from OTVision.track.model.filebased.frame_chunk import FinishedChunk
from OTVision.track.model.filebased.frame_group import get_output_file
from OTVision.track.model.track_exporter import FinishedTracksExporter


class FinishedChunkTrackExporter(FinishedTracksExporter[FinishedChunk]):

    def __init__(self, file_type: str = CONFIG[DEFAULT_FILETYPE][TRACK]) -> None:
        super().__init__(file_type)

    def get_detection_dicts(self, container: FinishedChunk) -> list[dict]:
        return container.to_detection_dicts()

    def get_result_path(self, container: FinishedChunk) -> Path:
        return get_output_file(container.file, self.file_type)

    def get_metadata(self, container: FinishedChunk) -> dict:
        return container.metadata

    def get_frame_group_id(self, container: FinishedChunk) -> int:
        return container.frame_group_id
