import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from OTVision.config import CONFIG, DEFAULT_FILETYPE, DETECTIONS, TRACK
from OTVision.dataformat import (
    DATA,
    FRAME,
    FRAME_GROUP,
    INPUT_FILE_PATH,
    METADATA,
    TRACK_ID,
    TRACKING,
    TRACKING_RUN_ID,
)
from OTVision.helpers.files import write_json
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


F = TypeVar("F")  # Finished container: e.g. FinishedFrame or FinishedChunk


class FinishedTracksExporter(ABC, Generic[F]):

    @abstractmethod
    def get_detection_dicts(self, container: F) -> list[dict]:
        pass

    @abstractmethod
    def get_result_path(self, container: F) -> Path:
        pass

    @abstractmethod
    def get_metadata(self, container: F) -> dict:
        pass

    @abstractmethod
    def get_frame_group_id(self, container: F) -> int:
        pass

    def export(
        self, tracking_run_id: str, stream: Iterator[F], overwrite: bool
    ) -> None:
        for container in stream:
            self.export_frames(container, tracking_run_id, overwrite)

    def export_frames(
        self, container: F, tracking_run_id: str, overwrite: bool
    ) -> None:
        file_path = self.get_result_path(container)
        file_type = CONFIG[DEFAULT_FILETYPE][TRACK]

        det_dicts = self.reindex(self.get_detection_dicts(container))

        output = self.build_output(
            det_dicts,
            self.get_metadata(container),
            tracking_run_id,
            self.get_frame_group_id(container),
        )

        write_json(
            dict_to_write=output,
            file=Path(file_path),
            filetype=file_type,
            overwrite=overwrite,
        )

        log.info(f"Successfully tracked and wrote {file_path}")

    def reindex(self, det_dicts: list[dict]) -> list[dict]:
        min_frame_no = min(det[FRAME] for det in det_dicts)
        reindexed_dets = [
            {**det, **{FRAME: det[FRAME] - min_frame_no}}
            for det in det_dicts  # TODO is 0 or 1 the first desired index
        ]

        if len(reindexed_dets) == 0:
            return []

        assert len({detection[INPUT_FILE_PATH] for detection in reindexed_dets}) == 1

        reindexed_dets.sort(
            key=lambda detection: (
                detection[INPUT_FILE_PATH],
                detection[FRAME],
                detection[TRACK_ID],
            )
        )

        return reindexed_dets

    def build_output(
        self,
        detections: list[dict],
        metadata: dict,
        tracking_run_id: str,
        frame_group_id: int,
    ) -> dict:
        metadata = metadata
        metadata[TRACKING][TRACKING_RUN_ID] = tracking_run_id
        metadata[TRACKING][FRAME_GROUP] = frame_group_id
        return {METADATA: metadata, DATA: {DETECTIONS: detections}}
