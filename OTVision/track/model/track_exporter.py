import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from tqdm import tqdm

from OTVision.application.config import DEFAULT_FILETYPE, TRACK
from OTVision.config import CONFIG
from OTVision.dataformat import (
    DATA,
    DETECTIONS,
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

    def __init__(self, file_type: str = CONFIG[DEFAULT_FILETYPE][TRACK]):
        self.file_type = file_type

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
            filetype=self.file_type,
            overwrite=overwrite,
        )

        log.info(f"Successfully tracked and wrote {file_path}")

    @staticmethod
    def reindex(det_dicts: list[dict]) -> list[dict]:
        min_frame_no = min(det[FRAME] for det in det_dicts)

        det_dicts_progress = tqdm(
            det_dicts,
            desc="reindex TrackedDetections",
            total=len(det_dicts),
            leave=False,
        )
        reindexed_dets = [
            {**det, **{FRAME: det[FRAME] - min_frame_no + 1}}
            for det in det_dicts_progress
        ]

        if len(reindexed_dets) == 0:
            return []

        if len({detection[INPUT_FILE_PATH] for detection in reindexed_dets}) > 1:
            raise ValueError("Expect detections from only a single source file")

        reindexed_dets.sort(
            key=lambda detection: (
                detection[INPUT_FILE_PATH],
                detection[FRAME],
                detection[TRACK_ID],
            )
        )

        return reindexed_dets

    @staticmethod
    def build_output(
        detections: list[dict],
        metadata: dict,
        tracking_run_id: str,
        frame_group_id: int,
    ) -> dict:
        metadata[TRACKING][TRACKING_RUN_ID] = tracking_run_id
        metadata[TRACKING][FRAME_GROUP] = frame_group_id
        return {METADATA: metadata, DATA: {DETECTIONS: detections}}
