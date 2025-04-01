import logging

from tqdm import tqdm

from OTVision.application.config import Config
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.helpers.files import get_files
from OTVision.helpers.input_types import check_types
from OTVision.helpers.log import LOGGER_NAME
from OTVision.track.id_generator import StrIdGenerator, tracking_run_uuid_generator
from OTVision.track.model.track_exporter import FinishedTracksExporter
from OTVision.track.tracker.filebased_tracking import UnfinishedChunksBuffer

log = logging.getLogger(LOGGER_NAME)


class OtvisionTrack:
    @property
    def config(self) -> Config:
        return self._get_current_config.get()

    def __init__(
        self,
        get_current_config: GetCurrentConfig,
        track_exporter: FinishedTracksExporter,
        unfinished_chunks_buffer: UnfinishedChunksBuffer,
        tracking_run_id_generator: StrIdGenerator = tracking_run_uuid_generator,
    ) -> None:
        self._get_current_config = get_current_config
        self._track_exporter = track_exporter
        self._buffer = unfinished_chunks_buffer
        self._tracking_run_id_generator = tracking_run_id_generator

    def start(self) -> None:
        check_types(
            self.config.track.sigma_l,
            self.config.track.sigma_h,
            self.config.track.sigma_iou,
            self.config.track.t_min,
            self.config.track.t_miss_max,
        )

        detections_files = get_files(
            paths=self.config.track.paths, filetypes=[self.config.filetypes.detect]
        )

        start_msg = f"Start tracking of {len(detections_files)} detections files"
        log.info(start_msg)
        print(start_msg)

        if not detections_files:
            log.warning(
                f"No files of type '{self.config.filetypes.detect}' " "found to track!"
            )
            return

        tracking_run_id = self._tracking_run_id_generator()
        finished_chunk_stream = self._buffer.group_and_track(detections_files)

        finished_chunk_progress = tqdm(
            finished_chunk_stream, desc="export FrameChunk", total=len(detections_files)
        )
        self._track_exporter.export(
            tracking_run_id, iter(finished_chunk_progress), self.config.track.overwrite
        )
