import uuid
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm

from OTVision import dataformat
from OTVision.config import (
    CONFIG,
    IOU,
    OVERWRITE,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    T_MIN,
    T_MISS_MAX,
    TRACK,
)
from OTVision.track.exporter.filebased_exporter import FinishedChunkTrackExporter
from OTVision.track.parser.chunk_parser_plugins import JsonChunkParser
from OTVision.track.parser.frame_group_parser_plugins import (
    TimeThresholdFrameGroupParser,
)
from OTVision.track.tracker.filebased_tracking import (
    GroupedFilesTracker,
    UnfinishedChunksBuffer,
)
from OTVision.track.tracker.tracker_plugin_iou import IouParameters, IouTracker


def track_id_generator() -> Iterator[int]:
    ID: int = 0
    while True:
        ID += 1
        yield ID


STR_ID_GENERATOR = Callable[[], str]


def tracker_metadata(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: float, t_miss_max: float
) -> dict:
    return {
        dataformat.NAME: "IOU",
        dataformat.SIGMA_L: sigma_l,
        dataformat.SIGMA_H: sigma_h,
        dataformat.SIGMA_IOU: sigma_iou,
        dataformat.T_MIN: t_min,
        dataformat.T_MISS_MAX: t_miss_max,
    }


def main(
    paths: list[Path],
    sigma_l: float = CONFIG[TRACK][IOU][SIGMA_L],
    sigma_h: float = CONFIG[TRACK][IOU][SIGMA_H],
    sigma_iou: float = CONFIG[TRACK][IOU][SIGMA_IOU],
    t_min: int = CONFIG[TRACK][IOU][T_MIN],
    t_miss_max: int = CONFIG[TRACK][IOU][T_MISS_MAX],
    overwrite: bool = CONFIG[TRACK][OVERWRITE],
    tracking_run_id_generator: STR_ID_GENERATOR = lambda: str(uuid.uuid4()),
) -> None:

    iou_tracker: IouTracker[Path] = IouTracker(
        parameters=IouParameters(sigma_l, sigma_h, sigma_iou, t_min, t_miss_max)
    )

    chunk_parser = JsonChunkParser()
    group_parser = TimeThresholdFrameGroupParser(
        tracker_data=tracker_metadata(sigma_l, sigma_h, sigma_iou, t_min, t_miss_max)
    )

    file_tracker = GroupedFilesTracker(
        tracker=iou_tracker,
        chunk_parser=chunk_parser,
        frame_group_parser=group_parser,
        id_generator_factory=lambda _: track_id_generator(),
        overwrite=True,
    )

    buffer = UnfinishedChunksBuffer(
        tracker=file_tracker,
        keep_discarded=True,
    )

    exporter = FinishedChunkTrackExporter()

    tracking_run_id = tracking_run_id_generator()
    finished_chunk_stream = buffer.group_and_track(paths)

    finished_chunk_progress = tqdm(
        finished_chunk_stream, desc="export FrameChunk", total=len(paths)
    )
    exporter.export(tracking_run_id, iter(finished_chunk_progress), overwrite)
