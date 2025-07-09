from datetime import datetime, timedelta
from pathlib import Path

from OTVision.application.config import (
    COL_WIDTH,
    CONF,
    CONVERT,
    CRF,
    DATETIME_FORMAT,
    DEFAULT_FILETYPE,
    DELETE_INPUT,
    DETECT,
    DETECT_END,
    DETECT_START,
    ENCODING_SPEED,
    EXPECTED_DURATION,
    FLUSH_BUFFER_SIZE,
    FONT,
    FONT_SIZE,
    FPS_FROM_FILENAME,
    FRAME_WIDTH,
    GUI,
    HALF_PRECISION,
    IMG,
    IMG_SIZE,
    INPUT_FPS,
    IOU,
    LOCATION_X,
    LOCATION_Y,
    LOG,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
    NORMALIZED,
    OUTPUT_FILETYPE,
    OUTPUT_FPS,
    OVERWRITE,
    PATHS,
    REFPTS,
    ROTATION,
    RUN_CHAINED,
    SEARCH_SUBDIRS,
    SIGMA_H,
    SIGMA_IOU,
    SIGMA_L,
    START_TIME,
    STREAM,
    STREAM_NAME,
    STREAM_SAVE_DIR,
    STREAM_SOURCE,
    T_MIN,
    T_MISS_MAX,
    TRACK,
    TRANSFORM,
    UNDISTORT,
    VID,
    VIDEO_CODEC,
    WEIGHTS,
    WINDOW,
    WRITE_VIDEO,
    YOLO,
    Config,
    ConvertConfig,
    DetectConfig,
    StreamConfig,
    TrackConfig,
    YoloConfig,
    _DefaultFiletype,
    _GuiConfig,
    _GuiWindowConfig,
    _LogConfig,
    _TrackIouConfig,
    _TransformConfig,
    _UndistortConfig,
)
from OTVision.domain.serialization import Deserializer


class ConfigParser:

    def __init__(self, deserializer: Deserializer) -> None:
        self._deserialize = deserializer

    def parse(self, file: Path) -> Config:
        data = self._deserialize.deserialize(file)
        return self.parse_from_dict(data)

    def parse_from_dict(self, d: dict) -> Config:
        log_dict = d.get(LOG)
        default_filtetype_dict = d.get(DEFAULT_FILETYPE)
        convert_dict = d.get(CONVERT)
        detect_dict = d.get(DETECT)
        track_dict = d.get(TRACK)
        undistort_dict = d.get(UNDISTORT)
        transform_dict = d.get(TRANSFORM)
        gui_dict = d.get(GUI)
        stream_config_dict = d.get(STREAM)

        log_config = self.parse_log_config(log_dict) if log_dict else Config.log
        default_filetype = (
            self.parse_default_filetype(default_filtetype_dict)
            if default_filtetype_dict
            else Config.default_filetype
        )
        convert_config = (
            self.parse_convert_config(convert_dict) if convert_dict else Config.convert
        )
        detect_config = (
            self.parse_detect_config(detect_dict) if detect_dict else Config.detect
        )
        track_config = (
            self.parse_track_config(track_dict) if track_dict else Config.track
        )
        undistort_config = (
            self.parse_undistort_config(undistort_dict)
            if undistort_dict
            else Config.undistort
        )
        transform_config = (
            self.parse_transform_config(transform_dict)
            if transform_dict
            else Config.transform
        )
        gui_config = self.parse_gui_config(gui_dict) if gui_dict else Config.gui
        stream_config = None
        if stream_config_dict is not None:
            stream_config = self.parse_stream_config(stream_config_dict)

        return Config(
            log=log_config,
            search_subdirs=d.get(SEARCH_SUBDIRS, Config.search_subdirs),
            default_filetype=default_filetype,
            convert=convert_config,
            detect=detect_config,
            track=track_config,
            undistort=undistort_config,
            transform=transform_config,
            gui=gui_config,
            stream=stream_config,
        )

    def parse_log_config(self, data: dict) -> _LogConfig:
        return _LogConfig(
            data.get(LOG_LEVEL_CONSOLE, _LogConfig.log_level_console),
            data.get(LOG_LEVEL_FILE, _LogConfig.log_level_file),
        )

    def parse_default_filetype(self, data: dict) -> _DefaultFiletype:
        return _DefaultFiletype(
            data.get(VID, _DefaultFiletype.video),
            data.get(IMG, _DefaultFiletype.image),
            data.get(DETECT, _DefaultFiletype.detect),
            data.get(TRACK, _DefaultFiletype.track),
            data.get(REFPTS, _DefaultFiletype.refpts),
        )

    def parse_convert_config(self, data: dict) -> ConvertConfig:
        sources = self.parse_sources(data.get(PATHS, []))
        return ConvertConfig(
            sources,
            data.get(RUN_CHAINED, ConvertConfig.run_chained),
            data.get(OUTPUT_FILETYPE, ConvertConfig.output_filetype),
            data.get(INPUT_FPS, ConvertConfig.input_fps),
            data.get(OUTPUT_FPS, ConvertConfig.output_fps),
            data.get(FPS_FROM_FILENAME, ConvertConfig.fps_from_filename),
            data.get(DELETE_INPUT, ConvertConfig.delete_input),
            data.get(ROTATION, ConvertConfig.rotation),
            data.get(OVERWRITE, ConvertConfig.overwrite),
        )

    def parse_sources(self, sources: list[str]) -> list[str]:
        return [str(Path(source).expanduser()) for source in sources]

    def parse_detect_config(self, data: dict) -> DetectConfig:
        yolo_config_dict = data.get(YOLO)
        yolo_config = (
            self.parse_yolo_config(yolo_config_dict)
            if yolo_config_dict
            else DetectConfig.yolo_config
        )
        sources = self.parse_sources(data.get(PATHS, []))

        expected_duration = data.get(EXPECTED_DURATION, None)
        if expected_duration is not None:
            expected_duration = timedelta(seconds=int(expected_duration))

        start_time = self._parse_start_time(data)
        return DetectConfig(
            paths=sources,
            run_chained=data.get(RUN_CHAINED, DetectConfig.run_chained),
            yolo_config=yolo_config,
            expected_duration=expected_duration,
            overwrite=data.get(OVERWRITE, DetectConfig.overwrite),
            half_precision=data.get(HALF_PRECISION, DetectConfig.half_precision),
            start_time=start_time,
            detect_start=data.get(DETECT_START, DetectConfig.detect_start),
            detect_end=data.get(DETECT_END, DetectConfig.detect_end),
            write_video=data.get(WRITE_VIDEO, DetectConfig.write_video),
            video_codec=data.get(VIDEO_CODEC, DetectConfig.video_codec),
            encoding_speed=data.get(ENCODING_SPEED, DetectConfig.encoding_speed),
            crf=data.get(CRF, DetectConfig.crf),
        )

    def parse_yolo_config(self, data: dict) -> YoloConfig:
        return YoloConfig(
            weights=data.get(WEIGHTS, YoloConfig.weights),
            conf=data.get(CONF, YoloConfig.conf),
            iou=data.get(IOU, YoloConfig.iou),
            img_size=data.get(IMG_SIZE, YoloConfig.img_size),
            normalized=data.get(NORMALIZED, YoloConfig.normalized),
        )

    @staticmethod
    def _parse_start_time(d: dict) -> datetime | None:
        if start_time := d.get(START_TIME, DetectConfig.start_time):
            return datetime.strptime(start_time, DATETIME_FORMAT)
        return start_time

    def parse_track_config(self, data: dict) -> TrackConfig:
        iou_config_dict = data.get(IOU)
        iou_config = (
            self.parse_track_iou_config(iou_config_dict)
            if iou_config_dict
            else TrackConfig.iou
        )
        sources = self.parse_sources(data.get(PATHS, []))

        return TrackConfig(
            sources,
            data.get(RUN_CHAINED, TrackConfig.run_chained),
            iou_config,
            data.get(OVERWRITE, TrackConfig.overwrite),
        )

    def parse_track_iou_config(self, data: dict) -> _TrackIouConfig:
        return _TrackIouConfig(
            data.get(SIGMA_L, _TrackIouConfig.sigma_l),
            data.get(SIGMA_H, _TrackIouConfig.sigma_h),
            data.get(SIGMA_IOU, _TrackIouConfig.sigma_iou),
            data.get(T_MIN, _TrackIouConfig.t_min),
            data.get(T_MISS_MAX, _TrackIouConfig.t_miss_max),
        )

    def parse_undistort_config(self, data: dict) -> _UndistortConfig:
        return _UndistortConfig(
            data.get(OVERWRITE, _UndistortConfig.overwrite),
        )

    def parse_transform_config(self, d: dict) -> _TransformConfig:
        sources = self.parse_sources(d.get(PATHS, []))
        return _TransformConfig(
            sources,
            d.get(RUN_CHAINED, _TransformConfig.run_chained),
            d.get(OVERWRITE, _TransformConfig.overwrite),
        )

    def parse_gui_config(self, data: dict) -> _GuiConfig:
        window_config_dict = data.get(WINDOW)
        window_config = (
            self.parse_gui_window_config(window_config_dict)
            if window_config_dict
            else _GuiConfig.window_config
        )

        return _GuiConfig(
            font=data.get(FONT, _GuiConfig.font),
            font_size=data.get(FONT_SIZE, _GuiConfig.font_size),
            window_config=window_config,
            frame_width=data.get(FRAME_WIDTH, _GuiConfig.frame_width),
            col_width=data.get(COL_WIDTH, _GuiConfig.col_width),
        )

    def parse_gui_window_config(self, data: dict) -> _GuiWindowConfig:
        return _GuiWindowConfig(
            data.get(LOCATION_X, _GuiWindowConfig.location_x),
            data.get(LOCATION_Y, _GuiWindowConfig.location_y),
        )

    def parse_stream_config(self, data: dict) -> StreamConfig:
        name = data[STREAM_NAME]
        source = data[STREAM_SOURCE]
        save_dir = Path(data[STREAM_SAVE_DIR])
        flush_buffer_size = int(data[FLUSH_BUFFER_SIZE])
        return StreamConfig(
            name=name,
            source=source,
            save_dir=save_dir,
            flush_buffer_size=flush_buffer_size,
        )
