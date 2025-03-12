from pathlib import Path

from OTVision.application.get_current_config import GetCurrentConfig


class DetectionFileSavePathProvider:
    """Provides a mechanism to generate file save paths for detections.

    This class is responsible for deriving the appropriate filenames for
    detection files based on the source and current configuration
    settings. It utilizes a configuration provider to dynamically fetch
    the required configuration values.

    Args:
        get_current_config (GetCurrentConfig): Retrieve the current application
            configuration.
    """

    def __init__(self, get_current_config: GetCurrentConfig) -> None:
        self._get_current_config = get_current_config

    def provide(self, source: str) -> Path:
        config = self._get_current_config.get()
        return derive_filename(
            video_file=Path(source),
            detect_suffix=config.filetypes.detect,
            detect_start=config.detect.detect_start,
            detect_end=config.detect.detect_end,
        )


def derive_filename(
    video_file: Path,
    detect_suffix: str,
    detect_start: int | None = None,
    detect_end: int | None = None,
) -> Path:
    """
    Generates a filename for detection files by appending specified start and end
    markers and a suffix to the stem of the input video file.

    Args:
        video_file (Path): The input video file whose filename is to be modified.
        detect_start (int | None): The starting marker to append to the filename.
            If None, no starting marker will be appended.
        detect_end (int | None): The ending marker to append to the filename. If None,
            no ending marker will be appended.
        detect_suffix (str): The file suffix to apply to the derived filename.

    Returns:
        Path: The modified video file path with the updated stem and suffix applied.
    """
    cutout = ""
    if detect_start is not None:
        cutout += f"_start_{detect_start}"
    if detect_end is not None:
        cutout += f"_end_{detect_end}"
    new_stem = f"{video_file.stem}{cutout}"
    return video_file.with_stem(new_stem).with_suffix(detect_suffix)
