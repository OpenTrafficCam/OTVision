from OTVision.application.config import Config, TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.application.update_current_config import UpdateCurrentConfig


class UpdateCurrentTrackConfig:
    """Use case to update the current track configuration with a TrackConfig object"""

    def __init__(
        self,
        get_current_config: GetCurrentConfig,
        update_current_config: UpdateCurrentConfig,
    ) -> None:
        self._get_current_config = get_current_config
        self._update_current_config = update_current_config

    def update(self, track_config: TrackConfig) -> None:
        """Update current configuration with a TrackConfig object.

        Args:
            track_config (TrackConfig): TrackConfig object to update current
                configuration with.
        """
        updated_config = self._update_with(track_config)
        self._update_current_config.update(updated_config)

    def _update_with(self, track_config: TrackConfig) -> Config:
        current_config = self._get_current_config.get()
        return Config(
            log=current_config.log,
            search_subdirs=current_config.search_subdirs,
            default_filetype=current_config.default_filetype,
            filetypes=current_config.filetypes,
            last_paths=current_config.last_paths,
            convert=current_config.convert,
            detect=current_config.detect,
            track=track_config,
            undistort=current_config.undistort,
            transform=current_config.transform,
            gui=current_config.gui,
            stream=current_config.stream,
        )
