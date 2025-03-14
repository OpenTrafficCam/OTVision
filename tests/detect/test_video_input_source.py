from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from unittest.mock import MagicMock, Mock, _Call, call, patch

import pytest
from av import VideoFrame

from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.config import DATETIME_FORMAT, Config, DetectConfig
from OTVision.detect.video_input_source import VideoSource
from OTVision.domain.frame import Frame, FrameKeys
from tests.utils.mocking import create_mocks

FPS = 20
SIDE_DATA = Mock()


@pytest.fixture
def cyclist_mp4(test_data_dir: Path) -> Path:
    return test_data_dir / "Testvideo_Cars-Cyclist_FR20_2020-01-01_00-00-00.mp4"


@dataclass
class Given:
    config: Config
    subject: Mock
    get_current_config: Mock
    frame_rotator: Mock
    timestamper_factory: Mock
    timestampers: list[Mock]
    save_path_provider: Mock
    get_files: Mock
    get_fps: Mock
    input_files: list[Path]
    detection_files: list[Path]
    rotated_frames_per_video: list[list[Mock]]
    timestamped_frames_per_video: list[list[Frame]]
    video_frames_per_video: list[list[Mock]]
    mock_av: Mock | None = None

    @property
    def all_video_frames(self) -> list[Mock]:
        return list(chain.from_iterable(self.video_frames_per_video))

    @property
    def all_rotated_frames(self) -> list[Mock]:
        return list(chain.from_iterable(self.rotated_frames_per_video))

    @property
    def all_timestamped_frames(self) -> list[Frame]:
        return list(chain.from_iterable(self.timestamped_frames_per_video))


class TestVideoSource:
    @patch("OTVision.detect.video_input_source.get_fps")
    @patch("OTVision.detect.video_input_source.get_files")
    def test_produce_with_real_video_file(
        self,
        mock_get_files: Mock,
        mock_get_fps: Mock,
        cyclist_mp4: Path,
    ) -> None:
        amount_of_frames = 60
        given = setup_args(
            mock_get_files, mock_get_fps, [cyclist_mp4], True, amount_of_frames
        )
        target = setup(given)
        actual = list(target.produce())

        assert actual == given.all_timestamped_frames
        given.get_files.assert_called_once_with(
            paths=[cyclist_mp4],
            filetypes=given.config.filetypes.video_filetypes.to_list(),
        )
        given.get_fps.assert_called_once_with(cyclist_mp4)
        given.timestamper_factory.create_video_timestamper.assert_called_once_with(
            video_file=cyclist_mp4,
            expected_duration=given.config.detect.expected_duration,
        )

        assert given.frame_rotator.rotate.call_count == amount_of_frames
        for call_args in given.frame_rotator.rotate.call_args_list:
            actual_video_frame, actual_side_data = call_args.args
            assert isinstance(actual_video_frame, VideoFrame)
            assert actual_side_data == {}

        assert given.timestampers[0].stamp.call_args_list == [
            call(
                {
                    FrameKeys.data: rotated_frame,
                    FrameKeys.frame: frame_number,
                    FrameKeys.source: str(cyclist_mp4),
                }
            )
            for frame_number, rotated_frame in enumerate(
                given.all_rotated_frames, start=1
            )
        ]

    @patch("OTVision.detect.video_input_source.VideoSource.notify_observers")
    @patch("OTVision.detect.video_input_source.av")
    @patch("OTVision.detect.video_input_source.get_fps")
    @patch("OTVision.detect.video_input_source.get_files")
    def test_produce_with_multiple_video_files(
        self,
        mock_get_files: Mock,
        mock_get_fps: Mock,
        mock_av: Mock,
        mock_notify_observers: Mock,
    ) -> None:
        amount_of_frames_per_video = 60
        input_files = [
            Path("Testvideo1_FR20_2020-01-01_00-00-00.mp4"),
            Path("Testvideo2_FR20_2020-01-01_00-00-03.mp4"),
        ]
        given = setup_args(
            mock_get_files,
            mock_get_fps,
            input_files,
            True,
            amount_of_frames_per_video,
            mock_av,
        )
        target = setup(given)
        actual = list(target.produce())

        assert actual == given.all_timestamped_frames
        assert_get_files_called(given)
        assert_get_fps_called(given)
        assert_timestamper_factory_called(given)
        assert_frame_rotator_called(given)
        for index, input_file in enumerate(input_files):
            assert given.timestampers[index].stamp.call_args_list == [
                call(
                    {
                        FrameKeys.data: rotated_frame,
                        FrameKeys.frame: frame_number,
                        FrameKeys.source: str(input_file),
                    }
                )
                for frame_number, rotated_frame in enumerate(
                    given.rotated_frames_per_video[index], start=1
                )
            ]
        assert mock_notify_observers.call_args_list == [
            call(input_file, FPS) for input_file in input_files
        ]

    @patch("OTVision.detect.video_input_source.log")
    @patch("OTVision.detect.video_input_source.VideoSource.notify_observers")
    @patch("OTVision.detect.video_input_source.av")
    @patch("OTVision.detect.video_input_source.get_fps")
    @patch("OTVision.detect.video_input_source.get_files")
    def test_produce_video_skipped_when_no_start_date_found_in_file_name(
        self,
        mock_get_files: Mock,
        mock_get_fps: Mock,
        mock_av: Mock,
        mock_notify_observers: Mock,
        mock_log: Mock,
    ) -> None:
        amount_of_frames_per_video = 5
        input_file = Path("Video_without_start_date.mp4")
        given = setup_args(
            mock_get_files,
            mock_get_fps,
            [input_file],
            True,
            amount_of_frames_per_video,
            mock_av,
        )

        target = setup(given)
        actual = list(target.produce())
        assert actual == []
        mock_log.warning.assert_called_once_with(
            f"Video file name of '{input_file}' "
            f"must include date and time in format: {DATETIME_FORMAT}"
        )
        assert_get_files_called(given)
        given.get_fps.assert_not_called()
        given.timestamper_factory.create_video_timestamper.assert_not_called()
        given.frame_rotator.rotate.assert_not_called()
        for timestamper in given.timestampers:
            timestamper.stamp.assert_not_called()
        mock_notify_observers.assert_not_called()

    @patch("OTVision.detect.video_input_source.log")
    @patch("OTVision.detect.video_input_source.VideoSource.notify_observers")
    @patch("OTVision.detect.video_input_source.av")
    @patch("OTVision.detect.video_input_source.get_fps")
    @patch("OTVision.detect.video_input_source.get_files")
    def test_produce_skip_video_when_overwrite_not_allowed(
        self,
        mock_get_files: Mock,
        mock_get_fps: Mock,
        mock_av: Mock,
        mock_notify_observers: Mock,
        mock_log: Mock,
        cyclist_mp4: Path,
    ) -> None:
        amount_of_frames_per_video = 5
        given = setup_args(
            mock_get_files,
            mock_get_fps,
            [cyclist_mp4],
            False,
            amount_of_frames_per_video,
            mock_av,
        )

        target = setup(given)
        actual = list(target.produce())
        assert actual == []
        mock_log.warning.assert_called_once_with(
            f"{cyclist_mp4.with_suffix(".otdet")} already exists. "
            "To overwrite, set overwrite to True"
        )
        assert_get_files_called(given)
        given.get_fps.assert_not_called()
        given.timestamper_factory.create_video_timestamper.assert_not_called()
        given.frame_rotator.rotate.assert_not_called()
        for timestamper in given.timestampers:
            timestamper.stamp.assert_not_called()
        mock_notify_observers.assert_not_called()

    @patch("OTVision.detect.video_input_source.VideoSource.notify_observers")
    @patch("OTVision.detect.video_input_source.av")
    @patch("OTVision.detect.video_input_source.get_fps")
    @patch("OTVision.detect.video_input_source.get_files")
    def test_detection_start_and_end_are_considered(
        self,
        mock_get_files: Mock,
        mock_get_fps: Mock,
        mock_av: Mock,
        mock_notify_observers: Mock,
    ) -> None:
        detect_start = 1
        detect_end = 2
        expected_detect_start_in_frames = 20
        expected_detect_end_in_frames = 40
        total_frames = 60
        input_file = Path("path/to/Video_FR20_2020-01-01_00-00-00.mp4")
        given = setup_args(
            get_files=mock_get_files,
            get_fps=mock_get_fps,
            video_files=[input_file],
            detect_overwrite=True,
            amount_frames_per_video=total_frames,
            mock_av=mock_av,
            detect_start=detect_start,
            detect_end=detect_end,
        )

        target = setup(given)
        actual = list(target.produce())

        assert actual == given.all_timestamped_frames
        assert_get_files_called(given)
        assert_get_fps_called(given)
        assert_timestamper_factory_called(given)
        assert given.frame_rotator.rotate.call_args_list == [
            call(frame, SIDE_DATA)
            for frame in given.all_video_frames[
                expected_detect_start_in_frames - 1 : expected_detect_end_in_frames - 1
            ]
        ]
        expected_calls = []
        for frame_number, rotated_image in enumerate(given.all_rotated_frames, start=1):
            if (
                expected_detect_start_in_frames
                <= frame_number
                < expected_detect_end_in_frames
            ):
                expected_calls.append(
                    create_expected_frame_call(
                        data=given.all_rotated_frames[frame_number - 1],
                        frame_number=frame_number,
                        source=input_file,
                    )
                )
            else:
                create_expected_frame_call(
                    data=None,
                    frame_number=frame_number,
                    source=input_file,
                )
        mock_notify_observers.assert_called_once_with(input_file, FPS)


def create_expected_frame_call(
    data: Mock | None, frame_number: int, source: Path
) -> _Call:
    return call(
        {
            FrameKeys.data: data,
            FrameKeys.frame: frame_number,
            FrameKeys.source: str(source),
        }
    )


def assert_get_files_called(given: Given) -> None:
    given.get_files.assert_called_once_with(
        paths=given.input_files,
        filetypes=given.config.filetypes.video_filetypes.to_list(),
    )


def assert_get_fps_called(given: Given) -> None:
    assert given.get_fps.call_args_list == [
        call(input_file) for input_file in given.input_files
    ]


def assert_timestamper_factory_called(given: Given) -> None:
    assert given.timestamper_factory.create_video_timestamper.call_args_list == [
        call(
            video_file=input_file,
            expected_duration=given.config.detect.expected_duration,
        )
        for input_file in given.input_files
    ]


def assert_frame_rotator_called(given: Given) -> None:
    expected = [call(frame, SIDE_DATA) for frame in given.all_video_frames]
    assert given.frame_rotator.rotate.call_args_list == expected


def setup(given: Given) -> VideoSource:
    return VideoSource(
        subject=given.subject,
        get_current_config=given.get_current_config,
        frame_rotator=given.frame_rotator,
        timestamper_factory=given.timestamper_factory,
        save_path_provider=given.save_path_provider,
    )


def setup_args(
    get_files: Mock,
    get_fps: Mock,
    video_files: list[Path],
    detect_overwrite: bool,
    amount_frames_per_video: int,
    mock_av: Mock | None = None,
    detect_start: int | None = None,
    detect_end: int | None = None,
) -> Given:
    config = create_config(video_files, detect_overwrite, detect_start, detect_end)
    detection_files = [_file.with_suffix(".otdet") for _file in video_files]

    video_frames_per_video: list[list[Mock]] = []
    rotated_frames_per_video: list[list[Mock]] = []
    timestamped_frames_per_video: list[list[Frame]] = []
    timestampers_per_video = []

    for _ in video_files:
        rotated_frames_per_video.append(create_mocks(amount_frames_per_video))
        timestamped_frames: list[Frame] = create_mocks(amount_frames_per_video)
        timestamped_frames_per_video.append(timestamped_frames)
        timestampers_per_video.append(create_timestamper(timestamped_frames))
        if mock_av:
            video_frames_per_video.append(create_mocks(amount_frames_per_video))

    get_files.return_value = video_files
    get_fps.return_value = FPS
    if mock_av:
        configure_mock_av(mock_av, video_frames_per_video)

    total_rotated_frames = list(chain.from_iterable(rotated_frames_per_video))

    return Given(
        config=config,
        subject=Mock(),
        get_current_config=create_get_current_config(config),
        frame_rotator=create_frame_rotator(total_rotated_frames),
        timestamper_factory=create_timestamper_factory(timestampers_per_video),
        timestampers=timestampers_per_video,
        save_path_provider=create_save_path_provider(detection_files),
        get_files=get_files,
        get_fps=get_fps,
        input_files=video_files,
        detection_files=detection_files,
        video_frames_per_video=video_frames_per_video,
        rotated_frames_per_video=rotated_frames_per_video,
        timestamped_frames_per_video=timestamped_frames_per_video,
        mock_av=mock_av,
    )


def create_config(
    video_files: list[Path],
    detect_overwrite: bool,
    detect_start: int | None = None,
    detect_end: int | None = None,
) -> Config:
    detect_config = DetectConfig(
        paths=video_files,
        overwrite=detect_overwrite,
        detect_start=detect_start,
        detect_end=detect_end,
    )
    return Config(detect=detect_config)


def create_get_current_config(config: Config) -> Mock:
    mock = Mock(spec=GetCurrentConfig)
    mock.get.return_value = config
    return mock


def create_frame_rotator(rotated_frames: list[Mock]) -> Mock:
    mock = Mock()
    mock.rotate.side_effect = rotated_frames
    return mock


def create_timestamper_factory(timestamper: list[Mock]) -> Mock:
    mock = Mock()
    mock.create_video_timestamper.side_effect = timestamper
    return mock


def create_timestamper(timestamped_frames: list[Frame]) -> Mock:
    mock = Mock()
    mock.stamp.side_effect = timestamped_frames
    return mock


def create_save_path_provider(detection_files: list[Path]) -> Mock:
    mock = Mock()
    mock.provide.side_effect = detection_files
    return mock


def configure_mock_av(mock_av: Mock, video_frames_per_video: list[list[Mock]]) -> None:
    container = MagicMock()
    context_manager_container = MagicMock()
    mock_av.open.return_value = container
    container.__enter__.return_value = context_manager_container
    context_manager_container.streams.video[0].side_data = SIDE_DATA
    context_manager_container.decode.side_effect = [
        iter(video_frames) for video_frames in video_frames_per_video
    ]
