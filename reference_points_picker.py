import argparse

from OTVision.helpers.log import get_logger
from OTVision.transform.reference_points_picker import ReferencePointsPicker

log = get_logger(__name__)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference Points Picker")
    parser.add_argument(
        "-video_file",
        type=str,
        help="Path to the video in which reference points are to be clicked",
    )
    parser.add_argument(
        "-image_file",
        type=str,
        help="Path to the image in which reference points are to be clicked",
    )
    return parser.parse_args()


def main() -> None:
    kwargs = vars(parse())
    log.info("Call reference points picker from command line")
    log.info(kwargs)
    ReferencePointsPicker(**kwargs)


if __name__ == "__main__":
    main()
