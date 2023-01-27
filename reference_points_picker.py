import argparse

from OTVision.helpers.log import log
from OTVision.transform.reference_points_picker import ReferencePointsPicker


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
    log.info("Start reference points picker gui")
    log.debug(kwargs)
    ReferencePointsPicker(**kwargs)
    log.info("Stop reference points picker gui")


if __name__ == "__main__":
    main()
