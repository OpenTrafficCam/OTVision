"""
OTVision script to call the detect main with arguments parsed from command line
"""

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from pathlib import Path

from OTVision.config import Config
from OTVision.detect.builder import DetectBuilder
from OTVision.helpers.files import check_if_all_paths_exist


def main(argv: list[str] | None = None) -> None:  # sourcery skip: assign-if-exp
    builder = DetectBuilder(argv=argv)
    cli_args = builder.detect_cli_parser.parse()
    config = builder.update_detect_config_with_ci_args.update(
        config=builder.get_config.get(cli_args)
    )
    log = builder.configure_logger.configure(
        config,
        log_file=cli_args.logfile,
        logfile_overwrite=cli_args.logfile_overwrite,
    )

    log.info("Call detect from command line")
    log.info(f"Arguments: {vars(cli_args)}")

    try:
        builder.update_current_config.update(config)
        detect = builder.build()
        detect.start()

    except FileNotFoundError:
        log.exception(f"One of the following files cannot be found: {cli_args.paths}")
        raise
    except Exception:
        log.exception("")
        raise


def _extract_paths(config: Config) -> list[Path]:
    if paths := config.detect.paths:
        converted = [Path(path) for path in paths]
        check_if_all_paths_exist(converted)
        return converted

    raise IOError(
        "No paths have been passed as command line args."
        "No paths have been defined in the user config."
    )


if __name__ == "__main__":
    main()
