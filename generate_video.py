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


from OTVision.plugin.generate_video import GenerateVideoBuilder


def main(argv: list[str] | None = None) -> None:  # sourcery skip: assign-if-exp
    builder = GenerateVideoBuilder(argv=argv)
    cli_args = builder.detect_cli_parser.parse()
    config = builder.update_detect_config_with_ci_args.update(
        config=builder.get_config.get(cli_args)
    )
    log = builder.configure_logger.configure(
        config,
        log_file=cli_args.logfile,
        logfile_overwrite=cli_args.logfile_overwrite,
    )

    log.info("Call video generate from command line")
    log.info(f"Arguments: {vars(cli_args)}")

    try:
        builder.update_current_config.update(config)
        generate_video = builder.build_generate_video()
        generate_video.generate()

    except FileNotFoundError:
        log.exception(f"One of the following files cannot be found: {cli_args.paths}")
        raise
    except Exception:
        log.exception("")
        raise


if __name__ == "__main__":
    main()
