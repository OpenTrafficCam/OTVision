"""
OTVision tool to select reference points for transformation
from pixel to world coordinates
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


import json
import logging
import tkinter as tk
from pathlib import Path
from random import randrange
from tkinter import ttk
from tkinter.simpledialog import Dialog
from typing import Any, Union

import cv2

from OTVision.helpers.files import is_image, is_video
from OTVision.helpers.log import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


class ReferencePointsPicker:
    """Class to pick reference points in pixel coordinates for transform subpackage.

    Instructions for using the gui:
        Hold left mouse button      Find spot for reference point with magnifier
        Release left mouse button   Mark reference point
        CTRL + Z                    Unmark last reference point
        CTRL + Y                    Remark last reference point
        CTRL + S                    Save image with markers
        CTRL + N                    Show next frame (disabled if image was loaded)
        CTRL + R                    Show random frame (disabled if image was loaded)
        ESC                         Close window and return reference points
    """

    def __init__(
        self,
        file: Path,
        title: str = "Reference Points Picker",
        popup_root: Union[tk.Tk, None] = None,
    ):
        # Attributes
        self.title = title
        self.left_button_down = False
        self.refpts: dict = {}
        self.historic_refpts: dict = {}
        self.file = file
        self.refpts_file = file.with_suffix(".otrfpts")
        self.image = None
        self.video = None
        self.popup_root = popup_root

        # Initial method calls
        self.update_base_image()
        self.show()

    # ----------- Handle OpenCV gui -----------

    def update_base_image(self, random_frame: bool = False):
        log.debug("update base image")
        if is_image(self.file):
            self.get_image()
            self.video = None
        elif is_video(self.file):
            self.get_frame_frome_video(random_frame=random_frame)
        else:
            raise ValueError("The file path provided has to be an image or a video")
        self.draw_refpts()

    def get_image(self):
        """Set base_image with image from a file

        Raises:
            ImageWontOpenError: If image wont open
        """
        try:
            self.base_image = cv2.imread(self.image_file)
        except Exception as e:
            raise ImageWontOpenError(
                f"Error opening this image file: {self.image_file}"
            ) from e

    def get_frame_frome_video(self, random_frame: bool):
        """Set base image with next or random frame from a video

        Args:
            random_frame (bool): If true gets a random frame from the video.
                If false gets the next frame of the video.

        Raises:
            VideoWontOpenError: If video wont open
            FrameNotAvailableError: If frame cannt be read
        """
        if not self.video:
            self.video = cv2.VideoCapture(str(self.file))
        if not self.video.isOpened():
            raise VideoWontOpenError(f"Error opening this video file: {self.file}")
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if random_frame:
            frame_nr = randrange(0, total_frames)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
        else:
            frame_nr = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            if frame_nr >= total_frames:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.base_image = self.video.read()
        if not ret:
            raise FrameNotAvailableError("Video Frame cannot be read correctly")

    def update_image(self):
        """Show the current image"""
        if not self.left_button_down:
            log.debug("update image")
            cv2.imshow(self.title, self.image)

    def show(self):
        """Enter OpenCV event loop"""
        cv2.imshow(self.title, self.image)
        cv2.setMouseCallback(self.title, self.handle_mouse_events)

        while True:
            # wait for a key press to close the window (0 = indefinite loop)
            key = cv2.waitKey(-1) & 0xFF  # BUG: #150 on mac

            window_visible = (
                cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
            )
            if key == 27 or not window_visible:
                break  # Exit loop and collapse OpenCV window
            else:
                self.handle_keystrokes(key)

        cv2.destroyAllWindows()

    def handle_mouse_events(
        self,
        event: int,
        x_px: int,
        y_px: int,
        flags,  # TODO: Correct type hint
        params: Any,  # TODO: Correct type hint
    ):
        """Read the current mouse position with a left click and write it
        to the end of the array refpkte and increases the counter by one"""
        if event == cv2.EVENT_LBUTTONUP:
            self.left_button_down = False
            self.add_refpt(x_px, y_px)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_down = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.left_button_down:
                self.draw_magnifier(x_px, y_px)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.left_button_down:
                self.zoom_magnifier(x_px, y_px)

    def handle_keystrokes(self, key: int):
        if key == 26:  # ctrl + z
            self.undo_last_refpt()
        elif key == 25:  # ctrl + y
            self.redo_last_refpt()
        elif key == 14:  # ctrl + n
            self.update_base_image(random_frame=False)
        elif key == 18:  # ctrl + r
            self.update_base_image(random_frame=True)
        elif key == 23:  # ctrl+w
            self._write_refpts()

    # ----------- Handle connections -----------

    def add_refpt(self, x_px: int, y_px: int):
        """Add a reference point

        Args:
            x_px (int): x coordinate [px] (horizontal distance from the left)
            y_px (int): y coordinate [px] (vertical distance from the top)
        """
        log.debug("add refpt")
        new_refpt_px = {"x_px": x_px, "y_px": y_px}
        self.draw_refpts(temp_refpt=new_refpt_px)
        if new_refpt_utm := self.get_refpt_utm_from_popup():
            new_refpt = {**new_refpt_px, **new_refpt_utm}
            self.refpts = self.append_refpt(refpts=self.refpts, new_refpt=new_refpt)
            self._log_refpts()
        self.draw_refpts()

    def undo_last_refpt(self):
        """Remove the last reference point"""
        if self.refpts:
            log.debug("undo last refpt")
            self.refpts, undone_refpt = self.pop_refpt(refpts=self.refpts)
            log.debug(self.refpts)
            log.debug(undone_refpt)
            self.historic_refpts = self.append_refpt(
                refpts=self.historic_refpts, new_refpt=undone_refpt
            )
            self.draw_refpts()
            self._log_refpts()
        else:
            log.debug("refpts empty, cannot undo last refpt")

    def redo_last_refpt(self):
        """Again add the last removed reference point"""
        if self.historic_refpts:
            log.debug("redo last refpt")
            self.historic_refpts, redone_refpt = self.pop_refpt(
                refpts=self.historic_refpts
            )
            self.append_refpt(refpts=self.refpts, new_refpt=redone_refpt)
            self.draw_refpts()
            self._log_refpts()
        else:
            log.debug("no historc refpts, cannot redo last refpt")

    def get_refpt_utm_from_popup(self) -> dict:
        """Open a popup to enter utm lat and lon coordinates of the reference point

        Returns:
            dict: dict of utm coordinates including hemisphere and utm zone
        """
        # TODO: Refactor to smaller methods
        if not self.popup_root:
            self.popup_root = tk.Tk()
            self.popup_root.overrideredirect(True)
            self.popup_root.withdraw()
        refpt_utm_correct = False
        try_again = False
        while not refpt_utm_correct:
            # Get utm part of refpt from pupup
            new_refpt_utm = DialogUTMCoordinates(
                parent=self.popup_root, try_again=try_again
            ).coords_utm
            # Check utm part of refpt
            log.debug(new_refpt_utm)
            if new_refpt_utm:  # BUG: ValueError: could not convert string to float: ''
                # BUG: Cancel cancels whole refpt
                # (even pixel coordinates, otherwise stuck in infinite loop)
                hemisphere_correct = isinstance(
                    new_refpt_utm["hemisphere"], str
                ) and new_refpt_utm["hemisphere"] in ["N", "S"]
                zone_correct = isinstance(new_refpt_utm["zone_utm"], int) and (
                    1 <= new_refpt_utm["zone_utm"] <= 60
                )
                lon_utm_correct = isinstance(
                    new_refpt_utm["lon_utm"], (float, int)
                ) and (100000 <= new_refpt_utm["lon_utm"] <= 900000)
                lat_utm_correct = isinstance(
                    new_refpt_utm["lat_utm"], (float, int)
                ) and (0 <= new_refpt_utm["lat_utm"] <= 10000000)
                if (
                    hemisphere_correct
                    and zone_correct
                    and lon_utm_correct
                    and lat_utm_correct
                ):
                    refpt_utm_correct = True
                else:
                    try_again = True
            else:
                break
        return new_refpt_utm

    # ----------- Edit refpts dict -----------

    def append_refpt(self, refpts: dict, new_refpt: dict) -> dict:
        """Append a reference point to a dict of reference points.
            Used for valid reference points as well as for deleted reference points.

        Args:
            refpts (dict): old reference points
            new_refpt (dict): new reference point

        Returns:
            dict: updated reference points
        """
        log.debug("append refpts")
        new_idx = len(refpts) + 1
        refpts[new_idx] = new_refpt
        return refpts

    def pop_refpt(self, refpts: dict) -> tuple[dict, dict]:
        """Remove reference point from a dict of reference points.
            Used for valid reference points as well as for deleted reference points.

        Args:
            refpts (dict): old reference points

        Returns:
            dict: new reference points
            dict: removed reference point
        """
        log.debug("pop refpts")
        popped_refpt = refpts.popitem()[1]
        return refpts, popped_refpt

    # ----------- Draw on image -----------

    def draw_magnifier(self, x_px, y_px):
        log.debug("#TODO: draw magnifier")

    def zoom_magnifier(self, x_px, y_px):
        log.debug(" # TODO: zoom magnifier")

    def draw_refpts(self, temp_refpt: Union[dict, None] = None):
        """Draw all the reference points an self.image

        Args:
            temp_refpt (dict, optional): possible reference point from left button down.
                Defaults to None.
        """
        FONT = cv2.FONT_ITALIC
        FONT_SIZE_REL = 0.02
        MARKER_SIZE_REL = 0.02
        FONT_SIZE_PX = round(self.base_image.shape[0] * FONT_SIZE_REL)
        MARKER_SIZE_PX = round(self.base_image.shape[0] * MARKER_SIZE_REL)
        log.debug("draw refpts")
        self.image = self.base_image.copy()
        refpts = self.refpts.copy()
        if temp_refpt:
            refpts = self.append_refpt(refpts=refpts, new_refpt=temp_refpt)
        for idx, refpt in refpts.items():
            x_px = refpt["x_px"]
            y_px = refpt["y_px"]
            marker_bottom = (x_px, y_px + MARKER_SIZE_PX)
            marker_top = (x_px, y_px - MARKER_SIZE_PX)
            marker_left = (x_px - MARKER_SIZE_PX, y_px)
            marker_right = (x_px + MARKER_SIZE_PX, y_px)
            cv2.line(self.image, marker_bottom, marker_top, (0, 0, 255), 1)
            cv2.line(self.image, marker_left, marker_right, (0, 0, 255), 1)
            cv2.putText(
                self.image,
                str(idx),
                marker_top,
                FONT,
                cv2.getFontScaleFromHeight(FONT, FONT_SIZE_PX),
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        self.update_image()

    # ----------- Complete job -----------

    def _write_refpts(self):
        """Write reference points to a json file"""
        log.debug("write refpts")
        # Only necessary if the reference points picker is used as standalone tool
        with open(self.refpts_file, "w") as f:
            json.dump(self.refpts, f, indent=4)

    def write_image(self):
        """This is done via on-board resources of OpenCV"""
        # Alternative: Own implementation using cv2.imwrite()
        # Problem: Ctrl+S shortcut is occupied by this
        pass

    # ----------- Helper functions -----------

    def _log_refpts(self):
        """Helper for logging"""
        log.debug("-------------------------")
        log.debug("refpts:")
        log.debug(self.refpts)
        log.debug("-------------------------")
        log.debug("historic refpts:")
        log.debug(self.historic_refpts)
        log.debug("-------------------------")


class DialogUTMCoordinates(Dialog):
    """Modificatoin of tkinter.simpledialog.Dialog to get utm coordinates and zone and
    hemisphere of sreference point
    """

    def __init__(self, try_again=False, **kwargs):
        self.coords_utm = None
        self.try_again = try_again
        super().__init__(**kwargs)

    def body(
        self, master: tk.Frame
    ):  # sourcery skip: assign-if-exp, swap-if-expression
        # Labels
        if not self.try_again:
            text_provide = "Provide reference point\nin UTM coordinates!"
        else:
            text_provide = "No valid UTM!\nProvide reference point\nin UTM coordinates!"
        tk.Label(master, text=text_provide).grid(row=0, columnspan=3)
        tk.Label(master, text="UTM zone:").grid(row=1, sticky="e")
        tk.Label(master, text="Longitude (E):").grid(row=2, sticky="e")
        tk.Label(master, text="Latitude (N):").grid(row=3, sticky="e")

        # Zone
        zones: list = list(range(1, 61))
        self.combo_zone = ttk.Combobox(
            master=master, values=zones, width=3, state="readonly"
        )
        self.combo_zone.set(32)
        self.combo_zone.grid(row=1, column=1, sticky="ew")

        # Hemisphere
        hemispheres = ["N", "S"]
        self.combo_hemisphere = ttk.Combobox(
            master=master, values=hemispheres, width=2, state="readonly"
        )
        self.combo_hemisphere.set("N")
        self.combo_hemisphere.grid(row=1, column=2, sticky="ew")

        self.input_lon = tk.Entry(master, width=5)
        self.input_lat = tk.Entry(master, width=5)
        self.input_lon.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.input_lat.grid(row=3, column=1, columnspan=2, sticky="ew")

        return self.combo_zone  # initial focus

    def apply(self):
        self.zone = int(self.combo_zone.get())
        self.hemisphere = self.combo_hemisphere.get()
        self.lon = float(self.input_lon.get())
        self.lat = float(self.input_lat.get())
        self.coords_utm = {
            "hemisphere": self.hemisphere,
            "zone_utm": self.zone,
            "lon_utm": self.lon,
            "lat_utm": self.lat,
        }
        log.debug(self.coords_utm)


class NoPathError(Exception):
    pass


class ImageWontOpenError(Exception):
    pass


class VideoWontOpenError(Exception):
    pass


class FrameNotAvailableError(Exception):
    pass
