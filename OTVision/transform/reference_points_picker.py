import json
import logging
import tkinter as tk
from pathlib import Path
from random import randrange
from tkinter import ttk
from tkinter.simpledialog import Dialog, askfloat

import cv2


class ReferencePointsPicker:
    """Class to pick reference points in pixel coordinates for transform subpackage.

    Instructions for using the gui:
        Hold left mouse button      Find perfect spot for reference point with magnifier
        Release left mouse button   Mark reference point
        CTRL + Z                    Unmark last reference point
        CTRL + Y                    Remark last reference point
        CTRL + S                    Save image with markers
        ESC                         Close window and return reference points
    """

    def __init__(self, title=None, image_path=None, video_path=None, popup_root=None):

        # Attributes
        self.title = title or "Click reference points"
        self.left_button_down = False
        self.refpts = {}
        self.historic_refpts = {}
        self.image_path = image_path
        self.video_path = video_path
        self.refpts_path = Path(image_path or video_path).with_suffix(".otrfpts")
        self.image = None
        self.video = None
        self.popup_root = popup_root

        # Initial method calls
        self.update_base_image()
        self.show()

    # ----------- Handle OpenCV gui -----------

    def update_base_image(self, random_frame=False):
        print("update base image")
        if self.image_path:
            try:
                self.base_image = cv2.imread(self.image_path)
            except:
                raise ImageWontOpenError(
                    f"Error opening this image file: {self.image_path}"
                )
            self.video = None
        elif self.video_path:
            if not self.video:
                self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise VideoWontOpenError(
                    f"Error opening this video file: {self.video_path}"
                )
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
        else:
            raise NoPathError("Neither image nor video path was specified")
        self.draw_refpts()

    def update_image(self):
        if not self.left_button_down:
            print("update image")
            cv2.imshow(self.title, self.image)

    def show(self):

        cv2.imshow(self.title, self.image)
        cv2.setMouseCallback(self.title, self.handle_mouse_events)

        while True:

            # wait for a key press to close the window (0 = indefinite loop)
            key = cv2.waitKey(-1) & 0xFF

            window_visible = (
                cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
            )
            if key == 27 or not window_visible:
                break  # Exit loop and collapse OpenCV window
            else:
                self.handle_keystrokes(key)

        cv2.destroyAllWindows()

    def handle_mouse_events(self, event, x_px, y_px, flags, params):
        """Reads the current mouse position with a left click and writes it
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

    def handle_keystrokes(self, key):
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

    def add_refpt(self, x_px, y_px):
        print("add refpt")
        new_refpt_px = {"x_px": x_px, "y_px": y_px}
        self.draw_refpts(temp_refpt=new_refpt_px)
        if new_refpt_utm := self.get_refpt_utm_from_popup():
            new_refpt = {**new_refpt_px, **new_refpt_utm}
            self.refpts = self.append_refpt(refpts=self.refpts, new_refpt=new_refpt)
            self._print_refpts()
        self.draw_refpts()

    def undo_last_refpt(self):

        if self.refpts:
            print("undo last refpt")
            self.refpts, undone_refpt = self.pop_refpt(refpts=self.refpts)
            print(self.refpts)
            print(undone_refpt)
            self.historic_refpts = self.append_refpt(
                refpts=self.historic_refpts, new_refpt=undone_refpt
            )
            self.draw_refpts()
            self._print_refpts()
        else:
            print("refpts empty, cannot undo last refpt")

    def redo_last_refpt(self):
        if self.historic_refpts:
            print("redo last refpt")
            self.historic_refpts, redone_refpt = self.pop_refpt(
                refpts=self.historic_refpts
            )
            self.append_refpt(refpts=self.refpts, new_refpt=redone_refpt)
            self.draw_refpts()
            self._print_refpts()
        else:
            print("no historc refpts, cannot redo last refpt")

    def get_refpt_utm_from_popup(self):
        if not self.popup_root:
            self.popup_root = tk.Tk()
            self.popup_root.overrideredirect(1)
            self.popup_root.withdraw()
        refpt_utm_correct = False
        while not refpt_utm_correct:
            # Get utm part of refpt from pupup
            new_refpt_utm = DialogUTMCoordinates(parent=self.popup_root).coords_utm
            # Check utm part of refpt
            print(new_refpt_utm)
            if new_refpt_utm:  # BUG: ValueError: could not convert string to float: ''
                # TODO: Cancel cancels whole refpt (even pixel coordinates, otherwise stuck in infinite loop)
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
                break
        return new_refpt_utm

    # ----------- Edit refpts dict -----------

    def append_refpt(self, refpts, new_refpt):
        print("append refpts")
        new_idx = len(refpts) + 1
        refpts[new_idx] = new_refpt
        return refpts

    def pop_refpt(self, refpts):
        print("pop refpts")
        popped_refpt = refpts.popitem()[1]
        return refpts, popped_refpt

    # ----------- Draw on image -----------

    def draw_magnifier(self, x_px, y_px):
        # TODO
        print("draw magnifier")

    def zoom_magnifier(self, x_px, y_px):
        # TODO
        print("zoom magnifier")

    def draw_refpts(self, temp_refpt=None):
        FONT = cv2.FONT_ITALIC
        FONT_SIZE_REL = 0.02
        MARKER_SIZE_REL = 0.02
        FONT_SIZE_PX = round(self.base_image.shape[0] * FONT_SIZE_REL)
        MARKER_SIZE_PX = round(self.base_image.shape[0] * MARKER_SIZE_REL)
        print("draw refpts")
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
        print("write refpts")
        # Only necessary if the reference points picker is used as standalone tool
        with open(self.refpts_path, "w") as f:
            json.dump(self.refpts, f, indent=4)

    def write_image(self):
        """This is done via on-board resources of OpenCV"""
        # Alternative: Own implementation using cv2.imwrite()
        # Problem: Ctrl+S shortcut is occupied by this
        pass

    # ----------- Helper function -----------

    def _print_refpts(self):
        print("-------------------------")
        print("refpts:")
        print(self.refpts)
        print("-------------------------")
        print("historic refpts:")
        print(self.historic_refpts)
        print("-------------------------")


def main(title, image_path=None, video_path=None):
    logging.info("Start picker")
    return ReferencePointsPicker(
        title=title, image_path=image_path, video_path=video_path
    ).refpts


class DialogUTMCoordinates(Dialog):
    def __init__(self, **kwargs):
        self.coords_utm = None
        super().__init__(**kwargs)

    # root = tk.Tk()
    # root.overrideredirect(1)
    # root.withdraw()
    # coords = DialogUTMCoordinates(root)

    def body(self, master):

        # Labels
        tk.Label(master, text="Provide reference point\nin UTM coordinates!").grid(
            row=0, columnspan=3
        )
        tk.Label(master, text="UTM zone:").grid(row=1, sticky="e")
        tk.Label(master, text="Longitude (E):").grid(row=2, sticky="e")
        tk.Label(master, text="Latitude (N):").grid(row=3, sticky="e")

        # Zone
        zones = list(range(1, 61))
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

        # Coordinates
        # vcmd_lon = (
        #     self.register(self.validate_lon),
        #     "%d",
        #     "%i",
        #     "%P",
        #     "%s",
        #     "%S",
        #     "%v",
        #     "%V",
        #     "%W",
        # )
        # ivcmd_lon = (self.register(self.on_invalid),)
        # vcmd_lat = (
        #     self.register(self.validate_lat),
        #     # "%d",
        #     # "%i",
        #     "%P",
        #     # "%s",
        #     # "%S",
        #     # "%v",
        #     # "%V",
        #     # "%W",
        # )
        self.input_lon = tk.Entry(
            master, width=5
        )  # , validate="focusout", validatecommand=vcmd_lon
        self.input_lat = tk.Entry(
            master, width=5
        )  # , validate="focusout", validatecommand=vcmd_lat
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
        print(self.coords_utm)

    # def validate_lon(self, d, i, P, s, S, v, V, W):
    #     print("end", "OnValidate:\n")
    #     print("end", f"d='{d}'\n")
    #     print("end", f"i='{i}'\n")
    #     print("end", f"P='{P}'\n")
    #     print("end", f"s='{s}'\n")
    #     print("end", f"S='{S}'\n")
    #     print("end", f"v='{v}'\n")
    #     print("end", f"V='{V}'\n")
    #     print("end", f"W='{W}'\n")
    #     try:
    #         lon = float(P)
    #         in_range = 100000 < lon < 900000
    #         print("In Range:")
    #         print(in_range)
    #         return in_range
    #     except ValueError:
    #         print("ValueError")
    #         return False

    # def on_invalid_lon(self):
    #     pass

    # def validate_lat(self, P):
    #     try:
    #         lat = float(P)
    #         return 0 < lat < 10000000
    #     except ValueError:
    #         return False


class NoPathError(Exception):
    pass


class ImageWontOpenError(Exception):
    pass


class VideoWontOpenError(Exception):
    pass


class FrameNotAvailableError(Exception):
    pass


if __name__ == "__main__":
    # main(title="My refpt picker", image_path=r"tests\data\Radeberg_CamView.png")
    main(
        title="My refpt picker",
        video_path=r"tests\data\Testvideo_FR20_Cars-Cyclist.mp4",
    )
