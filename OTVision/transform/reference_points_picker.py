import json
import logging
from pathlib import Path
from random import randrange

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

    def __init__(self, title=None, image_path=None, video_path=None):

        # Attributes
        self.title = title or "Click reference points"
        self.left_button_down = False
        self.refpts = {}
        self.image_path = image_path
        self.video_path = video_path
        self.refpts_path = Path(image_path or video_path).with_suffix(".otrfpts")
        self.image = None
        self.video = None
        self.update_base_image()
        self.historic_refpts = {}

        # Initial method calls
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

    def handle_mouse_events(self, event, x, y, flags, params):
        """Reads the current mouse position with a left click and writes it
        to the end of the array refpkte and increases the counter by one"""
        if event == cv2.EVENT_LBUTTONUP:
            self.left_button_down = False
            self.add_refpt(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_down = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.left_button_down:
                self.draw_magnifier(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.left_button_down:
                self.zoom_magnifier(x, y)

    def handle_keystrokes(self, key):
        # TODO
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

    def add_refpt(self, x, y):
        print("add refpt")
        self.refpts = self.append_refpt(refpts=self.refpts, x=x, y=y)
        self.draw_refpts()
        self._print_refpts()

    def undo_last_refpt(self):

        if self.refpts:
            print("undo last refpt")
            self.refpts, undone_refpt = self.pop_refpt(refpts=self.refpts)
            print(self.refpts)
            print(undone_refpt)
            self.historic_refpts = self.append_refpt(
                refpts=self.historic_refpts,
                x=undone_refpt["x_px"],
                y=undone_refpt["y_px"],
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
            self.add_refpt(x=redone_refpt["x_px"], y=redone_refpt["y_px"])
        else:
            print("no historc refpts, cannot redo last refpt")

    # ----------- Edit refpts dict -----------

    def append_refpt(self, refpts, x, y):
        print("append refpts")
        new_idx = len(refpts) + 1
        refpts[new_idx] = {"x": x, "y": y}
        return refpts

    def pop_refpt(self, refpts):
        print("pop refpts")
        popped_refpt = refpts.popitem()[1]
        return refpts, popped_refpt

    # ----------- Draw on image -----------

    def draw_magnifier(self, x, y):
        # TODO
        print("draw magnifier")

    def zoom_magnifier(self, x, y):
        # TODO
        print("zoom magnifier")

    def draw_refpts(self):
        FONT = cv2.FONT_ITALIC
        FONT_SIZE_REL = 0.02
        MARKER_SIZE_REL = 0.02
        FONT_SIZE_PX = round(self.base_image.shape[0] * FONT_SIZE_REL)
        MARKER_SIZE_PX = round(self.base_image.shape[0] * MARKER_SIZE_REL)
        print("draw refpts")
        self.image = self.base_image.copy()
        for idx, refpt in self.refpts.items():
            x = refpt["x_px"]
            y = refpt["y_px"]
            marker_bottom = (x, y + MARKER_SIZE_PX)
            marker_top = (x, y - MARKER_SIZE_PX)
            marker_left = (x - MARKER_SIZE_PX, y)
            marker_right = (x + MARKER_SIZE_PX, y)
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
