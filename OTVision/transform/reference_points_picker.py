from pathlib import Path

import cv2

# TODO: from OTVision.config import CONFIG


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

    def __init__(self, title=None, image_path=None):

        # Attributes
        self.title = title or "Click reference points"
        # TODO: Path(CONFIG["TESTDATAFOLDER"]) / r"Radeberg_CamView.png"
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.refpts = {}
        self.historic_refpts = {}

        self.drawing = False

        # Initial method calls
        self.show()

    # ----------- Handle gui -----------

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
                self.return_refpts()
                break  # Exit loop and collapse OpenCV window
            else:
                self.handle_keystrokes(key)

        cv2.destroyAllWindows()

    def handle_mouse_events(self, event, x, y, flags, params):
        """Reads the current mouse position with a left click and writes it
        to the end of the array refpkte and increases the counter by one"""
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.add_refpt(x, y)
            self.update_image()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.draw_magnifier(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.drawing == True:
                self.zoom_magnifier(x, y)

    def handle_keystrokes(self, key):
        # TODO
        if key == 26:
            self.undo_last_refpt()
        elif key == 25:
            self.redo_last_refpt()

    def update_image(self):
        if not self.drawing:
            print("update image")
            cv2.imshow(self.title, self.image)

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
                refpts=self.historic_refpts, x=undone_refpt["x"], y=undone_refpt["y"]
            )
            self._print_refpts()
            self.draw_refpts()
        else:
            print("refpts empty, cannot undo last refpt")

    def redo_last_refpt(self):
        if self.historic_refpts:
            print("redo last refpt")
            self.historic_refpts, redone_refpt = self.pop_refpt(
                refpts=self.historic_refpts
            )
            self.add_refpt(x=redone_refpt["x"], y=redone_refpt["y"])
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
        # TODO
        print("draw refpts")

    # ----------- Complete job -----------

    def return_refpts(self):
        # TODO
        print("return refpts")

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


def main(title, image_path):
    refpts = ReferencePointsPicker(title=title, image_path=image_path).refpts
    print(refpts)
    return refpts


if __name__ == "__main__":
    main(title="My refpt picker", image_path=r"tests\data\Radeberg_CamView.png")
