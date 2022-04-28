from pathlib import Path

import cv2

# TODO: from OTVision.config import CONFIG


class RefptsPicker:
    """Class to pick reference points in pixel coordinates for transform subpackage."""

    def __init__(self, title=None, image_path=None):

        # Attributes
        self.title = title or "Click reference points"
        # TODO: Path(CONFIG["TESTDATAFOLDER"]) / r"Radeberg_CamView.png"
        self.image_path = image_path or r"tests\data\Radeberg_CamView.png"
        self.image = cv2.imread(image_path or self.image_path)
        self.refpts = {}

        self.drawing = False

        # Initial method calls
        self.show()

    def show(self):

        cv2.imshow(self.title, self.image)
        cv2.setMouseCallback(self.title, self.get_mouse_events)

        while True:
            # wait for a key press to close the window (0 = indefinite loop)
            key = cv2.waitKey(1)

            # Exit the loop
            window_visible = (
                cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
            )
            if key == 27 or not window_visible:
                break

        cv2.destroyAllWindows()

    def get_mouse_events(self, event, x, y, flags, params):
        """Reads the current mouse position with a left click and writes it
        to the end of the array refpkte and increases the counter by one"""
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.extend_refpts(x, y)
            # cv2.destroyWindow("Lupe")
            self.draw()
            print("Left button up")
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("Left button down")
            self.drawing = True
        # TODO: Magnifier
        # elif event == cv2.EVENT_MOUSEMOVE:
        #     if drawing == True:
        #     self.magnifier(x, y)
        # elif event == cv2.EVENT_MOUSEWHEEL:
        #     if drawing == True:
        # self.zoom_magnifier(x, y)

    def draw(self):
        if not self.drawing:
            print("draw new image")
            cv2.imshow(self.title, self.image)

    def extend_refpts(self, x, y):
        print("new refpts")

    def write_image(self):
        # TODO: cv2.imwrite()
        pass


if __name__ == "__main__":
    refpts_picker = RefptsPicker("My improved refpt picker")
