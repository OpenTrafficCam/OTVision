# -*- coding: utf-8 -*-
import tkinter as tk


class FrameRefptsCanvas(tk.LabelFrame):
    def __init__(self, image, **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.layout()

    def layout(self):
        self.canvas = tk.Canvas(master=self, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")


class FrameRefptsControls(tk.LabelFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input


def debug():
    window = tk.Tk()
    image = tk.PhotoImage(file=r"tests\data\Radeberg_CamView.png")
    frame_refpts_canvas = FrameRefptsCanvas(master=window, image=image)
    frame_refpts_canvas.pack()
    window.mainloop()


if __name__ == "__main__":
    debug()
