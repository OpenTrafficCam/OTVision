"""
Module to call yolov5/detect.py with arguments
"""


# Copyright (C) 2020 OpenTrafficCam Contributors
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
# GNU General Public License for more detectionsails.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch
import os
import pathlib


def detect_yolov5(
    input_path,
    output_folder,
    output_subfolder="run",
    overwrite=False,
    classes=[],
    detect_path=None,
    weights="yolov5s.pt",
    download_all_weights_again=False,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    inference_size=640,
    augment=False,
    save_txt=True,
    save_conf=True,
    display_results=False,
):
    """[Perform detection in images, videos or streams using yolov5]

    Args:
        input_path (str): [folder or file or URL of input image(s) or videos(s)]
        output_folder (str): [folder where subfolders with results will be stored]
        output_subfolder (str, optional): [Output folder path]. Defaults to "run".
        overwrite (bool, optional): [Overwrite output folder ]. Defaults to False.
        classes (list, optional): [List of classes to detect]. Defaults to [].
        detect_path (str, optional): [path to yolov5/detect.py]. Defaults to None
        weights (str, optional): [weights file to use]. Defaults to "yolov5s.pt".
        download_all_weights_again (bool, optional): [Just in case]. Defaults to False.
        confidence_threshold (float, optional): [Can be left low]. Defaults to 0.25.
        iou_threshold (float, optional): [We have to try it out]. Defaults to 0.45.
        inference_size (int, optional): [Image downsizing]. Defaults to 640.
        augment (bool, optional): [Have to read in paper about this]. Defaults to False.
        save_txt (bool, optional): [Whether or not to save labels?]. Defaults to True.
        save_conf (bool, optional): [Save confidences with labels?]. Defaults to True.
        display_results (bool, optional): [Doesnt work by now]. Defaults to False.
    """

    # Construct path to detect.py if none was given
    if detect_path is None:
        OpenTrafficCam_path = pathlib.Path(__file__).parents[3]
        detect_path = os.fspath(OpenTrafficCam_path) + r"\yolov5\detect.py"
    print("Path to yolov5/detect.py: " + detect_path)

    # Convert arguments to yolov5 inference format
    if download_all_weights_again:
        download_all_weights_again_str = " --update "
    else:
        download_all_weights_again_str = ""
    if augment:
        augment_str = " --augment "
    else:
        augment_str = ""
    if overwrite:
        overwrite_str = " --exist-ok "
    else:
        overwrite_str = ""
    if save_txt:  # !Bug: Saves single text for every frame
        save_txt_str = " --save-txt "
    else:
        save_txt_str = ""
    if save_conf:
        save_conf_str = " --save-conf "
    else:
        save_conf_str = ""
    if display_results:
        display_results_str = " --view-img "
    else:
        display_results_str = ""

    # Transfer classes to yolo class numbers
    classes_dict = {"Person": 0, "Car": 2}
    classes_str = ""
    for i, class_ in enumerate(classes):
        print(class_)
        if i == 0:
            classes_str += " --classes"
        classes_str += " " + str(classes_dict.get(class_))

    # Inform about usage of GPU no. or CPU
    print(
        "Setup complete. Using torch %s %s"
        % (
            torch.__version__,
            torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU",
        )
    )

    # Call the detect.py module in yolov5 package with custom arguments
    os.system(
        "python "
        + detect_path
        + classes_str
        + " --weights "
        + weights
        + download_all_weights_again_str
        + " --img-size "
        + str(inference_size)
        + " --conf "
        + str(confidence_threshold)
        + " --iou-thres "
        + str(iou_threshold)
        + augment_str
        + " --source "
        + input_path
        + " --project "
        + output_folder
        + " --name "
        + output_subfolder
        + overwrite_str
        + save_txt_str
        + save_conf_str
        + display_results_str
    )


if __name__ == "__main__":
    input_path = r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2021-01-23_yolov5\videos"
    output_folder = r"C:\Users\Baerwolff\Desktop\Lenovo_Arbeit\2021-01-23_yolov5\videos"
    output_subfolder = "detections"
    weights = "yolov5s.pt"
    classes = ["Person", "Car"]
    detect_yolov5(
        input_path=input_path,
        output_folder=output_folder,
        output_subfolder=output_subfolder,
        weights=weights,
        classes=classes,
    )
