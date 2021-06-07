import sys
import json
from pathlib import Path

try:
    testsPath = Path(__file__).parent.resolve()
    otvision_path = testsPath.parent.joinpath("OTVision")
    sys.path.append(str(otvision_path))

    from config import CONFIG
    from track import track, iou

except Exception as e:
    print(str(e))
    sys.exit("Could not import required OTVision modules")

testdatafolder = CONFIG["TESTDATAFOLDER"]
detections_file = testsPath.joinpath("data/Testvideo_FR20_Cars-Cyclist.otdet")

detections, dir, filename = track.read(detections_file)
new_detections, tracks_finished, vehIDs_finished = iou.track_iou(
    detections["data"], t_min=0, sigma_h=0.5
)

# TODO: This should probably be fixed otherwise, such that the output
#       data of track_iou is harmonized with the input of track.write()
#       and that wrapping of new_detections into a data field is obsolete
new_detections = {"data": new_detections}
track.write(new_detections, Path(detections_file).with_suffix(".ottrk"))

# %%
new_detections

# %%
tracks_finished

# %%
vehIDs_finished

# %%
CONFIG["TRACK"]["IOU"]

# %%
tracks_file = Path(detections_file).with_suffix(".ottrk")
track.write(new_detections, tracks_file)

tracks_file = str(Path(testdatafolder) / "Testvideo_FR20_Cars-Cyclist.ottrk")
print(tracks_file)
with open(tracks_file) as f:
    tracks = json.load(f)
print(tracks["data"].keys())

# %%
def get_three_names():
    name1 = "Alex"
    name2 = "Betty"
    name3 = "Caden"
    return name1, name2, name3
names = get_three_names()
print(names)
print(type(names))

# %%
