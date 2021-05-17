# %%
# Add OTVision to path
from pathlib import Path
import sys

otvision_path = str(Path(__file__).parents[1] / "OTVision")
print(otvision_path)
sys.path.insert(1, otvision_path)

# %%
# Get test data folder
from config import CONFIG

testdatafolder = CONFIG["TESTDATAFOLDER"]
testdatafolder

# %%
# track
from track import track, iou

detections_file = str(Path(testdatafolder) / "Testvideo_FR20_Cars-Cyclist.otdet")
# detections_file = r"V:\Stud_Arbeiten\DA_Kollascheck\relevante Videos\Radeberg\raspberrypi_FR20_2020-02-20_12-00-00.otdet"
detections, dir, filename = track.read(detections_file)
new_detections, tracks_finished, vehIDs_finished = iou.track_iou(
    detections["data"], t_min=0, sigma_h=0.5
)

# %%
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

# %%
import json
from pathlib import Path

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
