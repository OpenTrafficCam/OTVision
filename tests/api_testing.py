# %%
# Add OTVision to path
import sys
from pathlib import Path

otvision_path = str(Path(__file__).parents[1] / "OTVision")
print(otvision_path)
sys.path.insert(1, otvision_path)

# %%
# track
from track import track

track.main()

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
