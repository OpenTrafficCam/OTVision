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
testdatafilename = "Testvideo_FR20_Cars-Cyclist"
detections_file = Path(testdatafolder).joinpath(testdatafilename)

detections, dir, filename = track.read(detections_file.with_suffix(".otdet"))
new_detections, tracks_finished, vehIDs_finished = iou.track_iou(
    detections["data"], t_min=0, sigma_h=0.5
)

# TODO: This should probably be fixed otherwise, such that the output
#       data of track_iou is harmonized with the input of track.write()
#       and that wrapping of new_detections into a data field is obsolete
new_detections = {"data": new_detections}
track.write(new_detections, Path(detections_file).with_suffix(".ottrk"))

tracks_file = str(Path(testdatafolder) / "Testvideo_FR20_Cars-Cyclist.ottrk")
with open(tracks_file) as f:
    tracks = json.load(f)
    frames = tracks["data"].keys()
    print("Successfully read in {} frames with indices: {}"
          .format(len(frames), ' '.join(frames)))
