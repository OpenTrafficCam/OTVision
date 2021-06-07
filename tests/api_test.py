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

detections_file_in = (Path(testdatafolder) / testdatafilename).with_suffix(".otdet")
detections_file_out = (Path(testdatafolder) / testdatafilename).with_suffix(".ottrk")

detections_in, _, _ = track.read(detections_file_in)
detections_out, tracks_finished, vehIDs_finished = iou.track_iou(
    detections_in["data"], t_min=0, sigma_h=0.5
)

# TODO: This should probably be fixed otherwise, such that the output
#       data of track_iou is harmonized with the input of track.write()
#       and that wrapping of new_detections into a data field is obsolete
detections_out = {"data": detections_out}
track.write(detections_out, detections_file_out)

tracks_file = str(detections_file_out)
with open(tracks_file) as f:
    tracks = json.load(f)
    frames = tracks["data"].keys()
    print("Successfully read in {} frames with indices: {}"
          .format(len(frames), ' '.join(frames)))
