METADATA: str = "metadata"
OTDET_VERSION: str = "otdet_version"
OTVISION_VERSION: str = "otvision_version"
VIDEO: str = "video"
DETECTION: str = "detection"
TRACK: str = "trk"
FILENAME: str = "filename"
FILETYPE: str = "filetype"
WIDTH: str = "width"
HEIGHT: str = "height"
RECORDED_FPS: str = "recorded_fps"
FRAMES: str = "frames"
RECORDED_START_DATE: str = "recorded_start_date"
LENGTH: str = "length"
NUMBER_OF_FRAMES: str = "number_of_frames"

DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
INPUT_FILE_PATH: str = "input_file_path"
DATA: str = "data"
DETECTIONS: str = "detections"
CLASS: str = "class"
CLASSIFIED: str = "classified"
FRAME: str = "frame"
OCCURRENCE: str = "occurrence"
LABEL: str = "label"
CONFIDENCE: str = "conf"
X: str = "x"
Y: str = "y"
W: str = "w"
H: str = "h"

TRACK_ID: str = "track-id"

# Detector config
MODEL: str = "model"
CHUNKSIZE: str = "chunksize"
NORMALIZED_BBOX: str = "normalized_bbox"
# Detektor model config
NAME: str = "name"
WEIGHTS: str = "weights"
IOU_THRESHOLD: str = "iou"
IMAGE_SIZE: str = "image_size"
MAX_CONFIDENCE: str = "max_confidence"
HALF_PRECISION: str = "half_precision"
CLASSES: str = "classes"

# Tracker config
TRACKER: str = "tracker"
SIGMA_L: str = "sigma_l"
SIGMA_H: str = "sigma_h"
SIGMA_IOU: str = "sigma_iou"
T_MIN: str = "t_min"
T_MISS_MAX: str = "t_miss_max"

# iou
BBOXES: str = "bboxes"
CENTER: str = "center"
AGE: str = "age"
MAX_CLASS: str = "max_class"
MAX_CONF: str = "max_conf"
FIRST: str = "first"
FINISHED: str = "finished"
START_FRAME: str = "start_frame"
