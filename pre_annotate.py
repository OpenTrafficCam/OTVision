from OTVision import pre_annotation

if __name__ == "__main__":
    file_path = (
        "/Users/michaelheilig/Downloads/task_800x600_cloudy_h5m_aov60deg_"
        + "intersection_priority_mondercangeintersection5-2022_08_19_10_23_31"
        + "-yolo 1.1.zip"
    )
    model_weights = "yolov5s.pt"
    chunk_size = 200
    classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
    }

    pre_annotation.main(file_path, model_weights, chunk_size, classes)
