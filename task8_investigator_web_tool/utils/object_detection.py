from ultralytics import YOLO
import numpy as np

model = None

def load_model(model_path="models/yolo_objects_model/"):
    global model
    model = YOLO(model_path)

def detect_objects(image):
    results = model(image)[0]
    bboxes = []
    masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    class_labels = []
    confidences = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        masks[y1:y2, x1:x2] = 255
        bboxes.append([x1, y1, x2, y2])
        class_labels.append(int(cls))
        confidences.append(float(conf))

    return bboxes, masks, class_labels, confidences
