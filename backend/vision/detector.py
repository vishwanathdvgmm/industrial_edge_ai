import os
import numpy as np
from ultralytics import YOLO
from vision.preprocess import preprocess_frame
from agent.state import Detection

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))

_model: YOLO | None = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def run_inference(
    bgr_frame: np.ndarray, camera_id: str = "cam0"
) -> tuple[list[Detection], str, np.ndarray]:
    """
    Run YOLOv8 inference on a BGR frame.
    Returns (filtered_detections, saved_frame_path, annotated_bgr_frame)
    Only detections with confidence >= CONF_THRESHOLD are returned.
    """
    model = get_model()
    _, frame_path = preprocess_frame(bgr_frame, camera_id)

    results = model(bgr_frame, conf=CONF_THRESHOLD, verbose=False)[0]

    detections: list[Detection] = []
    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            class_id = int(box.cls[0])
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": conf,
            })

    annotated = results.plot()
    return detections, frame_path, annotated
