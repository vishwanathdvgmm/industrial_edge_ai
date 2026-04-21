import cv2
import numpy as np
import uuid
from pathlib import Path

FRAMES_DIR = Path("data/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = (640, 640)

def preprocess_frame(bgr_frame: np.ndarray, camera_id: str = "cam0") -> tuple[np.ndarray, str]:
    """
    Resize to 640x640, convert BGR→RGB, save as lossless PNG.
    Returns (normalized_rgb_array, saved_png_path)
    """
    resized = cv2.resize(bgr_frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    fname = f"{camera_id}_{uuid.uuid4().hex[:8]}.png"
    path = FRAMES_DIR / fname
    # Save in BGR (OpenCV convention) — PNG is lossless so no artifact risk
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    normalized = rgb.astype(np.float32) / 255.0
    return normalized, str(path)

def draw_detections(bgr_frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on frame for dashboard streaming."""
    frame = bgr_frame.copy()
    h, w = frame.shape[:2]
    sx, sy = w / 640, h / 640

    for det in detections:
        x1, y1, x2, y2 = (int(det["bbox"][0] * sx), int(det["bbox"][1] * sy),
                           int(det["bbox"][2] * sx), int(det["bbox"][3] * sy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (79, 156, 249), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 156, 249), 1, cv2.LINE_AA)
    return frame

def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    """Encode BGR frame as JPEG bytes for WebSocket streaming."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
