import asyncio
import cv2
import os
import numpy as np
from typing import AsyncGenerator

def _open_cap(url: str) -> cv2.VideoCapture:
    if str(url).lower() == "auto":
        # Auto-detect the first working camera module
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Auto-detected working camera at index {i}")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                cap.release()
        raise RuntimeError("Auto-detect failed: No working camera module found.")

    try:
        source = int(url)
    except ValueError:
        source = url  # e.g., http://192.168.1.5:4747/video (DroidCam)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if isinstance(source, str) else cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, str) and "4747" in source:
        # Retry with DroidCam specific pure MJPEG endpoint
        alt_source = source + "/force" if not source.endswith("/force") else source
        cap = cv2.VideoCapture(alt_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

import threading
import time

class CameraStream:
    """Continuously pulls frames in a background thread to prevent OpenCV buffer latency."""
    def __init__(self, url):
        self.cap = _open_cap(url)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                # Read as fast as possible to keep the OpenCV buffer empty
                self.ret, self.frame = self.cap.read()
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

async def frame_generator(camera_id: str = "cam0") -> AsyncGenerator[np.ndarray, None]:
    """
    Async generator yielding BGR frames at SAMPLE_FPS.
    Reads from a dedicated CameraStream thread to guarantee zero latency.
    """
    loop = asyncio.get_event_loop()
    
    # Forcefully bypass os.environ cache and read directly from .env file
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    camera_url = "0"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('CAMERA_URL='):
                    camera_url = line.split('=', 1)[1].strip().strip('"').strip("'")
    
    sample_fps = int(os.getenv("SAMPLE_FPS", "5"))
    interval = 1.0 / sample_fps

    # Open the camera stream in a background thread to prevent frame queuing
    stream = await loop.run_in_executor(None, CameraStream, camera_url)
    
    try:
        # Wait up to 10 seconds for the first frame to arrive over the network
        import time
        start_time = time.time()
        while stream.frame is None:
            if time.time() - start_time > 10.0:
                raise RuntimeError(f"Connected to {camera_url} but failed to read any frames after 10 seconds.")
            await asyncio.sleep(0.1)
        
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                # If camera suddenly drops connection
                print(f"Camera dropped connection: {camera_url}")
                break
            
            # Yield a copy of the latest frame
            yield frame.copy()
            await asyncio.sleep(interval)
    finally:
        stream.release()
