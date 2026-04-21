"""
Industrial Edge AI — FastAPI Backend
Entrypoint: uvicorn main:app --reload
"""
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from agent.graph import agent_graph
from agent.state import AgentState
from db.mongo import (
    insert_defect_event,
    get_defect_history,
    get_events_paginated,
    get_trend_data,
    get_system_config,
)
from db.gridfs_helper import store_frame_file, store_pdf, get_file_bytes
from pdf.generator import generate_pdf
from vision.capture import frame_generator
from vision.detector import run_inference
from vision.preprocess import draw_detections, encode_jpeg

# ── WebSocket Connection Manager ───────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception as e:
                print(f"WebSocket send failed: {e}. Dropping client.")
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)
            try:
                await ws.close()
            except Exception:
                pass

manager = ConnectionManager()
pipeline_running = False
pipeline_task: asyncio.Task | None = None

# ── Pipeline Loop ──────────────────────────────────────────────────────────────

import concurrent.futures

_pdf_process_pool = None

def get_pdf_pool():
    global _pdf_process_pool
    if _pdf_process_pool is None:
        _pdf_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    return _pdf_process_pool

async def process_defect_event(initial_state: AgentState, camera_id: str, line_id: str):
    """Background task to run the LangGraph agent and save results to DB without blocking the video feed."""
    loop = asyncio.get_event_loop()
    try:
        # ── Step 4: Run agent ──
        try:
            result: AgentState = await loop.run_in_executor(None, agent_graph.invoke, initial_state)
        except Exception as e:
            print(f"⚠️ LLM Network Error (Groq): {e}. Using safe fallback values.")
            result = initial_state
            # Assign default values so the pipeline can continue safely
            result["defect_type"] = initial_state["raw_detections"][0]["class_name"].upper()
            result["severity"] = "LOW"
            result["zone"] = "UNKNOWN"
            result["confidence"] = initial_state["raw_detections"][0]["confidence"]
            result["cause_hypothesis"] = "AI Network Error: Failed to reach LLM API. Check connection."
            result["action"] = "LOG_ONLY"
            result["action_rationale"] = "Network error during analysis."
            result["report_payload"] = {}
            result["event_id"] = f"err_{int(loop.time())}"

        # ── Step 5: Store image in GridFS ──
        image_id = await loop.run_in_executor(
            None, store_frame_file, initial_state["frame_path"], result.get("event_id", "unknown")
        )

        # ── Step 6: Generate + store PDF (MUST use ProcessPool to avoid GIL freeze) ──
        report = result.get("report_payload", {})
        pdf_bytes = await loop.run_in_executor(get_pdf_pool(), generate_pdf, report)
        pdf_id = await loop.run_in_executor(
            None, store_pdf, pdf_bytes, f"{result.get('event_id','')}.pdf",
            {"event_id": result.get("event_id")}
        )

        # ── Step 7: Persist defect event ──
        doc = {
            "event_id": result.get("event_id"),
            "timestamp": result.get("timestamp"),
            "camera_id": camera_id,
            "line_id": line_id,
            "defect_type": result["defect_type"],
            "severity": result["severity"],
            "zone": result["zone"],
            "confidence": result["confidence"],
            "cause_hypothesis": result["cause_hypothesis"],
            "cause_confidence": result.get("cause_confidence", 0.5),
            "action": result["action"],
            "action_rationale": result["action_rationale"],
            "image_gridfs_id": image_id,
            "pdf_gridfs_id": pdf_id,
            "report_payload": report,
        }
        await loop.run_in_executor(None, insert_defect_event, doc)

        # ── Step 8: Broadcast event to dashboard ──
        await manager.broadcast({
            "type": "event",
            "data": {k: v for k, v in doc.items() if k not in ("report_payload", "_id")},
        })
    except Exception as e:
        import traceback
        print(f"\n❌ AGENT TASK CRASHED: {e}")
        traceback.print_exc()

async def run_pipeline_loop(camera_id: str = "cam0", line_id: str = "line_01"):
    """
    Continuously:
      1. Read frame from camera
      2. Run YOLOv8 inference
      3. Broadcast annotated frame over WebSocket
      4. Trigger agent in background (if not already running)
    """
    global pipeline_running
    config = get_system_config()
    agent_task = None

    try:
        async for bgr_frame in frame_generator(camera_id):
            if not pipeline_running:
                break

            # ── Step 1: Inference ──
            loop = asyncio.get_event_loop()
            detections, frame_path, annotated = await loop.run_in_executor(
                None, run_inference, bgr_frame, camera_id
            )

            # ── Step 2: Stream annotated frame ──
            jpeg_bytes = encode_jpeg(annotated)
            await manager.broadcast({
                "type": "frame",
                "camera_id": camera_id,
                "frame": base64.b64encode(jpeg_bytes).decode(),
                "detection_count": len(detections),
            })

            # ── Step 3: Gate check ──
            if not detections:
                continue

            # ── Step 4: Fire background agent task (Debounced) ──
            if agent_task is not None and not agent_task.done():
                continue  # Skip if we are currently generating a report

            initial_state: AgentState = {
                "frame_path": frame_path,
                "camera_id": camera_id,
                "line_id": line_id,
                "raw_detections": detections,
                "defect_type": "",
                "severity": "",
                "confidence": 0.0,
                "zone": "",
                "cause_hypothesis": "",
                "cause_confidence": 0.0,
                "action": "",
                "action_rationale": "",
                "report_payload": None,
                "image_gridfs_id": None,
                "pdf_gridfs_id": None,
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": None,
            }

            agent_task = asyncio.create_task(process_defect_event(initial_state, camera_id, line_id))
    except Exception as e:
        import traceback
        print(f"\n❌ PIPELINE CRASHED: {e}")
        traceback.print_exc()
    finally:
        pipeline_running = False

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global pipeline_running, pipeline_task
    pipeline_running = False
    if pipeline_task:
        pipeline_task.cancel()


# ── App Init ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Industrial Edge AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pipeline": pipeline_running}


@app.post("/pipeline/start")
async def start_pipeline(camera_id: str = "cam0", line_id: str = "line_01"):
    global pipeline_running, pipeline_task
    if pipeline_running:
        return {"status": "already_running"}
    pipeline_running = True
    pipeline_task = asyncio.create_task(run_pipeline_loop(camera_id, line_id))
    return {"status": "started"}


@app.post("/pipeline/stop")
async def stop_pipeline():
    global pipeline_running, pipeline_task
    pipeline_running = False
    if pipeline_task:
        pipeline_task.cancel()
        pipeline_task = None
    return {"status": "stopped"}


class DetectRequest(BaseModel):
    camera_id: str = "cam0"
    line_id: str = "line_01"
    image_base64: Optional[str] = None   # optional: send frame from client


@app.post("/detect")
async def detect_once(req: DetectRequest, background_tasks: BackgroundTasks):
    """
    Single-shot detection endpoint.
    If image_base64 provided: decode and run pipeline on it.
    Otherwise: grab one frame from the camera.
    """
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Grab a single frame from camera using the robust capture module
        from vision.capture import _open_cap
        loop = asyncio.get_event_loop()
        def _get_one_frame():
            cap = _open_cap(os.getenv("CAMERA_URL", "auto"))
            ret, frame = cap.read()
            cap.release()
            return ret, frame
            
        ret, bgr_frame = await loop.run_in_executor(None, _get_one_frame)
        if not ret:
            raise HTTPException(status_code=503, detail="Camera unavailable")

    loop = asyncio.get_event_loop()
    detections, frame_path, annotated = await loop.run_in_executor(
        None, run_inference, bgr_frame, req.camera_id
    )

    if not detections:
        return {"status": "no_defect", "detections": 0}

    initial_state: AgentState = {
        "frame_path": frame_path,
        "camera_id": req.camera_id,
        "line_id": req.line_id,
        "raw_detections": detections,
        "defect_type": "", "severity": "", "confidence": 0.0, "zone": "",
        "cause_hypothesis": "", "cause_confidence": 0.0,
        "action": "", "action_rationale": "",
        "report_payload": None, "image_gridfs_id": None, "pdf_gridfs_id": None,
        "timestamp": datetime.utcnow().isoformat(), "event_id": None,
    }

    result: AgentState = await loop.run_in_executor(None, agent_graph.invoke, initial_state)

    image_id = await loop.run_in_executor(
        None, store_frame_file, frame_path, result.get("event_id", "")
    )
    report = result.get("report_payload", {})
    pdf_bytes = await loop.run_in_executor(None, generate_pdf, report)
    pdf_id = await loop.run_in_executor(
        None, store_pdf, pdf_bytes, f"{result.get('event_id','')}.pdf", {}
    )

    doc = {
        "event_id": result.get("event_id"),
        "timestamp": result.get("timestamp"),
        "camera_id": req.camera_id,
        "line_id": req.line_id,
        "defect_type": result["defect_type"],
        "severity": result["severity"],
        "zone": result["zone"],
        "confidence": result["confidence"],
        "cause_hypothesis": result["cause_hypothesis"],
        "cause_confidence": result.get("cause_confidence", 0.5),
        "action": result["action"],
        "action_rationale": result["action_rationale"],
        "image_gridfs_id": image_id,
        "pdf_gridfs_id": pdf_id,
        "report_payload": report,
    }
    await loop.run_in_executor(None, insert_defect_event, doc)
    await manager.broadcast({"type": "event", "data": {k: v for k, v in doc.items() if k != "report_payload"}})

    return doc


@app.get("/events")
def events(limit: int = 50, skip: int = 0):
    return get_events_paginated(limit=limit, skip=skip)


@app.get("/trend")
def trend(hours: int = 24):
    return get_trend_data(hours=hours)


@app.get("/config")
def config():
    return get_system_config()


@app.get("/report/{event_id}/pdf")
def download_pdf(event_id: str):
    """Download generated PDF for a defect event."""
    events = get_events_paginated(limit=1, skip=0)
    # Find by event_id
    from db.mongo import get_collection
    col = get_collection("defect_events")
    doc = col.find_one({"event_id": event_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Event not found")
    pdf_id = doc.get("pdf_gridfs_id")
    if not pdf_id:
        raise HTTPException(status_code=404, detail="PDF not generated")
    pdf_bytes = get_file_bytes(pdf_id)
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={event_id}.pdf"})


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive pings
    except WebSocketDisconnect:
        manager.disconnect(ws)
