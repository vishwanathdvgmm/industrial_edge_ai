# Industrial Edge AI: v1.0 Production Stabilization Plan

**Deadline:** April 23, 2026 | **Current Status:** Beta (single-camera, partially stable)

This plan outlines the concrete, prioritized changes required to reach a stable, submission-ready v1.0. Items are ordered by impact — fix the most user-visible bugs first.

---

## Current Known Issues (Priority Ordered)

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Live event panel requires page reload | High — UX broken | ✅ Fixed (ObjectId serialization) |
| 2 | "Root cause analysis inconclusive" (JSON hallucination) | High — broken output | ⚠️ Partial fix (regex fallback) |
| 3 | Video feed freezes for 2–4s on detection | High — unstable | ✅ Fixed (ProcessPool for PDF) |
| 4 | Timezone shows UTC instead of local time | Medium — confusing | ✅ Fixed (frontend locale parsing) |
| 5 | WebSocket zombie on disconnect | Medium — crashes server | ✅ Fixed |
| 6 | Old frames in PDF reports (buffer latency) | Medium — wrong data | ✅ Fixed (CameraStream thread) |
| 7 | Single camera only | Medium — hackathon demo OK | ⬜ Planned for v1.0 |
| 8 | LLM network drop crashes agent task silently | Medium | ✅ Fixed (try/except fallback) |
| 9 | Heavy load / high FPS causes browser WS crash | Medium | ✅ Fixed (SAMPLE_FPS tuning docs) |

---

## Phase 1 — LLM Stability (Root Cause Analysis) 🔥 HIGHEST PRIORITY

**Goal:** Eliminate "Root cause analysis inconclusive" and JSON parse errors permanently.

### 1.1 — Structured Outputs via Pydantic

Replace the current error-prone `json.loads()` + regex fallback in all three LLM nodes with LangChain's `.with_structured_output()`. This enforces schema at the API level — the LLM is forced to return valid JSON or the library retries automatically.

**Files to modify:**
- `backend/agent/nodes/classifier.py`
- `backend/agent/nodes/root_cause.py`
- `backend/agent/nodes/action.py`

**Pattern:**
```python
from pydantic import BaseModel, Field

class ClassificationOutput(BaseModel):
    defect_type: str = Field(description="One of: SCRATCH, CRACK, RUST, DENT, HOLE, DISCOLORATION, DEFORMATION, CONTAMINATION")
    severity: str = Field(description="One of: CRITICAL, MEDIUM, LOW")
    zone: str = Field(description="One of: SURFACE, EDGE, JOINT, CORE, COATING")
    confidence: float = Field(ge=0.0, le=1.0)

# In classify_defect():
structured_llm = llm.with_structured_output(ClassificationOutput)
result = structured_llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
```

### 1.2 — Enriched Few-Shot Prompts

Add 2–3 real industrial examples to the system prompt in each node so the LLM has context for ambiguous camera input (dark images, partial objects, etc.).

---

## Phase 2 — Multi-Camera Support

**Goal:** Support N simultaneous camera streams from the same FastAPI server.

### 2.1 — Backend: Dictionary-based Pipeline Manager

Replace the single global `pipeline_running: bool` and `pipeline_task` with:

```python
# main.py
active_pipelines: dict[str, asyncio.Task] = {}
```

Update `/pipeline/start` to accept `camera_url` and `camera_id` as parameters:

```python
@app.post("/pipeline/start")
async def start_pipeline(camera_id: str, line_id: str, camera_url: str = "auto"):
    if camera_id in active_pipelines and not active_pipelines[camera_id].done():
        return {"status": "already_running", "camera_id": camera_id}
    active_pipelines[camera_id] = asyncio.create_task(
        run_pipeline_loop(camera_id, line_id, camera_url)
    )
    return {"status": "started", "camera_id": camera_id}
```

Update `run_pipeline_loop` and `frame_generator` to accept `camera_url` as a direct argument instead of reading from `.env`.

### 2.2 — Frontend: Multi-Camera Grid

Update `LiveFeed.jsx` to render a responsive CSS grid that shows all active camera feeds simultaneously. Each `frame` WebSocket message includes `camera_id`, which can be used to route the frame to the correct grid cell.

---

## Phase 3 — Performance (Low Latency)

### 3.1 — ONNX Model Export

Convert `yolov8n.pt` to `yolov8n.onnx` for a **2–3x inference speedup** on CPU with zero code changes (Ultralytics handles it automatically):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

Set `YOLO_MODEL=yolov8n.onnx` in `.env`. The `detector.py` already handles ONNX if Ultralytics finds the file.

### 3.2 — Agent Cooldown / Debounce

Currently, if the same object stays in frame, the agent fires on every detection cycle. Add a per-camera, per-class cooldown:

```python
# In run_pipeline_loop
last_agent_trigger: dict[str, float] = {}   # key: f"{camera_id}:{class_name}"
AGENT_COOLDOWN_SEC = 10.0

def should_trigger_agent(camera_id, class_name) -> bool:
    key = f"{camera_id}:{class_name}"
    now = time.time()
    if now - last_agent_trigger.get(key, 0) < AGENT_COOLDOWN_SEC:
        return False
    last_agent_trigger[key] = now
    return True
```

### 3.3 — Async MongoDB with Motor

Replace synchronous PyMongo `run_in_executor` calls with the native async `motor` driver to free up the thread pool for video processing:

```bash
pip install motor
```

```python
from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient(MONGO_URI)
# All await-able — no run_in_executor needed
await col.insert_one(doc)
```

**Files:** `backend/db/mongo.py`, `backend/db/gridfs_helper.py`

### 3.4 — Configurable Worker Pool

Add `MAX_PDF_WORKERS` to `.env` so the ProcessPool scales with available CPU cores:

```env
MAX_PDF_WORKERS=2   # increase for multi-camera servers
```

---

## Phase 4 — Real-World Robustness

### 4.1 — LLM Retry with Exponential Backoff

Replace the single-try LLM call with an automatic retry on network errors:

```python
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    retry=tenacity.retry_if_exception_type(groq.APIConnectionError),
)
def invoke_llm(llm, messages):
    return llm.invoke(messages)
```

### 4.2 — Frame Quality Gate

Before sending a frame to YOLO, check basic quality metrics (brightness, blur) and skip frames that are too dark or too blurry. This prevents YOLO from generating false positives on corrupted or partially-received USB frames.

```python
def is_frame_valid(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness > 20 and sharpness > 50
```

### 4.3 — Health Monitoring Endpoint

Extend `/health` to return per-camera pipeline status, LLM connectivity, and MongoDB connectivity:

```json
{
  "status": "ok",
  "pipelines": { "cam0": "running", "cam1": "stopped" },
  "llm": "connected",
  "mongodb": "connected",
  "uptime_seconds": 3600
}
```

---

## Verification Plan

### After Phase 1 (LLM Stability)
- [ ] Run pipeline for 5 minutes. Zero "Root cause analysis inconclusive" messages in terminal.
- [ ] Verify all three event card fields (type, cause, action) are populated in the UI.

### After Phase 2 (Multi-Camera)
- [ ] Start pipeline with `camera_id=cam0` and `camera_id=cam1` simultaneously.
- [ ] Verify both streams appear in the frontend grid.
- [ ] Verify events from both cameras appear in the event list with correct `camera_id` labels.

### After Phase 3 (Performance)
- [ ] Measure time from object placement to defect event appearing in UI — target < 5 seconds.
- [ ] Run for 30 minutes with 2 cameras. Verify no memory growth, no frame drops, no crashes.

### After Phase 4 (Robustness)
- [ ] Temporarily block internet. Verify pipeline stays alive and shows "AI Network Error" events.
- [ ] Cover camera lens. Verify system handles no-detection frames gracefully without errors.

---

## Submission Checklist (April 23)

- [x] README updated to professional enterprise level
- [x] Single camera pipeline fully operational
- [x] Live dashboard with WebSocket streaming
- [x] PDF report generation with MongoDB storage
- [x] LLM fallback for network errors
- [x] Local timezone display in UI
- [ ] Structured LLM outputs (Phase 1) — **do before submission**
- [ ] Multi-camera support (Phase 2) — time permitting
- [ ] ONNX acceleration (Phase 3.1) — quick win, 1-line change
- [ ] Agent cooldown (Phase 3.2) — recommended

