# Industrial Edge AI — Defect Detection System

<div align="center">
_An end-to-end agentic AI system for real-time industrial defect detection, root cause analysis, and automated reporting._

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://docs.ultralytics.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green?logo=mongodb)](https://www.mongodb.com/)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Camera Configuration](#camera-configuration)
7. [Environment Variables](#environment-variables)
8. [Running the System](#running-the-system)
9. [API Reference](#api-reference)
10. [WebSocket Protocol](#websocket-protocol)
11. [Agent Pipeline Deep Dive](#agent-pipeline-deep-dive)
12. [Project Structure](#project-structure)
13. [Performance Tuning](#performance-tuning)
14. [Troubleshooting](#troubleshooting)

---

## Overview

Industrial Edge AI is a production-grade agentic system that monitors factory camera feeds in real time. When a defect is detected by a computer vision model, a **4-node LangGraph stateful AI agent** automatically:

1. **Classifies** the defect type, severity, and location zone
2. **Reasons** about the most probable root cause using historical context
3. **Recommends** an action (`HALT_LINE`, `FLAG_QC`, or `LOG_ONLY`)
4. **Generates** a full PDF inspection report

All of this happens within seconds. Operators monitor the entire factory floor through a live React dashboard with real-time WebSocket streaming.

> **Current Status:** Beta — Single-camera pipeline fully operational. Multi-camera, ONNX acceleration, and structured LLM output are planned for v1.0 release.

---

## System Architecture

```mermaid
graph TD
    CAM["📷 Camera Source\n(USB / DroidCam / IP)"]
    CS["CameraStream Thread\n(Zero-Latency Buffer)"]
    YOLO["YOLOv8 Inference\nconf ≥ threshold gate"]
    WS_OUT["WebSocket Broadcast\nAnnotated JPEG Frame"]
    AGENT["LangGraph Agent\n(background asyncio Task)"]
    N1["Node 1: Classifier\ntype · severity · zone"]
    N2["Node 2: Root Cause\nhypothesis + DB history"]
    N3["Node 3: Action\nHALT / FLAG / LOG"]
    N4["Node 4: Reporter\nreport_payload"]
    PDF["PDF Generator\n(ProcessPool — no GIL freeze)"]
    MONGO["MongoDB\ndefect_events collection"]
    GRIDFS["GridFS\nFrame PNGs + PDF blobs"]
    REACT["React Dashboard\nLive Feed · Events · Trend Chart"]

    CAM --> CS --> YOLO
    YOLO -->|"every frame"| WS_OUT --> REACT
    YOLO -->|"detection ≥ threshold"| AGENT
    AGENT --> N1 --> N2 --> N3 --> N4
    N4 --> PDF --> GRIDFS
    N4 --> MONGO
    MONGO --> WS_OUT
    WS_OUT --> REACT
```

---

## Technology Stack

| Layer              | Technology                                         | Notes                                |
| ------------------ | -------------------------------------------------- | ------------------------------------ |
| Computer Vision    | YOLOv8n (Ultralytics)                              | Nano model — fastest on CPU          |
| Agent Orchestrator | LangGraph (stateful graph)                         | 4-node pipeline with shared state    |
| LLM                | Groq / Gemini / OpenAI                             | Pluggable via `LLM_PROVIDER` env var |
| Backend            | FastAPI + asyncio                                  | Async WebSocket streaming            |
| Database           | MongoDB 7.0 + GridFS                               | Document store + binary file storage |
| Frontend           | React 18 (Vite) + Recharts                         | Live WebSocket dashboard             |
| PDF Reports        | xhtml2pdf                                          | Pure-Python, no system dependencies  |
| Camera Capture     | OpenCV + Threaded CameraStream                     | Zero-latency buffer draining         |
| Concurrency        | asyncio + ThreadPoolExecutor + ProcessPoolExecutor | CPU-bound PDF in separate process    |

---

## Prerequisites

Ensure the following are installed before you begin:

| Requirement | Version  | Download                                       |
| ----------- | -------- | ---------------------------------------------- |
| Python      | ≥ 3.12   | https://www.python.org/downloads/              |
| Node.js     | ≥ 20 LTS | https://nodejs.org/                            |
| MongoDB     | ≥ 7.0    | https://www.mongodb.com/try/download/community |
| Git         | any      | https://git-scm.com/                           |

> **Windows Note:** Install MongoDB as a Windows Service so it starts automatically. Alternatively, use Docker (see below) or [MongoDB Atlas](https://www.mongodb.com/atlas) (free cloud tier).

---

## Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-org/industrial_edge_ai.git
cd industrial_edge_ai
```

### Step 2 — Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

### Step 3 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` in your editor and fill in your credentials. See the [Environment Variables](#environment-variables) section for a full reference.

**Minimum required:**

```env
GROQ_API_KEY=gsk_your_key_here     # Get free at console.groq.com
CAMERA_URL=0                        # 0 = first USB webcam, or DroidCam index
```

### Step 4 — Frontend Setup

```bash
cd ../frontend
npm install
```

### Step 5 — Start MongoDB

**Option A — Local MongoDB (Windows Service):**

```bash
# MongoDB should already be running as a Windows service after installation.
# Verify with:
net start MongoDB
```

**Option B — Docker:**

```bash
docker run -d -p 27017:27017 --name mongo mongo:7.0
```

**Option C — MongoDB Atlas (Cloud):**
Set `MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/` in your `.env` file.

---

## Camera Configuration

The system supports multiple camera input types. Set `CAMERA_URL` in your `.env`:

| Mode                 | `CAMERA_URL` value                          | Use Case                                    |
| -------------------- | ------------------------------------------- | ------------------------------------------- |
| Auto-detect          | `auto`                                      | First working webcam found automatically    |
| USB Webcam           | `0`, `1`, `2`                               | Indexed system webcam                       |
| DroidCam (USB)       | `1` or `2`                                  | Phone as webcam via DroidCam virtual driver |
| DroidCam (Wi-Fi/USB) | `http://192.168.x.x:4747/video`             | DroidCam over network                       |
| RTSP IP Camera       | `rtsp://user:pass@192.168.1.100:554/stream` | Professional industrial IP camera           |
| MJPEG Stream         | `http://192.168.x.x:8080/video`             | Generic MJPEG stream                        |

### DroidCam Setup (Phone as Camera — Recommended for Testing)

1. Install [DroidCam](https://www.dev47apps.com/) on your Android phone.
2. Install [DroidCam Desktop Client](https://www.dev47apps.com/) on your laptop.
3. Connect your phone via USB and enable USB Debugging.
4. Open the DroidCam desktop app → click **USB** → click **Start**.
5. Set `CAMERA_URL=1` in `.env` (or `2` if index 1 doesn't work).

> **Important:** Close other apps (Windows Phone Link, Teams) that may lock the camera device before starting the pipeline.

---

## Environment Variables

Full reference for `backend/.env`:

```env
# ── LLM Provider ─────────────────────────────────────────────────────────────
# Options: groq | gemini | openai
LLM_PROVIDER=groq

# API Keys — only the key for your chosen provider is required
GROQ_API_KEY=gsk_...           # Free at console.groq.com
# GEMINI_API_KEY=AIza...       # Free at aistudio.google.com
# OPENAI_API_KEY=sk-...

# LLM Model Selection
# Groq:   llama-3.3-70b-versatile | mixtral-8x7b-32768 | llama3-70b-8192
# Gemini: gemini-2.0-flash | gemini-1.5-flash
# OpenAI: gpt-4o-mini | gpt-4o
LLM_MODEL=llama-3.3-70b-versatile

# ── Database ──────────────────────────────────────────────────────────────────
MONGO_URI=mongodb://localhost:27017
MONGO_DB=edge_ai

# ── Computer Vision ───────────────────────────────────────────────────────────
# Model: yolov8n.pt (fastest) | yolov8s.pt (balanced) | yolov8m.pt (accurate)
YOLO_MODEL=yolov8n.pt

# Confidence gate — only detections above this score trigger the AI agent
CONF_THRESHOLD=0.65

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_URL=0

# Frames per second sent to the frontend dashboard
# Recommended: 15-30. Higher values increase bandwidth, not accuracy.
SAMPLE_FPS=15
```

---

## Running the System

Open **two separate terminals**.

### Terminal 1 — Backend

```bash
cd backend
.venv\Scripts\activate      # Windows
# or: source .venv/bin/activate  (Linux/macOS)
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

The dashboard will be available at: `http://localhost:5173`

### Using the Dashboard

1. Open `http://localhost:5173` in your browser.
2. Click **▶ Start Pipeline** in the top-right corner.
3. The live camera feed will appear within ~2 seconds.
4. Point the camera at an object. The AI will detect and classify it.
5. Defect event cards will appear in the **Defect Events** panel in real time.
6. Click any event card to download its full PDF inspection report.
7. Click **■ Stop** to halt the pipeline.

---

## API Reference

### Health

| Method | Endpoint  | Description                        |
| ------ | --------- | ---------------------------------- |
| `GET`  | `/health` | Returns system and pipeline status |

**Response:**

```json
{ "status": "ok", "pipeline": true }
```

### Pipeline Control

| Method | Endpoint                                         | Description                      |
| ------ | ------------------------------------------------ | -------------------------------- |
| `POST` | `/pipeline/start?camera_id=cam0&line_id=line_01` | Start continuous camera pipeline |
| `POST` | `/pipeline/stop`                                 | Stop the active pipeline         |

**Response:**

```json
{ "status": "started" }
```

### Detection

| Method | Endpoint  | Description                     |
| ------ | --------- | ------------------------------- |
| `POST` | `/detect` | Run a single one-shot detection |

**Request Body (optional — omit to capture from camera):**

```json
{
	"camera_id": "cam0",
	"line_id": "line_01",
	"image_base64": "<base64 encoded image string>"
}
```

### Events & Reports

| Method | Endpoint                  | Description                          |
| ------ | ------------------------- | ------------------------------------ |
| `GET`  | `/events?limit=50&skip=0` | Paginated list of defect events      |
| `GET`  | `/trend?hours=24`         | Hourly defect counts for trend chart |
| `GET`  | `/config`                 | Current system configuration         |
| `GET`  | `/report/{event_id}/pdf`  | Download the PDF report for an event |

**Event Object Schema:**

```json
{
	"event_id": "A3F1B2C4",
	"timestamp": "2026-04-21T20:15:32",
	"camera_id": "cam0",
	"line_id": "line_01",
	"defect_type": "CONTAMINATION",
	"severity": "MEDIUM",
	"zone": "SURFACE",
	"confidence": 0.87,
	"cause_hypothesis": "Inadequate cleaning protocol detected...",
	"cause_confidence": 0.85,
	"action": "FLAG_QC",
	"action_rationale": "Medium severity requires quality control inspection.",
	"image_gridfs_id": "...",
	"pdf_gridfs_id": "..."
}
```

---

## WebSocket Protocol

Connect to `ws://localhost:8000/ws` for real-time streaming.

The server sends two types of messages:

### Frame Message (every tick)

```json
{
	"type": "frame",
	"camera_id": "cam0",
	"frame": "<base64 JPEG image>",
	"detection_count": 1
}
```

### Event Message (on defect detection)

```json
{
  "type": "event",
  "data": { ...event object... }
}
```

**Keep-Alive:** Send any text (e.g., `"ping"`) to the WebSocket every 20 seconds to prevent connection timeout.

**Reconnection:** The frontend automatically reconnects within 2 seconds if the connection is lost.

---

## Agent Pipeline Deep Dive

The AI agent is a 4-node **LangGraph stateful graph**. Each node passes enriched state to the next.

```
Camera Frame
    │
    ▼
YOLOv8 (conf ≥ CONF_THRESHOLD gate)
    │  Detection fails gate → continue to next frame
    ▼
Node 1: classify_defect (classifier.py)
    │  Input:  raw_detections, camera_id, line_id, [frame image]
    │  Output: defect_type, severity, zone, confidence
    ▼
Node 2: find_root_cause (root_cause.py)
    │  Input:  defect_type, severity + MongoDB defect history
    │  Output: cause_hypothesis, cause_confidence
    ▼
Node 3: recommend_action (action.py)
    │  Input:  severity (CRITICAL → safety override, no LLM)
    │  Output: action (HALT_LINE | FLAG_QC | LOG_ONLY), action_rationale
    ▼
Node 4: generate_report (reporter.py)
    │  Input:  full agent state
    │  Output: report_payload (structured JSON for PDF)
    ▼
ProcessPool → PDF bytes → GridFS
MongoDB → defect_events collection
WebSocket → React Dashboard
```

### Safety Override

If `severity == CRITICAL`, the system **bypasses the LLM entirely** for Node 3 and issues a `HALT_LINE` command immediately. This is a deterministic hard-coded safety rule to ensure critical defects are always actioned within milliseconds, regardless of network or LLM latency.

### Severity Definitions

| Severity | Meaning                              | Action    |
| -------- | ------------------------------------ | --------- |
| CRITICAL | Structural/safety risk               | HALT_LINE |
| MEDIUM   | Quality issue — may escape to market | FLAG_QC   |
| LOW      | Cosmetic — no functional impact      | LOG_ONLY  |

---

## Project Structure

```
industrial_edge_ai/
├── backend/
│   ├── agent/
│   │   ├── nodes/
│   │   │   ├── classifier.py      # Node 1: Defect type, severity, zone
│   │   │   ├── root_cause.py      # Node 2: Root cause hypothesis
│   │   │   ├── action.py          # Node 3: Action recommendation
│   │   │   └── reporter.py        # Node 4: Report payload builder
│   │   ├── graph.py               # LangGraph graph definition
│   │   └── state.py               # Shared AgentState TypedDict
│   ├── db/
│   │   ├── mongo.py               # PyMongo helpers (events, trend, config)
│   │   └── gridfs_helper.py       # GridFS image/PDF storage
│   ├── pdf/
│   │   ├── generator.py           # xhtml2pdf report generator
│   │   └── template.html          # Jinja2 HTML template for reports
│   ├── vision/
│   │   ├── capture.py             # CameraStream thread + frame_generator
│   │   ├── detector.py            # YOLOv8 inference wrapper
│   │   └── preprocess.py          # Frame annotation + JPEG encoding
│   ├── llm.py                     # Pluggable LLM provider factory
│   ├── main.py                    # FastAPI app, WebSocket, pipeline loop
│   ├── .env                       # Configuration (never commit this)
│   ├── .env.example               # Template for environment variables
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TopBar.jsx         # Status bar + pipeline controls
│   │   │   ├── LiveFeed.jsx       # Camera feed with detection overlay
│   │   │   ├── EventList.jsx      # Defect event cards (real-time)
│   │   │   └── TrendChart.jsx     # 24h defect trend (Recharts)
│   │   ├── hooks/
│   │   │   └── useWebSocket.js    # WebSocket hook with auto-reconnect
│   │   └── App.jsx                # Root component and state management
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml             # One-command Docker deployment
└── README.md
```

---

## Performance Tuning

### Recommended Settings by Deployment Target

| Environment        | `SAMPLE_FPS` | `CONF_THRESHOLD` | `YOLO_MODEL` |
| ------------------ | ------------ | ---------------- | ------------ |
| Demo / Testing     | `15`         | `0.50`           | `yolov8n.pt` |
| Single Camera Prod | `15`         | `0.65`           | `yolov8s.pt` |
| Multi-Camera Prod  | `10`         | `0.70`           | `yolov8n.pt` |
| High-Accuracy Prod | `5`          | `0.75`           | `yolov8m.pt` |

### Understanding the Latency Budget

Every frame goes through three stages. Latency at each stage on a mid-range laptop CPU:

| Stage                      | Approximate Time | Notes                                  |
| -------------------------- | ---------------- | -------------------------------------- |
| OpenCV frame read          | ~5–10 ms         | Zero-latency background thread         |
| YOLOv8n inference          | ~30–80 ms        | Main bottleneck on CPU                 |
| JPEG encode + WS broadcast | ~5 ms            | Async, non-blocking                    |
| LLM API call (Groq)        | ~1–3 seconds     | Background task — does NOT block feed  |
| PDF generation             | ~500 ms – 2 s    | Separate process — does NOT block feed |

> **Key insight:** The live video feed and the AI report generation run in completely separate execution contexts. The video stream is never blocked by the AI thinking.

### Lowering Inference Latency with ONNX (v1.0 roadmap)

The current YOLOv8 model runs via PyTorch (`.pt`). Exporting to ONNX format provides a **2–3x CPU speedup**:

```bash
# Export the ONNX model (run once)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

# Then set in .env:
YOLO_MODEL=yolov8n.onnx
```

---

## Troubleshooting

### Camera feed is stuck / frozen

- Make sure no other application (Phone Link, Teams, OBS) is using the camera.
- Try changing `CAMERA_URL` from `0` to `1` or `2` in `.env`.
- Restart the pipeline by clicking **Stop** then **Start**.
- Reduce `SAMPLE_FPS` to `10` or `15` to reduce USB bandwidth pressure.

### "Root cause analysis inconclusive" in event cards

- This is caused by the LLM returning a comma inside a JSON string value, which breaks standard JSON parsing.
- The system has a fallback regex parser for this, but it may fail on very malformed responses.
- **Workaround:** Switch to `LLM_MODEL=mixtral-8x7b-32768` in `.env`, which tends to produce cleaner JSON.
- **Planned fix:** Upgrade to structured outputs (`.with_structured_output()`) in v1.0.

### Defect events panel not updating live

- Ensure the WebSocket connection is `OPEN` — check the browser console for errors.
- The frontend auto-reconnects within 2 seconds if the connection drops.
- This was previously caused by a MongoDB `ObjectId` serialization bug — now patched.

### "groq.APIConnectionError: Connection error"

- This is a transient internet connectivity issue to the Groq API.
- The system now handles this gracefully with a fallback event saved to MongoDB.
- Your live camera feed will not be interrupted.
- If persistent, check your internet connection or try switching `LLM_PROVIDER=gemini`.

### Backend crashes with `ValueError: list.remove(x): x not in list`

- This was a WebSocket connection manager race condition. It is now patched.
- If you see this error, update to the latest version of `main.py`.

### getSize: Not a float '-0.02em' in logs

- This is a harmless CSS warning from the `xhtml2pdf` library. It does not affect the system in any way and can be safely ignored.

---

<div align="center">

**Built for Cognizant Technoverse · April 2026**

</div>
