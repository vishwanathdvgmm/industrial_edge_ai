# Industrial Edge AI — Production System

**Cognizant Technoverse · Agent Builder Challenge · April 2026**

End-to-end defect detection GenAI agent: IP Camera → YOLOv8 → LangGraph → MongoDB → React Dashboard → WeasyPrint PDF

---

## Stack

| Layer              | Technology                        |
| ------------------ | --------------------------------- |
| CV Model           | YOLOv8 (Ultralytics)              |
| Agent Orchestrator | LangGraph (4-node stateful graph) |
| LLM                | Groq, Gemini, or OpenAI           |
| Backend            | FastAPI + WebSocket               |
| Database           | MongoDB + GridFS                  |
| Frontend           | React (Vite) + Recharts           |
| PDF Reports        | WeasyPrint                        |

---

## Quick Start

### 1. Backend

```bash
cd backend
cp .env.example .env          # add your GROQ_API_KEY (or Gemini/OpenAI)
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. MongoDB

```bash
# via Docker
docker run -d -p 27017:27017 --name mongo mongo:7.0

# or use MongoDB Atlas — just set MONGO_URI in .env
```

### 4. Test single detection

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"cam0","line_id":"line_01"}'
```

### 5. Start the full pipeline (continuous)

```bash
curl -X POST http://localhost:8000/pipeline/start
```

---

## API Endpoints

| Method | Endpoint                 | Description                        |
| ------ | ------------------------ | ---------------------------------- |
| GET    | `/health`                | Health check                       |
| POST   | `/pipeline/start`        | Start continuous camera loop       |
| POST   | `/pipeline/stop`         | Stop pipeline                      |
| POST   | `/detect`                | Single-shot detection              |
| GET    | `/events?limit=50`       | Recent defect events               |
| GET    | `/trend?hours=24`        | Hourly defect trend data           |
| GET    | `/config`                | System config                      |
| GET    | `/report/{event_id}/pdf` | Download PDF report                |
| WS     | `/ws`                    | WebSocket for live frames + events |

---

## Agent Pipeline

```
Camera Frame
    │
    ▼
YOLOv8 Inference (conf ≥ 0.65 gate)
    │
    ▼
[LangGraph Stateful Agent]
    │
    ├─ Node 1: Defect Classifier    → type, severity, zone
    ├─ Node 2: Root Cause Reasoner  → cause_hypothesis (+ MongoDB history)
    ├─ Node 3: Action Recommender   → HALT_LINE | FLAG_QC | LOG_ONLY
    └─ Node 4: Report Generator     → full report_payload
    │
    ▼
MongoDB (defect_events) + GridFS (images + PDFs)
    │
    ▼
WebSocket broadcast → React Dashboard
```

---

## Project Structure

```
industrial_edge_ai/
├── backend/
│   ├── main.py               ← FastAPI app + WebSocket + pipeline loop
│   ├── requirements.txt
│   ├── .env.example
│   ├── agent/
│   │   ├── state.py          ← AgentState TypedDict
│   │   ├── graph.py          ← LangGraph compiled graph
│   │   └── nodes/
│   │       ├── classifier.py ← Node 1
│   │       ├── root_cause.py ← Node 2
│   │       ├── action.py     ← Node 3
│   │       └── reporter.py   ← Node 4
│   ├── vision/
│   │   ├── capture.py        ← Async RTSP/USB frame generator
│   │   ├── preprocess.py     ← BGR→RGB, resize, normalize
│   │   └── detector.py       ← YOLOv8 wrapper
│   ├── db/
│   │   ├── mongo.py          ← MongoDB client + queries
│   │   └── gridfs_helper.py  ← Image/PDF binary storage
│   └── pdf/
│       ├── generator.py      ← WeasyPrint renderer
│       └── template.html     ← Report HTML template
├── frontend/
│   └── src/
│       ├── App.jsx           ← Root component
│       ├── index.css         ← Design system
│       ├── components/
│       │   ├── TopBar.jsx    ← Status + controls
│       │   ├── LiveFeed.jsx  ← Camera stream
│       │   ├── EventList.jsx ← Defect event cards
│       │   └── TrendChart.jsx← Recharts 24h trend
│       └── hooks/
│           └── useWebSocket.js ← WS + API helpers
└── docker-compose.yml
```

---

## Environment Variables

| Variable         | Default                     | Description                                     |
| ---------------- | --------------------------- | ----------------------------------------------- |
| `LLM_PROVIDER`   | `groq`                      | Set to `groq`, `gemini`, or `openai`            |
| `GROQ_API_KEY`   | —                           | Required if using Groq                          |
| `GEMINI_API_KEY` | —                           | Required if using Gemini                        |
| `OPENAI_API_KEY` | —                           | Required if using OpenAI                        |
| `LLM_MODEL`      | `llama-3.3-70b-versatile`   | Model name for the selected provider            |
| `MONGO_URI`      | `mongodb://localhost:27017` | MongoDB connection string                       |
| `MONGO_DB`       | `edge_ai`                   | Database name                                   |
| `YOLO_MODEL`     | `yolov8n.pt`                | Model file (nano for speed, small for accuracy) |
| `CONF_THRESHOLD` | `0.65`                      | Minimum confidence to trigger agent             |
| `CAMERA_URL`     | `auto`                      | `auto`, `0`, or IP URL (e.g., DroidCam IP)      |
| `SAMPLE_FPS`     | `5`                         | Frames per second to process                    |
