import json
import re
import base64
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from llm import get_llm, supports_vision

llm = get_llm(temperature=0, max_tokens=512)

SYSTEM = (
    "You are an industrial defect classifier. Analyze detection metadata and classify "
    "the defect.\n"
    "Defect types: SCRATCH, CRACK, RUST, DENT, HOLE, DISCOLORATION, DEFORMATION, CONTAMINATION\n"
    "Severity: CRITICAL (structural/safety risk) | MEDIUM (quality issue) | LOW (cosmetic)\n"
    "Zone: SURFACE | EDGE | JOINT | CORE | COATING\n"
    "Respond ONLY with valid JSON."
)

def classify_defect(state: AgentState) -> AgentState:
    """Node 1 — Classify defect type, severity, and zone using LLM."""
    det = state["raw_detections"][0]

    text_prompt = (
        f"Camera: {state['camera_id']} | Line: {state['line_id']}\n"
        f"Detected class: {det['class_name']} | Confidence: {det['confidence']:.2f}\n"
        f"Bounding box (640px space): {[round(v) for v in det['bbox']]}\n\n"
        'Return JSON: {"defect_type":"...","severity":"...","zone":"...","confidence":0.00}'
    )

    content: list = [{"type": "text", "text": text_prompt}]

    # Attach image only for providers that support vision (Gemini, GPT-4o)
    frame_path = Path(state.get("frame_path", ""))
    if frame_path.exists() and supports_vision():
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}})

    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=content)])

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        try:
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

    return {
        **state,
        "defect_type": data.get("defect_type", det["class_name"].upper()),
        "severity": data.get("severity", "MEDIUM"),
        "zone": data.get("zone", "SURFACE"),
        "confidence": float(data.get("confidence", det["confidence"])),
    }
