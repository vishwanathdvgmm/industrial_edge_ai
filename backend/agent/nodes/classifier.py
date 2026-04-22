import base64
from pathlib import Path
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from llm import get_llm, supports_vision

# ── Pydantic output schema ─────────────────────────────────────────────────────

class ClassificationOutput(BaseModel):
    """Structured output for industrial defect classification."""
    defect_type: Literal[
        "SCRATCH", "CRACK", "RUST", "DENT", "HOLE",
        "DISCOLORATION", "DEFORMATION", "CONTAMINATION", "NONE"
    ] = Field(description="The specific type of defect observed.")
    severity: Literal["CRITICAL", "MEDIUM", "LOW"] = Field(
        description="CRITICAL=structural/safety risk, MEDIUM=quality issue, LOW=cosmetic only."
    )
    zone: Literal["SURFACE", "EDGE", "JOINT", "CORE", "COATING"] = Field(
        description="The zone of the product where the defect is located."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this classification from 0.0 to 1.0."
    )

# ── Few-shot system prompt ─────────────────────────────────────────────────────

SYSTEM = """\
You are an expert industrial quality control AI. Your job is to classify manufacturing defects \
detected by a computer vision system on a factory production line.

Use the YOLO detection label and bounding box as your primary signal. Apply industrial domain \
knowledge to infer the most likely defect type, severity, and zone.

## Severity Guidelines
- CRITICAL: Defect that compromises structural integrity or poses a safety hazard (e.g., crack through \
a load-bearing joint, deep hole in a pressure vessel wall).
- MEDIUM: Defect that degrades quality and may cause product failure or customer returns (e.g., surface \
rust, deep scratch on a sealing surface, dent on a functional component).
- LOW: Cosmetic defect with no functional impact (e.g., minor surface discoloration, light scratch on \
a non-critical area, paint chip).

## Zone Guidelines
- SURFACE: Flat outer face of the component.
- EDGE: Perimeter or boundary of the component.
- JOINT: Weld seam, bolt hole, or connection point.
- CORE: Internal or structural body of the component.
- COATING: Paint, plating, or protective layer.

## Few-Shot Examples
Example 1:
  Detected: "scratch" (conf 0.82) on a metal panel, bbox top-left region
  → defect_type: SCRATCH, severity: LOW, zone: SURFACE, confidence: 0.80

Example 2:
  Detected: "crack" (conf 0.91) on a weld joint, bbox center region
  → defect_type: CRACK, severity: CRITICAL, zone: JOINT, confidence: 0.90

Example 3:
  Detected: "rust" (conf 0.76) on a pipe edge, bbox left region
  → defect_type: RUST, severity: MEDIUM, zone: EDGE, confidence: 0.75

Example 4:
  Detected: "keyboard" (conf 0.71) — it is a standard object, not a defect itself
  → defect_type: NONE, severity: LOW, zone: SURFACE, confidence: 0.90

Example 5:
  Detected: "laptop" (conf 0.85) — but the image shows the screen is shattered
  → defect_type: CRACK, severity: CRITICAL, zone: SURFACE, confidence: 0.95
"""

# ── LLM (structured output mode) ──────────────────────────────────────────────

_base_llm = get_llm(temperature=0, max_tokens=256)

# .with_structured_output() forces the LLM to return a valid Pydantic object.
# If the LLM fails, LangChain raises OutputParserException — no JSON regex needed.
structured_llm = _base_llm.with_structured_output(ClassificationOutput)


# ── Node function ──────────────────────────────────────────────────────────────

def classify_defect(state: AgentState) -> AgentState:
    """Node 1 — Classify defect type, severity, and zone using structured LLM output."""
    det = state["raw_detections"][0]

    text_prompt = (
        f"Camera: {state['camera_id']} | Line: {state['line_id']}\n"
        f"YOLO detected: '{det['class_name']}' with confidence {det['confidence']:.2f}\n"
        f"Bounding box (x1,y1,x2,y2 in 640px space): {[round(v) for v in det['bbox']]}\n\n"
        f"If you can see the image, inspect the '{det['class_name']}' for any visible damage "
        f"(cracks, scratches, rust, dents). If it is damaged, classify the defect. "
        f"If the object appears completely undamaged or you cannot see the image, classify the defect as NONE."
    )

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=text_prompt)]

    # Attach image for providers that support vision (Gemini, GPT-4o)
    frame_path = Path(state.get("frame_path", ""))
    if frame_path.exists() and supports_vision():
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        messages[1] = HumanMessage(content=[
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
        ])

    try:
        result: ClassificationOutput = structured_llm.invoke(messages)
        return {
            **state,
            "defect_type": result.defect_type,
            "severity": result.severity,
            "zone": result.zone,
            "confidence": result.confidence,
        }
    except Exception as e:
        # Absolute last-resort fallback — structured output failed entirely
        print(f"⚠️ Classifier structured output failed: {e}. Using YOLO label as fallback.")
        return {
            **state,
            "defect_type": det["class_name"].upper(),
            "severity": "MEDIUM",
            "zone": "SURFACE",
            "confidence": float(det["confidence"]),
        }
