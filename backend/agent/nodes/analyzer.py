import base64
from pathlib import Path
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from llm import get_llm, supports_vision
from db.mongo import get_defect_history

# ── Pydantic output schema ─────────────────────────────────────────────────────

class AnalysisOutput(BaseModel):
    """Unified structured output for industrial defect classification and root cause analysis."""
    defect_type: Literal[
        "SCRATCH", "CRACK", "RUST", "DENT", "HOLE",
        "DISCOLORATION", "DEFORMATION", "CONTAMINATION", "NONE"
    ] = Field(description="The specific type of defect observed.")
    severity: Literal["CRITICAL", "MEDIUM", "LOW"] = Field(
        description="CRITICAL=structural risk, MEDIUM=quality issue, LOW=cosmetic only."
    )
    zone: Literal["SURFACE", "EDGE", "JOINT", "CORE", "COATING"] = Field(
        description="The zone of the product where the defect is located."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this classification from 0.0 to 1.0."
    )
    cause_hypothesis: str = Field(
        description=(
            "One clear, specific, technically accurate sentence describing the most probable "
            "root cause of the defect. Must reference the defect type, zone, and manufacturing context."
        ),
        min_length=20,
    )
    cause_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this root cause hypothesis from 0.0 (speculative) to 1.0 (highly certain)."
    )
    action_rationale: str = Field(
        description=(
            "One clear, professional sentence explaining WHY the recommended action is required, "
            "based on the defect type, severity, and root cause."
        ),
        min_length=20,
    )

# ── Safety-first action map ───────────────────────────────────────────────────
SEVERITY_ACTION_MAP = {
    "CRITICAL": "HALT_LINE",
    "MEDIUM":   "FLAG_QC",
    "LOW":      "LOG_ONLY",
}

# ── Few-shot system prompt ─────────────────────────────────────────────────────

SYSTEM = """\
You are an expert industrial quality control AI. Your job is to classify manufacturing defects \
detected by a computer vision system and immediately perform root cause analysis (RCA).

## Output Guidelines
1. You must classify the defect (defect_type, severity, zone).
2. You must provide a ONE SENTENCE root cause hypothesis (cause_hypothesis).
3. You must provide a ONE SENTENCE action rationale (action_rationale) justifying the response.
4. STRICT RULE: NEVER mention the camera, field of view, or camera alignment. Always reason about \
   the manufacturing process (tooling wear, cleaning failure, material handling error, etc.).
5. If the object appears undamaged (or you cannot see the image), classify it as NONE (LOW severity).

## Action Mapping Context (For your rationale)
- CRITICAL severity always results in HALT_LINE.
- MEDIUM severity always results in FLAG_QC.
- LOW severity (or NONE) always results in LOG_ONLY.

## Few-Shot Examples
Example 1:
  Detected object: "laptop", but image shows a shattered screen.
  → defect_type: CRACK, severity: CRITICAL, zone: SURFACE, confidence: 0.95
  → cause_hypothesis: "A critical crack on the surface zone suggests severe mechanical impact during the casing assembly process."
  → action_rationale: "The line must be halted because a shattered screen poses a direct safety hazard and total functional failure."

Example 2:
  Detected object: "keyboard", image shows no damage.
  → defect_type: NONE, severity: LOW, zone: SURFACE, confidence: 0.90
  → cause_hypothesis: "No defect was detected; the surface variation is within normal material tolerances."
  → action_rationale: "This event is logged for trend analysis as there is no functional risk."
"""

# ── LLM (structured output mode) ──────────────────────────────────────────────
_base_llm = get_llm(temperature=0.1, max_tokens=300)
structured_llm = _base_llm.with_structured_output(AnalysisOutput)

# ── Node function ──────────────────────────────────────────────────────────────

def analyze_defect(state: AgentState) -> AgentState:
    """Unified Node — Classify, Root Cause, and Rationale in ONE API call."""
    det = state["raw_detections"][0]

    # Fetch recent defect history
    history = get_defect_history(camera_id=state["camera_id"], limit=5)
    history_block = ""
    if history:
        lines = [f"  - {h.get('defect_type', '?')} ({h.get('severity', '?')})" for h in history]
        history_block = "\n\nRecent history (latest first):\n" + "\n".join(lines)

    text_prompt = (
        f"Camera: {state['camera_id']} | Line: {state['line_id']}\n"
        f"YOLO detected: '{det['class_name']}' with confidence {det['confidence']:.2f}\n"
        f"Bounding box: {[round(v) for v in det['bbox']]}{history_block}\n\n"
        f"If you can see the image, inspect the '{det['class_name']}' for any visible damage. "
        f"If damaged, classify the defect and provide a root cause. If undamaged, classify as NONE."
    )

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=text_prompt)]

    # Attach image
    frame_path = Path(state.get("frame_path", ""))
    if frame_path.exists() and supports_vision():
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        messages[1] = HumanMessage(content=[
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
        ])

    try:
        result: AnalysisOutput = structured_llm.invoke(messages)
        
        # Deterministic action mapping
        action = SEVERITY_ACTION_MAP.get(result.severity, "LOG_ONLY")
        if result.severity == "CRITICAL":
            action = "HALT_LINE"

        return {
            **state,
            "defect_type": result.defect_type,
            "severity": result.severity,
            "zone": result.zone,
            "confidence": result.confidence,
            "cause_hypothesis": result.cause_hypothesis,
            "cause_confidence": result.cause_confidence,
            "action": action,
            "action_rationale": result.action_rationale,
        }
    except Exception as e:
        print(f"⚠️ Unified analyzer failed: {e}. Using deterministic fallback.")
        return _rule_based_fallback(state, det)

def _rule_based_fallback(state: AgentState, det: dict) -> AgentState:
    """Safe fallback when LLM API rate limits are hit."""
    return {
        **state,
        "defect_type": det["class_name"].upper(),
        "severity": "LOW",
        "zone": "SURFACE",
        "confidence": float(det["confidence"]),
        "cause_hypothesis": "API Rate Limit Exceeded. Manual inspection required.",
        "cause_confidence": 0.5,
        "action": "LOG_ONLY",
        "action_rationale": "Event logged manually due to API connectivity failure.",
    }
