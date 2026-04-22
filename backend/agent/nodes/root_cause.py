from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from db.mongo import get_defect_history
from llm import get_llm

# ── Pydantic output schema ─────────────────────────────────────────────────────

class RootCauseOutput(BaseModel):
    """Structured output for industrial root cause analysis."""
    cause_hypothesis: str = Field(
        description=(
            "One clear, specific, technically accurate sentence describing the most probable "
            "root cause of the defect. Must reference the defect type, zone, and manufacturing context."
        ),
        min_length=20,
    )
    cause_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this hypothesis from 0.0 (speculative) to 1.0 (highly certain)."
    )

# ── Few-shot system prompt ─────────────────────────────────────────────────────

SYSTEM = """\
You are a senior industrial process engineer performing root cause analysis (RCA) for a \
factory quality inspection system. Your analysis will be printed in an official defect report.

## Your Role
Given a detected defect type, severity, zone, and recent defect history, you must identify \
the most probable ROOT CAUSE in the manufacturing process itself.

## STRICT RULES
1. NEVER mention the camera, camera field of view, camera alignment, or detection system.
2. NEVER say the root cause is "inconclusive" or "unknown".
3. ALWAYS frame the cause as a manufacturing process issue: tooling wear, cleaning failure, \
   material handling error, process parameter drift, environmental contamination, operator error, etc.
4. ONE sentence only. Be specific and technical.
5. If the defect type is NONE or LOW severity, reason that minor surface irregularities are \
   consistent with normal process variation or early-stage tooling wear.

## Defect-to-Process Mapping (use as reasoning guide)
- CONTAMINATION → Cleaning protocol failure, air filtration issue, coolant contamination, \
  foreign particles from upstream machining.
- SCRATCH → Abrasive contact from worn conveyor guides, improper fixture clamping, \
  rough handling during transfer.
- CRACK → Excessive clamping torque, thermal stress, material fatigue, improper heat treatment.
- RUST → Inadequate corrosion protection, moisture ingress, failed drying stage.
- DENT → Impact from tooling, drop during handling, excessive pressing force.
- DISCOLORATION → Heat exposure beyond spec, chemical contamination, coating process drift.
- DEFORMATION → Excessive forming force, die misalignment, incorrect material grade.
- HOLE → Tooling damage, drill bit misalignment, material void from casting defect.
- NONE (low confidence) → Minor surface irregularity consistent with normal process variation \
  or early-stage tooling wear that does not yet meet rejection criteria.

## Few-Shot Examples
Example 1 — CONTAMINATION, MEDIUM, SURFACE, recurring pattern:
→ "Recurring surface contamination on cam0's inspection zone indicates a systemic failure \
in the upstream parts washing station, likely due to a clogged filtration nozzle or \
insufficient cleaning agent concentration."

Example 2 — SCRATCH, LOW, EDGE:
→ "Light edge scratches are consistent with worn conveyor side guides in the transfer section \
immediately before this inspection station, causing intermittent abrasive contact."

Example 3 — NONE, LOW, SURFACE, first occurrence:
→ "No significant defect was detected; the minor surface variation observed is within normal \
process tolerance and consistent with expected material grain or surface finish variation at \
this production stage."

Example 4 — CRACK, CRITICAL, JOINT:
→ "A structural crack at the joint zone most likely results from excessive clamping torque \
or thermal cycling stress during the welding operation at this station, indicating tooling \
calibration or process parameter review is required."
"""

# ── LLM (structured output mode) ──────────────────────────────────────────────

_base_llm = get_llm(temperature=0.2, max_tokens=256)
structured_llm = _base_llm.with_structured_output(RootCauseOutput)


# ── Node function ──────────────────────────────────────────────────────────────

def find_root_cause(state: AgentState) -> AgentState:
    """Node 2 — Root cause reasoning with structured output and MongoDB history."""

    # Fetch recent defect history for this camera
    history = get_defect_history(camera_id=state["camera_id"], limit=10)
    history_block = ""
    if history:
        lines = [
            f"  - {h.get('defect_type', '?')} ({h.get('severity', '?')}) at {h.get('timestamp', '?')[:16]}"
            for h in history
        ]
        history_block = "\n\nRecent defect history (latest first):\n" + "\n".join(lines)

    prompt = (
        f"Current defect to analyze:\n"
        f"  Type: {state['defect_type']} | Severity: {state['severity']}\n"
        f"  Zone: {state['zone']} | Camera: {state['camera_id']} | Line: {state['line_id']}\n"
        f"  Detection confidence: {state['confidence']:.2f}"
        f"{history_block}\n\n"
        f"Identify the single most probable root cause for this defect."
    )

    try:
        result: RootCauseOutput = structured_llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=prompt),
        ])
        return {
            **state,
            "cause_hypothesis": result.cause_hypothesis,
            "cause_confidence": result.cause_confidence,
        }
    except Exception as e:
        # Absolute last-resort — structured output failed. Build a meaningful fallback.
        print(f"⚠️ Root cause structured output failed: {e}. Using rule-based fallback.")
        fallback = _rule_based_hypothesis(state)
        return {
            **state,
            "cause_hypothesis": fallback,
            "cause_confidence": 0.40,
        }


def _rule_based_hypothesis(state: AgentState) -> str:
    """Deterministic fallback that always produces a meaningful hypothesis without the LLM."""
    defect = state.get("defect_type", "UNKNOWN")
    zone = state.get("zone", "SURFACE")
    severity = state.get("severity", "LOW")
    camera = state.get("camera_id", "cam0")

    rule_map = {
        "CONTAMINATION": f"Surface contamination in the {zone} zone on {camera} likely indicates an upstream cleaning or filtration failure.",
        "CRACK":         f"A {severity.lower()}-severity crack in the {zone} zone suggests excessive mechanical stress or tooling misalignment at this station.",
        "RUST":          f"Rust detected in the {zone} zone indicates inadequate corrosion protection or prolonged exposure to moisture.",
        "SCRATCH":       f"Scratch marks in the {zone} zone are likely caused by improper handling, abrasive contact, or worn conveyor components.",
        "DENT":          f"A dent in the {zone} zone suggests an impact event during transport, pressing, or assembly at this station.",
        "DISCOLORATION": f"Discoloration in the {zone} zone may result from heat exposure, chemical contamination, or an inconsistent coating process.",
        "DEFORMATION":   f"Deformation in the {zone} zone is likely caused by excessive force during forming, pressing, or clamping operations.",
        "HOLE":          f"An unintended hole in the {zone} zone suggests tooling damage, drilling misalignment, or material fatigue.",
        "NONE":          f"No manufacturing defect detected — the camera on {camera} may be observing a non-product object or an area outside the inspection zone.",
    }
    return rule_map.get(defect, f"Defect type '{defect}' in the {zone} zone requires manual inspection to determine root cause.")
