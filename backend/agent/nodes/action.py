from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from llm import get_llm

# ── Pydantic output schema ─────────────────────────────────────────────────────

class ActionOutput(BaseModel):
    """Structured output for industrial action recommendation."""
    action_rationale: str = Field(
        description=(
            "One clear, professional sentence explaining WHY this specific action was chosen "
            "for this defect, referencing the defect type, severity, and root cause."
        ),
        min_length=20,
    )

# ── Safety-first action map — deterministic, LLM cannot override ───────────────

SEVERITY_ACTION_MAP = {
    "CRITICAL": "HALT_LINE",
    "MEDIUM":   "FLAG_QC",
    "LOW":      "LOG_ONLY",
}

# ── Few-shot system prompt ─────────────────────────────────────────────────────

SYSTEM = """\
You are an industrial process control AI for a factory line. Your job is to write a \
professional, one-sentence rationale explaining why a specific corrective action was taken \
for a detected defect.

## Context
The action itself is already determined by safety policy — you only need to justify it:
- HALT_LINE: Production line stopped immediately. Used for CRITICAL defects only.
- FLAG_QC: Item flagged for quality control inspection. Used for MEDIUM severity defects.
- LOG_ONLY: Event logged for trend analysis. Used for LOW severity cosmetic defects.

## Rules
1. Be specific — reference the defect type, zone, and root cause.
2. Be professional — use factory/quality control language.
3. One sentence only — no lists, no markdown, no additional explanation.

## Few-Shot Examples
Example 1 (FLAG_QC, CONTAMINATION, MEDIUM):
  Root cause: "Inadequate cleaning protocol is introducing contaminants onto the surface."
  → rationale: "This component has been flagged for quality control inspection because \
medium-severity surface contamination may compromise adhesion or coating integrity downstream."

Example 2 (HALT_LINE, CRACK, CRITICAL):
  Root cause: "Excessive clamping torque caused a crack in the joint zone."
  → rationale: "The production line has been halted immediately because a critical structural \
crack in the joint zone poses a direct safety and product integrity risk that cannot proceed \
to the next assembly stage."

Example 3 (LOG_ONLY, DISCOLORATION, LOW):
  Root cause: "Minor heat exposure during curing caused surface discoloration."
  → rationale: "This cosmetic discoloration event has been logged for trend analysis as it \
poses no functional or structural risk to the component."
"""

# ── LLM (structured output mode) ──────────────────────────────────────────────

_base_llm = get_llm(temperature=0, max_tokens=150)
structured_llm = _base_llm.with_structured_output(ActionOutput)


# ── Node function ──────────────────────────────────────────────────────────────

def recommend_action(state: AgentState) -> AgentState:
    """Node 3 — Recommend action. CRITICAL is a hard safety override that bypasses the LLM."""
    severity = state["severity"]
    action = SEVERITY_ACTION_MAP.get(severity, "LOG_ONLY")

    # ── SAFETY OVERRIDE: CRITICAL never touches the LLM ──────────────────────
    if severity == "CRITICAL":
        return {
            **state,
            "action": "HALT_LINE",
            "action_rationale": (
                f"SAFETY OVERRIDE: Critical {state['defect_type']} defect detected in the "
                f"{state['zone']} zone on {state['camera_id']}. "
                "Production line halted immediately per safety policy — LLM reasoning bypassed."
            ),
        }

    # ── MEDIUM / LOW: LLM writes the rationale, policy sets the action ────────
    prompt = (
        f"Defect: {state['defect_type']} | Severity: {severity} | Zone: {state['zone']}\n"
        f"Root cause: {state['cause_hypothesis']}\n"
        f"Prescribed action: {action}\n\n"
        f"Write one professional sentence justifying why '{action}' is the correct response."
    )

    try:
        result: ActionOutput = structured_llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=prompt),
        ])
        rationale = result.action_rationale
    except Exception as e:
        print(f"⚠️ Action structured output failed: {e}. Using rule-based rationale.")
        rationale = _rule_based_rationale(action, state)

    return {
        **state,
        "action": action,
        "action_rationale": rationale,
    }


def _rule_based_rationale(action: str, state: AgentState) -> str:
    """Deterministic fallback rationale when LLM is unavailable."""
    defect = state.get("defect_type", "defect")
    zone = state.get("zone", "surface")
    severity = state.get("severity", "LOW")

    if action == "FLAG_QC":
        return (
            f"This component is flagged for quality control inspection because a "
            f"{severity.lower()}-severity {defect.lower()} in the {zone.lower()} zone "
            f"may affect downstream quality or customer acceptance."
        )
    elif action == "LOG_ONLY":
        return (
            f"This {defect.lower()} event in the {zone.lower()} zone is logged for "
            f"trend analysis as its {severity.lower()} severity poses no immediate "
            f"functional or safety risk."
        )
    else:
        return f"{action} applied per severity policy for {defect} in {zone} zone."
