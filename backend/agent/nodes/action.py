import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from llm import get_llm

llm = get_llm(temperature=0, max_tokens=256)

# Safety-first severity → action map. CRITICAL cannot be overridden by LLM.
SEVERITY_ACTION_MAP = {
    "CRITICAL": "HALT_LINE",
    "MEDIUM": "FLAG_QC",
    "LOW": "LOG_ONLY",
}

def recommend_action(state: AgentState) -> AgentState:
    """Node 3 — Determine action. CRITICAL is a safety override that bypasses LLM."""
    severity = state["severity"]
    action = SEVERITY_ACTION_MAP.get(severity, "LOG_ONLY")

    # Safety override — no LLM involvement for CRITICAL
    if severity == "CRITICAL":
        return {
            **state,
            "action": "HALT_LINE",
            "action_rationale": (
                f"SAFETY OVERRIDE: Critical {state['defect_type']} detected in {state['zone']} zone. "
                "Immediate line halt issued. LLM reasoning bypassed per safety policy."
            ),
        }

    # For MEDIUM/LOW, LLM provides the rationale (action itself is deterministic)
    prompt = (
        f"Defect: {state['defect_type']} | Severity: {severity}\n"
        f"Root cause: {state['cause_hypothesis']}\n"
        f"Zone: {state['zone']}\n\n"
        f"The action is {action}. Provide a one-sentence professional rationale.\n"
        f'{{"action":"{action}","action_rationale":"<rationale>"}}'
    )

    response = llm.invoke([
        SystemMessage(content="You are an industrial process control AI. Respond ONLY with valid JSON."),
        HumanMessage(content=prompt),
    ])

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
        "action": action,
        "action_rationale": data.get("action_rationale", f"{action} applied per severity policy (LLM parsing failed)."),
    }
