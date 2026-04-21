import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from llm import get_llm

llm = get_llm(temperature=0.2, max_tokens=512)

SYSTEM = (
    "You are an industrial root cause analysis expert. "
    "Analyze defect patterns and manufacturing context to identify the most probable root cause. "
    "Be specific, technical, and actionable. Respond ONLY with valid JSON."
)

def find_root_cause(state: AgentState, defect_history: list | None = None) -> AgentState:
    """Node 2 — LLM root cause reasoning with optional MongoDB history injection."""
    history_block = ""
    if defect_history:
        lines = [
            f"  - {h.get('defect_type','?')} ({h.get('severity','?')}) on {h.get('timestamp','?')[:10]}"
            for h in defect_history[-10:]
        ]
        history_block = "\n\nRecent defect history (last 10):\n" + "\n".join(lines)

    prompt = (
        f"Current defect:\n"
        f"  Type: {state['defect_type']} | Severity: {state['severity']}\n"
        f"  Zone: {state['zone']} | Camera: {state['camera_id']}\n"
        f"  Confidence: {state['confidence']:.2f}"
        f"{history_block}\n\n"
        'Return JSON: {"cause_hypothesis":"<one technical sentence>","cause_confidence":0.00}'
    )

    response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        try:
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception:
            print(f"Failed to parse root cause JSON from LLM: {response.content}")
            data = {}

    return {
        **state,
        "cause_hypothesis": data.get("cause_hypothesis", "Root cause analysis inconclusive due to LLM parsing error."),
        "cause_confidence": float(data.get("cause_confidence", 0.5)),
    }
