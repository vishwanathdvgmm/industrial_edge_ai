import uuid
from datetime import datetime
from agent.state import AgentState

def generate_report(state: AgentState) -> AgentState:
    """Node 4 — Assemble structured report payload. PDF generation is async post-agent."""
    report_id = str(uuid.uuid4())[:8].upper()

    report_payload = {
        "report_id": report_id,
        "generated_at": datetime.utcnow().isoformat(),
        "camera_id": state["camera_id"],
        "line_id": state["line_id"],
        "defect": {
            "type": state["defect_type"],
            "severity": state["severity"],
            "zone": state["zone"],
            "confidence": state["confidence"],
        },
        "analysis": {
            "cause_hypothesis": state["cause_hypothesis"],
            "cause_confidence": state.get("cause_confidence", 0.5),
        },
        "decision": {
            "action": state["action"],
            "rationale": state["action_rationale"],
        },
        "raw_detections": state["raw_detections"],
        "frame_path": state.get("frame_path", ""),
    }

    return {
        **state,
        "report_payload": report_payload,
        "event_id": report_id,
    }
