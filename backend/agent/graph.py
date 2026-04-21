from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.classifier import classify_defect
from agent.nodes.root_cause import find_root_cause
from agent.nodes.action import recommend_action
from agent.nodes.reporter import generate_report
from db.mongo import get_defect_history

def _root_cause_with_history(state: AgentState) -> AgentState:
    """Inject MongoDB history before calling root cause node."""
    history = get_defect_history(camera_id=state["camera_id"], limit=10)
    return find_root_cause(state, defect_history=history)

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_defect)
    graph.add_node("root_cause", _root_cause_with_history)
    graph.add_node("action_node", recommend_action)
    graph.add_node("report", generate_report)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "root_cause")
    graph.add_edge("root_cause", "action_node")
    graph.add_edge("action_node", "report")
    graph.add_edge("report", END)

    return graph.compile()

# Singleton — compiled once at import
agent_graph = build_agent_graph()
