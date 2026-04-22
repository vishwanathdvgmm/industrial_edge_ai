from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.analyzer import analyze_defect
from agent.nodes.reporter import generate_report

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze_defect)
    graph.add_node("report", generate_report)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)

    return graph.compile()

# Singleton — compiled once at import
agent_graph = build_agent_graph()
