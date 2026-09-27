"""
LangGraph Hybrid Knowledge Workflow for Demo 4.
Constructs and compiles the multi-branch state graph.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from graph.router import router_node
from graph.rag_node import rag_node
from graph.general_node import general_node
from graph.synthesizer import synthesizer_node


def route_decision(state: AgentState) -> Literal["out_of_scope_path", "rag_path", "hybrid_path"]:
    """Determines next node path based on router classification."""
    route = state.get("route", "HYBRID")
    if route == "OUT_OF_SCOPE":
        return "out_of_scope_path"
    elif route == "RAG":
        return "rag_path"
    else:
        return "hybrid_path"


def hybrid_fanout_rag(state: AgentState):
    """Executes RAG node as part of hybrid flow."""
    return rag_node(state)


def hybrid_fanout_general(state: AgentState):
    """Executes General node as part of hybrid flow."""
    return general_node(state)


def build_workflow():
    """Builds and compiles the complete LangGraph Hybrid Knowledge graph."""
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("general_node", general_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Add Start -> Router
    workflow.add_edge(START, "router")

    # 3. Add Conditional Branching from Router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "out_of_scope_path": "synthesizer",  # Directly to synthesizer to produce clean refusal
            "rag_path": "rag_node",
            "hybrid_path": "rag_node",  # Hybrid runs RAG first, then General
        },
    )


    # In hybrid mode: rag_node proceeds to general_node
    # In single modes: rag_node proceeds directly to synthesizer
    def after_rag_step(state: AgentState) -> Literal["general_node", "synthesizer"]:
        if state.get("route") == "HYBRID":
            return "general_node"
        return "synthesizer"

    workflow.add_conditional_edges(
        "rag_node",
        after_rag_step,
        {
            "general_node": "general_node",
            "synthesizer": "synthesizer",
        },
    )

    # General node proceeds to synthesizer
    workflow.add_edge("general_node", "synthesizer")

    # Synthesizer completes the graph
    workflow.add_edge("synthesizer", END)

    # Compile executable graph
    return workflow.compile()


# Compiled LangGraph Application
hybrid_graph_app = build_workflow()
