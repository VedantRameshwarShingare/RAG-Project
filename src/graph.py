from langgraph.graph import StateGraph
from rag import process_query

def process_node(state):
    query = state["query"]

    answer, confidence, intent = process_query(query)

    state["answer"] = answer
    state["confidence"] = confidence
    state["intent"] = intent

    return state

def route(state):
    if state["intent"] == "greeting":
        return "final"

    if state["confidence"] < 0.5:
        return "hitl"

    return "final"

def hitl_node(state):
    import time
    print("📩 Escalated to Human Support...")
    time.sleep(2)

    state["answer"] = "Our human agent will contact you shortly."
    return state

def final_node(state):
    return state

def build_graph():   # ✅ IMPORTANT NAME
    graph = StateGraph(dict)

    graph.add_node("process", process_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("process")

    graph.add_conditional_edges("process", route)
    graph.add_edge("hitl", "final")

    return graph.compile()