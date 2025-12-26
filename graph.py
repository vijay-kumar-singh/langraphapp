from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
import os

# Load API key from environment (Render Env Vars)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

@tool
def add(a: int, b: int):
    """Add two integers."""
    return a + b

tools = [add]


# ─────────────────────────────────────────────
# LLM NODE
# ─────────────────────────────────────────────
def llm_step(state: MessagesState):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}


# ─────────────────────────────────────────────
# STOP CONDITION (CRITICAL FIX)
# ─────────────────────────────────────────────
def should_continue(state: MessagesState):
    """Decide whether to keep looping or stop."""

    # Last message from LLM or tool
    last = state["messages"][-1]

    # CASE 1: The LLM wants to call a tool → continue to tools node
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # CASE 2: No more tool calls → STOP
    return END


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("llm", llm_step)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("llm")

    # After llm → check stop condition
    graph.add_conditional_edges("llm", should_continue)

    # After tools → go back to llm
    graph.add_edge("tools", "llm")

    # Prevent runaway loops
    return graph.compile(recursion_limit=10)


workflow = build_graph()
