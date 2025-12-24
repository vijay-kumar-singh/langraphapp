from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
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

def llm_step(state: MessagesState):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}

def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("llm", llm_step)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("llm")
    graph.add_edge("llm", "tools")
    graph.add_edge("tools", "llm")

    return graph.compile()

workflow = build_graph()
