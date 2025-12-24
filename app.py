from fastapi import FastAPI
from graph import workflow

app = FastAPI()

@app.get("/")
def root():
    return {"message": "LangGraph FastAPI backend is running!"}

@app.post("/run")
def run_graph(message: str):
    result = workflow.invoke({"messages": [message]})
    return {"response": result["messages"][-1].content}
