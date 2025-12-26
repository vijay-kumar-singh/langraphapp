from fastapi import FastAPI
from pydantic import BaseModel
from graph import workflow
import os

app = FastAPI()

class Input(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "LangGraph FastAPI backend is running!"}

@app.post("/run")
def run_graph(data: Input):
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    try:
        # Call LangGraph workflow
        result = workflow.invoke({"messages": [data.message]})
        return {"response": result["messages"][-1].content}
    except Exception as e:
        # Catch errors instead of returning 500
        return {"error": str(e)}
