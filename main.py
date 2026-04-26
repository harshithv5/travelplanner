from fastapi import FastAPI
from pydantic import BaseModel
from agents.orchestrator import run as orchestrate

app = FastAPI(title="TravelStack AI")


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "TravelStack AI running"}


@app.post("/chat")
def chat(request: ChatRequest):
    return {"response": orchestrate(request.query)}
