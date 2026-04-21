from fastapi import APIRouter
from core.agent import TravelAgent

router = APIRouter()
agent = TravelAgent()

@router.post('/chat')
def chat(query: str):
    return agent.run(query)
