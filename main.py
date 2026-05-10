import os
import time
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables BEFORE importing any module that depends on them
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.orchestrator import run as orchestrate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("travelstack")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    started_at = time.time()
    logger.info("=" * 60)
    logger.info("TravelStack AI — starting up")
    logger.info("=" * 60)

    required_envs = ["MISTRAL_API_KEY", "ORS_API_KEY", "RAPIDAPI_KEY"]
    for key in required_envs:
        present = bool(os.getenv(key))
        logger.info(f"env {key}: {'OK' if present else 'MISSING'}")

    optional_envs = [
        "GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
    ]
    for key in optional_envs:
        if os.getenv(key):
            logger.info(f"env {key}: OK (optional)")

    app.state.started_at = started_at
    logger.info("Orchestrator agent loaded and ready")
    yield

    uptime = time.time() - started_at
    logger.info(f"TravelStack AI — shutting down after {uptime:.1f}s")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TravelStack AI",
    description="Travel planning orchestrator — discovers places, finds hotels, plans routes.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language travel planning request.",
        examples=[
            "Plan a 3-day trip to Meghalaya from 2026-06-01 to 2026-06-03 "
            "for 4 adults in 2 rooms by car. We like waterfalls and viewpoints."
        ],
    )


class ChatResponse(BaseModel):
    response: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    uptime = time.time() - app.state.started_at
    return {
        "service": "TravelStack AI",
        "status":  "running",
        "uptime_seconds": round(uptime, 1),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural-language travel planning request to the orchestrator agent.
    The agent gathers info, runs discovery + hotel search in parallel, and plans
    the day-wise route.
    """
    started = time.time()
    logger.info(f"/chat <- {request.query[:120]}{'...' if len(request.query) > 120 else ''}")

    try:
        response_text = orchestrate(request.query)
    except Exception as exc:
        logger.exception("orchestrator failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = int((time.time() - started) * 1000)
    logger.info(f"/chat -> {elapsed_ms}ms")

    return ChatResponse(response=response_text, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
