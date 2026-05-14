import json
import re
from strands import Agent
from models.llms import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.geocode import geocode
from tools_source.fetch_places import fetch_places
from langfuse import get_client
from dotenv import load_dotenv
from prompts.discovery_agent import SYSTEM_PROMPT

load_dotenv()
langfuse = get_client()


def _parse_places(text: str) -> list[dict]:
    """Extract a JSON array of place objects from the agent's raw output.

    Handles plain JSON, ```json ... ``` fences, and stray prose around the array.
    """
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())
    bracket = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket:
        candidates.append(bracket.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("places"), list):
            return [p for p in parsed["places"] if isinstance(p, dict)]
    return []


discovery_agent = Agent(
    model=mistral,
    tools=[geocode, fetch_places],
    system_prompt=SYSTEM_PROMPT,
)


def run(destination_query: str) -> list[dict]:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="discovery-agent",
        input={"query": destination_query},
    ) as span:
        result = discovery_agent(destination_query)
        places = _parse_places(str(result))
        span.update(output={"places": places})
        langfuse.flush()
        return places


if __name__ == "__main__":
    sample_input = [
        {"place": "Ladakh", "user_preference": "", "ideal_count": 5},
        
    ]
    places = run(json.dumps(sample_input))
    print(f"Returned {len(places)} places:\n")
    print(json.dumps(places, indent=2))
