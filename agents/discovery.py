import json
from strands import Agent
from models import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.geocode import geocode
from tools_source.fetch_places import fetch_places, reset_session, get_stored_places
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

SYSTEM_PROMPT = """You are the TravelStack Discovery Agent — a specialist in uncovering interesting places for travellers.

## How to work
1. Call `geocode(place_name)` with the user's destination to get lat/lng.
2. Call `fetch_places(lat, lng)` with those coordinates.
3. Read the user's request carefully and decide which places to return:
   - If the user asks for everything or has no specific preference → return all IDs.
   - If the user mentions specific interests (e.g. waterfalls, museums, food, historic) → return only IDs whose name or category match that interest.
4. Return your final answer as a JSON array of the selected IDs only:
["p1", "p4", "p7"]

Rules:
- Only use IDs that appear in the fetch_places results.
- Never fabricate or rename IDs.
- Do not include any text or markdown outside the JSON array."""

discovery_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[geocode, fetch_places],
    system_prompt=SYSTEM_PROMPT,
)


def _expand_ids(raw_output: str) -> list[dict]:
    """Parse selected IDs from agent output and expand to full place objects."""
    store = get_stored_places()
    try:
        ids = json.loads(raw_output)
        if isinstance(ids, list):
            return [store[i] for i in ids if i in store]
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: return everything in the store
    return list(store.values())


def _format_output(places: list[dict]) -> str:
    if not places:
        return "No places found."
    lines = []
    for i, p in enumerate(places, start=1):
        lines.append(f"{i}. {p['name']}  [{p['category']}]")
        if p.get("description"):
            lines.append(f"   {p['description']}")
        lines.append(f"   Lat: {p['lat']},  Lng: {p['lng']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def run(destination_query: str) -> str:
    reset_session()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="discovery-agent",
        input={"query": destination_query}
    ) as span:
        result = discovery_agent(destination_query)
        places = _expand_ids(str(result))
        output = _format_output(places)
        span.update(output={"places": places})
        langfuse.flush()
        return output


run(destination_query="What are the places in meghalaya and i prefer water falls ")
