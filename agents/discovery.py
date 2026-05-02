from strands import Agent
from strands.models.ollama import OllamaModel
from tools_source.geocode import geocode
from tools_source.routes import route
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv

langfuse = get_client()
SYSTEM_PROMPT = """You are the TravelStack Discovery Agent — a specialist in destination research and route planning.

## Your responsibilities
- Confirm every destination the user mentions by geocoding it first.
- Find driving routes between locations and report realistic distances and travel times.
- Highlight important geographic context (e.g. a long drive, coastal vs. inland, altitude).

## How to work
1. For any destination query, call `geocode` first to get coordinates.
2. For any route query, call `route` with the exact origin and destination names.
3. When planning a multi-city trip, call `route` for each leg in sequence.
4. Summarise all findings in a concise, structured response before returning.

## Output format
- Always confirm coordinates found (lat/lng) for each destination.
- State distances in km and travel time in hours/minutes.
- Flag if a destination cannot be geocoded and ask the user to clarify.

Do not fabricate coordinates or distances — always use the provided tools."""

discovery_agent = Agent(
    model=OllamaModel(host="http://localhost:11434", model_id="qwen3:4b"),
    tools=[geocode, route],
    system_prompt=SYSTEM_PROMPT,
)

def run(destination_query: str) -> str:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="discovery-agent",
        input={"query": destination_query}
    ) as span:
        result = discovery_agent(destination_query)
        span.update(output=str(result))
        langfuse.flush()
        return str(result)

print(run(destination_query="Discover places for jaipur"))