from strands import Agent
from strands.models.ollama import OllamaModel
from tools_source.search_tavily import places
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

SEARCH_CATEGORIES = [
    "hidden gems",
    "street food",
    "historic sites",
    "viewpoints",
    "local experiences",
    "nature",
    "markets bazaars",
]

SYSTEM_PROMPT = """You are the TravelStack Discovery Agent — a specialist in uncovering interesting places and local experiences for travellers.

## Your responsibilities
- Discover places in a city based on the user's request and preferences.
- Cover a diverse range of categories: hidden gems, street food, historic sites, viewpoints, local experiences, nature, and markets & bazaars.
- Use the `places` tool to search for each relevant category, passing the city name and category as inputs.

## Available search categories
- "hidden gems"       — off the beaten path, rarely visited spots
- "street food"       — local food markets, stalls, and authentic eateries
- "historic sites"    — forts, temples, stepwells, heritage buildings
- "viewpoints"        — scenic spots, sunrise/sunset points, panoramas
- "local experiences" — pottery, cooking classes, night walks, craft workshops
- "nature"            — parks, lakes, hills, gardens
- "markets bazaars"   — local shopping, handicrafts, bazaars

## How to work
1. Identify which categories are relevant to the user's request and preferences.
2. Call `places(city, category)` for each relevant category — run all needed searches.
3. From the raw search results (title, content, url), extract individual place names and details.
4. Deduplicate — if the same place appears under multiple categories, keep it once under the best-fitting category.
5. Format the final output as a JSON array (see Output format below).

## Output format
Return ONLY a valid JSON array. Each element must have exactly these fields:
[
  {
    "place": "<name of the place>",
    "category": "<one of the search categories above>",
    "description": "<one concise sentence about what makes this place worth visiting>",
    "link": "<source URL from the search result>"
  }
]

Do not include any text, explanation, or markdown outside the JSON array.
Do not fabricate places — only include what the tool results contain."""

discovery_agent = Agent(
    model=OllamaModel(host="http://localhost:11434", model_id="qwen3:4b"),
    tools=[places],
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

print(run(destination_query="What are places i can visit in jaipur and give me soeme food places and iconic structures and beautiful place here"))