import json
from strands import Agent
from models import cerebras  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.search_tavily import places, get_stored_results, get_call_tracker, clear_session
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
- Use the `places` tool for each relevant category. The tool returns a compact summary — full data is stored internally.
## Available search categories
- "hidden gems"       — off the beaten path, rarely visited spots
- "street food"       — local food markets, stalls, and authentic eateries
- "historic sites"    — forts, temples, stepwells, heritage buildings
- "viewpoints"        — scenic spots, sunrise/sunset points, panoramas
- "local experiences" — pottery, cooking classes, night walks, craft workshops
- "nature"            — parks, lakes, hills, gardens
- "markets bazaars"   — local shopping, handicrafts, bazaars
## How to decide max_results
| Trip / Request type                        | max_results per category |
|--------------------------------------------|--------------------------|
| 1-day trip or "quick picks"                | 1–2                      |
| 2–3 day trip or "a few places"             | 2–3                      |
| 4–7 day trip or "comprehensive" / "full"   | 4–5                      |
| No duration mentioned, general explore     | 3 (default)              |
If the user wants to explore many places, max is 5 per category.
## How to work
1. Read the request: identify the city, relevant categories, and trip duration.
2. Decide max_results from the table above.
3. Call `places(city, category, max_results)` for each relevant category.
4. Once all tool calls are done, respond with: DISCOVERY_COMPLETE
Do NOT attempt to format the final JSON — the system builds it from stored data."""

discovery_agent = Agent(
    model=cerebras,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[places],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=10,
)


def _print_tracking_report() -> None:
    tracker = get_call_tracker()
    print("\n" + "=" * 55)
    print(f"  TOOL CALL TRACKING  ({len(tracker)} calls)")
    print("=" * 55)
    total_raw = total_extracted = 0
    for t in tracker:
        print(
            f"  [{t['call_index']:>2}] {t['category']:<20} | "
            f"raw: {t['raw_count']}  extracted: {t['extracted_count']}  "
            f"@ {t['timestamp']}"
        )
        total_raw += t["raw_count"]
        total_extracted += t["extracted_count"]
    print("-" * 55)
    print(f"       Total raw results : {total_raw}")
    print(f"       Total places found: {total_extracted}")
    print("=" * 55 + "\n")


def _deduplicate(places_list: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in places_list:
        key = p["place"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def run(destination_query: str) -> str:
    clear_session()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="discovery-agent",
        input={"query": destination_query}
    ) as span:
        discovery_agent(destination_query)

        final_places = _deduplicate(get_stored_results())
        output = json.dumps(final_places, indent=2, ensure_ascii=False)

        _print_tracking_report()

        span.update(output=output)
        langfuse.flush()
        return output


run(destination_query="I have schedule to travel to meghalaya give me the best nature for travel of 3 days all best places i can cover")
