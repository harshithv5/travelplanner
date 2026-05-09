import json
from strands import Agent
from models import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.hotels import find_hotels, reset_session, get_stored_hotels
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

SYSTEM_PROMPT = """You are the TravelStack Hotel Agent — a specialist in selecting the right accommodation for the user.

You receive a JSON input with these fields:
- place        : destination city or area name — required
- checkin      : check-in date in YYYY-MM-DD format — required
- checkout     : check-out date in YYYY-MM-DD format — required
- adults       : number of adult guests (default 1)
- rooms        : number of rooms required (default 1)
- preferences  : free-text user preference (optional, e.g. "pet friendly", "with pool",
                 "budget under 5000", "luxury", "couple stay")

## How to work
1. Call `find_hotels(city, checkin, checkout, adults, rooms)` exactly once using the provided place as the city.
2. The tool returns compact summaries `[{hotel_id, description}]`. Read each description carefully.
3. Match against the user's `preferences` (if given):
   - "pet friendly"     → only pet-friendly hotels
   - "couple"           → only couple-friendly hotels
   - "with pool"        → only hotels with a pool
   - "luxury"           → 4+ star hotels
   - "budget"           → cheaper hotels
   - "free cancellation"→ only hotels with free cancellation
   Combine multiple preferences with AND logic when listed together.
4. If no preferences are given, return ALL hotel_ids from the tool output.
5. Return your final answer as a JSON array of selected hotel_ids:
   ["45123132", "98765432"]

Rules:
- Call `find_hotels` exactly once. Never repeat a tool call.
- Only use hotel_ids that appear in the find_hotels results.
- Never fabricate or rename hotel_ids.
- Do not include any text or markdown outside the JSON array."""

hotel_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[find_hotels],
    system_prompt=SYSTEM_PROMPT,
)


def _expand_ids(raw_output: str) -> list[dict]:
    """Parse selected hotel_ids from agent output and expand to full hotel objects."""
    store = get_stored_hotels()
    try:
        ids = json.loads(raw_output)
        if isinstance(ids, list):
            return [store[i] for i in ids if i in store]
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback — return everything stored
    return list(store.values())


def run(
    place: str,
    checkin: str,
    checkout: str,
    adults: int = 1,
    rooms: int = 1,
    preferences: str | None = None,
) -> list[dict]:
    """
    Find and filter hotels matching the user's preferences.

    Args:
        place:       Destination city or area name.
        checkin:     Check-in date (YYYY-MM-DD).
        checkout:    Check-out date (YYYY-MM-DD).
        adults:      Number of adult guests (default 1).
        rooms:       Number of rooms required (default 1).
        preferences: Free-text preference (e.g. "pet friendly with pool").

    Returns:
        List of full hotel dicts that match the user's preferences. Each dict
        contains every field the tool returned (name, stars, rating, lat/lng,
        price, facilities, rooms, etc.) for use by downstream agents.
    """
    reset_session()

    payload: dict = {
        "place":    place,
        "checkin":  checkin,
        "checkout": checkout,
        "adults":   adults,
        "rooms":    rooms,
    }
    if preferences:
        payload["preferences"] = preferences

    query = f"Find hotels matching this request:\n{json.dumps(payload, indent=2)}"

    with langfuse.start_as_current_observation(
        as_type="span",
        name="hotel-agent",
        input=payload,
    ) as span:
        result = hotel_agent(query)
        hotels = _expand_ids(str(result))
        span.update(output={"hotels": hotels})
        langfuse.flush()
        return hotels


# print(json.dumps(
#     run(
#         place="Shillong",
#         checkin="2026-06-01",
#         checkout="2026-06-03",
#         adults=4,
#         rooms=2,
#         preferences="Any regular high rated hotel",
#     ),
#     indent=2
# ))
