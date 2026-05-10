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

## STRICT EXECUTION POLICY — READ FIRST

You MUST call `find_hotels` EXACTLY ONCE in your entire run. After that call returns,
you MUST NOT call any tool again under any circumstance. Your next response after
the single tool call MUST be the final JSON array — nothing else.

If the first `find_hotels` call returns even one hotel_id, your job is done — pick the
matching ids and return the array. Do NOT retry. Do NOT call again to get more hotels,
better hotels, different parameters, or because the descriptions look short. The tool
result is final.

If the tool somehow returns zero hotels, return `[]` and stop. Never retry.

## How to work

1. Call `find_hotels(city, checkin, checkout, adults, rooms)` ONCE with:
   - city = place
   - checkin, checkout, adults, rooms from the input

2. The tool returns `{"total": N, "hotels": [{hotel_id, description}, ...]}`.
   Read each description.

3. Match against the user's `preferences`:
   - "pet friendly"     → keep only hotels whose description mentions pet-friendly
   - "couple"           → keep only couple-friendly
   - "with pool"        → keep only hotels with a pool
   - "luxury"           → 4+ star hotels
   - "budget"           → cheaper hotels
   - "free cancellation"→ only with free cancellation
   - generic preference (e.g. "any well-rated comfortable hotel") → keep ALL hotel_ids
   Combine multiple specific preferences with AND logic.

4. If preferences is missing, empty, or generic → return ALL hotel_ids from the result.

5. Return your final answer as a plain JSON array of selected hotel_ids — NO tool call,
   NO prose, NO markdown:
   ["45123132", "98765432"]

## Hard rules

- find_hotels is called EXACTLY ONCE. Never twice. Never retried.
- After the first tool call, your next message MUST be the JSON array final answer.
- Only use hotel_ids that appear in the find_hotels result.
- Never fabricate, rename, or guess hotel_ids.
- If filtering would yield zero ids, fall back to returning ALL ids from the result.
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
