import json
import re
from datetime import date
from strands import Agent
from models.llms import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.hotels import find_hotels
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

SYSTEM_PROMPT = """You are the TravelStack Hotel Agent — a specialist in selecting the right accommodation for the user.

You receive a JSON input with these fields:
- place        : destination city — REQUIRED
- state        : state / region (optional; pass it when known)
- country      : country (optional; pass it when known)
- checkin      : check-in date in YYYY-MM-DD format (optional — see DEFAULTS)
- checkout     : check-out date in YYYY-MM-DD format (optional — see DEFAULTS)
- adults       : number of adult guests (optional — see DEFAULTS)
- rooms        : number of rooms required (optional — see DEFAULTS)
- preferences  : free-text user preference (optional, e.g. "pet friendly", "with pool",
                 "budget under 5000", "luxury", "couple stay")
- hotel_name   : if the user asked for a SPECIFIC hotel by name, pass it here.
                 Otherwise leave it empty.
- today        : ISO date the request was made on. Use this as the reference
                 for computing any missing date defaults.

## DEFAULTS — apply silently when the user did NOT specify a value

If the user is just browsing hotels in a place without committing to a trip,
fill in sensible defaults and CALL THE TOOL once. Never ask the user for
clarification — assume the defaults below.

- checkin  missing → today + 2 days
- checkout missing → checkin + 2 days  (a 2-night stay)
- adults   missing → 2
- rooms    missing → 1

If only ONE of checkin/checkout is given, derive the other so the stay is
2 nights. Never call the tool with an empty checkin/checkout — always
materialise concrete YYYY-MM-DD values from `today` first.

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

1. Call `find_hotels` ONCE with:
   - city            = place
   - state           = state from input (if provided)
   - country         = country from input (if provided)
   - checkin, checkout, adults, rooms from the input
   - user_preference = preferences from input (if provided)
   - hotel_name      = hotel_name from input (only if the user named a specific hotel)

2. The tool returns `{"total": N, "hotels": [<full hotel object>, ...]}`.
   Each hotel object has: hotel_id, name, stars, rating, rating_word, review_count,
   lat, lng, description, facilities (couple_friendly, pet_friendly, parking,
   wifi, pool, restaurant), rooms (live availability), price_per_night, currency.
   Read these fields to evaluate each hotel.

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

5. Return your final answer as a plain JSON array of the selected hotel OBJECTS,
   taken verbatim from the tool result (same keys, same values). No tool call,
   no prose, no markdown.

## Hard rules

- find_hotels is called EXACTLY ONCE. Never twice. Never retried.
- After the first tool call, your next message MUST be the JSON array final answer.
- Only use hotel objects that appear in the find_hotels result — never invent or alter fields.
- If filtering would yield zero hotels, fall back to returning ALL hotels from the result.
- Do not include any text or markdown outside the JSON array."""

hotel_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[find_hotels],
    system_prompt=SYSTEM_PROMPT,
)


def _parse_hotels(text: str) -> list[dict]:
    """Extract a JSON array of hotel objects from the agent's raw output.

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
            return [h for h in parsed if isinstance(h, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("hotels"), list):
            return [h for h in parsed["hotels"] if isinstance(h, dict)]
    return []


def run(
    place: str,
    checkin: str | None = None,
    checkout: str | None = None,
    state: str = "",
    country: str = "",
    adults: int | None = None,
    rooms: int | None = None,
    preferences: str | None = None,
    hotel_name: str = "",
) -> list[dict]:
    """Find hotels matching the user's request.

    Only `place` is strictly required — every other field falls back to a
    sensible default if not provided, mirroring what the prompt instructs the
    LLM to do. The defaults applied in the prompt and here must stay in sync.

    Args:
        place:       Destination city or area name. Required.
        checkin:     Check-in date YYYY-MM-DD. If omitted, today + 2 days.
        checkout:    Check-out date YYYY-MM-DD. If omitted, checkin + 2 days.
        state:       Optional state / region.
        country:     Optional country.
        adults:      Number of adult guests. Default 2.
        rooms:       Number of rooms. Default 1.
        preferences: Free-text preference (e.g. "pet friendly with pool").
        hotel_name:  Specific hotel name if the user asked for one.
    """
    today = date.today()

    payload: dict = {
        "place":    place,
        "checkin":  checkin or "",
        "checkout": checkout or "",
        "adults":   adults if adults is not None else "",
        "rooms":    rooms if rooms is not None else "",
        "today":    today.isoformat(),
    }
    if state:
        payload["state"] = state
    if country:
        payload["country"] = country
    if preferences:
        payload["preferences"] = preferences
    if hotel_name:
        payload["hotel_name"] = hotel_name

    query = f"Find hotels matching this request:\n{json.dumps(payload, indent=2)}"

    with langfuse.start_as_current_observation(
        as_type="span",
        name="hotel-agent",
        input=payload,
    ) as span:
        result = hotel_agent(query)
        hotels = _parse_hotels(str(result))
        span.update(output={"hotels": hotels})
        langfuse.flush()
        return hotels


if __name__ == "__main__":
    # No dates, no occupancy — exercise the default-fill path through the agent.
    hotels = run(
        place="Shillong",
        state="Meghalaya",
        country="India",
        preferences="any well-rated comfortable hotel",
    )
    print(f"\nAgent returned {len(hotels)} hotels:\n")
    print(json.dumps(hotels, indent=2, default=str))
