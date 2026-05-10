import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from memory.session_memory import RedisSessionHandler
from memory.pg_memory import save_conversation, add_user_preference
from agents.orchestrator import run as run_orchestrator

load_dotenv()

# ---------------------------------------------------------------------------
# Cerebras client (direct OpenAI-compatible, no Strands)
# ---------------------------------------------------------------------------
_cerebras = OpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY", ""),
    base_url="https://api.cerebras.ai/v1",
)
_MODEL = "llama-3.3-70b"

# ---------------------------------------------------------------------------
# Redis handler (singleton for the process)
# ---------------------------------------------------------------------------
_redis = RedisSessionHandler(ttl_seconds=3600)

# ---------------------------------------------------------------------------
# Required session fields before orchestrator is invoked
# ---------------------------------------------------------------------------
_REQUIRED = ("place", "checkin", "checkout", "adults", "rooms")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are TravelStack Assistant — a friendly conversational agent that helps users plan trips.

Your only job in this layer is to GATHER the information needed to build a travel plan and keep the
user engaged. You do NOT plan routes, search hotels, or call any travel APIs here.

## Required fields you must collect (ask one at a time if missing):
- place           : destination city or region
- checkin         : check-in / trip start date (YYYY-MM-DD)
- checkout        : check-out / trip end date (YYYY-MM-DD)
- adults          : number of adults
- rooms           : number of rooms

## Optional session fields — capture if mentioned, never demand:
- days            : number of travel days (integer)
- mode_of_travel  : "car" | "bike" | "public_transport"
- user_preference : "ideal" (relaxed) | "cover_as_much_as_possible" (packed)
- hotel_preference: free-text note about hotel type (e.g. "luxury", "budget with pool")
- places_per_day  : how many places to visit per day
- max_km_per_day  : max travel distance per day in km
- place_preferences: {"visited": [...], "optional": [...]}
- destination_query: interests + destination (e.g. "waterfalls and viewpoints in Meghalaya")

## How to respond
- Be warm and conversational.
- Ask for ONE missing required field at a time.
- Acknowledge interests naturally and capture them in destination_query.
- When all required fields are present, confirm the plan summary to the user.

## Output format — return ONLY valid JSON, no markdown, no extra text:
{
  "session": {
    "<field_name>": <value>
  },
  "response": "<your conversational message to the user>",
  "user_preference": "<detected long-term preference string or null>"
}

Rules:
- "session"         : only fields newly extracted or updated in THIS turn. {} if nothing new.
- "response"        : the message to send back to the user.
- "user_preference" : short string if a genuine long-term travel preference was detected
                      (e.g. "prefers budget stays", "likes nature trips with family").
                      Set to null if nothing meaningful was detected.
                      This is a permanent profile note — only set when genuinely applicable.
"""


def _build_context_block(session: dict | None) -> str:
    """Render current session state as a context block for the prompt."""
    if not session:
        return "Current session state: (empty — no information collected yet)"

    lines = ["Current session state (already collected):"]
    for key, value in session.items():
        if value not in (None, "", [], {}):
            lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")

    missing = [f for f in _REQUIRED if not session.get(f)]
    if missing:
        lines.append(f"Still missing required fields: {', '.join(missing)}")
    else:
        lines.append("All required fields are present — ready to build the itinerary.")

    return "\n".join(lines)


def _all_required_present(session: dict) -> bool:
    return all(session.get(f) for f in _REQUIRED)


def _build_orchestrator_query(session: dict) -> str:
    """Assemble a natural-language planning query from collected session fields."""
    parts = []

    dest  = session.get("destination_query") or session.get("place", "")
    place = session.get("place", "")
    parts.append(f"Plan a trip to {dest or place}.")

    checkin  = session.get("checkin", "")
    checkout = session.get("checkout", "")
    if checkin and checkout:
        parts.append(f"Dates: {checkin} to {checkout}.")

    days = session.get("days")
    if days:
        parts.append(f"Trip duration: {days} days.")

    adults = session.get("adults", 1)
    rooms  = session.get("rooms", 1)
    parts.append(f"{adults} adults in {rooms} room(s).")

    mode = session.get("mode_of_travel")
    if mode:
        parts.append(f"Mode of travel: {mode}.")

    pref = session.get("user_preference")
    if pref:
        parts.append(f"Trip pace: {pref}.")

    hotel_pref = session.get("hotel_preference")
    if hotel_pref:
        parts.append(f"Hotel preference: {hotel_pref}.")

    ppd = session.get("places_per_day")
    if ppd:
        parts.append(f"Places per day: {ppd}.")

    max_km = session.get("max_km_per_day")
    if max_km:
        parts.append(f"Max km per day: {max_km}.")

    place_prefs = session.get("place_preferences")
    if isinstance(place_prefs, dict):
        visited  = place_prefs.get("visited", [])
        optional = place_prefs.get("optional", [])
        if visited:
            parts.append(f"Already visited (exclude): {', '.join(visited)}.")
        if optional:
            parts.append(f"Optional places: {', '.join(optional)}.")

    return " ".join(parts)


def run_user_query(user_id: str, session_id: str, user_query: str) -> str:
    """
    Process a user message in the conversational gathering phase.

    Flow:
      1. Load session state from Redis.
      2. Inject state into the Cerebras prompt.
      3. Parse {session, response, user_preference} from the LLM reply.
      4. Update Redis with new session fields.
      5. Persist user_preference to Postgres if detected.
      6. Save conversation turn to Postgres.
      7. If all required fields are now present, invoke the orchestrator
         and return the full itinerary.

    Returns:
        Conversational reply, or the full travel plan when session is complete.
    """
    user_session_id = f"{user_id}:{session_id}"

    # 1 — Load current session state from Redis
    current_session: dict = _redis.get(user_session_id) or {}

    # 2 — Build context-aware prompt and call Cerebras
    context_block     = _build_context_block(current_session)
    augmented_message = f"{context_block}\n\nUser message: {user_query}"

    completion = _cerebras.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": augmented_message},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    raw = completion.choices[0].message.content.strip()

    # 3 — Parse JSON response
    try:
        parsed              = json.loads(raw)
        new_session_fields  = parsed.get("session") or {}
        reply: str          = parsed.get("response") or raw
        user_pref: str | None = parsed.get("user_preference") or None
    except (json.JSONDecodeError, AttributeError):
        new_session_fields = {}
        reply              = raw
        user_pref          = None

    # 4 — Update Redis with new session fields
    if new_session_fields:
        _redis.update(user_session_id, new_session_fields)

    # 5 — Persist user_preference to Postgres if detected
    if user_pref:
        add_user_preference(user_id=user_id, preference=user_pref)

    # 6 — Save conversation turn to Postgres
    save_conversation(
        user_id=user_id,
        session_id=session_id,
        user_question=user_query,
        system_response=reply,
    )

    # 7 — If all required fields are now present, invoke the orchestrator
    merged_session = {**current_session, **new_session_fields}
    if _all_required_present(merged_session):
        orchestrator_query = _build_orchestrator_query(merged_session)
        itinerary = run_orchestrator(orchestrator_query)

        save_conversation(
            user_id=user_id,
            session_id=session_id,
            user_question="[orchestrator invoked]",
            system_response=itinerary,
        )

        return itinerary

    return reply
