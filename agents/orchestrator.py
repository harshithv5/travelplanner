from concurrent.futures import ThreadPoolExecutor
from strands import Agent, tool
from models import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from agents import discovery as discovery_agent_module
from agents import hotels as hotels_agent_module
from agents import route_agent as route_agent_module
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

# Fallback preference used when the user gives no specific hotel ask.
# Keeping it general ensures the hotel agent never returns an empty list
# due to overly strict matching.
_DEFAULT_HOTEL_PREFERENCE = "any well-rated comfortable hotel"


# ---------------------------------------------------------------------------
# Sub-agents exposed as tools
# ---------------------------------------------------------------------------

@tool
def discover_places(destination_query: str) -> list[dict]:
    """
    Discover places of interest at a destination — attractions, museums, viewpoints,
    historic sites, parks, markets, spiritual sites, and natural landmarks.

    The query should describe both the destination AND any user interests so the
    discovery agent can filter by category. Examples:
      - "Places in Meghalaya"
      - "Waterfalls and viewpoints in Cherrapunji"
      - "Historic sites and museums in Shillong"

    Args:
        destination_query: Natural language string with destination and interests.

    Returns:
        List of place dicts: {id, name, category, lat, lng, description}.
        Pass this list directly to plan_routes as the `places` argument.
    """
    return discovery_agent_module.run(destination_query)


@tool
def find_hotels(
    place: str,
    checkin: str,
    checkout: str,
    adults: int = 1,
    rooms: int = 1,
    preferences: str | None = None,
) -> list[dict]:
    """
    Find available hotels at a destination matching user preferences.
    Returns full hotel details (location, price, facilities, room types) so the
    output can be passed directly to plan_routes for day-wise hotel assignment.

    Args:
        place:       Destination city or area name (e.g. "Shillong", "Goa").
        checkin:     Check-in date in YYYY-MM-DD format.
        checkout:    Check-out date in YYYY-MM-DD format (must be after checkin).
        adults:      Total adult guests (default 1).
        rooms:       Number of rooms required (default 1).
                     Example: 4 adults in 2 rooms → adults=4, rooms=2.
        preferences: Free-text preference describing hotel type. Examples:
                       "couple friendly with pool"
                       "budget under 5000 per night"
                       "luxury 4-star with free cancellation"
                       "pet friendly"
                     Pass None or omit if user has no specific hotel preferences.

    Returns:
        List of hotel dicts: {hotel_id, name, stars, rating, lat, lng,
        price_per_night, facilities, rooms, ...}.
        Pass this list directly to plan_routes as the `hotels` argument.
    """
    return hotels_agent_module.run(
        place=place,
        checkin=checkin,
        checkout=checkout,
        adults=adults,
        rooms=rooms,
        preferences=preferences or _DEFAULT_HOTEL_PREFERENCE,
    )


@tool
def plan_routes(
    places: list[dict],
    hotels: list[dict] | None = None,
    mode_of_travel: str = "car",
    user_preference: str = "ideal",
    hotel_preference: str | None = None,
    days: int | None = None,
    places_per_day: int | None = None,
    max_km_per_day: int | None = None,
    place_preferences: dict | None = None,
) -> dict:
    """
    Build a complete day-wise travel itinerary using places (from discover_places)
    and a hotel pool (from find_hotels). Uses nearest-neighbour routing to minimise
    backtracking, assigns the optimal hotel from the pool to each day's cluster,
    and returns full route legs (distance + duration) for every leg including the
    return to hotel.

    ALWAYS call this LAST, after discover_places and find_hotels have completed.

    Args:
        places:            Output from discover_places — list of {name, lat, lng, ...}.
        hotels:            Output from find_hotels — list of {name, lat, lng, ...}.
                           Treated as a pool; the agent picks which to use per day.
        mode_of_travel:    "car" | "bike" | "public_transport" (default "car").
        user_preference:   "ideal" | "cover_as_much_as_possible" (default "ideal").
                           "ideal" = relaxed pace; "cover_as_much_as_possible" = packed days.
        hotel_preference:  Free-text note on hotel choice, e.g.
                           "stay near Cherrapunji on day 1, luxury hotel last night".
                           Pass None if no specific hotel preference.
        days:              Preferred number of travel days (soft target).
        places_per_day:    Hard limit on places per day (default 3).
        max_km_per_day:    Override default km budget per day (user-specified).
        place_preferences: Dict with two optional keys:
                             "visited"  → list of place names already visited (excluded)
                             "optional" → list of place names to include only if feasible

    Returns:
        Day-wise itinerary dict: {mode_of_travel, daily_km_budget, places_per_day,
        days[], hotels_used, hotels_unused, unvisited_places, excluded_places,
        total_distance_km, total_duration_mins}.
    """
    return route_agent_module.run(
        places=places,
        mode_of_travel=mode_of_travel,
        user_preference=user_preference,
        hotels=hotels,
        hotel_preference=hotel_preference,
        days=days,
        places_per_day=places_per_day,
        max_km_per_day=max_km_per_day,
        place_preferences=place_preferences,
    )


@tool
def gather_destination_info(
    destination_query: str,
    place: str,
    checkin: str,
    checkout: str,
    adults: int = 1,
    rooms: int = 1,
    hotel_preferences: str | None = None,
) -> dict:
    """
    Convenience tool that runs discover_places AND find_hotels in PARALLEL.
    Use this instead of calling the two separately when you already have all
    user info gathered — it's faster because both API-heavy calls run concurrently.

    Args:
        destination_query: Query for discover_places (destination + interests).
        place:             Destination city for find_hotels.
        checkin:           Check-in date (YYYY-MM-DD).
        checkout:          Check-out date (YYYY-MM-DD).
        adults:            Adult guests (default 1).
        rooms:             Number of rooms (default 1).
        hotel_preferences: Free-text hotel preference, optional.

    Returns:
        {"places": [...], "hotels": [...]} — feed both directly to plan_routes.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        places_future = executor.submit(
            discovery_agent_module.run,
            destination_query,
        )
        hotels_future = executor.submit(
            hotels_agent_module.run,
            place=place,
            checkin=checkin,
            checkout=checkout,
            adults=adults,
            rooms=rooms,
            preferences=hotel_preferences or _DEFAULT_HOTEL_PREFERENCE,
        )
        return {
            "places": places_future.result(),
            "hotels": hotels_future.result(),
        }


# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are TravelStack — a travel planning orchestrator that builds full, day-by-day, route-optimised itineraries by coordinating four specialist tools.

Each tool wraps a sub-agent. Tool calls are EXPENSIVE (each runs a full LLM agent loop). Use them strictly and minimally — never speculatively, never repeatedly.

## Your tools

1. **gather_destination_info(destination_query, place, checkin, checkout, adults, rooms, hotel_preferences)** — PRIMARY
   Runs discover_places + find_hotels IN PARALLEL inside one call.
   This is the DEFAULT path for steps 2 and 3 of the workflow. Always prefer this
   over calling discover_places + find_hotels separately.

2. **discover_places(destination_query)** — FALLBACK ONLY
   Use ONLY if you specifically need to re-run discovery (e.g. user adds a new
   destination later in the conversation). Do NOT use this in the standard flow —
   gather_destination_info already calls it internally.

3. **find_hotels(...)** — FALLBACK ONLY
   Use ONLY if you specifically need to re-run hotel search (e.g. user changes
   dates or preferences mid-conversation). Do NOT use this in the standard flow.

4. **plan_routes(places, hotels, mode_of_travel, user_preference, hotel_preference,
                 days, places_per_day, max_km_per_day, place_preferences)** — FINAL STEP
   Builds the day-wise itinerary. MUST be called AFTER hotels and places are
   already in hand. Never call it before gather_destination_info / find_hotels.

---

## Tool call budget (strict)

In a normal flow you should make EXACTLY 2 tool calls per planning request:
  1. gather_destination_info  (ONCE)
  2. plan_routes              (ONCE, after step 1)

That's it. Do not call discovery or hotels again unless the user explicitly
changes destination, dates, occupancy, or preferences mid-conversation.

If you find yourself wanting to re-run a tool, ask: did the user actually change
something? If not, use the data already returned.

---

## Strict ordering

PHASE 1 (no tools) → PHASE 2 (gather_destination_info) → PHASE 3 (plan_routes) → PHASE 4 (present)

- Routes is STRICTLY the LAST tool call. Never run plan_routes before hotels exist.
- Discovery/hotels are STRICTLY before routes. Never run them after routes.

---

## PHASE 1 — Gather ALL information FIRST (NO tool calls)

Required from the user:
- Destination (city/region)
- Check-in date (YYYY-MM-DD)
- Check-out date (YYYY-MM-DD)
- Number of adults
- Number of rooms

If any required field is missing, ask ONE concise targeted question. Do NOT
call any tool until all required fields are present.

OPTIONAL fields — try to gather but DO NOT block on these:
- Mode of travel (car / bike / public_transport — default: car)
- Trip pace (ideal / cover_as_much_as_possible — default: ideal)
- Place interests (waterfalls, museums, food, historic, etc.)
- Hotel preferences (couple friendly, pool, luxury, budget, etc.)
- Places per day (default: 3)
- Already visited places (to exclude)
- Optional places (include only if convenient)
- Hotel preference notes for routing (e.g. "luxury for last night")

## PHASE 2 — Single parallel call to gather_destination_info

Build the call:
- destination_query  = "<destination> + <interests if any>"  (e.g. "Waterfalls and viewpoints in Meghalaya")
- place              = the destination city
- checkin / checkout = the dates
- adults / rooms     = occupancy

Hotel preference handling — CAREFUL:
- If the user gave a specific hotel preference → pass it verbatim as `hotel_preferences`.
- If the user gave NO hotel preference (or it is empty/unclear) → pass the
  generic fallback `"any well-rated comfortable hotel"` as `hotel_preferences`.
- NEVER pass an empty string or None — always pass either the user's preference
  or the generic fallback. This guarantees the hotel agent returns usable results.

## PHASE 3 — One call to plan_routes

After Phase 2 returns `{places, hotels}`, call plan_routes ONCE with:
- places            = the full places list returned in Phase 2
- hotels            = the full hotels list returned in Phase 2
- mode_of_travel    = user-given or "car"
- user_preference   = user-given or "ideal"
- hotel_preference  = user's free-text routing note about hotel choice (or null)
- days              = (checkout - checkin) in days, or what user explicitly said
- places_per_day    = user-given or 3
- max_km_per_day    = user-given or null
- place_preferences = {"visited": [...], "optional": [...]} when user mentioned any

## PHASE 4 — Present the final plan

Format clearly:
- Day-by-day:
  • Day N — Hotel: <name>
  • Places visited: <list>
  • Route legs with distance + duration
  • Day total km / duration
- Hotels used vs unused
- Excluded places (already visited)
- Unvisited places (couldn't fit) with reason
- Grand totals
- One short prose summary

---

## Hard rules

- NEVER call any tool before required user info is complete.
- NEVER call plan_routes before hotels are obtained.
- NEVER call discovery or hotels after plan_routes has run.
- NEVER call the same tool twice unless the user actually changed inputs.
- NEVER pass empty/None for hotel_preferences — fall back to "any well-rated comfortable hotel".
- NEVER fabricate place/hotel names, coordinates, distances, or durations.
- If a tool returns empty results, explain the issue and ask the user to refine input."""

orchestrator_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[discover_places, find_hotels, gather_destination_info, plan_routes],
    system_prompt=SYSTEM_PROMPT,
)


def run(user_input: str) -> str:
    """
    Process a user travel-planning request through the orchestrator.

    Args:
        user_input: Natural language travel request.

    Returns:
        The orchestrator's reply — either a clarifying question (if info is
        missing) or the final formatted itinerary.
    """
    with langfuse.start_as_current_observation(
        as_type="span",
        name="orchestrator",
        input={"query": user_input},
    ) as span:
        response = str(orchestrator_agent(user_input))
        span.update(output={"response": response})
        langfuse.flush()
        return response


# if __name__ == "__main__":
#     print(run(
#         "I want to plan a 2-day trip to Chikkamagluru from 2026-06-01 to 2026-06-03. "
#         "We are 3 adults in 2 rooms, travelling by car. relaxed trip probaly limited places per day and more relaxing , any set of good hotels and places would be fine"
#     ))
