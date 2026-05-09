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
        preferences=preferences,
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
            preferences=hotel_preferences,
        )
        return {
            "places": places_future.result(),
            "hotels": hotels_future.result(),
        }


# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are TravelStack — a travel planning orchestrator that builds full, day-by-day, route-optimised itineraries by coordinating three specialist tools.

## Your tools

1. **discover_places(destination_query)**
   Discovers places of interest (attractions, museums, viewpoints, historic sites,
   parks, spiritual sites). Returns a list of places with lat/lng for routing.

2. **find_hotels(place, checkin, checkout, adults, rooms, preferences)**
   Finds available hotels matching user preferences. Returns full hotel details
   (lat/lng, price, facilities, rooms) for downstream route planning.

3. **gather_destination_info(...)**
   Convenience helper that runs discover_places + find_hotels IN PARALLEL.
   Prefer this whenever you have all the info needed for both — it is faster.

4. **plan_routes(places, hotels, mode_of_travel, user_preference, hotel_preference,
                days, places_per_day, max_km_per_day, place_preferences)**
   Builds the final day-wise itinerary from the places + hotels lists.
   Returns full route legs (distance + duration) per day with hotel assignments.

---

## Strict workflow

### PHASE 1 — Gather ALL information FIRST (no tool calls yet)

Before invoking ANY tool, make sure you have these from the user:

REQUIRED:
- Destination (city or region)
- Check-in date (YYYY-MM-DD)
- Check-out date (YYYY-MM-DD)
- Number of adults
- Number of rooms

OPTIONAL but ASK for:
- Mode of travel (car / bike / public_transport — default car)
- Trip pace (ideal / cover_as_much_as_possible — default ideal)
- Place interests (e.g. waterfalls, museums, food markets, historic sites)
- Hotel preferences (e.g. couple friendly, with pool, luxury, budget)
- Places per day preference (default 3)
- Already visited places (to exclude)
- Optional places (include only if convenient)
- Hotel preference notes (e.g. "luxury for last night", "near Cherrapunji on day 1")

If ANY required field is missing, ask ONE clear, targeted question to gather what's missing. Do NOT call any tool until ALL required fields are present.

### PHASE 2 — Run discovery + hotels in parallel

Once required info is complete, call **gather_destination_info** in a single tool call.
This runs discover_places + find_hotels concurrently.

(If you must call them separately for any reason, emit BOTH tool calls in the SAME response so the framework can run them in parallel — never serially.)

### PHASE 3 — Plan the route

Pass the `places` and `hotels` from Phase 2 to **plan_routes**, along with:
- mode_of_travel, user_preference (from user input or defaults)
- hotel_preference (free-text note, if user gave one)
- days (derived from check-in/out dates)
- places_per_day (user-given or default)
- place_preferences (dict with "visited" and "optional" lists if user mentioned any)

### PHASE 4 — Present the final plan

Format the result for the user with:
- Day-by-day breakdown:
  • Day N — Hotel: <name>
  • Places visited: <list>
  • Route legs with distance + duration
  • Day total km / duration
- Hotels used vs unused (from the pool)
- Excluded places (already visited)
- Unvisited places (couldn't fit) with reason
- Grand totals (km + duration)
- A short prose summary at the end

---

## Hard rules

- NEVER call any tool before all REQUIRED user info is gathered.
- NEVER fabricate place names, hotel names, coordinates, distances, or durations — use only what tools return.
- ALWAYS use gather_destination_info (parallel) over discover_places + find_hotels separately.
- ALWAYS call plan_routes LAST, exactly once, with the outputs of phase 2.
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


if __name__ == "__main__":
    print(run(
        "I want to plan a 3-day trip to Meghalaya from 2026-06-01 to 2026-06-03. "
        "We are 4 adults in 2 rooms, travelling by car. Interested in waterfalls, "
        "viewpoints, and historic sites. Prefer couple-friendly hotels with pool. "
        "We have already been to Shillong city centre, and Sualkuchi is optional. "
        "Please plan no more than 3 places per day."
    ))
