from concurrent.futures import ThreadPoolExecutor
from strands import Agent, tool
from models.llms import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
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

SYSTEM_PROMPT = """You are TravelStack — a travel planning orchestrator. All required trip details are already provided in the input. Your only job is to call the right tools in the right order and present the final itinerary.

## All required information is pre-collected. Do NOT ask any questions.

---

## Tool execution — EXACTLY 2 calls, in this order

### Step 1 — gather_destination_info (ONCE)
Call immediately with all inputs from the request:
- destination_query : destination + any interests (e.g. "waterfalls and viewpoints in Meghalaya")
- place             : destination city
- checkin           : check-in date (YYYY-MM-DD)
- checkout          : check-out date (YYYY-MM-DD)
- adults            : number of adults
- rooms             : number of rooms
- hotel_preferences : user's hotel preference, OR "any well-rated comfortable hotel" if none given —
                      NEVER pass null or empty string

### Step 2 — plan_routes (ONCE, immediately after Step 1 returns)
Use the exact output from gather_destination_info:
- places            = places list from Step 1
- hotels            = hotels list from Step 1
- mode_of_travel    = from input, default "car"
- user_preference   = from input, default "ideal"
- hotel_preference  = hotel routing note from input, or null
- days              = from input, or null
- places_per_day    = from input, default 3
- max_km_per_day    = from input, or null
- place_preferences = {"visited": [...], "optional": [...]} from input, or null

---

## Hard rules
- Call gather_destination_info FIRST, plan_routes SECOND. No other order.
- Never call any tool more than once.
- Never call discover_places or find_hotels separately — gather_destination_info handles both.
- Never fabricate places, hotels, distances, or durations.

---

## Present the final plan clearly

- **Day-by-day:**
  - Day N — Hotel: <name>
  - Places visited: <list>
  - Route legs with distance + duration
  - Day total km / duration
- Hotels used vs unused
- Excluded places (already visited)
- Unvisited places (couldn't fit)
- Grand total distance and duration
- One short prose summary of the trip"""

orchestrator_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[discover_places, find_hotels, gather_destination_info, plan_routes],
    system_prompt=SYSTEM_PROMPT,
)


def run(user_input: str) -> str:
    """
    Build a complete travel itinerary from a fully-formed planning request.
    All required fields must be present in user_input — the orchestrator
    does not ask clarifying questions.
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
