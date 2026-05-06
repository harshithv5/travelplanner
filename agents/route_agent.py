import json
from strands import Agent
from models import mistral  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
from tools_source.routes import route
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()

# Default daily travel budget in km by mode + preference
_DAILY_KM_BUDGET = {
    "car":              {"ideal": 120, "cover_as_much_as_possible": 200},
    "bike":             {"ideal": 40,  "cover_as_much_as_possible": 80},
    "public_transport": {"ideal": 60,  "cover_as_much_as_possible": 120},
}

_KM_FLEX = 25  # km the agent may exceed the daily budget to avoid splitting a last close place

SYSTEM_PROMPT = """You are the TravelStack Route Agent — a specialist in planning efficient, realistic travel itineraries.

You receive a JSON input with these fields:
- places            : list of {name, lat, lng} — required
- mode_of_travel    : "car" | "bike" | "public_transport"  (default: "car")
- user_preference   : "ideal" | "cover_as_much_as_possible"  (default: "ideal")
- hotels            : list of {day, name, lat, lng}  (optional — day-wise stays keyed by day number)
- days              : hard limit on travel days  (optional)
- daily_km_budget   : soft km cap per day  (system-provided; you may exceed by up to `km_flex` km if it meaningfully reduces leftover places)
- km_flex           : how many extra km you may use beyond `daily_km_budget` on any single day
- places_per_day    : max number of places to visit per day  (optional — hard limit per day)

---

## Planning — Always day-wise

### Step 1 — Determine day structure
- The `daily_km_budget` is a **soft cap**. You may exceed it by up to `km_flex` km per day if doing so allows you to include one more nearby place without leaving it stranded.
- `places_per_day` (if given) is a **hard limit** — never exceed it regardless of km budget.
- If `days` is provided, it is a **hard limit** — do not plan more days than this. Any places that cannot fit go into `unvisited_places`.
- If `days` is not provided, use as many days as naturally needed to cover all places.
- Hotel assignment: use the hotel matching the day number. If only one hotel is given, use it for all days. If a day number has no hotel entry, reuse the last hotel provided — except for the final day if no hotel is given for it (treat that as end-of-trip: no return leg).

### Step 2 — For each day
- Start from the day's hotel (if one exists).
- Use nearest-neighbour to order the day's places starting from the hotel.
- Call `route(origin_lat, origin_lng, dest_lat, dest_lng)` for every leg: hotel→place1, place1→place2, ..., lastPlace→hotel.
- **End-of-trip rule**: if the current day has no hotel (i.e. it is the final day and no hotel was provided for it), do NOT add a return leg — the trip ends at the last place visited.
- Track cumulative km as you add each leg. Stop adding places to the day when the next place would push the total beyond `daily_km_budget + km_flex` or when `places_per_day` is reached.

### Step 3 — Return ONLY this JSON (no text outside it):
{
  "mode_of_travel": "<mode>",
  "user_preference": "<preference>",
  "daily_km_budget": <budget>,
  "days": [
    {
      "day": 1,
      "hotel": "<hotel name or null>",
      "places_visited": ["<name>", ...],
      "route": [
        {"from": "<hotel name>", "to": "<place1 name>", "distance_km": <km>, "duration_mins": <mins>},
        {"from": "<place1 name>", "to": "<place2 name>", "distance_km": <km>, "duration_mins": <mins>},
        {"from": "<last place>", "to": "<hotel name>", "distance_km": <km>, "duration_mins": <mins>}
      ],
      "day_total_distance_km": <sum of all route legs>,
      "day_total_duration_mins": <sum of all route legs>
    },
    ...
  ],
  "unvisited_places": ["<name>", ...],
  "total_distance_km": <grand total across all days>,
  "total_duration_mins": <grand total across all days>
}

---

Rules:
- Call `route` only for actual legs — never speculatively.
- Every place must appear in exactly one day's `places_visited` OR in `unvisited_places` — never both, never missing.
- On days WITH a hotel: the last route entry must be the return to hotel.
- On the final day WITHOUT a hotel: no return leg — route ends at the last place.
- Do not include any text or markdown outside the JSON object."""

route_agent = Agent(
    model=mistral,  # swap: ollama | groq | groq_litellm | gemini | mistral | mistral_json | cerebras
    tools=[route],
    system_prompt=SYSTEM_PROMPT,
)


def run(
    places: list[dict],
    mode_of_travel: str = "car",
    user_preference: str = "ideal",
    hotels: list[dict] | None = None,
    days: int | None = None,
    places_per_day: int | None = None,
    max_km_per_day: int | None = None,
) -> dict:
    """
    Plan a day-wise travel itinerary for the given places.

    Args:
        places:          List of {name, lat, lng} dicts (output from discovery agent).
        mode_of_travel:  "car" | "bike" | "public_transport"  (default: "car")
        user_preference: "ideal" | "cover_as_much_as_possible"  (default: "ideal")
        hotels:          Optional list of {day, name, lat, lng} for day-wise stays.
                         If the last day has no hotel entry, it is treated as end-of-trip.
        days:            Hard limit on number of travel days. Remaining places go to unvisited.
        places_per_day:  Hard limit on how many places to visit per day.
        max_km_per_day:  Override the default km budget per day (user-specified).

    Returns:
        Day-wise route plan dict.
    """
    mode = mode_of_travel if mode_of_travel in _DAILY_KM_BUDGET else "car"
    pref = user_preference if user_preference in ("ideal", "cover_as_much_as_possible") else "ideal"
    budget = max_km_per_day if max_km_per_day and max_km_per_day > 0 else _DAILY_KM_BUDGET[mode][pref]

    payload: dict = {
        "places":          [{"name": p["name"], "lat": p["lat"], "lng": p["lng"]} for p in places],
        "mode_of_travel":  mode,
        "user_preference": pref,
        "daily_km_budget": budget,
        "km_flex":         _KM_FLEX,
    }
    if hotels:
        payload["hotels"] = hotels
    if days:
        payload["days"] = days
    if places_per_day:
        payload["places_per_day"] = places_per_day

    query = f"Plan the travel itinerary for this input:\n{json.dumps(payload, indent=2)}"

    with langfuse.start_as_current_observation(
        as_type="span",
        name="route-agent",
        input=payload
    ) as span:
        result = route_agent(query)
        try:
            output = json.loads(str(result))
        except (json.JSONDecodeError, TypeError):
            output = {"raw": str(result)}
        span.update(output=output)
        langfuse.flush()
        return output


_TEST_PLACES = [
    {"id": "p1", "name": "Sohra (Cherrapunji)",       "category": "attraction", "lat": 25.2777336, "lng": 91.7292416},
    {"id": "p2", "name": "Shillong",                  "category": "attraction", "lat": 25.5759931, "lng": 91.8827872},
    {"id": "p3", "name": "Mazar of Hazrat Shahjalal", "category": "spiritual",  "lat": 24.9021885, "lng": 91.8663671},
    {"id": "p4", "name": "Mawsynram",                 "category": "attraction", "lat": 25.2988198, "lng": 91.5824514},
    {"id": "p5", "name": "Don Bosco Square",          "category": "historic",   "lat": 25.5698816, "lng": 91.8935312},
    {"id": "p6", "name": "Sualkuchi",                 "category": "attraction", "lat": 26.1699129, "lng": 91.5708517},
    # {"id": "p7", "name": "Assam State Museum",        "category": "museum",     "lat": 26.1852883, "lng": 91.752382},
    # {"id": "p8", "name": "Guwahati Planetarium",      "category": "museum",     "lat": 26.1914795, "lng": 91.7519783},
]

_TEST_HOTELS = [
    {"day": 1, "name": "Polo Orchid Resort", "lat": 25.2780, "lng": 91.7300},
    {"day": 2, "name": "Hotel Ri Kynjai",    "lat": 25.5740, "lng": 91.8820},
    {"day": 3, "name": "Vivanta Guwahati",   "lat": 26.1900, "lng": 91.7520},
    # Day 4 has no hotel → treated as end-of-trip, no return leg    
]

print(json.dumps(
    run(
        places=_TEST_PLACES,
        mode_of_travel="car",
        user_preference="ideal",
        hotels=_TEST_HOTELS,
        days=4,
        places_per_day=2,
    ),
    indent=2
))
