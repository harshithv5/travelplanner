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

_KM_FLEX        = 25  # km the agent may exceed the daily budget to fit one more nearby place
_DEFAULT_PPD    = 3   # default places per day if user doesn't specify
_MAX_EXTRA_DAYS = 2   # extra days allowed beyond `days` hint before spilling to unvisited

SYSTEM_PROMPT = """You are the TravelStack Route Agent — a specialist in planning efficient, realistic travel itineraries.

You receive a JSON input with these fields:
- places            : list of {name, lat, lng, status} — required
                      status values:
                        "active"   → must be planned (core places)
                        "visited"  → user has already been here; exclude entirely
                        "optional" → include only if it fits naturally; skip otherwise
- mode_of_travel    : "car" | "bike" | "public_transport"  (default: "car")
- user_preference   : "ideal" | "cover_as_much_as_possible"  (default: "ideal")
- hotels            : list of {name, lat, lng} — pool of hotels finalized by the user.
                      The agent decides which hotel to stay at on which day(s).
- hotel_preference  : free-text user note about hotel choice (e.g. "prefer luxury for last 2 nights",
                      "stay near Shillong on day 1") — apply if given, otherwise ignore.
- days              : preferred number of travel days  (soft target)
- max_extra_days    : extra days beyond `days` allowed before spilling to unvisited_places
- daily_km_budget   : soft km cap per day
- km_flex           : extra km you may exceed `daily_km_budget` per day
- places_per_day    : hard max places per day

---

## Step 0 — Classify places before planning

Split the place list into three buckets:

1. **excluded** — status = "visited"
   → Remove from planning entirely. Collect names into `excluded_places` in output.

2. **core** — status = "active"
   → Must be scheduled. These fill the itinerary first.

3. **optional** — status = "optional"
   → Hold aside. After core places are scheduled, attempt to slot optional places in
     only if ALL of the following are true for that day:
     - Adding the optional place does not push the day's total km beyond `daily_km_budget + km_flex`
     - The day's `places_per_day` count has not been reached
     - The optional place is geographically close to an already-planned place or hotel that day
   → Optional places that do not fit in any day go into `unvisited_places` (not `excluded_places`).

---

## Step 1 — Build the itinerary from core places

- Use `places_per_day` as a hard cap per day (default 3).
- `daily_km_budget` is a soft cap; you may exceed by `km_flex` km to avoid leaving one nearby core place stranded.
- If `days` is given: try to fit all core places within `days`. If they overflow, use up to `days + max_extra_days` days. Core places still unscheduled after that go to `unvisited_places`.
- If `days` is not given: use however many days are naturally needed.

---

## Step 2 — Assign hotels to days (flexible — agent's call)

The `hotels` list is a **pool**, not a fixed day-wise mapping. You decide which hotel to use on which day(s).

For each day's place cluster:
1. Pick the hotel from the pool that minimises total daily travel distance (closest to the day's centroid or first/last place).
2. Multi-night stays are allowed and encouraged — if consecutive days visit places near the same hotel, keep the user at that hotel for both nights to avoid pointless hotel switches.
3. Apply `hotel_preference` (free text) if it specifies anything overriding (e.g. "luxury for last night", "near beach"). Use your judgement to honour intent. If unclear or not provided, optimise purely on distance.
4. The same hotel can be assigned to as many days as makes sense. A hotel may also go unused if its location is not optimal for any day.
5. **End-of-trip rule**: the very last day may have NO hotel if the user is departing — only set hotel = null on the final day if doing so produces a more efficient plan (e.g. last cluster is far from any hotel and the user is heading home). When hotel = null for a day, do NOT add a return leg.

Record which hotels were used and which were skipped in the output.

---

## Step 3 — Slot optional places

After finalising the core schedule and hotel assignments, revisit each day and attempt to insert optional places:
- Prefer days that have spare capacity (places count < places_per_day AND cumulative km < daily_km_budget).
- For each optional place, find the best insertion position that minimises extra distance.
- Include it only if extra distance keeps the day within `daily_km_budget + km_flex` and places_per_day cap is respected.
- Optional places not slotted anywhere → `unvisited_places`.

---

## Step 4 — Build each day's route

- Start from the day's hotel (if any).
- Use nearest-neighbour to order places starting from the hotel.
- Call `route(origin_lat, origin_lng, dest_lat, dest_lng)` for every leg including the return to hotel.
- End-of-trip rule: if a day has no hotel, skip the return leg.

---

## Step 5 — Return ONLY this JSON (no text outside it):
{
  "mode_of_travel": "<mode>",
  "user_preference": "<preference>",
  "daily_km_budget": <budget>,
  "places_per_day": <limit>,
  "hotel_preference_applied": "<short note on how hotel_preference shaped the plan, or null>",
  "days": [
    {
      "day": 1,
      "hotel": "<hotel name or null>",
      "places_visited": ["<name>", ...],
      "route": [
        {"from": "<hotel>", "to": "<place1>", "distance_km": <km>, "duration_mins": <mins>},
        {"from": "<place1>", "to": "<place2>", "distance_km": <km>, "duration_mins": <mins>},
        {"from": "<last place>", "to": "<hotel>", "distance_km": <km>, "duration_mins": <mins>}
      ],
      "day_total_distance_km": <sum>,
      "day_total_duration_mins": <sum>
    }
  ],
  "hotels_used": ["<hotel name>", ...],
  "hotels_unused": ["<hotel name>", ...],
  "unvisited_places": ["<names that couldn't fit>"],
  "excluded_places": ["<names with status=visited>"],
  "total_distance_km": <grand total>,
  "total_duration_mins": <grand total>
}

---

Rules:
- Call `route` only for actual legs — never speculatively.
- Every non-excluded place must appear in exactly one day's `places_visited` OR in `unvisited_places`.
- On days WITH a hotel: last route entry must be the return to hotel.
- On days WITHOUT a hotel: no return leg.
- Every hotel from the pool must appear in either `hotels_used` or `hotels_unused`.
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
    hotel_preference: str | None = None,
    days: int | None = None,
    places_per_day: int | None = None,
    max_km_per_day: int | None = None,
    place_preferences: dict | None = None,
) -> dict:
    """
    Plan a day-wise travel itinerary for the given places.

    Args:
        places:            List of {name, lat, lng} dicts (output from discovery agent).
        mode_of_travel:    "car" | "bike" | "public_transport"  (default: "car")
        user_preference:   "ideal" | "cover_as_much_as_possible"  (default: "ideal")
        hotels:            Pool of finalized hotels — list of {name, lat, lng}.
                           The agent decides which hotel to use on which day(s).
        hotel_preference:  Free-text note about hotel choice (e.g. "luxury for last 2 nights").
        days:              Preferred number of travel days (up to max_extra_days flex).
        places_per_day:    Hard limit on places per day (default: 3).
        max_km_per_day:    Override the default km budget per day.
        place_preferences: Dict with optional keys:
                             "visited"  → list of place names already seen (excluded from plan)
                             "optional" → list of place names to include only if feasible

    Returns:
        Day-wise route plan dict.
    """
    mode   = mode_of_travel if mode_of_travel in _DAILY_KM_BUDGET else "car"
    pref   = user_preference if user_preference in ("ideal", "cover_as_much_as_possible") else "ideal"
    budget = max_km_per_day if max_km_per_day and max_km_per_day > 0 else _DAILY_KM_BUDGET[mode][pref]

    prefs    = place_preferences or {}
    visited  = set(prefs.get("visited", []))
    optional = set(prefs.get("optional", []))

    place_list = []
    for p in places:
        name = p["name"]
        if name in visited:
            status = "visited"
        elif name in optional:
            status = "optional"
        else:
            status = "active"
        place_list.append({"name": name, "lat": p["lat"], "lng": p["lng"], "status": status})

    payload: dict = {
        "places":          place_list,
        "mode_of_travel":  mode,
        "user_preference": pref,
        "daily_km_budget": budget,
        "km_flex":         _KM_FLEX,
        "places_per_day":  places_per_day or _DEFAULT_PPD,
        "max_extra_days":  _MAX_EXTRA_DAYS,
    }
    if hotels:
        payload["hotels"] = [
            {"name": h["name"], "lat": h["lat"], "lng": h["lng"]} for h in hotels
        ]
    if hotel_preference:
        payload["hotel_preference"] = hotel_preference
    if days:
        payload["days"] = days

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


# _TEST_PLACES = [
#     {"id": "p1", "name": "Sohra (Cherrapunji)",       "category": "attraction", "lat": 25.2777336, "lng": 91.7292416},
#     {"id": "p2", "name": "Shillong",                  "category": "attraction", "lat": 25.5759931, "lng": 91.8827872},
#     {"id": "p3", "name": "Mazar of Hazrat Shahjalal", "category": "spiritual",  "lat": 24.9021885, "lng": 91.8663671},
#     {"id": "p4", "name": "Mawsynram",                 "category": "attraction", "lat": 25.2988198, "lng": 91.5824514},
#     {"id": "p5", "name": "Don Bosco Square",          "category": "historic",   "lat": 25.5698816, "lng": 91.8935312},
#     {"id": "p6", "name": "Sualkuchi",                 "category": "attraction", "lat": 26.1699129, "lng": 91.5708517},
#     {"id": "p7", "name": "Assam State Museum",        "category": "museum",     "lat": 26.1852883, "lng": 91.752382},
#     {"id": "p8", "name": "Guwahati Planetarium",      "category": "museum",     "lat": 26.1914795, "lng": 91.7519783},
# ]

# # Flat pool of finalized hotels — agent picks which to use on which day(s)
# _TEST_HOTELS = [
#     {"name": "Polo Orchid Resort, Cherrapunji", "lat": 25.2780, "lng": 91.7300},
#     {"name": "Hotel Ri Kynjai, Shillong",       "lat": 25.5740, "lng": 91.8820},
#     {"name": "Vivanta Guwahati",                "lat": 26.1900, "lng": 91.7520},
#     {"name": "Radisson Blu Guwahati",           "lat": 26.1500, "lng": 91.7700},
# ]

# print(json.dumps(
#     run(
#         places=_TEST_PLACES,
#         mode_of_travel="car",
#         user_preference="ideal",
#         hotels=_TEST_HOTELS,
#         hotel_preference="prefer staying near Cherrapunji for the first night, and a comfortable Guwahati hotel for the last night",
#         days=3,
#         places_per_day=3,
#         place_preferences={
#             "visited":  ["Shillong", "Don Bosco Square"],
#             "optional": ["Sualkuchi", "Guwahati Planetarium"],
#         },
#     ),
#     indent=2
# ))
