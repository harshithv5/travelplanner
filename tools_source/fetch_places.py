import asyncio
import httpx
from strands import tool
from models.llms import summarize_tool_output

# ---------------------------------------------------------------------------
# Session store — full place data (with lat/lng) keyed by ID.
# The LLM only ever sees the compact summarized list.
# ---------------------------------------------------------------------------
_place_store: dict[str, dict] = {}


def reset_session() -> None:
    _place_store.clear()


def get_stored_places() -> dict[str, dict]:
    return dict(_place_store)


async def _get_places_osm(lat: float, lng: float, radius_m: int) -> list[dict]:
    query = f"""
    [out:json];
    (
      node["historic"](around:{radius_m},{lat},{lng});
      node["tourism"="attraction"](around:{radius_m},{lat},{lng});
      node["tourism"="viewpoint"](around:{radius_m},{lat},{lng});
      node["tourism"="museum"](around:{radius_m},{lat},{lng});
      node["amenity"="marketplace"](around:{radius_m},{lat},{lng});
      node["leisure"="park"](around:{radius_m},{lat},{lng});
      node["shop"="mall"](around:{radius_m},{lat},{lng});
      node["amenity"="place_of_worship"](around:{radius_m},{lat},{lng});
      node["tourism"="gallery"](around:{radius_m},{lat},{lng});
      node["natural"="peak"](around:{radius_m},{lat},{lng});
    );
    out body 50;
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "travelstack-app/1.0"
            }
        )
        data = response.json()

    raw = []
    for i, element in enumerate(data.get("elements", []), start=1):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        if not name:
            continue

        if tags.get("historic"):
            category = "historic"
        elif tags.get("tourism") == "viewpoint":
            category = "viewpoint"
        elif tags.get("tourism") == "museum":
            category = "museum"
        elif tags.get("amenity") == "marketplace":
            category = "food_market"
        elif tags.get("leisure") == "park":
            category = "park"
        elif tags.get("amenity") == "place_of_worship":
            category = "spiritual"
        elif tags.get("tourism") == "gallery":
            category = "art_gallery"
        elif tags.get("natural") == "peak":
            category = "nature"
        else:
            category = "attraction"

        place_id = f"p{i}"

        _place_store[place_id] = {
            "id":       place_id,
            "name":     name,
            "category": category,
            "lat":      element.get("lat"),
            "lng":      element.get("lon"),
        }

        raw.append({"id": place_id, "name": name, "category": category})

    return raw


def _summarize(raw: list[dict]) -> list[dict]:
    if not raw:
        return []

    lines = "\n".join(f"{p['id']} | {p['name']} | {p['category']}" for p in raw)
    prompt = (
        "You are summarizing travel places. For each place write a one-sentence description "
        "of what makes it worth visiting.\n"
        "Return JSON: {\"places\": [{\"id\": \"...\", \"name\": \"...\", "
        "\"category\": \"...\", \"description\": \"...\"}]}\n\n"
        f"Places:\n{lines}"
    )

    result = summarize_tool_output(
        prompt=prompt,
        generation_config={"max_tokens": 800, "response_format": {"type": "json_object"}},
    )
    if isinstance(result, dict) and result.get("places"):
        return result["places"]
    return [{"id": p["id"], "name": p["name"], "category": p["category"], "description": ""} for p in raw]


_SEARCH_RADIUS_M = 100000

@tool
def fetch_places(lat: float, lng: float) -> dict:
    """
    Fetch nearby places of interest from OpenStreetMap for given coordinates.
    Returns attractions, historic sites, viewpoints, museums, parks, markets, and more.

    Args:
        lat: Latitude of the location.
        lng: Longitude of the location.

    Returns:
        A dict with keys: total (int), places (list of {id, name, category, description}).
        Use the returned IDs to reference places in your final answer.
    """
    raw = asyncio.run(_get_places_osm(lat=lat, lng=lng, radius_m=_SEARCH_RADIUS_M))
    compact = _summarize(raw)
    return {"total": len(compact), "places": compact}
