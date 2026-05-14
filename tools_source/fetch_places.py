import asyncio
import hashlib
import json
import httpx
from strands import tool
from memory.knowledge_base import KnowledgeBase
from models.llms import groq_client, GROQ_MODEL_ID
from prompts.description_enhancer import SYSTEM_PROMPT as _ENHANCER_SYSTEM_PROMPT, USER_TEMPLATE as _ENHANCER_USER_TEMPLATE
from dotenv import load_dotenv
load_dotenv()


_SEARCH_RADIUS_M = 100_000  # 100 km — maximum practical OSM radius
_COLLECTION = "Discovery"
_kb = KnowledgeBase()


def _enhance_descriptions(places: list[dict]) -> dict[str, str]:
    """Return {place_id: enhanced_description} via Groq llama-3.3-70b-versatile.

    Falls back to an empty dict on any error so callers keep the original OSM
    descriptions and continue.
    """
    if not places:
        return {}

    payload = [
        {
            "id":          p.get("id"),
            "name":        p.get("name"),
            "category":    p.get("category"),
            "state":       p.get("state"),
            "country":     p.get("country"),
            "description": p.get("description", ""),
        }
        for p in places
    ]

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": _ENHANCER_SYSTEM_PROMPT},
                {"role": "user",   "content": _ENHANCER_USER_TEMPLATE.format(
                    places_json=json.dumps(payload, ensure_ascii=False)
                )},
            ],
            temperature=0.4,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        parsed = json.loads(raw)
        entries = parsed.get("descriptions", []) if isinstance(parsed, dict) else []
        return {
            str(e["id"]): e["description"].strip()
            for e in entries
            if isinstance(e, dict) and e.get("id") is not None and e.get("description")
        }
    except Exception:
        return {}


async def _reverse_geocode(lat: float, lng: float) -> tuple[str, str]:
    """Return (state, country) from Nominatim for the centre coordinate."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "json", "lat": lat, "lon": lng},
                headers={"User-Agent": "travelstack-app/1.0"},
            )
            addr = r.json().get("address", {})
            return addr.get("state", ""), addr.get("country", "")
    except Exception:
        return "", ""


def _resolve_category(tags: dict) -> str:
    if tags.get("historic"):
        return "historic"
    if tags.get("tourism") == "viewpoint":
        return "viewpoint"
    if tags.get("tourism") == "museum":
        return "museum"
    if tags.get("amenity") == "marketplace":
        return "food_market"
    if tags.get("leisure") == "park":
        return "park"
    if tags.get("amenity") == "place_of_worship":
        return "spiritual"
    if tags.get("tourism") == "gallery":
        return "art_gallery"
    if tags.get("natural") == "peak":
        return "nature"
    return "attraction"


async def _get_places_osm(lat: float, lng: float, state: str, country: str) -> list[dict]:
    query = f"""
    [out:json];
    (
      node["historic"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["tourism"="attraction"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["tourism"="viewpoint"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["tourism"="museum"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["amenity"="marketplace"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["leisure"="park"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["shop"="mall"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["amenity"="place_of_worship"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["tourism"="gallery"](around:{_SEARCH_RADIUS_M},{lat},{lng});
      node["natural"="peak"](around:{_SEARCH_RADIUS_M},{lat},{lng});
    );
    out body 50;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    data = None
    last_error = None
    async with httpx.AsyncClient(timeout=60) as client:
        for url in endpoints:
            try:
                response = await client.post(
                    url,
                    data={"data": query},
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "travelstack-app/1.0",
                    },
                )
                if response.status_code != 200:
                    last_error = f"{url} -> HTTP {response.status_code}: {response.text[:200]}"
                    continue
                data = response.json()
                break
            except Exception as e:
                last_error = f"{url} -> {type(e).__name__}: {e}"
                continue

    if data is None:
        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")

    places = []
    for i, element in enumerate(data.get("elements", []), start=1):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        if not name:
            continue

        category = _resolve_category(tags)

        description = (
            tags.get("description:en")
            or tags.get("description")
            or tags.get("note:en")
            or tags.get("note")
            or f"{category.replace('_', ' ').title()} in {state or country or 'the region'}"
        )

        places.append({
            "id":          f"p{i}",
            "name":        name,
            "category":    category,
            "description": description,
            "state":       state,
            "country":     country,
            "lat":         element.get("lat"),
            "lng":         element.get("lon"),
        })

    return places


def _place_point_id(name: str, state: str, country: str) -> int:
    """Stable integer ID derived from name + region — avoids duplicates on re-insert."""
    digest = hashlib.md5(f"{name}{state}{country}".encode()).hexdigest()
    return int(digest[:15], 16)


def fetch_places_raw(lat: float, lng: float, preference: str = "") -> dict:
    """
    Fetch places for (lat, lng), using Qdrant cache when available.

    Flow:
      1. Reverse-geocode to get state + country.
      2. If Qdrant already has places for that region:
           - preference given → hybrid semantic search on description
           - no preference   → return all cached places for the region
      3. On cache miss: fetch from OSM, write results to Qdrant, return.
    """
    state, country = asyncio.run(_reverse_geocode(lat, lng))

    _kb.ensure_collection(_COLLECTION, keyword_index_fields=["state", "country"])
    filters = {k: v for k, v in {"state": state, "country": country}.items() if v}

    if filters:
        if preference:
            cached = _kb.search(_COLLECTION, preference, top_k=20, metadata_filters=filters)
        else:
            cached = _kb.dense_search(
                _COLLECTION, "tourist attraction place", top_k=50, metadata_filters=filters
            )

        if cached:
            places = [hit["metadata"] for hit in cached]
            return {"total": len(places), "places": places}

    # Cache miss — hit OSM and persist results
    places = asyncio.run(_get_places_osm(lat, lng, state, country))

    if places:
        enhanced = _enhance_descriptions(places)
        if enhanced:
            for p in places:
                better = enhanced.get(str(p["id"]))
                if better:
                    p["description"] = better

        points = [
            {
                "id":       _place_point_id(p["name"], state, country),
                "text":     f"{p['name']} — {p['description']}",
                "metadata": p,
            }
            for p in places
        ]
        _kb.insert_batch(_COLLECTION, points)

    return {"places": places}



@tool
def fetch_places(lat: float, lng: float, preference: str = "") -> dict:
    """
    Fetch nearby places of interest from OpenStreetMap for given coordinates.
    Returns attractions, historic sites, viewpoints, museums, parks, markets, and more.
    Results are cached in Qdrant; subsequent calls for the same region skip the OSM fetch.
    If a preference is provided (e.g. "waterfall", "historic temple"), semantic search is
    used to rank cached places by relevance to that preference.

    Args:
        lat:        Latitude of the location.
        lng:        Longitude of the location.
        preference: Optional interest filter (e.g. "museum", "waterfall", "street food").

    Returns:
        Dict with keys: total (int), places (list of {id, name, category, description, state, country, lat, lng}).
        Use the returned IDs to reference places in your final answer.
        

    Fetch travel places for a resolved location.

    Call ONLY after geocode resolution.
    Call ONCE per destination/category combination.
    Do NOT call repeatedly unless user explicitly asks for broader results.
    Returns top ranked places.

        
    """
    result = fetch_places_raw(lat, lng, preference)
    return json.dumps(result, separators=(",", ":"))


# if __name__ == "__main__":
#     import json
#     # Shillong, Meghalaya — change coordinates to test another location
#     result = fetch_places_raw(lat=25.5788, lng=91.8933)
#     print(f"Total places found: {result['total']}\n")
#     print(json.dumps(result["places"], indent=2))
