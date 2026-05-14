import asyncio
import hashlib
import json
import os
from datetime import date, timedelta

import httpx
from strands import tool

from memory.knowledge_base import KnowledgeBase
from models.llms import groq_client, GROQ_MODEL_ID
from prompts.hotel_description_enhancer import (
    SYSTEM_PROMPT as _ENHANCER_SYSTEM_PROMPT,
    USER_TEMPLATE  as _ENHANCER_USER_TEMPLATE,
)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "booking-com15.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"

_COLLECTION = "Hotels"
_kb = KnowledgeBase()


def _headers() -> dict:
    return {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key":  RAPIDAPI_KEY,
    }


def _hotel_point_id(hotel_id: str) -> int:
    """Stable integer ID for Qdrant derived from the Booking hotel_id."""
    digest = hashlib.md5(hotel_id.encode()).hexdigest()
    return int(digest[:15], 16)


# ---------------------------------------------------------------------------
# Booking.com API helpers
# ---------------------------------------------------------------------------

async def _get_destination(client: httpx.AsyncClient, city: str) -> tuple[str, str]:
    """Resolve city name to Booking.com dest_id and dest_type."""
    resp = await client.get(
        f"{BASE_URL}/api/v1/hotels/searchDestination",
        params={"query": city},
        headers=_headers(),
    )
    data = resp.json()
    first = data["data"][0]
    return first["dest_id"], first["dest_type"]


async def _get_description(client: httpx.AsyncClient, hotel_id: str) -> str:
    try:
        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/getDescriptionAndInfo",
            params={"hotel_id": hotel_id},
            headers=_headers(),
        )
        data = resp.json()
        descriptions = data.get("data", {}).get("description", [])
        for item in descriptions:
            if item.get("descriptiontype_id") == 6:
                return item.get("description", "").strip()
        return ""
    except Exception:
        return ""


async def _get_hotel_facilities(client: httpx.AsyncClient, hotel_id: str) -> dict:
    try:
        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/getFacilitiesByHotelId",
            params={"hotel_id": hotel_id},
            headers=_headers(),
        )
        data = resp.json()
        facilities = data.get("data", [])
        facility_names = [f.get("name", "").lower() for f in facilities if f.get("name")]
        return {
            "couple_friendly": any("couple" in f for f in facility_names),
            "pet_friendly":    any("pet" in f or "animal" in f for f in facility_names),
            "parking":         any("parking" in f for f in facility_names),
            "wifi":            any("wifi" in f or "internet" in f for f in facility_names),
            "pool":            any("pool" in f or "swimming" in f for f in facility_names),
            "restaurant":      any("restaurant" in f for f in facility_names),
            "facilities_raw":  facility_names,
        }
    except Exception:
        return {}


async def _reverse_geocode(lat: float | None, lng: float | None) -> tuple[str, str]:
    """Return (state, country) from Nominatim for the given lat/lng."""
    if lat is None or lng is None:
        return "", ""
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


def _extract_block_price(block: dict) -> float:
    """Booking-com15 puts prices on each block (a bookable room offer) in
    several shapes depending on the response variant. Try the common ones."""
    if not isinstance(block, dict):
        return 0.0
    for key in ("product_price_breakdown", "price_breakdown", "priceBreakdown"):
        pb = block.get(key)
        if isinstance(pb, dict):
            for inner in ("gross_amount", "all_inclusive_amount", "grossPrice", "gross_price"):
                amt = pb.get(inner)
                if isinstance(amt, dict):
                    val = amt.get("value") or amt.get("amount_unrounded") or amt.get("amount_rounded")
                    if val:
                        return float(val)
                elif isinstance(amt, (int, float)) and amt:
                    return float(amt)
    for key in ("finalPrice", "min_price", "gross_price"):
        amt = block.get(key)
        if isinstance(amt, dict):
            val = amt.get("price") or amt.get("value")
            if val:
                return float(val)
        elif isinstance(amt, (int, float)) and amt:
            return float(amt)
    return 0.0


async def _get_room_list(
    client: httpx.AsyncClient,
    hotel_id: str,
    checkin: str,
    checkout: str,
    adults: int,
    rooms: int,
) -> list[dict]:
    """Live room availability + pricing for the requested dates. Never cached.

    Booking returns:
      - data.rooms  : {room_id: {name, bed types, max occupancy, ...}}
      - data.block  : list of bookable offers, each tied to a room_id with a price.
    We join the two and keep the cheapest available block per room.
    """
    try:
        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/getRoomList",
            params={
                "hotel_id":       hotel_id,
                "arrival_date":   checkin,
                "departure_date": checkout,
                "adults":         adults,
                "room_qty":       rooms,
                "currency_code":  "INR",
            },
            headers=_headers(),
        )
        data = resp.json().get("data", {})
        rooms_data = data.get("rooms", {}) or {}
        blocks     = data.get("block", []) or []

        # Cheapest block price per room_id
        price_by_room: dict[str, float] = {}
        for b in blocks:
            rid = str(b.get("room_id") or b.get("roomId") or "")
            if not rid:
                continue
            price = _extract_block_price(b)
            if price <= 0:
                continue
            current = price_by_room.get(rid)
            if current is None or price < current:
                price_by_room[rid] = price

        out: list[dict] = []
        for rid, r in rooms_data.items():
            bed_options = r.get("bed_configurations") or []
            bed_type = ""
            if bed_options and isinstance(bed_options, list):
                first = bed_options[0] or {}
                beds = first.get("bed_types") or []
                if beds:
                    bed_type = beds[0].get("name") or beds[0].get("name_with_count") or ""
            out.append({
                "room_id":       str(rid),
                "room_name":     r.get("name") or r.get("roomName") or "",
                "max_occupancy": r.get("max_occupancy") or r.get("maxOccupancy") or adults,
                "bed_type":      bed_type or r.get("bedType") or "",
                "price":         price_by_room.get(str(rid), 0.0),
            })
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Description enhancer — turns the raw blurb + full facility list into one
# semantically rich paragraph used both as the embedding text and as the
# stored description.
# ---------------------------------------------------------------------------

def _build_raw_description_payload(hotels: list[dict]) -> list[dict]:
    return [
        {
            "hotel_id":         h.get("hotel_id"),
            "name":             h.get("name"),
            "stars":            h.get("stars"),
            "rating":           h.get("rating"),
            "rating_word":      h.get("rating_word"),
            "city":             h.get("city"),
            "state":            h.get("state"),
            "country":          h.get("country"),
            "raw_description":  h.get("description", ""),
            "facilities":       (h.get("facilities") or {}).get("facilities_raw", []),
        }
        for h in hotels
    ]


def _enhance_hotel_descriptions(hotels: list[dict]) -> dict[str, str]:
    """Return {hotel_id: enhanced_description} via Groq llama-3.3-70b-versatile.

    Falls back to {} on any error — callers keep the raw Booking blurb.
    """
    if not hotels:
        return {}

    payload = _build_raw_description_payload(hotels)
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": _ENHANCER_SYSTEM_PROMPT},
                {"role": "user",   "content": _ENHANCER_USER_TEMPLATE.format(
                    hotels_json=json.dumps(payload, ensure_ascii=False)
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
            str(e["hotel_id"]): e["description"].strip()
            for e in entries
            if isinstance(e, dict) and e.get("hotel_id") is not None and e.get("description")
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metadata fetch — date-agnostic hotel info (cached in Qdrant)
# ---------------------------------------------------------------------------

async def _fetch_hotel_metadata(search_query: str, city_key: str) -> list[dict]:
    """Fetch the top hotels for `search_query` along with their static metadata
    (name, location, stars, rating, description, facilities, photo).

    Args:
        search_query: Free-text used for Booking's destination resolver. Can be
                      a city ("Shillong") or a specific hotel name.
        city_key:     Normalised city slug stored in metadata so cache filters work.
    """
    # searchHotels requires dates; use a placeholder month-out window. The
    # returned price/availability is discarded — we only want stable fields.
    future = date.today() + timedelta(days=30)
    placeholder_in  = future.isoformat()
    placeholder_out = (future + timedelta(days=1)).isoformat()

    async with httpx.AsyncClient(timeout=30) as client:
        dest_id, dest_type = await _get_destination(client, search_query)

        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/searchHotels",
            params={
                "dest_id":        dest_id,
                "search_type":    dest_type,
                "arrival_date":   placeholder_in,
                "departure_date": placeholder_out,
                "adults":         2,
                "room_qty":       1,
                "currency_code":  "INR",
                "sort_by":        "popularity",
            },
            headers=_headers(),
        )
        data = resp.json()
        raw_hotels = data.get("data", {}).get("hotels", [])[:5]

        hotels: list[dict] = []
        for h in raw_hotels:
            prop = h.get("property", {})
            hid = str(prop.get("id", ""))
            if not hid:
                continue
            hotels.append({
                "hotel_id":     hid,
                "name":         prop.get("name"),
                "stars":        prop.get("propertyClass") or 0,
                "rating":       prop.get("reviewScore") or 0,
                "rating_word":  prop.get("reviewScoreWord") or "Not rated",
                "review_count": prop.get("reviewCount") or 0,
                "lat":          prop.get("latitude"),
                "lng":          prop.get("longitude"),
                "photo":        (prop.get("photoUrls") or [None])[0],
                "city":         city_key,
                "state":        "",
                "country":      "",
                "description":  "",
                "facilities":   {},
            })

        if hotels:
            enrich = await asyncio.gather(*[
                asyncio.gather(
                    _get_description(client, h["hotel_id"]),
                    _get_hotel_facilities(client, h["hotel_id"]),
                    _reverse_geocode(h["lat"], h["lng"]),
                )
                for h in hotels
            ])
            for h, (desc, fac, (state, country)) in zip(hotels, enrich):
                h["description"] = desc
                h["facilities"]  = fac
                h["state"]       = state
                h["country"]     = country

    # Now that every hotel has its raw blurb + full facility list + region,
    # generate the enriched semantic description via Groq.
    enhanced = _enhance_hotel_descriptions(hotels)
    for h in hotels:
        better = enhanced.get(str(h["hotel_id"]))
        if better:
            h["description"] = better

    return hotels


async def _fetch_availability_for_all(
    hotels: list[dict],
    checkin: str,
    checkout: str,
    adults: int,
    rooms: int,
) -> list[list[dict]]:
    """Fetch live room availability for every hotel in parallel."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await asyncio.gather(*[
            _get_room_list(client, h["hotel_id"], checkin, checkout, adults, rooms)
            for h in hotels
        ])


# ---------------------------------------------------------------------------
# Qdrant cache helpers
# ---------------------------------------------------------------------------

_KB_INDEX_FIELDS = ["city", "state", "country"]


def _hotels_from_cache(
    *,
    query_text: str,
    filters: dict,
    top_k: int = 20,
) -> list[dict]:
    """Retrieve cached hotels.

    - If `query_text` is provided → hybrid (dense + sparse RRF) search with
      the metadata filters, so embeddings are computed and the result is
      ranked by semantic match to the preference / hotel name.
    - If `query_text` is empty → no embeddings; just scroll the collection
      using the metadata filters and return every matching hotel.
    """
    _kb.ensure_collection(_COLLECTION, keyword_index_fields=_KB_INDEX_FIELDS)

    if query_text:
        hits = _kb.search(
            _COLLECTION,
            query_text=query_text,
            top_k=top_k,
            metadata_filters=filters or None,
        )
        return [h["metadata"] for h in hits]

    if not filters:
        return []
    rows = _kb.scroll(_COLLECTION, metadata_filters=filters, limit=top_k)
    return [r["metadata"] for r in rows]


def _persist_hotels(hotels: list[dict]) -> None:
    """Persist hotel metadata. The embedding text is the enriched semantic
    description so semantic search can match preferences like "pet-friendly
    with a pool" against the actual prose."""
    if not hotels:
        return
    _kb.ensure_collection(_COLLECTION, keyword_index_fields=_KB_INDEX_FIELDS)
    points = [
        {
            "id":       _hotel_point_id(h["hotel_id"]),
            "text":     h.get("description") or h.get("name") or "",
            "metadata": {
                "hotel_id":     h.get("hotel_id"),
                "name":         h.get("name"),
                "stars":        h.get("stars"),
                "rating":       h.get("rating"),
                "rating_word":  h.get("rating_word"),
                "review_count": h.get("review_count"),
                "lat":          h.get("lat"),
                "lng":          h.get("lng"),
                "city":         h.get("city"),
                "state":        h.get("state"),
                "country":      h.get("country"),
                "photo":        h.get("photo"),
                "description":  h.get("description"),
                "facilities":   h.get("facilities") or {},
            },
        }
        for h in hotels
    ]
    _kb.insert_batch(_COLLECTION, points)


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

@tool
def find_hotels(
    city: str,
    checkin: str,
    checkout: str,
    state: str = "",
    country: str = "",
    adults: int = 2,
    rooms: int = 1,
    user_preference: str = "",
    hotel_name: str = "",
) -> dict:
    """Find hotels in a city and return their live room availability and prices.

    Use this tool whenever the user wants to:
      - browse hotels in a place (with or without preferences),
      - check availability and pricing for specific dates,
      - look up a specific hotel by name.

    Caching behaviour:
      The tool first looks the city up in a Qdrant collection of previously
      seen hotels and only falls back to Booking.com when nothing is cached.
      Room availability and pricing are ALWAYS fetched live for the supplied
      check-in / check-out dates — they are never cached.

    Args:
        city:            REQUIRED. The destination city (e.g. "Shillong",
                         "Chikkamagalur", "Goa"). Plain city name, no commas.
        checkin:         REQUIRED. Check-in date in YYYY-MM-DD. If the user
                         did not specify a date, the agent should default to
                         two days from today.
        checkout:        REQUIRED. Check-out date in YYYY-MM-DD. If the user
                         did not specify a duration, the agent should default
                         to two nights after the check-in date.
        state:           Optional. State / region name (e.g. "Karnataka").
                         Pass it when you know it — it narrows the cache lookup.
        country:         Optional. Country name (e.g. "India"). Pass it when
                         known.
        adults:          Number of adult guests. Default 2 if the user did
                         not specify.
        rooms:           Number of rooms required. Default 1 if the user did
                         not specify.
        user_preference: Free-text preference used for semantic search over
                         hotel descriptions. Pass the user's words directly
                         (e.g. "pet friendly with a pool", "budget under 5000",
                         "luxury couple stay"). Leave empty when the user has
                         no preference — the tool will then return every
                         hotel matching the location filters.
        hotel_name:      Only set this when the user asked for ONE SPECIFIC
                         hotel by name (e.g. "Coffee Grove Resort"). It is
                         used both for semantic search and for post-filtering
                         results to that hotel.

    Returns:
        {"total": N, "hotels": [<hotel>, ...]} — each hotel object contains:
            hotel_id, name, stars, rating, rating_word, review_count,
            lat, lng, city, state, country, description (semantic),
            facilities (couple_friendly, pet_friendly, parking, wifi, pool,
            restaurant, facilities_raw), rooms (live list of available room
            types with bed_type, max_occupancy, price), price_per_night
            (cheapest live price for the dates), currency, photo.
    """
    city_key = (city or "").strip().lower()

    filters: dict = {}
    if city_key:
        filters["city"] = city_key
    if state:
        filters["state"] = state
    if country:
        filters["country"] = country

    # Embeddings only when we have something meaningful to match on; otherwise
    # plain filter scroll over the collection.
    query_text = (hotel_name or user_preference or "").strip()

    hotels = _hotels_from_cache(query_text=query_text, filters=filters)

    if hotel_name:
        hn = hotel_name.strip().lower()
        matched = [h for h in hotels if hn in (h.get("name") or "").lower()]
        hotels = matched or []

    if not hotels:
        # Cache miss — query Booking. Use hotel_name as the destination query
        # when given so Booking can resolve directly to that property.
        search_query = hotel_name.strip() if hotel_name else city
        hotels = asyncio.run(_fetch_hotel_metadata(search_query, city_key))
        _persist_hotels(hotels)
        if hotel_name:
            hn = hotel_name.strip().lower()
            hotels = [h for h in hotels if hn in (h.get("name") or "").lower()] or hotels

    if hotels:
        availability = asyncio.run(
            _fetch_availability_for_all(hotels, checkin, checkout, adults, rooms)
        )
        for h, room_list in zip(hotels, availability):
            h["rooms"]           = room_list
            prices               = [r.get("price", 0) for r in room_list if r.get("price")]
            h["price_per_night"] = min(prices) if prices else 0
            h["currency"]        = "INR"

    return {"total": len(hotels), "hotels": hotels}


# if __name__ == "__main__":
#     # Quick manual smoke test — hits Booking RapidAPI + Groq + Qdrant.
#     # Run twice to verify the warm path: first call populates Qdrant, second
#     # call should skip the metadata fetch and only re-query room availability.
#     from datetime import date as _date, timedelta as _td

#     _CITY     = "Chikkamagalur"
#     _CHECKIN  = (_date.today() + _td(days=14)).isoformat()
#     _CHECKOUT = (_date.today() + _td(days=16)).isoformat()

#     print(f"\n--- find_hotels({_CITY}, {_CHECKIN} -> {_CHECKOUT}) ---")
#     result = find_hotels(
#         city=_CITY,
#         checkin=_CHECKIN,
#         checkout=_CHECKOUT,
#         state="Karnataka",
#         country="India",
#         adults=2,
#         rooms=1,
#         user_preference="comfortable stay with good rating",
#     )
#     print(f"Total: {result['total']}")
#     for h in result["hotels"]:
#         print(
#             f"\n[{h.get('hotel_id')}] {h.get('name')} "
#             f"({h.get('stars')}*, rating {h.get('rating')})"
#         )
#         print(f"  city/state/country : {h.get('city')} / {h.get('state')} / {h.get('country')}")
#         print(f"  lat,lng            : {h.get('lat')}, {h.get('lng')}")
#         print(f"  price/night        : {h.get('price_per_night')} {h.get('currency')}")
#         print(f"  rooms available    : {len(h.get('rooms') or [])}")
#         print(f"  description        : {(h.get('description') or '')[:240]}")
