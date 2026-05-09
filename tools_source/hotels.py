import asyncio
import os
import httpx
from strands import tool
from models.llms import summarize_tool_output

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "booking-com15.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"

# ---------------------------------------------------------------------------
# Session store — full hotel data keyed by hotel_id.
# The LLM only ever sees the compact summarized list.
# ---------------------------------------------------------------------------
_hotel_store: dict[str, dict] = {}


def reset_session() -> None:
    _hotel_store.clear()


def get_stored_hotels() -> dict[str, dict]:
    return dict(_hotel_store)


def _headers() -> dict:
    return {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }


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
    """Fetch general description for a single hotel."""
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
    """Get amenities and friendliness flags."""
    try:
        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/getFacilitiesByHotelId",
            params={"hotel_id": hotel_id},
            headers=_headers(),
        )
        data = resp.json()
        facilities = data.get("data", [])

        facility_names = [f.get("name", "").lower() for f in facilities]

        return {
            "couple_friendly": any("couple" in f for f in facility_names),
            "pet_friendly":    any("pet" in f or "animal" in f for f in facility_names),
            "parking":         any("parking" in f for f in facility_names),
            "wifi":            any("wifi" in f or "internet" in f for f in facility_names),
            "pool":            any("pool" in f or "swimming" in f for f in facility_names),
            "restaurant":      any("restaurant" in f for f in facility_names),
            "facilities_raw":  facility_names[:10],
        }
    except Exception:
        return {}


async def _get_room_list(
    client: httpx.AsyncClient,
    hotel_id: str,
    checkin: str,
    checkout: str,
    adults: int,
    rooms: int
) -> list:
    """Get available room types for given occupancy."""
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
        data = resp.json()
        rooms_data = data.get("data", {}).get("rooms", {})

        room_list = []
        for room in rooms_data.values():
            room_list.append({
                "room_name":     room.get("roomName", ""),
                "max_occupancy": room.get("maxOccupancy", adults),
                "bed_type":      room.get("bedType", ""),
                "price":         room.get("priceBreakdown", {})
                                     .get("grossPrice", {})
                                     .get("value", 0),
            })
        return room_list
    except Exception:
        return []


async def _search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    adults: int,
    rooms: int,
) -> list:
    async with httpx.AsyncClient(timeout=30) as client:

        # Step 1 — resolve destination
        dest_id, dest_type = await _get_destination(client, city)

        # Step 2 — search hotels
        resp = await client.get(
            f"{BASE_URL}/api/v1/hotels/searchHotels",
            params={
                "dest_id":        dest_id,
                "search_type":    dest_type,
                "arrival_date":   checkin,
                "departure_date": checkout,
                "adults":         adults,
                "room_qty":       rooms,
                "currency_code":  "INR",
                "sort_by":        "popularity",
            },
            headers=_headers(),
        )
        data = resp.json()

        # Step 3 — extract top 5
        raw_hotels = data.get("data", {}).get("hotels", [])[:5]
        hotels = []
        for hotel in raw_hotels:
            prop = hotel.get("property", {})
            hotels.append({
                "hotel_id":          str(prop.get("id", "")),
                "name":              prop.get("name"),
                "stars":             prop.get("propertyClass") or 0,
                "rating":            prop.get("reviewScore") or 0,
                "rating_word":       prop.get("reviewScoreWord") or "Not rated",
                "review_count":      prop.get("reviewCount") or 0,
                "lat":               prop.get("latitude"),
                "lng":               prop.get("longitude"),
                "price_per_night":   (
                    prop.get("priceBreakdown", {})
                        .get("grossPrice", {})
                        .get("value") or 0
                ),
                "currency":          "INR",
                "free_cancellation": prop.get("isFreeCancellation", False),
                "photo":             prop.get("photoUrls", [None])[0],
                "description":       "",
                "facilities":        {},
                "rooms":             [],
            })

        # Step 4 — enrich top hotel with full details (3 parallel calls)
        if hotels and hotels[0]["hotel_id"]:
            top_id = hotels[0]["hotel_id"]
            desc, facilities, room_list = await asyncio.gather(
                _get_description(client, top_id),
                _get_hotel_facilities(client, top_id),
                _get_room_list(client, top_id, checkin, checkout, adults, rooms),
            )
            hotels[0]["description"] = desc
            hotels[0]["facilities"]  = facilities
            hotels[0]["rooms"]       = room_list

        return hotels


def _summarize(hotels: list[dict]) -> list[dict]:
    """
    Build compact [{hotel_id, description}] summaries for the LLM.
    Each description covers facilities, rating, price, and key practical points —
    no photos, no marketing prose, kept short.
    """
    if not hotels:
        return []

    lines = []
    for h in hotels:
        facilities = h.get("facilities") or {}
        flag_summary = ", ".join(
            k for k in ("couple_friendly", "pet_friendly", "parking", "wifi", "pool", "restaurant")
            if facilities.get(k)
        ) or "no flagged amenities"

        rooms_summary = "; ".join(
            f"{r.get('room_name', '?')} ({r.get('bed_type', '')}, max {r.get('max_occupancy', '?')}, Rs.{r.get('price', 0)})"
            for r in (h.get("rooms") or [])
        ) or "rooms not detailed"

        lines.append(
            f"{h['hotel_id']} | {h['name']} | "
            f"{h.get('stars', 0)} stars | rating {h.get('rating', 0)} ({h.get('rating_word', '')}) | "
            f"Rs.{h.get('price_per_night', 0)}/night | "
            f"free_cancel={h.get('free_cancellation', False)} | "
            f"facilities: {flag_summary} | "
            f"rooms: {rooms_summary} | "
            f"raw_desc: {(h.get('description') or '')[:300]}"
        )

    prompt = (
        "You are summarizing hotels for a traveller. For each hotel, write ONE concise sentence "
        "covering its key facilities, star rating, price per night, room types, and whether it "
        "suits couples/pets. No photos, no marketing fluff, no long descriptions — just the "
        "practical facts a traveller needs to choose.\n"
        'Return JSON: {"hotels": [{"hotel_id": "...", "description": "..."}]}\n\n'
        "Hotels:\n" + "\n".join(lines)
    )

    result = summarize_tool_output(
        prompt=prompt,
        generation_config={"max_tokens": 800, "response_format": {"type": "json_object"}},
    )
    if isinstance(result, dict) and result.get("hotels"):
        return result["hotels"]

    # Fallback — derive minimal description from raw fields
    return [
        {
            "hotel_id":    h["hotel_id"],
            "description": f"{h['name']} — {h.get('stars', 0)} star, rated {h.get('rating', 0)}, Rs.{h.get('price_per_night', 0)}/night",
        }
        for h in hotels
    ]


@tool
def find_hotels(
    city: str,
    checkin: str,
    checkout: str,
    adults: int = 1,
    rooms: int = 1,
) -> dict:
    """
    Search for available hotels in a city for the given dates and occupancy.
    Returns compact [{hotel_id, description}] summaries — full hotel data is
    stored internally and can be expanded by hotel_id at the agent level.

    Args:
        city:     Destination city (e.g. "Goa", "Jaipur", "Shillong").
        checkin:  Check-in date YYYY-MM-DD. Must be today or future.
        checkout: Check-out date YYYY-MM-DD. Must be after checkin.
        adults:   Total number of adult guests.
        rooms:    Number of rooms required (default 1).
                  Example: 4 adults in 2 rooms → adults=4, rooms=2.

    Returns:
        Dict with keys: total (int), hotels (list of {hotel_id, description}).
        Use the returned hotel_ids to reference hotels in your final answer.
    """
    raw = asyncio.run(
        _search_hotels(
            city=city,
            checkin=checkin,
            checkout=checkout,
            adults=adults,
            rooms=rooms,
        )
    )

    # Store full hotel data internally, keyed by hotel_id
    for h in raw:
        if h.get("hotel_id"):
            _hotel_store[h["hotel_id"]] = h

    # Build compact summaries for the LLM
    compact = _summarize(raw)

    # Write the LLM-generated summary back into the store for reference
    for c in compact:
        hid = c.get("hotel_id")
        if hid in _hotel_store:
            _hotel_store[hid]["summary"] = c.get("description", "")

    return {"total": len(compact), "hotels": compact}
