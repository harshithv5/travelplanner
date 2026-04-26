import asyncio
import os
import httpx
from strands import tool

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "booking-com15.p.rapidapi.com"


async def search_hotels(city: str, checkin: str, checkout: str, adults: int = 1) -> list:
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    async with httpx.AsyncClient() as client:
        dest_resp = await client.get(
            f"https://{RAPIDAPI_HOST}/api/v1/hotels/searchDestination",
            params={"query": city},
            headers=headers,
        )
        dest_data = dest_resp.json()

    dest_id = dest_data["data"][0]["dest_id"]
    dest_type = dest_data["data"][0]["dest_type"]

    params = {
        "dest_id": dest_id,
        "search_type": dest_type,
        "arrival_date": checkin,
        "departure_date": checkout,
        "adults": adults,
        "room_qty": 1,
        "currency_code": "INR",
        "sort_by": "popularity",
    }
    async with httpx.AsyncClient() as client:
        hotels_resp = await client.get(
            f"https://{RAPIDAPI_HOST}/api/v1/hotels/searchHotels",
            params=params,
            headers=headers,
        )
        data = hotels_resp.json()

    hotels = []
    for hotel in data.get("data", {}).get("hotels", [])[:10]:
        prop = hotel.get("property", {})
        hotels.append({
            "name": prop.get("name"),
            "rating": prop.get("reviewScore"),
            "review_count": prop.get("reviewCount"),
            "price_per_night": prop.get("priceBreakdown", {}).get("grossPrice", {}).get("value"),
            "currency": "INR",
            "photo": prop.get("photoUrls", [None])[0],
        })
    return hotels


@tool
def find_hotels(city: str, checkin: str, checkout: str, adults: int = 1) -> list:
    """
    Search for available hotels in a city for the given dates.
    Returns top 10 hotels sorted by popularity with pricing in INR.

    Args:
        city: Destination city name (e.g. "Goa", "Jaipur").
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        adults: Number of adult guests (default 1).

    Returns:
        A list of hotel dicts with: name, rating, review_count,
        price_per_night (INR), currency, photo URL.
    """
    return asyncio.run(search_hotels(city, checkin, checkout, adults))
