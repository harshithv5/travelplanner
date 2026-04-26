import asyncio
import httpx
from strands import tool


async def get_coordinates(place_name: str) -> dict:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "travelstack-app"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        data = response.json()
    if not data:
        return None
    return {
        "place": place_name,
        "lat": float(data[0]["lat"]),
        "lng": float(data[0]["lon"]),
    }


@tool
def geocode(place_name: str) -> dict:
    """
    Convert a place name into geographic coordinates (latitude and longitude).
    Use this to confirm a destination exists and get its precise location.

    Args:
        place_name: The name of the city, town, or landmark to look up.

    Returns:
        A dict with keys: place, lat, lng. Returns None if not found.
    """
    return asyncio.run(get_coordinates(place_name))
