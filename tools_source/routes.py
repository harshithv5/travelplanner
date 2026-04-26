import asyncio
import os
import httpx
from strands import tool

from tools_source.geocode import get_coordinates

ORS_API_KEY = os.getenv("ORS_API_KEY", "")


async def get_route(origin: str, destination: str) -> dict:
    origin_coords = await get_coordinates(origin)
    dest_coords = await get_coordinates(destination)

    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [
            [origin_coords["lng"], origin_coords["lat"]],
            [dest_coords["lng"], dest_coords["lat"]],
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=body, headers=headers)
        data = response.json()

    coordinates = data["features"][0]["geometry"]["coordinates"]
    summary = data["features"][0]["properties"]["summary"]
    return {
        "coordinates": coordinates,
        "distance_km": round(summary["distance"] / 1000, 2),
        "duration_mins": round(summary["duration"] / 60, 2),
    }


@tool
def route(origin: str, destination: str) -> dict:
    """
    Get the driving route between two places, including road distance and travel time.
    Use this whenever the user asks about travelling between two locations.

    Args:
        origin: Starting city or place name.
        destination: Ending city or place name.

    Returns:
        A dict with keys: coordinates (list), distance_km (float), duration_mins (float).
    """
    return asyncio.run(get_route(origin, destination))
