import asyncio
import os
import httpx
from strands import tool

ORS_API_KEY = os.getenv("ORS_API_KEY", "")


async def _get_route(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> dict:
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [
            [origin_lng, origin_lat],
            [dest_lng, dest_lat],
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=body, headers=headers)
        data = response.json()

    summary = data["features"][0]["properties"]["summary"]
    return {
        "distance_km":   round(summary["distance"] / 1000, 2),
        "duration_mins": round(summary["duration"] / 60, 2),
    }


@tool
def route(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> dict:
    """
    Get the driving route between two locations using their coordinates.

    Args:
        origin_lat: Latitude of the starting location.
        origin_lng: Longitude of the starting location.
        dest_lat:   Latitude of the destination.
        dest_lng:   Longitude of the destination.

    Returns:
        A dict with keys: distance_km (float), duration_mins (float).
    """
    return asyncio.run(_get_route(origin_lat, origin_lng, dest_lat, dest_lng))
