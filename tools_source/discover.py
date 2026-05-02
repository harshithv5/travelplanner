import httpx

import asyncio
from tools_source.geocode import get_coordinates




async def get_places_osm(lat: float, lng: float, radius_m: int = 10000) -> dict:
    """Get all places from OpenStreetMap for given coordinates"""
    
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
        print(response)
        data = response.json()
    print("data",response)
    places = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        if not name:
            continue

        # determine category
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

        places.append({
            "name": name,
            "category": category,
            "lat": element.get("lat"),
            "lng": element.get("lon"),
            "tags": {
                k: v for k, v in tags.items()
                if k in ["description", "opening_hours",
                          "fee", "wikipedia", "wikidata"]
            }
        })

    return {
        "total": len(places),
        "places": places
    }
    
    
async def test():
    coords = {"lat":"26.9154576" , "lng":"75.8189817"}
    result = await get_places_osm(coords["lat"], coords["lng"])
    print(f"Total places found: {result['total']}")
    for p in result["places"][:10]:
        print(f"{p['category']:15} | {p['name']}")

asyncio.run(test())