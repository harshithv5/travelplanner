from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def search_places(city: str, category: str) -> list:
    """Search for places and hidden gems for a city"""
    
    query = f"{category} in {city} locals recommend hidden gems 2026"
    
    results = client.search(
        query=query,
        max_results=5,
        search_depth="basic"
    )
    
    places = []
    for r in results["results"]:
        places.append({
            "title": r["title"],
            "content": r["content"],
            "url": r["url"]
        })
    
    return places


import asyncio

async def test():
    results = await search_places("Jaipur", "street food")
    for r in results:
        print(f"{r['title']}")
        print(f"{r['content'][:150]}")
        print("---")

asyncio.run(test())