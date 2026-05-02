from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from strands import tool

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

SEARCH_QUERIES = {
    "hidden gems":        lambda city: f"hidden gems secret places in {city} locals only off beaten path not touristy",
    "street food":        lambda city: f"best street food local food stalls markets in {city} authentic cheap locals eat",
    "historic sites":     lambda city: f"historic sites ancient temples forts stepwells in {city} worth visiting",
    "viewpoints":         lambda city: f"best viewpoints scenic spots sunrise sunset points in {city}",
    "local experiences":  lambda city: f"unique local experiences activities things to do in {city} not in guidebooks",
    "nature":             lambda city: f"parks lakes hills nature spots peaceful in {city}",
    "markets bazaars":    lambda city: f"local markets bazaars handicrafts shopping in {city} authentic",
}

async def search_places(city: str, category: str) -> list:
    """Search for places in a city for a given category."""
    query_fn = SEARCH_QUERIES.get(category)
    query = query_fn(city) if query_fn else f"{category} in {city}"

    results = client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )

    return [
        {"title": r["title"], "content": r["content"], "url": r["url"]}
        for r in results["results"]
    ]

@tool
def places(city: str, category: str) -> list:
    """
    Discover places in a city for a given category.

    Args:
        city: Name of the city to search in.
        category: One of — "hidden gems", "street food", "historic sites",
                  "viewpoints", "local experiences", "nature", "markets bazaars".

    Returns:
        A list of dicts with keys: title, content, url.
    """
    return asyncio.run(search_places(city=city, category=category))