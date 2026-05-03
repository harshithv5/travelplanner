import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from strands import tool
from models.llms import summarize_tool_output

load_dotenv()

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

# ---------------------------------------------------------------------------
# In-session memory — populated by each places() tool call.
# _tool_call_store : full structured data (used for final output)
# _call_tracker    : lightweight metadata log (used for tracking display)
# ---------------------------------------------------------------------------
_tool_call_store: list[dict] = []
_call_tracker:    list[dict] = []


def get_stored_results() -> list[dict]:
    """Return all extracted place entries from every tool call this session."""
    places = []
    for entry in _tool_call_store:
        for p in entry["extracted_places"]:
            places.append({
                "place":       p["place"],
                "category":    entry["category"],
                "description": p["description"],
                "link":        p["url"],
            })
    return places


def get_call_tracker() -> list[dict]:
    return _call_tracker


def clear_session() -> None:
    _tool_call_store.clear()
    _call_tracker.clear()


async def _search(city: str, category: str, max_results: int) -> list:
    query_fn = SEARCH_QUERIES.get(category)
    query = query_fn(city) if query_fn else f"{category} in {city}"
    results = client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        {"title": r["title"], "content": r["content"], "url": r["url"]}
        for r in results["results"]
    ]


@tool
def places(city: str, category: str, max_results: int = 3) -> str:
    """
    Discover places in a city for a given category.

    Args:
        city: Name of the city to search in.
        category: One of — "hidden gems", "street food", "historic sites",
                  "viewpoints", "local experiences", "nature", "markets bazaars".
        max_results: Number of results to fetch (1–10). Scale down for short
                     trips, scale up for comprehensive lists.

    Returns:
        A compact text summary of places found (for LLM context efficiency).
        Full raw results are stored internally for final output.
    """
    call_index = len(_tool_call_store) + 1

    raw_results = asyncio.run(_search(city=city, category=category, max_results=max_results))

    summarized = summarize_tool_output(city=city, category=category, results=raw_results)

    _tool_call_store.append({
        "call_index":       call_index,
        "city":             city,
        "category":         category,
        "max_results":      max_results,
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "raw_results":      raw_results,
        "extracted_places": summarized["places"],
    })

    _call_tracker.append({
        "call_index":          call_index,
        "city":                city,
        "category":            category,
        "raw_count":           len(raw_results),
        "extracted_count":     len(summarized["places"]),
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
    })

    return summarized["summary"]
