import httpx


async def get_weather(lat: float, lng: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m,weathercode,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 7,
        "timezone": "auto",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()
