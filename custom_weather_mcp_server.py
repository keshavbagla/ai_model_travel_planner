from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Weather Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


@mcp.tool()
def get_current_weather(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=15,
    )

    data = response.json()

    if response.status_code != 200:
        return {
            "error": data.get(
                "message",
                "Unable to fetch current weather",
            ),
            "city": city,
        }

    return {
        "city": data["name"],
        "country": data.get("sys", {}).get("country"),
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=15,
    )

    data = response.json()

    if response.status_code != 200:
        return {
            "error": data.get(
                "message",
                "Unable to fetch weather forecast",
            ),
            "city": city,
        }

    forecast = []

    for item in data.get("list", [])[:5]:
        forecast.append(
            {
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"],
            }
        )

    return {
        "city": data.get("city", {}).get("name", city),
        "country": data.get("city", {}).get("country"),
        "forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run()
