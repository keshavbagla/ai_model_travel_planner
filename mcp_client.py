import os
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

load_dotenv(override=True)

from config import (
    AVIATIONSTACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

BASE_DIR = Path(__file__).resolve().parent

AVIATION_MCP_PYTHON = os.getenv("AVIATION_MCP_PYTHON")

if not AVIATION_MCP_PYTHON:
    raise RuntimeError(
        "AVIATION_MCP_PYTHON is not set in .env"
    )

WEATHER_SERVER = BASE_DIR / "custom_weather_mcp_server.py"

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                f"https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            ),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": AVIATION_MCP_PYTHON,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY,
            },
        },
        "weather": {
            "transport": "stdio",
            "command": os.sys.executable,
            "args": [
                str(WEATHER_SERVER),
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)

search_tool = None
aviation_tools = {}
weather_tool = None
forecast_tool = None


async def initialize_mcp():
    global search_tool
    global aviation_tools

    if search_tool is not None and aviation_tools:
        return

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(f"- {tool.name}")

    print()

    search_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "tavily_search"
        ),
        None
    )

    if search_tool is None:
        raise RuntimeError(
            "Tavily search tool not found."
        )

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name not in {
            "tavily_search",
            "get_current_weather",
            "get_forecast",
        }
    }


async def tavily_mcp_search(query: str):
    await initialize_mcp()

    return await search_tool.ainvoke(
        {
            "query": query
        }
    )


async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    await initialize_mcp()

    tool = aviation_tools.get(tool_name)

    if not tool:
        return f"Tool unavailable: {tool_name}"

    return await tool.ainvoke(
        tool_args or {}
    )


async def get_airports():
    await initialize_mcp()

    tool = aviation_tools.get(
        "list_airports"
    )

    if not tool:
        return "Airport tool unavailable"

    return await tool.ainvoke({})


async def get_airlines():
    await initialize_mcp()

    tool = aviation_tools.get(
        "list_airlines"
    )

    if not tool:
        return "Airline tool unavailable"

    return await tool.ainvoke({})


async def initialize_weather_tools():
    global weather_tool
    global forecast_tool

    if (
        weather_tool is not None
        and forecast_tool is not None
    ):
        return

    tools = await client.get_tools()

    weather_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "get_current_weather"
        ),
        None
    )

    forecast_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "get_forecast"
        ),
        None
    )

    if weather_tool is None:
        raise RuntimeError(
            "get_current_weather MCP tool not found."
        )

    if forecast_tool is None:
        raise RuntimeError(
            "get_forecast MCP tool not found."
        )


def parse_mcp_json_result(result):
    if isinstance(result, dict):
        return result

    if isinstance(result, list):
        if not result:
            return {}

        for item in result:
            if not isinstance(item, dict):
                continue

            text_content = item.get("text")

            if text_content is None:
                continue

            if isinstance(text_content, dict):
                return text_content

            if isinstance(text_content, str):
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    return {
                        "error": "Invalid JSON returned by MCP weather tool",
                        "raw": text_content,
                    }

        return {
            "error": "No text content found in MCP response",
            "raw": result,
        }

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned by MCP weather tool",
                "raw": result,
            }

    return {
        "error": "Unexpected MCP response format",
        "raw": str(result),
    }


async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return parse_mcp_json_result(result)


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return parse_mcp_json_result(result)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def extract_destination(query: str):
    prompt = f"""
Extract only the destination city or country.

Query:
{query}

Return only destination name.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


async def test_weather(city: str):
    print("\n" + "=" * 60)
    print(f"WEATHER TEST: {city}")
    print("=" * 60)

    try:
        current_weather = await weather_mcp_search(city)

        print("\nCurrent Weather:")
        print(
            json.dumps(
                current_weather,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as e:
        print("\nCurrent weather error:")
        print(str(e))

    try:
        forecast = await forecast_mcp_search(city)

        print("\nForecast:")
        print(
            json.dumps(
                forecast,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as e:
        print("\nForecast error:")
        print(str(e))


async def main():
    await initialize_mcp()
    await initialize_weather_tools()

    print("\nMCP initialization completed successfully.")
    print("\nWeather tools are ready.")


if __name__ == "__main__":
    asyncio.run(main())
