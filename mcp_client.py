import os
import asyncio
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
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
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
    global search_tool, aviation_tools

    if search_tool is not None and aviation_tools:
        return

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name != "tavily_search"
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

    tool = aviation_tools.get("list_airports")

    if not tool:
        return "Airport tool unavailable"

    return await tool.ainvoke({})

async def get_airlines():
    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if not tool:
        return "Airline tool unavailable"

    return await tool.ainvoke({})

async def initialize_weather_tools():
    global weather_tool, forecast_tool

    if weather_tool is not None and forecast_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        tool
        for tool in tools
        if tool.name == "get_current_weather"
    )

    forecast_tool = next(
        tool
        for tool in tools
        if tool.name == "get_forecast"
    )

async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )

async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

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

async def main():
    await initialize_mcp()
    await initialize_weather_tools()

if __name__ == "__main__":
    asyncio.run(main())