import httpx
from mcp.server.fastmcp import FastMCP

# This represents an "Outer" 3rd-party MCP Server
mcp = FastMCP("Free Weather Server", host="127.0.0.1", port=8002)

@mcp.tool()
def get_live_weather(city: str) -> str:
    """Use this tool to get the real live weather for any city in the world."""
    try:
        # Hitting a real, free, public API that requires zero authentication!
        response = httpx.get(f"https://wttr.in/{city}?format=3")
        return f"Live Weather Result: {response.text}"
    except Exception as e:
        return f"Could not fetch weather: {e}"

if __name__ == "__main__":
    print("Starting Outer Weather MCP Server on http://127.0.0.1:8002/sse")
    mcp.run(transport="sse")
