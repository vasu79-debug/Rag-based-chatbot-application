import asyncio
from langchain.mcp import MCPAdapter

async def main():
    print("Testing MCP Adapter")
    async with MCPAdapter("http://127.0.0.1:8001/sse") as adapter:
        tools = await adapter.list_tools()
        print(tools)

if __name__ == "__main__":
    asyncio.run(main())
