import asyncio
from contextlib import AsyncExitStack
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(sse_client("http://127.0.0.1:8001/sse"))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        print("Success")
    except Exception as e:
        print(f"Exception: {repr(e)}")
    finally:
        await stack.aclose()

asyncio.run(main())
