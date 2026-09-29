import sys
import os
import asyncio

# Ensure backend directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.mcp_graph import mcp_graph_app
from langchain_core.messages import HumanMessage

async def test_api():
    state_input = {
        "question": "What time is my appointment tomorrow?",
        "session_id": "test_session_123",
        "route": "api"
    }
    result = await mcp_graph_app.ainvoke(state_input)
    print("AI Response:")
    print(result["final_message"])

if __name__ == "__main__":
    asyncio.run(test_api())
