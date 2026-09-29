import sys
import os
import asyncio

# Ensure backend directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.mcp_graph import mcp_graph_app
from langchain_core.messages import HumanMessage

async def test_rag():
    state_input = {
        "messages": [HumanMessage(content="What does Krify do? Tell me based on your knowledge base.")],
        "route": "api"
    }
    result = await mcp_graph_app.ainvoke(state_input)
    print("AI Response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(test_rag())
