from graph.workflow import build_workflow
from langchain_core.messages import HumanMessage
import asyncio

async def main():
    graph = build_workflow()
    state = {"question": "i need to develop a MobileApp", "history": []}
    result = await graph.ainvoke(state)
    print("\n\n--- ROUTING DECISION ---")
    print(result.get("route", "N/A"))
    print("\n\n--- RAG ANSWER ---")
    print(result.get("rag_answer", "N/A"))
    print("\n\n--- GENERAL ANSWER ---")
    print(result.get("general_answer", "N/A"))
    print("\n\n--- FINAL SYNTHESIZED OUTPUT ---")
    print(result.get("synthesizer_response", "N/A"))
    print("\n\n--- FINAL CONTENT (content) ---")
    print(result.get("content", "N/A"))

asyncio.run(main())
