import os
import sys
import json
import asyncio
import logging
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.tools import tool

from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from graph.llm_factory import get_chat_model
from config import settings
from rag.hybrid_retriever import hybrid_retriever

from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def search_local_knowledge_base(query: str) -> str:
    """Use this tool to search through the user's uploaded private PDFs and indexed websites for answers."""
    try:
        docs = hybrid_retriever.retrieve(query)
        if not docs:
            return "No relevant documents found in the knowledge base."
            
        results = []
        for d in docs:
            source = d.metadata.get("source", "Unknown Document")
            results.append(f"--- Source: {source} ---\n{d.page_content}\n")
            
        return "\n".join(results)
    except Exception as e:
        return f"Failed to search knowledge base: {str(e)}"

class MCPAgentApp:
    def __init__(self):
        # We need a powerful model for ReAct planning
        self.llm = get_chat_model(temperature=0.0, max_tokens=2048)
        self.system_prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION}.
Your tone is {settings.ASSISTANT_TONE}.
{settings.CUSTOM_SYSTEM_INSTRUCTIONS}

You are an autonomous ReAct agent.
You have access to PostgreSQL database tools for managing user subscriptions, AND a knowledge base tool (`search_knowledge_base`) to search through uploaded PDFs and websites.

Important Behavioral Rules:
1. Be highly conversational. If a user asks a general question like "can I add subscriptions?", respond enthusiastically and ask them for the required details (Provider, Plan name, Price, Billing cycle, Next payment date) *before* trying to use your database tools.
2. Do not proactively query the database to find missing subscriptions just because a user mentions a number (e.g., "I have 4 subscriptions"). Just guide them to provide the details.
3. When you have the details, confirm them with the user before saving.
4. When a user gives you a clear goal, THINK about what data you need, SELECT the appropriate tool, OBSERVE the result, and EVALUATE. Once complete, synthesize a final answer in nice markdown."""
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        db_path = os.path.join(backend_dir, "data", "chat_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_uri = f"sqlite:///{db_path}"
        
        self.mcp_session = None

    def get_history(self, session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.db_uri
        )

    async def ainvoke(self, state_input: dict):
        final_result = None
        async for event in self.astream(state_input):
            if event["type"] == "complete":
                final_result = event
        return final_result

    async def astream(self, state_input: dict):
        question = state_input["question"]
        session_id = state_input["session_id"]
        
        chat_history = self.get_history(session_id)
        history_messages = chat_history.messages
        
        yield {"type": "stage", "label": "Agent Planning..."}
        
        # 1. Start with our Native Tool(s)
        tools = [search_local_knowledge_base]
        
        # 2. Add MCP Tools if connected
        from main import mcp_state
        mcp_tools = []
        for url, conn in mcp_state["connections"].items():
            try:
                tools_from_server = await load_mcp_tools(conn["session"])
                mcp_tools.extend(tools_from_server)
                logger.info(f"Loaded {len(tools_from_server)} tools from MCP Server {url}.")
            except Exception as e:
                logger.error(f"Error loading MCP tools from {url}: {e}")
                yield {"type": "stage", "label": f"Warning: MCP Tool server {url} disconnected!"}
        
        if mcp_tools:
            tools.extend(mcp_tools)
        else:
            logger.warning("No active MCP session tools found.")
            yield {"type": "stage", "label": "Running with Local Tools only"}
        
        current_messages = [SystemMessage(content=self.system_prompt)] + history_messages + [HumanMessage(content=question)]
        final_message = ""

        # 3. Create and run the ReAct Agent Graph (It ALWAYS has at least the RAG tool now)
        agent_executor = create_react_agent(self.llm, tools)
        
        async for chunk in agent_executor.astream({"messages": current_messages}):
            if "agent" in chunk:
                message = chunk["agent"]["messages"][0]
                if message.tool_calls:
                    for tc in message.tool_calls:
                        yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                else:
                    final_message = message.content
            elif "tools" in chunk:
                yield {"type": "stage", "label": "Observing Tool Result..."}
        
        # Save to DB
        chat_history.add_user_message(question)
        if final_message:
            chat_history.add_message(AIMessage(content=final_message))
        
        yield {"type": "complete", "final_message": final_message}

mcp_graph_app = MCPAgentApp()
