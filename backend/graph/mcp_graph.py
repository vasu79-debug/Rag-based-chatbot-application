import os
import sys
import json
import asyncio
import logging
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from graph.llm_factory import get_chat_model
from config import settings

from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPAgentApp:
    def __init__(self):
        # We need a powerful model for ReAct planning
        self.llm = get_chat_model(temperature=0.0, max_tokens=2048)
        self.system_prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION}.
Your tone is {settings.ASSISTANT_TONE}.
{settings.CUSTOM_SYSTEM_INSTRUCTIONS}

You are an autonomous ReAct agent.
You have access to PostgreSQL database tools for managing user subscriptions.
When a user gives you a goal (like "reduce my spending"), you must:
1. THINK about what data you need.
2. SELECT the appropriate tool (e.g. get_subscriptions).
3. OBSERVE the tool's result.
4. EVALUATE if you have enough information to fulfill the goal.
5. If not, use another tool (like delete_subscription) or ask the user for clarification.
6. Once complete, synthesize a final answer in nice markdown."""
        
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
        
        # 1. Load the DB CRUD tools dynamically from MCP
        tools = await load_mcp_tools(self.mcp_session)
        logger.info(f"Loaded {len(tools)} tools for ReAct Agent.")
        
        # 2. Create the ReAct Agent Graph!
        # This handles the exact loop: Goal -> Plan -> Select Tool -> Execute -> Observe -> Evaluate
        agent_executor = create_react_agent(self.llm, tools)
        
        current_messages = [SystemMessage(content=self.system_prompt)] + history_messages + [HumanMessage(content=question)]
        
        final_message = ""
        
        # 3. Stream the Agent's thought process
        async for chunk in agent_executor.astream({"messages": current_messages}):
            if "agent" in chunk:
                # The LLM planned something
                message = chunk["agent"]["messages"][0]
                if message.tool_calls:
                    for tc in message.tool_calls:
                        yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                else:
                    final_message = message.content
            elif "tools" in chunk:
                # A tool was executed
                yield {"type": "stage", "label": "Observing Tool Result..."}
                
        # Save to DB
        chat_history.add_user_message(question)
        if final_message:
            chat_history.add_message(AIMessage(content=final_message))
        
        yield {"type": "complete", "final_message": final_message}

mcp_graph_app = MCPAgentApp()
