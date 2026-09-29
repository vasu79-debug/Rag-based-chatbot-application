import os
import sys
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from graph.llm_factory import get_chat_model
from config import settings

# We create a simple class to mimic the interface of LangGraph for main.py
class MCPAgentApp:
    def __init__(self):
        self.llm = get_chat_model(temperature=0.0, max_tokens=2048)
        self.system_prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION}.
Your tone is {settings.ASSISTANT_TONE}.
{settings.CUSTOM_SYSTEM_INSTRUCTIONS}

You are a highly capable API-enabled assistant. Your job is to understand the user request and answer it completely by selecting the correct tool.
If the user is missing required parameters for a tool, politely ask them for the missing information rather than guessing.
If the API returns an error or no records found, explain this clearly to the user in plain language.
If the API returns malformed data, mention that the service returned an unexpected format.
Explain the returned data in detailed, natural, human-friendly language. Provide comprehensive and thoroughly formatted responses."""
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Ensure data dir exists
        db_path = os.path.join(backend_dir, "data", "chat_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_uri = f"sqlite:///{db_path}"
        
        # This will be populated by main.py's lifespan
        self.mcp_session = None

    def get_history(self, session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.db_uri
        )

    async def ainvoke(self, state_input: dict):
        # We can just consume astream for the synchronous invoke
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
        
        # Dynamically load tools exposed by the MCP Server using the persistent session
        tools = await load_mcp_tools(self.mcp_session)
        
        llm_with_tools = self.llm.bind_tools(tools)
        
        user_msg = HumanMessage(content=question)
        
        # Prepend System Prompt and append history
        current_messages = [("system", self.system_prompt)] + history_messages + [user_msg]
        
        # 1. Ask LLM what to do
        yield {"type": "stage", "label": "Analyzing..."}
        response = await llm_with_tools.ainvoke(current_messages)
        
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            # No tool calls needed
            chat_history.add_user_message(question)
            chat_history.add_message(response)
            yield {"type": "complete", "final_message": response.content}
            return
        
        current_messages.append(response)
        
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 2. Execute the tool calls natively using MCP adapter
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"🛠️ LLM Selected Tool: '{tool_name}' with arguments: {tool_args}")
            yield {"type": "stage", "label": f"Running tool: {tool_name}..."}
            
            tool_instance = next((t for t in tools if t.name == tool_name), None)
            if tool_instance:
                try:
                    # Invoke tool on the MCP server
                    result = await tool_instance.ainvoke(tool_args)
                    logger.info(f"✅ Tool Response ('{tool_name}'): {str(result)}")
                    tool_msg = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                except Exception as e:
                    logger.error(f"❌ Tool Error ('{tool_name}'): {str(e)}")
                    tool_msg = ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
            else:
                logger.warning(f"⚠️ Tool Not Found: '{tool_name}'")
                tool_msg = ToolMessage(
                    content=f"Error: Tool {tool_name} not found.",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            current_messages.append(tool_msg)
        
        # 3. Final synthesis
        yield {"type": "stage", "label": "Synthesizing final response..."}
        final_response = await llm_with_tools.ainvoke(current_messages)
        
        # Save to DB
        chat_history.add_user_message(question)
        chat_history.add_message(final_response)
        
        yield {"type": "complete", "final_message": final_response.content}

mcp_graph_app = MCPAgentApp()
