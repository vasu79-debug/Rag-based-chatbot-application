import os
import sys
import json
import asyncio
import logging
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from graph.llm_factory import get_chat_model
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    async def process_intent(self, intent: str, history_messages, tools) -> dict:
        """Processes a single intent and returns its final text."""
        llm_with_tools = self.llm.bind_tools(tools)
        current_messages = [("system", self.system_prompt)] + history_messages + [HumanMessage(content=intent)]
        
        try:
            response = await llm_with_tools.ainvoke(current_messages)
            
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                return {"intent": intent, "result": response.content, "success": True}
                
            current_messages.append(response)
            
            # Execute tools sequentially for this intent
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"🛠️ LLM Selected Tool: '{tool_name}' with arguments: {tool_args}")
                
                tool_instance = next((t for t in tools if t.name == tool_name), None)
                if tool_instance:
                    try:
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
                
            # Final synthesis for this intent (using raw llm to guarantee a text response)
            final_response = await self.llm.ainvoke(current_messages)
            return {"intent": intent, "result": final_response.content, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Intent Processing Error: {str(e)}")
            return {"intent": intent, "result": f"An error occurred while processing this request: {str(e)}", "success": False}


    async def astream(self, state_input: dict):
        question = state_input["question"]
        session_id = state_input["session_id"]
        
        chat_history = self.get_history(session_id)
        history_messages = chat_history.messages
        
        # Dynamically load tools exposed by the MCP Server
        tools = await load_mcp_tools(self.mcp_session)
        
        # 1. Decompose Intents
        yield {"type": "stage", "label": "Decomposing user request into independent intents..."}
        
        decomposition_prompt = f"""You are an expert intent decomposer. 
Analyze the user's message and split it into independent requests. 
If the message contains only one request or they are dependent on each other, return just one intent.
If there are multiple independent intents, list each one clearly as a separate string.
Return your answer purely as a valid JSON array of strings, with no markdown formatting, no backticks, and no extra text.
Example 1: ["Show my annual subscriptions", "tell me my monthly spending", "explain how annual billing works"]
Example 2: ["What is the capital of France?"]
User message: {question}"""

        decomp_response = await self.llm.ainvoke([("system", decomposition_prompt)])
        
        try:
            # Clean up potential markdown formatting from LLM
            raw_text = decomp_response.content.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
            
            intents = json.loads(raw_text)
            if not isinstance(intents, list) or not intents:
                intents = [question]
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}. Falling back to single intent.")
            intents = [question]
            
        logger.info(f"Detected Intents: {intents}")
        
        if len(intents) == 1:
            yield {"type": "stage", "label": "Processing intent..."}
        else:
            yield {"type": "stage", "label": f"Processing {len(intents)} intents in parallel..."}
            
        # 2. Run intents in parallel
        tasks = [
            self.process_intent(intent, history_messages, tools)
            for intent in intents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 3. Combine responses
        yield {"type": "stage", "label": "Combining and synthesizing final response..."}
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if len(failed_results) == len(results) and len(results) > 0:
            final_message = "All parts of your request failed to process. Please try again."
        elif len(failed_results) > 0 and not settings.ALLOW_PARTIAL_SUCCESS:
            final_message = "Some parts of your request failed. We only allow full success. Please split your request and try again."
        else:
            if len(results) == 1:
                # If only one intent, just return the result
                final_message = results[0]["result"]
            else:
                # Combine multiple intents using LLM to format cleanly, or just format explicitly
                combined_text = "Here are the answers to your requests:\n\n"
                for i, res in enumerate(results, 1):
                    status = "✅" if res["success"] else "❌"
                    combined_text += f"### {status} Request {i}: {res['intent']}\n{res['result']}\n\n"
                final_message = combined_text

        # Save to DB
        chat_history.add_user_message(question)
        chat_history.add_message(AIMessage(content=final_message))
        
        yield {"type": "complete", "final_message": final_message}

mcp_graph_app = MCPAgentApp()
