import os
import sys
import json
import asyncio
import logging
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.tools import tool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


try:
    from laya.integrations.langchain import LayaTriage
except ImportError:
    LayaTriage = None
from graph.llm_factory import get_chat_model
from config import settings
from rag.hybrid_retriever import hybrid_retriever

from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Checkpointer for saving agent state
memory = MemorySaver()

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
You have access to a knowledge base tool (`search_knowledge_base`) and dynamically loaded MCP external tools to execute actions on behalf of the user.

Important Behavioral Rules (The Three-Part Discipline: Preview, Approve, Verify):
1. CONVERSATION FIRST: Before executing any transaction, you MUST ask the user for any missing details. Do not hallucinate or guess data, prices, or dates.
2. PREVIEW: Once you have all details, you MUST present a Preview in a Markdown table and explicitly ask "Are these details correct? (Yes/No)".
3. STRICT FORBIDDEN ACTION: You are STRICTLY FORBIDDEN from calling any transaction tools UNTIL the user has explicitly replied "yes" to your preview. If you jump straight to the tool, you fail.
4. VERIFY: After the tool is executed and approved, verify the output and synthesize a final answer.
5. ABORTS: If a tool returns a message that the action was "rejected" or "aborted", DO NOT retry the tool. Acknowledge the cancellation and ask how else you can help.
6. LIVE DATA: You DO have access to the internet via the `search_web` tool. If the user asks for live sports scores, weather, news, or any real-time data, you MUST use the `search_web` tool. NEVER say you don't have access to live data."""
        
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

    async def aresume(self, session_id: str, approval_status: str):
        # We need to recreate the exact same agent_executor setup (tools can't be empty)
        # Because we need the graph. Since tools are dynamic, we should fetch them again or cache them.
        # For this demo, we can just rebuild it quickly
        from main import mcp_state
        tools = [search_local_knowledge_base]
        WRITE_TOOLS = settings.write_tools_list
        
        for url, conn in mcp_state["connections"].items():
            try:
                tools_from_server = await conn["adapter"].list_tools()
                for t in tools_from_server:
                    if t.name in WRITE_TOOLS:
                        async def make_wrapper(original_tool=t):
                            async def wrapped_func(**kwargs):
                                response = interrupt({"type": "approval_request", "tool": original_tool.name, "args": kwargs})
                                if response == "approved":
                                    return await original_tool.ainvoke(kwargs)
                                else:
                                    return f"Action {original_tool.name} was rejected by the user. SYSTEM DIRECTIVE: DO NOT retry this tool. Inform the user that the operation was aborted."
                            return wrapped_func
                        wrapper_coro = await make_wrapper(t)
                        wrapped_tool = StructuredTool(name=t.name, description=t.description, args_schema=t.args_schema, coroutine=wrapper_coro, func=None)
                        tools.append(wrapped_tool)
                    else:
                        tools.append(t)
            except Exception:
                pass
                
        agent_executor = create_react_agent(self.llm, tools, checkpointer=memory, prompt=self.system_prompt)
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 12}
        
        final_message = ""
        try:
            # We resume the graph with the user's string ("approved" or "rejected")
            async for chunk in agent_executor.astream(Command(resume=approval_status), config=config):
                if "agent" in chunk:
                    message = chunk["agent"]["messages"][0]
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                    else:
                        final_message = message.content
                elif "tools" in chunk:
                    yield {"type": "stage", "label": "Observing Tool Result..."}
                elif "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value
                    yield {"type": "approval_needed", "data": interrupt_data}
                    return
        except Exception as e:
            logger.error(f"Graph resume error: {e}")
            yield {"type": "stage", "label": f"Error: {str(e)}"}
            return
            
        chat_history = self.get_history(session_id)
        if final_message:
            chat_history.add_message(AIMessage(content=final_message))
        
        yield {"type": "complete", "final_message": final_message}

    async def astream(self, state_input: dict):
        question = state_input["question"]
        session_id = state_input["session_id"]
        
        chat_history = self.get_history(session_id)
        history_messages = chat_history.messages
        
        # ==========================================
        # LESSON 1: MEMORY COMPRESSION (SUMMARIZATION)
        # ==========================================
        if len(history_messages) > 6:
            yield {"type": "stage", "label": f"Compressing {len(history_messages)} old messages to save tokens..."}
            
            summary_prompt = "Briefly summarize this conversation in 2-3 sentences. Focus on the user's core intent, facts provided, and actions taken:\n\n"
            for m in history_messages:
                role = "User" if m.type == "human" else "Agent"
                summary_prompt += f"{role}: {m.content}\n"
                
            try:
                # We use the main LLM here, but in production, we can use a smaller model 
                # (e.g., Llama-3-8B) to save even more cost on this summarization step.
                summary_result = await self.llm.ainvoke(summary_prompt)
                summary_text = summary_result.content
                
                # Clear the old database rows and save only the summary
                chat_history.clear()
                chat_history.add_message(SystemMessage(content=f"Previous conversation summary:\n{summary_text}"))
                
                # Refresh our local variable
                history_messages = chat_history.messages
                yield {"type": "stage", "label": "Memory Compressed Successfully!"}
            except Exception as e:
                logger.error(f"Memory compression failed: {e}")
                yield {"type": "stage", "label": "Memory compression skipped (error)."}
        
        yield {"type": "stage", "label": "Agent Planning..."}
        
        # 1. Start with our Native Tool(s)
        tools = [search_local_knowledge_base]
        
        # 2. Add MCP Tools if connected
        from main import mcp_state
        mcp_tools = []
        WRITE_TOOLS = settings.write_tools_list

        for url, conn in mcp_state["connections"].items():
            try:
                tools_from_server = await conn["adapter"].list_tools()
                for t in tools_from_server:
                    if t.name in WRITE_TOOLS:
                        # Wrap write tools with Human-in-the-Loop Interrupt
                        async def make_wrapper(original_tool=t):
                            async def wrapped_func(**kwargs):
                                # Pause execution and ask for UI approval
                                response = interrupt({
                                    "type": "approval_request",
                                    "tool": original_tool.name,
                                    "args": kwargs
                                })
                                if response == "approved":
                                    return await original_tool.ainvoke(kwargs)
                                else:
                                    return f"Action {original_tool.name} was rejected by the user. SYSTEM DIRECTIVE: DO NOT retry this tool. Inform the user that the operation was aborted."
                            return wrapped_func
                        
                        wrapper_coro = await make_wrapper(t)
                        wrapped_tool = StructuredTool(
                            name=t.name,
                            description=t.description,
                            args_schema=t.args_schema,
                            coroutine=wrapper_coro,
                            func=None # Async only
                        )
                        mcp_tools.append(wrapped_tool)
                    else:
                        mcp_tools.append(t)
                        
                logger.info(f"Loaded {len(tools_from_server)} tools from MCP Server {url}.")
            except Exception as e:
                logger.error(f"Error loading MCP tools from {url}: {e}")
                yield {"type": "stage", "label": f"Warning: MCP Tool server {url} disconnected!"}
        
        if mcp_tools:
            tools.extend(mcp_tools)
        else:
            logger.warning("No active MCP session tools found.")
            yield {"type": "stage", "label": "Running with Local Tools only"}
            
        # ==========================================
        # LAYA TRIAGE LAYER (SYSTEM-1 ROUTING)
        # ==========================================
        yield {"type": "stage", "label": "Laya Triage: Classifying Intent & Filtering Tools..."}
        triage_tools = tools # Default to all tools
        try:
            if LayaTriage:
                triage = LayaTriage(state_key="message")
                state = {"message": question}
                decision = await triage.ainvoke(state)
                
                # Retrieve the intent from the dictionary
                triage_dict = decision.get("triage", {})
                intent = triage_dict.get("intent", "other")
                
                # ==========================================
                # DOMAIN MAPPING (ChatGPT Architecture)
                # ==========================================
                domain_mapping = {
                    "subscription": ["get_subscriptions", "create_subscription", "delete_subscription", "get_alternative_plans", "get_usage_statistics", "search_public_subscription_data"],
                    "knowledge": ["search_local_knowledge_base"],
                    "billing": ["get_subscriptions"] 
                }
                
                if intent == "other" or intent not in domain_mapping:
                    # If Laya isn't sure, fallback to all tools so LLM can decide
                    triage_tools = tools 
                else:
                    # Filter tools EXACTLY as per the mapped domain
                    mapped_tool_names = domain_mapping[intent]
                    triage_tools = [t for t in tools if t.name in mapped_tool_names]
                    
                if len(triage_tools) == 0: 
                    triage_tools = tools # Safety fallback
        except Exception as e:
            logger.warning(f"Laya Triage failed, falling back to all tools: {e}")
            
        if len(triage_tools) < len(tools):
            yield {"type": "stage", "label": f"Triage Result: Pruned down to {len(triage_tools)} tools for LLM"}
        else:
            yield {"type": "stage", "label": "Triage Result: Complex Query -> Passing all tools to LLM"}
            
        # 3. Create the ReAct Agent Graph with Checkpointer (Using filtered tools!)
        agent_executor = create_react_agent(self.llm, triage_tools, checkpointer=memory, prompt=self.system_prompt)
        config = {"configurable": {"thread_id": session_id}, "recursion_limit":12}
        
        # Avoid duplicate messages in checkpointer state
        state = agent_executor.get_state(config)
        if not state.values:
            # Checkpointer is empty (e.g., server restart), seed with full history
            current_messages = history_messages + [HumanMessage(content=question)]
        else:
            # Checkpointer already has history, only send the new question
            current_messages = [HumanMessage(content=question)]
            
        final_message = ""
        
        try:
            # We must use resume if the graph was paused on this thread!
            # But for simplicity, we always send the messages unless we are explicitly resuming.
            # We will handle "resume" logic in a separate endpoint if needed, but for now we just run it.
            async for chunk in agent_executor.astream({"messages": current_messages}, config=config):
                if "agent" in chunk:
                    message = chunk["agent"]["messages"][0]
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                    else:
                        final_message = message.content
                elif "tools" in chunk:
                    yield {"type": "stage", "label": "Observing Tool Result..."}
                elif "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value
                    yield {"type": "approval_needed", "data": interrupt_data}
                    return # Stop execution here until resumed!
                    
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            yield {"type": "stage", "label": f"Error: {str(e)}"}
        
        # Save to DB only if we finished without interrupting
        chat_history.add_user_message(question)
        if final_message:
            chat_history.add_message(AIMessage(content=final_message))
        
        yield {"type": "complete", "final_message": final_message}

mcp_graph_app = MCPAgentApp()
