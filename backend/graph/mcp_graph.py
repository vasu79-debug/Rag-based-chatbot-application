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
    import laya
    laya_runner = laya.load("convaiinnovations/laya")
except ImportError:
    laya = None
    laya_runner = None
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
6. LIVE DATA: You DO have access to the internet via the `tavily_search` tool. If the user asks for live sports scores, weather, news, or any real-time data, you MUST use the `tavily_search` tool. NEVER say you don't have access to live data.
7. READ-ONLY EXCEPTION: You DO NOT need to ask for permission to use read-only tools (like getting weather, viewing subscriptions, or checking alternative plans). Only ask for permission for WRITE actions (create, delete, modify).
8. BROWSER CAPABILITY: You HAVE full access to a web browser using tools like `navigate`, `click`, `type_text`, and `get_page_content`. NEVER tell the user that you cannot browse the web or log into websites. If you need them to log in, simply navigate to the page and use `request_human_action`.
9. SEARCH LIMIT: When using any web search tool, you are allowed a MAXIMUM of 3 searches per request. If you cannot find the answer after 3 attempts, you MUST STOP searching and politely inform the user that you could not find the information. DO NOT loop endlessly."""
        
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
                
        # ADD BROWSER TOOLS
        from browser_tools import BROWSER_TOOLS
        @tool
        async def request_human_action(reason: str, details: str = "") -> str:
            """
            [capability: browser] [operation: read]
            Call this tool when you encounter a Login page, OTP, or CAPTCHA.
            It will pause execution and wait for human approval.
            """
            response = interrupt({
                "type": "approval_request",
                "tool": "request_human_action",
                "args": {"reason": reason, "details": details}
            })
            if response == "approved":
                from browser_tools import get_page_content
                new_content = await get_page_content.ainvoke({})
                return f"Human has successfully completed the action! Here is the new screen content after they logged in:\n\n{new_content}"
            else:
                return "Human rejected or aborted. SYSTEM DIRECTIVE: Stop the task."

        tools.extend(BROWSER_TOOLS)
        tools.append(request_human_action)
                
        agent_executor = create_react_agent(self.llm, tools, checkpointer=memory, prompt=self.system_prompt)
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 12}
        
        final_message = ""
        try:
            state = agent_executor.get_state(config)
            
            # Map approval status to all pending interrupted tasks
            resume_val = {}
            for task in state.tasks:
                if task.interrupts:
                    for i in task.interrupts:
                        resume_val[i.id] = approval_status
            
            # Fallback for single interrupt format
            if not resume_val:
                resume_val = approval_status
                
            # We resume the graph with the correct Command format
            async for chunk in agent_executor.astream(Command(resume=resume_val), config=config):
                if "agent" in chunk:
                    message = chunk["agent"]["messages"][0]
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            logger.info(f"LLM Calling Tool: {tc['name']} with args: {tc['args']}")
                            from browser_tools import get_latest_screenshot_base64
                            b64_img = await get_latest_screenshot_base64()
                            if b64_img:
                                yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}", "image": b64_img}
                            else:
                                yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                    else:
                        final_message = message.content
                        if hasattr(message, "response_metadata") and "token_usage" in message.response_metadata:
                            logger.info(f"Token Usage for turn: {message.response_metadata['token_usage']}")
                elif "tools" in chunk:
                    from browser_tools import get_latest_screenshot_base64
                    b64_img = await get_latest_screenshot_base64()
                    if b64_img:
                        yield {"type": "stage", "label": "Observing Tool Result...", "image": b64_img}
                    else:
                        yield {"type": "stage", "label": "Observing Tool Result..."}
                elif "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value
                    from browser_tools import get_latest_screenshot_base64
                    b64_img = await get_latest_screenshot_base64()
                    if b64_img:
                        yield {"type": "approval_needed", "data": interrupt_data, "image": b64_img}
                    else:
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
        # ADD BROWSER TOOLS
        # ==========================================
        from browser_tools import BROWSER_TOOLS
        
        @tool
        async def request_human_action(reason: str, details: str = "") -> str:
            """
            [capability: browser] [operation: read]
            Call this tool when you encounter a Login page, OTP, or CAPTCHA.
            It will pause execution and wait for human approval.
            """
            response = interrupt({
                "type": "approval_request",
                "tool": "request_human_action",
                "args": {"reason": reason, "details": details}
            })
            if response == "approved":
                from browser_tools import get_page_content
                # Immediately fetch the new DOM so the LLM doesn't have to!
                new_content = await get_page_content.ainvoke({})
                return f"Human has successfully completed the action! Here is the new screen content after they logged in:\n\n{new_content}"
            else:
                return "Human rejected or aborted. SYSTEM DIRECTIVE: Stop the task."

        tools.extend(BROWSER_TOOLS)
        tools.append(request_human_action)
        
        # ==========================================
        # PARSE DISTRIBUTED METADATA
        # ==========================================
        import re
        tool_metadata_map = {}
        for t in tools:
            # Default metadata
            meta = {"capability": "other", "operation": "read"}
            
            # Hardcoded fallbacks for existing tools without tags
            if "tavily" in t.name: meta["capability"] = "search"
            elif "knowledge" in t.name: meta["capability"] = "knowledge"
            elif t.name in ["navigate", "click", "type_text", "get_page_content"]: meta["capability"] = "browser"
            
            # Parse [capability: X] [operation: Y] from description
            if t.description:
                cap_match = re.search(r"\[capability:\s*([^\]]+)\]", t.description)
                if cap_match: meta["capability"] = cap_match.group(1).strip()
                
                op_match = re.search(r"\[operation:\s*([^\]]+)\]", t.description)
                if op_match: meta["operation"] = op_match.group(1).strip()
                
                desc_match = re.search(r"\[intent_desc:\s*([^\]]+)\]", t.description)
                if desc_match: meta["intent_desc"] = desc_match.group(1).strip()
                
            tool_metadata_map[t.name] = meta


        # ==========================================
        # LAYA SYSTEM-1 (DYNAMIC INTENT GENERATION)
        # ==========================================
        yield {"type": "stage", "label": "Laya System-1: Probabilistic Capability Routing..."}
        triage_tools = tools # Default to all tools
        try:
            if laya and laya_runner:
                # 1. Dynamically build Laya Criteria based on currently connected tools
                dynamic_criteria = {}
                # Provide strong fallback descriptions for core domains
                core_descriptions = {
                    "subscription": "Questions about subscriptions, plans, or account management",
                    "knowledge": "Questions requiring local knowledge-base or document information",
                    "search": "Requests requiring live web search, weather, news, or general real-time facts",
                    "browser": "Requests requiring specific browser interaction or UI navigation",
                    "github": "Requests to interact with GitHub repositories, code, or creating issues"
                }
                
                for t in tools:
                    t_meta = tool_metadata_map.get(t.name, {})
                    cap = t_meta.get("capability")
                    if cap and cap not in dynamic_criteria:
                        # Use custom intent_desc if parsed, else use core fallback, else generic
                        desc = t_meta.get("intent_desc")
                        if not desc:
                            desc = core_descriptions.get(cap, f"Requests relating to {cap}")
                        dynamic_criteria[cap] = desc
                        
                dynamic_criteria["other"] = "Anything else"
                
                # 2. Ask System-1 independent noul questions for EVERY capability (Multi-Intent Support)
                questions = {}
                for cap, desc in dynamic_criteria.items():
                    if cap == "other": continue
                    questions[f"needs_{cap}"] = {
                        "type": "noul",
                        "instructions": f"Does the user's request require interacting with {cap}? ({desc})"
                    }
                
                questions["needs_browser"] = {
                    "type": "noul",
                    "instructions": "Does this request explicitly require navigating, logging in, or interacting with a website via a browser? (Examples: 'go to spotify', 'open amazon', 'visit website')"
                }
                questions["needs_write"] = {
                    "type": "noul",
                    "instructions": "Does this request ask to modify, create, delete, or write data?"
                }
                
                decision_result = await asyncio.to_thread(
                    laya.decide,
                    laya_runner,
                    state={"message": question},
                    questions=questions,
                    return_details=True
                )
                
                ans = decision_result.answers
                browser_prob = ans["needs_browser"]["noul"]
                write_prob = ans["needs_write"]["noul"]
                
                # Calculate the highest scored intent just for logging/fallback
                highest_intent_prob = max([ans[k]["noul"] for k in ans if k.startswith("needs_") and k not in ["needs_browser", "needs_write"]], default=0.0)
                
                logger.info(f"System-1 Multi-Intent Triage: Top Intent={highest_intent_prob:.2f}, browser={browser_prob:.2f}, write={write_prob:.2f}")

                # 3. DYNAMIC PYTHON POLICY (Using Distributed Tool Metadata)
                allowed_tools = []
                
                for t in tools:
                    t_meta = tool_metadata_map.get(t.name, {})
                    cap = t_meta.get("capability", "other")
                    op = t_meta.get("operation", "read")
                    
                    is_allowed = False
                    
                    # 1. Multi-Intent Match: Allow any tool whose specific capability scored high
                    cap_prob = 0.0
                    if cap != "other" and f"needs_{cap}" in ans:
                        cap_prob = ans[f"needs_{cap}"]["noul"]
                        if cap_prob >= 0.35:
                            is_allowed = True
                    
                    # 0. Follow-up Context Fallback
                    # If NO intents scored highly (extremely confused, short follow up like "yes"), 
                    # allow ALL tools so the LLM doesn't lose context of the workflow it is in.
                    if highest_intent_prob < 0.35:
                        is_allowed = True
                    
                    # 2. Browser override
                    if cap == "browser" and browser_prob >= 0.4:
                        is_allowed = True
                        
                    # 3. Search / Knowledge Fallback (always keep them available for basic queries)
                    if cap in ["knowledge", "search"]:
                        is_allowed = True
                        
                    # 4. Security Check (Block write tools if highly unlikely to need write access)
                    if op == "write":
                        if write_prob < 0.4 and highest_intent_prob >= 0.35:
                            # Only block writes if we are highly confident this is a new, read-only request
                            is_allowed = False
                        
                    if is_allowed:
                        allowed_tools.append(t)
                        
                triage_tools = allowed_tools
                
                if len(triage_tools) == 0: 
                    triage_tools = tools # Absolute safety fallback
        except Exception as e:
            logger.warning(f"Laya System-1 failed, falling back to all tools: {e}")
            
        if len(triage_tools) < len(tools):
            yield {"type": "stage", "label": f"Triage Result: Pruned down to {len(triage_tools)} tools for LLM"}
        else:
            yield {"type": "stage", "label": "Triage Result: Complex Query -> Passing all tools to LLM"}
            
        # 3. Create the ReAct Agent Graph with Checkpointer
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
                            logger.info(f"LLM Calling Tool: {tc['name']} with args: {tc['args']}")
                            from browser_tools import get_latest_screenshot_base64
                            b64_img = await get_latest_screenshot_base64()
                            if b64_img:
                                yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}", "image": b64_img}
                            else:
                                yield {"type": "stage", "label": f"Selecting Tool: {tc['name']}"}
                    else:
                        final_message = message.content
                        if hasattr(message, "response_metadata") and "token_usage" in message.response_metadata:
                            logger.info(f"Token Usage for turn: {message.response_metadata['token_usage']}")
                elif "tools" in chunk:
                    from browser_tools import get_latest_screenshot_base64
                    b64_img = await get_latest_screenshot_base64()
                    if b64_img:
                        yield {"type": "stage", "label": "Observing Tool Result...", "image": b64_img}
                    else:
                        yield {"type": "stage", "label": "Observing Tool Result..."}
                elif "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value
                    from browser_tools import get_latest_screenshot_base64
                    b64_img = await get_latest_screenshot_base64()
                    if b64_img:
                        yield {"type": "approval_needed", "data": interrupt_data, "image": b64_img}
                    else:
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
