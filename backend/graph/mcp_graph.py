import os
import sys
import json
import asyncio
import logging
from typing import Annotated, TypedDict, List
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.tools import tool, StructuredTool
import sqlite3
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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

# Persistent Checkpointer: stores agent state and pending approvals on disk in checkpoints.db
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoints_db_path = os.path.join(backend_dir, "data", "checkpoints.db")
os.makedirs(os.path.dirname(checkpoints_db_path), exist_ok=True)

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
        current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")
        current_year = datetime.now().year
        self.system_prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION}.
Your tone is {settings.ASSISTANT_TONE}.
{settings.CUSTOM_SYSTEM_INSTRUCTIONS}

TODAY'S DATE: {current_date_str}. The current year is {current_year}.
CRITICAL DATE RULE: All future travel and hotel bookings MUST use the current year {current_year} or future years. NEVER use past years (like 2025 or earlier). When the user says "November 21", they mean {current_year}-11-21!

You are an autonomous ReAct agent.
You have access to a knowledge base tool (`search_local_knowledge_base`), live web search (`tavily_search`), and browser automation tools (`navigate`, `click`, `type_text`, `press_key`, `get_page_content`, `request_human_action`).

When to use Tools vs Direct Conversation:
- CONVERSATIONAL TRAVEL PLANNING: For general advice, safety questions, recommendations, itinerary creation, and travel tips, respond conversationally and naturally. Do NOT open the browser for simple conversational questions.
- LIVE WEB SEARCH: Use `tavily_search` when you need real-time data like train timings, flight schedules, current weather, ticket rules, or to locate the exact direct booking page of a hotel on MakeMyTrip, Agoda, OYO, or Booking.com.
- BROWSER AUTOMATION: Use browser tools (`navigate`, `click`, `type_text`, `get_page_content`) to actively search, navigate, and proceed through booking portals (e.g. MakeMyTrip, Agoda, Booking.com, Google Hotels, Skyscanner).
- HUMAN TAKEOVER: Call `request_human_action` ONLY when you reach the final GUEST DETAILS / PAYMENT screen, or when encountering a Login, CAPTCHA, or OTP.

Important Behavioral Rules (The Three-Part Discipline: Preview, Approve, Verify):
1. CONVERSATION FIRST: Before executing any transaction, you MUST ask the user for any missing details. Do not hallucinate or guess data, prices, or dates.
2. PREVIEW: Once you have all details, you MUST present a Preview in a Markdown table and explicitly ask "Are these details correct? (Yes/No)".
3. STRICT FORBIDDEN ACTION: You are STRICTLY FORBIDDEN from calling any transaction tools UNTIL the user has explicitly replied "yes" to your preview. If you jump straight to the tool, you fail.
4. VERIFY: After the tool is executed and approved, verify the output and synthesize a final answer.
5. ABORTS: If a tool returns a message that the action was "rejected" or "aborted", DO NOT retry the tool. Acknowledge the cancellation and ask how else you can help.
6. LIVE DATA: You DO have access to the internet via the `tavily_search` tool. If the user asks for live sports scores, weather, news, or any real-time data, you MUST use the `tavily_search` tool. NEVER say you don't have access to live data.
7. READ-ONLY EXCEPTION: You DO NOT need to ask for permission to use read-only tools (like getting weather, searching flights/hotels, viewing itineraries, or browsing pages). Only ask for permission for WRITE/BOOKING actions (confirming reservations, purchases, payments).
8. AUTONOMOUS BOOKING & DIRECT DEEP-LINKING (CHATGPT AGENT STYLE):
You are an autonomous browser agent. You MUST operate efficiently like ChatGPT Agent:
- DIRECT SEARCH URLs: NEVER navigate to an empty homepage to manually fill search forms, dates, and dropdowns by hand! That is slow, prone to popups, and causes endless loops.
  Always construct direct, pre-filled deep-link URLs:
  * Google Hotels (Fastest, cleanest, compares all portals without blocking):
    `https://www.google.com/travel/hotels/{{destination}}?q={{hotel_or_area}}&dates={{checkin_date}},{{checkout_date}}`
  * Booking.com:
    `https://www.booking.com/searchresults.en-us.html?ss={{destination}}&checkin={{checkin_date}}&checkout={{checkout_date}}&group_adults={{adults}}&no_rooms={{rooms}}`
  * Specific Property (OYO / Boutique / Resort): Use `tavily_search` to get the direct listing URL on MakeMyTrip, Agoda, or OYO, and navigate directly to that property!
- STRICT RULE — NEVER RE-CLICK SEARCH BARS OR DATES ONCE NAVIGATED:
  When you navigate to a deep-link search results page, the results are ALREADY loaded below!
  DO NOT touch, click, or reopen the search bar, calendar picker, or guest dropdown at the top of the page.
  Instead, immediately call `get_page_content` to read the list of hotels, find the property or room, and click the hotel link or "Select Room" / "View Deal".
- DISMISS PROMOTIONAL POPUPS: When on Booking.com or travel sites, if a 'Genius: Sign in, save money' or promotional banner popup appears with a close (X) button, dismiss it immediately using `press_key('Escape')` or `click("role=button[name='Dismiss sign-in info.']")` so you can view the hotels! DO NOT pause for human action on simple dismissible popups.
- Once you reach the room booking summary, or the GUEST DETAILS / PAYMENT form, call `request_human_action` for human takeover so the user can enter their personal details and complete the booking!
9. SEARCH LIMIT: When using any web search tool, you are allowed a MAXIMUM of 3 searches per request. If you cannot find the answer after 3 attempts, you MUST STOP searching and politely inform the user that you could not find the information. DO NOT loop endlessly.
10. EFFICIENT BROWSING: Navigate directly to the destination URL when known or found, rather than clicking through multi-step search engine pagination.
11. SEARCH EFFICIENCY: When calling tavily_search, ALWAYS set max_results=3. NEVER request 10 results. Make your search query concise to keep responses fast and token-efficient."""
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        db_path = os.path.join(backend_dir, "data", "chat_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_uri = f"sqlite:///{db_path}"
        
        self.mcp_session = None

    async def _wrap_tool(self, original_tool, write_tools: list) -> StructuredTool:
        """Wraps tools to add HITL interrupt approval or truncate noisy search results."""
        if original_tool.name in write_tools:
            async def wrapped_write(**kwargs):
                response = interrupt({"type": "approval_request", "tool": original_tool.name, "args": kwargs})
                if response == "approved":
                    return await original_tool.ainvoke(kwargs)
                else:
                    return f"Action {original_tool.name} was rejected by the user. SYSTEM DIRECTIVE: DO NOT retry this tool. Inform the user that the operation was aborted."
            return StructuredTool(
                name=original_tool.name,
                description=original_tool.description,
                args_schema=original_tool.args_schema,
                coroutine=wrapped_write,
                func=None,
            )
        elif "tavily" in original_tool.name:
            async def wrapped_tavily(**kwargs):
                # Clamp max_results to 3 to prevent token explosion
                if "max_results" in kwargs and kwargs["max_results"] > 3:
                    kwargs["max_results"] = 3
                res = await original_tool.ainvoke(kwargs)
                res_str = str(res)
                if len(res_str) > 1500:
                    return res_str[:1500] + "\n...[search results truncated to save tokens]"
                return res
            return StructuredTool(
                name=original_tool.name,
                description=original_tool.description,
                args_schema=original_tool.args_schema,
                coroutine=wrapped_tavily,
                func=None,
            )
        return original_tool

    def _prepare_agent_prompt(self, state: dict) -> list:
        """
        Dynamically pre-processes messages sent to LLM before each ReAct turn:
        1. Keeps the system prompt intact.
        2. Separates past completed dialogue turns from the current active turn.
        3. For past turns: retains clean (User, Assistant final answer) pairs, omitting old intermediate
           tool execution dumps (old Tavily searches, old page snapshots) which explode context size.
           Applies a sliding window of the last 6 clean messages (3 turns).
        4. For the current active turn: keeps all live tool calls and observations, but caps tool output length
           (max 1,500 chars) to prevent prompt token explosions.
        """
        msgs = state.get("messages", [])
        if not msgs:
            return [SystemMessage(content=self.system_prompt)]

        # Find the start of the current turn (last HumanMessage)
        last_human_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                last_human_idx = i
                break

        if last_human_idx == -1:
            last_human_idx = 0

        # 1. Clean past turns: keep only human prompts and final assistant text answers
        past_clean = []
        for i in range(last_human_idx):
            m = msgs[i]
            if isinstance(m, HumanMessage):
                past_clean.append(m)
            elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
                past_clean.append(m)

        # Sliding window: keep the last 6 clean messages (3 conversation turns)
        past_window = past_clean[-6:]

        # 2. Current active turn: keep intermediate steps but cap tool outputs
        curr_msgs = []
        for m in msgs[last_human_idx:]:
            if isinstance(m, ToolMessage):
                content_str = str(m.content)
                if len(content_str) > 1500:
                    m = ToolMessage(
                        content=content_str[:1500] + "\n...[output truncated to save tokens]",
                        tool_call_id=getattr(m, "tool_call_id", ""),
                        name=getattr(m, "name", None)
                    )
            curr_msgs.append(m)

        return [SystemMessage(content=self.system_prompt)] + past_window + curr_msgs

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
        from browser_tools import set_current_task_context, clear_kill_flag
        set_current_task_context(session_id)  # bind browser tool calls to this session's sandbox
        clear_kill_flag(session_id)  # a fresh turn always overrides a previous Emergency Stop
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
                    wrapped_t = await self._wrap_tool(t, WRITE_TOOLS)
                    tools.append(wrapped_t)
            except Exception:
                pass
                
        # ADD BROWSER TOOLS
        from browser_tools import BROWSER_TOOLS
        tools.extend(BROWSER_TOOLS)
                
        final_message = ""
        try:
            async with AsyncSqliteSaver.from_conn_string(checkpoints_db_path) as memory:
                agent_executor = create_react_agent(self.llm, tools, checkpointer=memory, prompt=self._prepare_agent_prompt)
                config = {"configurable": {"thread_id": session_id}, "recursion_limit": 80}
                state = await agent_executor.aget_state(config)
                
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

        from browser_tools import set_current_task_context, clear_kill_flag, reset_page
        set_current_task_context(session_id, question)  # bind browser tool calls to this session's sandbox & user task
        clear_kill_flag(session_id)  # a fresh turn always overrides a previous Emergency Stop
        await reset_page(session_id)  # don't let a new task start by showing a previous task's leftover page
        
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
                    wrapped_t = await self._wrap_tool(t, WRITE_TOOLS)
                    mcp_tools.append(wrapped_t)
                        
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
        tools.extend(BROWSER_TOOLS)
        
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
            elif t.name in ["navigate", "click", "type_text", "get_page_content", "request_human_action"]: meta["capability"] = "browser"
            
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
                    
                    # Core Capabilities: In this Browser Agent application, Browser, Search, and Knowledge tools are ALWAYS available
                    if cap in ["browser", "knowledge", "search"]:
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
            
        # 3. Create the ReAct Agent Graph with Async Checkpointer
        final_message = ""
        try:
            async with AsyncSqliteSaver.from_conn_string(checkpoints_db_path) as memory:
                agent_executor = create_react_agent(self.llm, triage_tools, checkpointer=memory, prompt=self._prepare_agent_prompt)
                config = {"configurable": {"thread_id": session_id}, "recursion_limit": 80}
                
                # Avoid duplicate messages in checkpointer state
                state = await agent_executor.aget_state(config)
                
                # RECOVERY: Check if the previous run was aborted or killed while an AIMessage had pending tool_calls
                if state.values and "messages" in state.values:
                    existing = state.values["messages"]
                    if existing and isinstance(existing[-1], AIMessage) and getattr(existing[-1], "tool_calls", None):
                        logger.warning(f"Resolving dangling tool_calls in checkpoint state to prevent INVALID_CHAT_HISTORY crash...")
                        recovery_tool_msgs = [
                            ToolMessage(
                                content="Action aborted or cancelled by user.",
                                tool_call_id=tc["id"],
                                name=tc.get("name", "tool")
                            )
                            for tc in existing[-1].tool_calls
                        ]
                        await agent_executor.aupdate_state(config, {"messages": recovery_tool_msgs})

                if not state.values:
                    # Checkpointer is empty (e.g., server restart), seed with full history
                    current_messages = history_messages + [HumanMessage(content=question)]
                else:
                    # Checkpointer already has history, only send the new question
                    current_messages = [HumanMessage(content=question)]
                
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
