import os
import sys
import json
import logging
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from graph.llm_factory import get_chat_model
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the State for our Workflow
class WorkflowState(TypedDict):
    question: str
    session_id: str
    action_type: str  # "analyze_cost", "chat"
    raw_subscriptions: list
    analysis_report: str
    final_message: str

class MCPAgentApp:
    def __init__(self):
        self.llm = get_chat_model(temperature=0.0, max_tokens=2048)
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(backend_dir, "data", "chat_history.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_uri = f"sqlite:///{db_path}"
        
        self.mcp_session = None
        
        # Build the LangGraph Workflow
        workflow = StateGraph(WorkflowState)
        
        # Add Nodes
        workflow.add_node("extractor", self.node_extractor)
        workflow.add_node("fetch_data", self.node_fetch_data)
        workflow.add_node("analyze", self.node_analyze)
        workflow.add_node("synthesize", self.node_synthesize)
        workflow.add_node("chat", self.node_chat)
        
        # Add Edges (Routing)
        workflow.set_entry_point("extractor")
        
        workflow.add_conditional_edges(
            "extractor",
            lambda state: state["action_type"],
            {
                "analyze_cost": "fetch_data",
                "chat": "chat"
            }
        )
        
        # Fixed path for the analyze_cost workflow
        workflow.add_edge("fetch_data", "analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", END)
        workflow.add_edge("chat", END)
        
        self.graph = workflow.compile()

    def get_history(self, session_id: str):
        return SQLChatMessageHistory(session_id=session_id, connection=self.db_uri)

    # --- Node 1: Extract Intent ---
    async def node_extractor(self, state: WorkflowState):
        logger.info("🟢 STEP 1: Extracting Intent")
        prompt = f"""Does the following user request ask about viewing, analyzing, or summarizing their subscriptions or costs?
Reply with exactly one word: 'YES' or 'NO'.
User request: {state['question']}"""
        
        response = await self.llm.ainvoke([("user", prompt)])
        action = "analyze_cost" if "YES" in response.content.upper() else "chat"
        return {"action_type": action}

    # --- Node 2: Fetch Data via MCP ---
    async def node_fetch_data(self, state: WorkflowState):
        logger.info("🟢 STEP 2: Fetching Data via MCP Tool")
        tools = await load_mcp_tools(self.mcp_session)
        
        # Find the specific tool we want
        get_subs_tool = next((t for t in tools if t.name == "get_subscriptions"), None)
        
        if get_subs_tool:
            raw_result = await get_subs_tool.ainvoke({})
            logger.info(f"MCP Tool returned: {raw_result}")
            try:
                # LangChain MCP tools return a string representation of a list of blocks
                import ast
                if isinstance(raw_result, str) and raw_result.startswith("["):
                    parsed_list = ast.literal_eval(raw_result)
                    json_str = parsed_list[0].get("text", "{}")
                    result_json = json.loads(json_str)
                else:
                    result_json = json.loads(raw_result)
                subs = result_json.get("data", [])
            except Exception as e:
                logger.error(f"Failed to parse JSON from MCP Tool: {e}")
                subs = []
        else:
            subs = []
            
        return {"raw_subscriptions": subs}

    # --- Node 3: Pure Python Analysis ---
    async def node_analyze(self, state: WorkflowState):
        logger.info("🟢 STEP 3: Analyzing Data with Python")
        subs = state.get("raw_subscriptions", [])
        
        if not subs:
            return {"analysis_report": "No subscriptions found in the database."}
            
        total_monthly = 0.0
        report = "System Analysis Data (For LLM Context):\n"
        
        for sub in subs:
            cost = sub["cost"]
            cycle = sub["billing_cycle"]
            
            # Normalize to monthly
            if cycle == "yearly":
                monthly_equivalent = cost / 12
            elif cycle == "weekly":
                monthly_equivalent = cost * 4.33
            else:
                monthly_equivalent = cost
                
            total_monthly += monthly_equivalent
            report += f"- {sub['service_name']}: ${cost} ({cycle}) -> ${monthly_equivalent:.2f}/month\n"
            
        report += f"\nTOTAL ESTIMATED MONTHLY SPEND: ${total_monthly:.2f}"
        return {"analysis_report": report}

    # --- Node 4: LLM Synthesis ---
    async def node_synthesize(self, state: WorkflowState):
        logger.info("🟢 STEP 4: Synthesizing Final Response")
        prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION} 
Answer the user's question based strictly on the following data analysis report. Format it nicely using markdown.
        
Data Report:
{state['analysis_report']}

User Question: {state['question']}"""
        
        response = await self.llm.ainvoke([("system", prompt)])
        return {"final_message": response.content}
        
    # --- Alternative Node: Normal Chat ---
    async def node_chat(self, state: WorkflowState):
        logger.info("🔵 STEP: Normal Chat")
        prompt = f"""You are {settings.ASSISTANT_NAME}, {settings.ASSISTANT_ROLE_DESCRIPTION}
Tone: {settings.ASSISTANT_TONE}
{settings.CUSTOM_SYSTEM_INSTRUCTIONS}

Reply to the user gracefully.
User: {state['question']}"""
        response = await self.llm.ainvoke([("system", prompt)])
        return {"final_message": response.content}

    # --- Orchestrator Methods ---
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
        
        # Initial State
        state = {
            "question": question,
            "session_id": session_id
        }
        
        yield {"type": "stage", "label": "Executing fixed pipeline workflow..."}
        
        # Run the LangGraph Workflow
        final_state = await self.graph.ainvoke(state)
        
        final_msg = final_state["final_message"]
        
        # Save to DB
        chat_history.add_user_message(question)
        chat_history.add_message(AIMessage(content=final_msg))
        
        yield {"type": "complete", "final_message": final_msg}

mcp_graph_app = MCPAgentApp()
