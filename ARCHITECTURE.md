# Demo 7A Architecture: Transactional AI with Human-in-the-Loop

This document outlines the architectural patterns used to achieve safe, transactional API executions using LangGraph, React, and MCP.

## 1. The Core Data Flow (HITL)

When a user requests a sensitive action (e.g., modifying database records), the system enforces a strict boundary between "planning" and "execution".

1. **User Request:** "Cancel my Netflix subscription."
2. **LLM Planning:** The ReAct agent identifies the need to call `delete_subscription`. It formats a conversational preview and yields text to the user: "Are you sure you want to cancel?"
3. **User Confirmation:** The user types "Yes".
4. **Tool Selection:** The agent attempts to invoke the `delete_subscription` tool.
5. **The Interceptor (`mcp_graph.py`):**
   - The graph dynamically checks the requested tool against `settings.write_tools_list`.
   - If a match is found, the graph calls `interrupt()`.
   - The graph execution is **suspended** and its exact state is saved into the LangGraph `MemorySaver` checkpointer.
   - The backend yields a special `approval_needed` event to the frontend over SSE.
6. **The Approval UI (`ChatView.jsx`):**
   - The React frontend intercepts the `approval_needed` payload.
   - It renders a hard boundary UI (Action Approval Box) displaying the Target Tool and its JSON Arguments.
   - The UI provides `[Approve]` and `[Reject]` buttons.
7. **The Resume API (`main.py`):**
   - Clicking a button triggers a POST request to `/api/chat/resume`.
   - The backend calls `aresume(session_id, action)`.
   - LangGraph retrieves the paused state from the checkpointer and injects the user's action.
8. **Execution & Verification:** 
   - If "approved", the graph executes the MCP tool, observes the database result, and synthesizes a final verified answer.
   - If "rejected", the graph gracefully aborts the tool execution.

## 2. Dynamic Tool Interception (The Wrapper)

Instead of hardcoding intercepts into the MCP server (which is decoupled), the graph dynamically wraps incoming MCP tools with an interrupt layer.

```python
if tool.name in WRITE_TOOLS:
    async def wrapped_func(**kwargs):
        response = interrupt({"type": "approval_request", "tool": tool.name, "args": kwargs})
        if response == "approved":
            return await tool.ainvoke(kwargs)
        else:
            return "Rejected by user."
```
*Why this is powerful:* The agent can connect to ANY 3rd party MCP server (e.g., GitHub, Jira, Slack). As long as the admin lists the sensitive tool names in `config.py` (`WRITE_TOOLS="create_issue,send_message"`), the system will automatically protect them with HITL without modifying any external code.

## 3. Configuration-Driven Agent Persona

The entire agent persona is abstracted into `backend/config.py`.
By altering `ASSISTANT_NAME`, `ASSISTANT_ROLE_DESCRIPTION`, and `WRITE_TOOLS`, this architecture can instantly pivot from a Finance Assistant to an IT Helpdesk or a Marketing Strategist, requiring zero code changes in `main.py` or the React frontend.
