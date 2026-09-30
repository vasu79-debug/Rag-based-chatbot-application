import json
import logging
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Live Data Server", host="127.0.0.1", port=8003)

@mcp.tool()
def search_web(query: str) -> str:
    """Use this tool to search the live internet for any information, like live cricket scores, news, or weather."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return json.dumps({"status": "error", "message": f"No public data found for '{query}'."})
                
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("href", "")
                })
                
            return json.dumps({
                "status": "success", 
                "query": query,
                "search_results": formatted_results
            })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to search the web: {str(e)}"})

if __name__ == "__main__":
    print("Starting Live Data MCP Server on http://127.0.0.1:8003/sse")
    mcp.run(transport="sse")
