import os
import json
import logging
from fastmcp import FastMCP
from github import Github

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("GitHub Server")

# Initialize PyGithub client using environment variable only
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    logger.critical("CRITICAL: GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set. Refusing to start server.")
    raise ValueError(
        "CRITICAL SECURITY CONFIGURATION: 'GITHUB_PERSONAL_ACCESS_TOKEN' environment variable is required. "
        "Do not hardcode tokens in the codebase. Please set it in your .env file or environment."
    )

g = Github(GITHUB_TOKEN)

@mcp.tool()
def create_github_issue(repo_name: str, title: str, body: str) -> str:
    """Create a new issue in a GitHub repository.
    [capability: github] [operation: write]
    repo_name MUST be in the exact format 'owner/repo' (e.g. 'vasu79-debug/Rag-based-chatbot-application').
    Use this tool when the user asks to report a bug or create an issue.
    """
    try:
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        return json.dumps({
            "status": "success", 
            "message": f"Issue created successfully at {issue.html_url}",
            "issue_number": issue.number
        })
    except Exception as e:
        logger.error(f"Failed to create GitHub issue: {e}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    logger.info("Starting GitHub MCP Server on http://127.0.0.1:8003/sse")
    mcp.run(transport="sse", host="127.0.0.1", port=8003)
