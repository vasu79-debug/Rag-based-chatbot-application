import os
import json
import logging
from fastmcp import FastMCP
from github import Github

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("GitHub Server")

# Initialize PyGithub client using the user's PAT
# Using the token provided by the user (partially masked token shouldn't be hardcoded if they provided the full one in env)
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "github_pat_11CMOD35Y0X4xxxew_GL2DDJnUndrelRZc3cc3kKY3IlBrmFrt8gTEhg4mxxxQbEYy3")
g = Github(GITHUB_TOKEN)

@mcp.tool()
def create_github_issue(repo_name: str, title: str, body: str) -> str:
    """Create a new issue in a GitHub repository.
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
