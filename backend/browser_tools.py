import asyncio
import os
import logging
from datetime import datetime
from urllib.parse import urlparse
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Page, Browser, Playwright

logger = logging.getLogger(__name__)

# Global Playwright state
_playwright: Playwright = None
_browser: Browser = None
_page: Page = None


AUDIT_LOG_FILE = "audit.log"
EVIDENCE_DIR = "evidence"

os.makedirs(EVIDENCE_DIR, exist_ok=True)

async def init_browser():
    """Initialize Playwright in headful mode so the user can interact."""
    global _playwright, _browser, _page
    if _playwright is None:
        _playwright = await async_playwright().start()
        # Headless mode (headless=True) so it runs entirely in the background
        _browser = await _playwright.chromium.launch(headless=False)
        # Create a new browser context with a reasonably sized viewport
        context = await _browser.new_context(viewport={'width': 1280, 'height': 800})
        _page = await context.new_page()
        logger.info("Playwright browser initialized in headful mode.")

async def close_browser():
    """Close the Playwright browser cleanly."""
    global _playwright, _browser, _page
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    _playwright = None
    _browser = None
    _page = None
    logger.info("Playwright browser closed.")

def log_audit(action: str, details: str):
    """Write all important actions to the audit log."""
    timestamp = datetime.now().isoformat()
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ACTION: {action} | DETAILS: {details}\n")
    logger.info(f"Audit Log: {action} - {details}")

import base64

async def take_screenshot(name: str) -> str:
    """Take a screenshot for evidence retention."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EVIDENCE_DIR}/{name}_{timestamp}.png"
    if _page and not _page.is_closed():
        await _page.screenshot(path=filename, full_page=False)
    return filename

async def get_latest_screenshot_base64() -> str:
    """Take a fast, low-quality screenshot and return as base64 string for the UI."""
    if _page and not _page.is_closed():
        try:
            image_bytes = await _page.screenshot(type='jpeg', quality=40)
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None
    return None

@tool
async def navigate(url: str) -> str:
    """
    Navigate to a specified URL.
    Use this tool to open websites.
    """
    if not _page or _page.is_closed():
        await init_browser()
        
    log_audit("NAVIGATE", f"URL: {url}")
    await _page.goto(url)
    await _page.wait_for_load_state('domcontentloaded')
    return f"Successfully navigated to {url}"

@tool
async def click(selector: str) -> str:
    """
    Click on an element in the browser. 
    Since you only see the text of the page, you MUST use Playwright text selectors to click things!
    Example: selector="text='Log In'" or selector="text='Submit'"
    Use this to click buttons, links, or open menus.
    """
    if not _page or _page.is_closed():
        return "Error: Browser not initialized or was closed. Navigate to a page first."
    
    log_audit("CLICK", f"Selector: {selector}")
    # Pre-action evidence
    await take_screenshot(f"pre_click_{selector.replace(' ', '_')}")
    
    try:
        await _page.click(selector, timeout=5000)
        # Post-action evidence
        await take_screenshot(f"post_click_{selector.replace(' ', '_')}")
        return f"Successfully clicked on {selector}"
    except Exception as e:
        return f"Failed to click on {selector}: {str(e)}"

@tool
async def type_text(selector: str, text: str) -> str:
    """
    Type text into an input field in the browser.
    You MUST use Playwright text selectors, name attributes, or placeholders.
    Example: selector="input[name='email']" or selector="input[placeholder='Search']"
    DO NOT use this tool for typing passwords! The user will take over for passwords.
    """
    if not _page or _page.is_closed():
        return "Error: Browser not initialized or was closed. Navigate to a page first."
    
    # Security: Do not log the exact text to avoid leaking PII in audit
    log_audit("TYPE", f"Selector: {selector}, Text Length: {len(text)}")
    await take_screenshot(f"pre_type_{selector.replace(' ', '_')}")
    
    try:
        await _page.fill(selector, text, timeout=5000)
        await take_screenshot(f"post_type_{selector.replace(' ', '_')}")
        return f"Successfully typed text into {selector}"
    except Exception as e:
        return f"Failed to type in {selector}: {str(e)}"

@tool
async def get_page_content() -> str:
    """
    Extract the text content of the current page.
    Use this to read what is currently visible on the screen.
    """
    if not _page or _page.is_closed():
        return "Error: Browser not initialized or was closed."
    
    log_audit("EXTRACT", "Extracted page content")
    try:
        # Extract innerText PLUS a summary of interactive elements
        js_script = """
        () => {
            let text = document.body.innerText;
            let interactives = '\\n\\n--- INTERACTIVE ELEMENTS ---\\n';
            document.querySelectorAll('input, button, a').forEach(el => {
                let name = el.innerText || el.placeholder || el.name || el.value || 'unnamed';
                name = name.trim().replace(/\\n/g, ' ').substring(0, 40);
                if(name) {
                    let type = el.tagName.toLowerCase();
                    let sel = el.id ? `#${el.id}` : (el.name ? `[name='${el.name}']` : `text='${name}'`);
                    interactives += `[${type}] "${name}" -> Use Selector: ${sel}\\n`;
                }
            });
            return text + interactives;
        }
        """
        content = await _page.evaluate(js_script)
        return content[:10000] # Limit size to avoid context overflow
    except Exception as e:
        return f"Failed to extract content: {str(e)}"

# A list of tools to export
BROWSER_TOOLS = [navigate, click, type_text, get_page_content]
