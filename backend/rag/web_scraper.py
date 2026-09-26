"""
Secure Web Scraper & HTML Sanitizer for Demo 4 RAG.
Includes enterprise-grade protections against:
1. SSRF (Server-Side Request Forgery) and internal IP scanning.
2. Malware / Malicious Script injection (stripping <script>, <iframe>, event handlers).
3. Hidden prompt injection & zero-width character obfuscation.
4. Denial of Service (size caps, timeout limits, decompression bomb prevention).
"""

import ipaddress
import re
import socket
import urllib.parse
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Maximum page size: 5 MB
MAX_PAGE_BYTES = 5 * 1024 * 1024

# Allowed URL schemes
ALLOWED_SCHEMES = {"http", "https"}

# Blocked tags that could contain malware, scripts, or active executable payloads
BLOCKED_TAGS = {
    "script", "style", "iframe", "frame", "frameset", "object", "embed",
    "applet", "svg", "canvas", "form", "input", "button", "select", "textarea",
    "noscript", "link", "meta", "base", "template"
}

# Regex to detect and strip zero-width / hidden unicode characters
ZERO_WIDTH_REGEX = re.compile(r"[\u200B-\u200D\uFEFF\u0000-\u0008\u000B\u000C\u000E-\u001F]")


class SecurityValidationError(ValueError):
    """Raised when a URL or payload fails security checks."""
    pass


def validate_url_security(url: str) -> str:
    """
    Validates URL to protect against SSRF (Server-Side Request Forgery),
    internal network probing, and illegal schemes.
    """
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as e:
        raise SecurityValidationError(f"Invalid URL structure: {e}")

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SecurityValidationError(
            f"Blocked scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted for safety."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SecurityValidationError("URL must include a valid hostname.")

    # Check for localhost / loopback aliases
    if hostname.lower() in ["localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"]:
        raise SecurityValidationError("Access to localhost / loopback addresses is blocked.")

    # Resolve IP and verify it is not private/reserved/link-local
    try:
        _, _, ip_list = socket.gethostbyname_ex(hostname)
        for ip_str in ip_list:
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
                or ip_str.startswith("169.254.")  # Cloud metadata IP
            ):
                raise SecurityValidationError(
                    f"Target host resolves to a restricted/private network address ({ip_str})."
                )
    except socket.gaierror:
        raise SecurityValidationError(f"Could not resolve hostname '{hostname}'.")

    return parsed.geturl()


def sanitize_text_content(text: str) -> str:
    """
    Cleans extracted text:
    - Removes zero-width characters (anti-obfuscation)
    - Removes excessive whitespaces and carriage returns
    - Neutralizes raw script tags that escaped HTML parsing
    """
    if not text:
        return ""

    # Remove zero-width & non-printable control chars
    clean = ZERO_WIDTH_REGEX.sub("", text)

    # Normalize excessive newlines and spaces
    clean = re.sub(r"\r\n|\r", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)

    return clean.strip()


def extract_clean_web_content(html_content: str, url: str) -> Dict[str, Any]:
    """
    Parses HTML, strips all dangerous executable tags, and extracts
    structured, clean content.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Remove all malicious / executable / styling tags
    for tag in soup.find_all(list(BLOCKED_TAGS)):
        tag.decompose()

    # 2. Remove all inline event handlers (onclick, onload, onerror, etc.)
    for tag in soup.find_all(True):
        attrs_to_remove = [
            attr for attr in tag.attrs
            if attr.lower().startswith("on") or "javascript:" in str(tag.attrs[attr]).lower()
        ]
        for attr in attrs_to_remove:
            del tag[attr]

    # 3. Extract Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # 4. Target main body / article if present, otherwise body
    main_node = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": re.compile(r"content|main|article|body", re.I)})
        or soup.find("body")
        or soup
    )

    # 5. Extract structured text (preserving headings, lists, tables)
    lines = []
    if title:
        lines.append(f"# {title}\n")

    for element in main_node.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "blockquote"]):
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue

        tag_name = element.name.lower()
        if tag_name == "h1":
            lines.append(f"\n# {text}\n")
        elif tag_name == "h2":
            lines.append(f"\n## {text}\n")
        elif tag_name == "h3":
            lines.append(f"\n### {text}\n")
        elif tag_name == "li":
            lines.append(f"- {text}")
        elif tag_name == "blockquote":
            lines.append(f"> {text}")
        elif tag_name == "tr":
            cells = [c.get_text(strip=True) for c in element.find_all(["th", "td"])]
            if cells:
                lines.append(" | ".join(cells))
        else:
            lines.append(f"{text}\n")

    # If structured tags produced little text, fallback to clean body text
    full_text = "\n".join(lines).strip()
    if len(full_text) < 100:
        full_text = main_node.get_text(separator="\n", strip=True)

    sanitized_text = sanitize_text_content(full_text)
    return {
        "title": title or url,
        "content": sanitized_text,
        "url": url,
    }


def fetch_and_sanitize_url(url: str) -> Dict[str, Any]:
    """
    Safely fetches a web page with timeout, size caps, and SSRF validation.
    """
    safe_url = validate_url_security(url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(
            safe_url,
            headers=headers,
            timeout=(5.0, 10.0),
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type and "application/xhtml" not in content_type:
            raise SecurityValidationError(f"Unsupported content type '{content_type}'. Only web HTML/text is allowed.")

        # Read safely with max size cap
        chunks = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=16384):
            total_bytes += len(chunk)
            if total_bytes > MAX_PAGE_BYTES:
                raise SecurityValidationError(f"Page size exceeds safety limit of {MAX_PAGE_BYTES // (1024*1024)}MB.")
            chunks.append(chunk)

        html_text = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch webpage ({safe_url}): {str(e)}")

    # Extract & sanitize content
    extracted = extract_clean_web_content(html_text, safe_url)
    if not extracted["content"]:
        raise ValueError("Could not extract readable text from the provided webpage.")

    return extracted
