/**
 * API Client for Demo 4 (Hybrid Knowledge AI).
 * Connects to the standalone FastAPI backend.
 */

const BASE_PATH = import.meta.env.BASE_URL === "/" ? "" : import.meta.env.BASE_URL.replace(/\/$/, "");
const BASE_URL = import.meta.env.VITE_API_URL || `${BASE_PATH}/api`;

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Backend service unavailable");
  return res.json();
}

export async function sendChatMessage(question, session_id = "default_session") {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, session_id }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}

export async function streamChatMessage(question, session_id = "default_session", onStage) {
  try {
    const res = await fetch(`${BASE_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, session_id }),
    });

    if (!res.ok || !res.body) {
      return sendChatMessage(question, session_id);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          try {
            const payload = JSON.parse(trimmed.slice(6));
            if (payload.type === "stage" && onStage) {
              onStage(payload);
            } else if (payload.type === "approval_needed") {
              if (onStage) onStage({ stage: "paused", label: "Agent paused for approval." });
              finalResult = payload;
            } else if (payload.type === "complete") {
              finalResult = payload.result;
            } else if (payload.type === "error") {
              throw new Error(payload.error || "Processing failed");
            }
          } catch (e) {
            if (e.message !== "Unexpected end of JSON input") {
              console.warn("SSE parse error:", e);
            }
          }
        }
      }
    }

    if (finalResult) {
      return finalResult;
    }
    return sendChatMessage(question, session_id);
  } catch (err) {
    return sendChatMessage(question, session_id);
  }
}

export async function fetchDocuments() {
  const res = await fetch(`${BASE_URL}/admin/documents`);
  if (!res.ok) throw new Error("Failed to fetch indexed documents");
  return res.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/admin/documents`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload and index document");
  }

  return res.json();
}

export async function ingestUrl(url) {
  const res = await fetch(`${BASE_URL}/admin/url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to index website URL");
  }

  return res.json();
}

export async function deleteDocument(docId) {
  const res = await fetch(`${BASE_URL}/admin/documents/${docId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to delete document");
  }

  return res.json();
}

export async function getMcpSettings() {
  const res = await fetch(`${BASE_URL}/settings/mcp`);
  if (!res.ok) throw new Error("Failed to get MCP settings");
  return res.json();
}

export async function updateMcpSettings(url) {
  const res = await fetch(`${BASE_URL}/settings/mcp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to update MCP settings");
  }
  return res.json();
}

export async function disconnectMcpSettings(url) {
  const res = await fetch(`${BASE_URL}/settings/mcp`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to disconnect MCP settings");
  }
  return res.json();
}

export async function fetchChatHistory(sessionId) {
  const res = await fetch(`${BASE_URL}/chat/history/${sessionId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history for session ${sessionId}`);
  }
  return res.json();
}

export async function resumeChat(sessionId, action, onChunk) {
  const res = await fetch(`${BASE_URL}/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, action: action }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to resume chat");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");

    for (let line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.replace("data: ", ""));
          onChunk(data);
        } catch (e) {
          console.error("Error parsing resume SSE chunk:", e);
        }
      }
    }
  }
}
