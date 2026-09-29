/**
 * API Client for Demo 4 (Hybrid Knowledge AI).
 * Connects to the standalone FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

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

export async function fetchChatHistory(sessionId) {
  const res = await fetch(`${BASE_URL}/chat/history/${sessionId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history for session ${sessionId}`);
  }
  return res.json();
}
