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

export async function sendChatMessage(question, history = []) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, history }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed with status ${res.status}`);
  }

  return res.json();
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

