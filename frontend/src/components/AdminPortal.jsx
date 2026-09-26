import React, { useState, useRef } from "react";

export default function AdminPortal({ documents, onUpload, onIngestUrl, onDelete, uploading }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [urlInput, setUrlInput] = useState("");
  const [urlStatus, setUrlStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setUploadError(null);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = async (e) => {
    e.preventDefault();
    setUploadError(null);
    if (e.target.files && e.target.files[0]) {
      await processFile(e.target.files[0]);
    }
  };

  const processFile = async (file) => {
    try {
      await onUpload(file);
    } catch (err) {
      setUploadError(err.message || "Failed to upload document");
    }
  };

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!urlInput.trim() || uploading) return;

    setUrlStatus(null);
    setUploadError(null);

    const targetUrl = urlInput.trim();
    if (!targetUrl.startsWith("http://") && !targetUrl.startsWith("https://")) {
      setUploadError("Please provide a valid URL starting with http:// or https://");
      return;
    }

    try {
      await onIngestUrl(targetUrl);
      setUrlStatus(`✓ Successfully indexed content from ${targetUrl}`);
      setUrlInput("");
    } catch (err) {
      setUploadError(err.message || "Failed to index website URL");
    }
  };

  return (
    <div className="admin-container">
      {/* 1. Ingestion Grid: Documents + Website URL */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "24px" }}>
        
        {/* Document Upload Box */}
        <div
          className={`upload-dropzone ${dragActive ? "dragging" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{ minHeight: "180px" }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc,.csv,.txt,.md"
            onChange={handleChange}
            style={{ display: "none" }}
          />
          <div className="dropzone-icon">📤</div>
          <h3 className="dropzone-title" style={{ fontSize: "15px" }}>
            {uploading ? "Indexing Document..." : "Upload Local Files"}
          </h3>
          <p className="dropzone-sub" style={{ fontSize: "12px" }}>
            PDF, Word (.docx), CSV, TXT, Markdown
          </p>
        </div>

        {/* Website / URL Ingestion Box */}
        <div className="upload-dropzone" style={{ minHeight: "180px", cursor: "default", borderStyle: "solid" }}>
          <div className="dropzone-icon">🌐</div>
          <h3 className="dropzone-title" style={{ fontSize: "15px" }}>
            Connect Website / Webpage
          </h3>
          <p className="dropzone-sub" style={{ fontSize: "12px", marginBottom: "12px" }}>
            Extracts & indexes clean text with <strong>SSRF, Malware & Script Protection</strong>
          </p>

          <form onSubmit={handleUrlSubmit} style={{ width: "100%", display: "flex", gap: "8px" }}>
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://example.com/about-us"
              disabled={uploading}
              style={{
                flex: 1,
                padding: "8px 12px",
                borderRadius: "8px",
                background: "rgba(255, 255, 255, 0.06)",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                color: "#fff",
                fontSize: "13px",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={!urlInput.trim() || uploading}
              className="tab-btn active"
              style={{ padding: "8px 14px", fontSize: "13px", cursor: "pointer", whiteSpace: "nowrap" }}
            >
              {uploading ? "Fetching..." : "Index URL ➔"}
            </button>
          </form>

          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "10px", fontSize: "11px", color: "var(--org-emerald)" }}>
            <span>🛡️ Anti-Malware & Script Stripping Active</span>
          </div>
        </div>

      </div>

      {uploadError && (
        <div style={{
          padding: "12px 16px",
          borderRadius: "8px",
          background: "rgba(239, 68, 68, 0.12)",
          border: "1px solid rgba(239, 68, 68, 0.3)",
          color: "#f87171",
          fontSize: "13px",
          marginBottom: "20px",
          fontWeight: "600"
        }}>
          ⚠️ {uploadError}
        </div>
      )}

      {urlStatus && (
        <div style={{
          padding: "12px 16px",
          borderRadius: "8px",
          background: "rgba(16, 185, 129, 0.12)",
          border: "1px solid rgba(16, 185, 129, 0.3)",
          color: "#34d399",
          fontSize: "13px",
          marginBottom: "20px",
          fontWeight: "600"
        }}>
          {urlStatus}
        </div>
      )}

      {/* 2. Document Library Table */}
      <div className="docs-table-card">
        <div className="table-header-bar">
          <h3 className="table-title">📚 Indexed Knowledge Corpus ({documents.length} Sources)</h3>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            Total Chunks: {documents.reduce((acc, d) => acc + d.chunks_count, 0)}
          </span>
        </div>

        {documents.length === 0 ? (
          <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)", fontSize: "14px" }}>
            No documents or websites indexed yet. Upload files or enter a website URL above to empower the hybrid assistant!
          </div>
        ) : (
          <table className="docs-table">
            <thead>
              <tr>
                <th>Knowledge Source Name</th>
                <th>Type</th>
                <th>Chunks in ChromaDB</th>
                <th>Pages / Sections</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.doc_id}>
                  <td style={{ fontWeight: "600" }}>{doc.filename}</td>
                  <td>
                    <span style={{
                      textTransform: "uppercase",
                      fontSize: "11px",
                      fontWeight: "700",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      background: doc.format === "web" ? "rgba(168, 85, 247, 0.15)" : "rgba(0, 164, 228, 0.15)",
                      color: doc.format === "web" ? "var(--gen-purple)" : "var(--krify-blue)"
                    }}>
                      {doc.format === "web" ? "🌐 Website" : doc.format}
                    </span>
                  </td>
                  <td>{doc.chunks_count} chunks</td>
                  <td>{doc.pages_count} pages</td>
                  <td>
                    <button
                      onClick={() => onDelete(doc.doc_id)}
                      className="btn-delete"
                      title="Purge vectors from ChromaDB"
                    >
                      Delete 🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

