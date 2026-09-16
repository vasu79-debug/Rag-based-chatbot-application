import React, { useState, useRef } from "react";

export default function AdminPortal({ documents, onUpload, onDelete, uploading }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);
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

  return (
    <div className="admin-container">
      {/* 1. Drag and Drop Uploader */}
      <div
        className={`upload-dropzone ${dragActive ? "dragging" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.doc,.csv,.txt,.md"
          onChange={handleChange}
          style={{ display: "none" }}
        />
        <div className="dropzone-icon">📤</div>
        <h3 className="dropzone-title">
          {uploading ? "Extracting, Chunking & Indexing Document..." : "Click or Drag & Drop Documents to Index"}
        </h3>
        <p className="dropzone-sub">
          Supports PDF, Word (.docx), CSV, and Text/Markdown. Automatically vectorizes with ChromaDB + BM25.
        </p>

        {uploadError && (
          <p style={{ color: "var(--danger-red)", marginTop: "12px", fontSize: "13px", fontWeight: "600" }}>
            ⚠️ {uploadError}
          </p>
        )}
      </div>

      {/* 2. Document Library Table */}
      <div className="docs-table-card">
        <div className="table-header-bar">
          <h3 className="table-title">📚 Indexed Knowledge Corpus ({documents.length} Files)</h3>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            Total Chunks: {documents.reduce((acc, d) => acc + d.chunks_count, 0)}
          </span>
        </div>

        {documents.length === 0 ? (
          <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)", fontSize: "14px" }}>
            No documents indexed yet. Upload PDF/DOCX files above to empower the hybrid assistant!
          </div>
        ) : (
          <table className="docs-table">
            <thead>
              <tr>
                <th>Document Name</th>
                <th>Format</th>
                <th>Chunks in ChromaDB</th>
                <th>Pages</th>
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
                      padding: "2px 6px",
                      borderRadius: "4px",
                      background: "rgba(0, 164, 228, 0.15)",
                      color: "var(--krify-blue)"
                    }}>
                      {doc.format}
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
