import React from "react";

export default function CitationsDrawer({ citation, onClose }) {
  if (!citation) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <h2 className="drawer-title">📑 Source Citation</h2>
          <button className="btn-close" onClick={onClose}>&times;</button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Document Source</span>
            <p style={{ fontSize: "15px", fontWeight: "700", color: "#fff" }}>{citation.source}</p>
          </div>

          <div style={{ display: "flex", gap: "20px" }}>
            <div>
              <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Page / Section</span>
              <p style={{ fontSize: "14px", color: "var(--org-emerald)", fontWeight: "600" }}>Page {citation.page}</p>
            </div>
            <div>
              <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>File Format</span>
              <p style={{ fontSize: "14px", color: "var(--krify-blue)", fontWeight: "600", textTransform: "uppercase" }}>{citation.format}</p>
            </div>
            <div>
              <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700" }}>Rerank Relevance</span>
              <p style={{ fontSize: "14px", color: "#fbbf24", fontWeight: "600" }}>{Math.round(citation.score * 100)}%</p>
            </div>
          </div>
        </div>

        <div>
          <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: "700", display: "block", marginBottom: "6px" }}>Extracted Context Passage</span>
          <div className="snippet-box">
            {citation.snippet}
          </div>
        </div>
      </div>
    </div>
  );
}
