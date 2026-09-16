import React from "react";

export default function Header({ activeTab, setActiveTab, healthData }) {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">⚡</div>
        <div>
          <h1 className="brand-title">
            Hybrid Knowledge AI
            <span className="brand-badge">Demo 4</span>
          </h1>
        </div>
      </div>

      <div className="header-actions">
        <button
          onClick={() => setActiveTab("chat")}
          className={`tab-btn ${activeTab === "chat" ? "active" : ""}`}
        >
          💬 Hybrid Assistant
        </button>

        <button
          onClick={() => setActiveTab("admin")}
          className={`tab-btn ${activeTab === "admin" ? "active" : ""}`}
        >
          📂 Admin Knowledge Base ({healthData?.indexed_documents_count || 0})
        </button>

        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>{healthData?.provider?.toUpperCase() || "AI"} Online</span>
        </div>
      </div>
    </header>
  );
}
