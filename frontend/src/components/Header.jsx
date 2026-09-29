import React from "react";

export default function Header({ activeTab, onNavigate, onNewChat, healthData }) {
  const isAdmin = activeTab === "admin";

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">
          <img
            src="/subzillo_logo.jpg"
            alt="Subzillo"
            style={{ width: "24px", height: "24px", objectFit: "contain", borderRadius: "4px" }}
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>
        <div>
          <h1 className="brand-title">
            {isAdmin ? (
              "Subscription Administration"
            ) : (
              <>
                Subscription Assistant
                <span className="brand-subtitle">
                  · Intelligent Subscription Tracking & Optimization
                </span>
              </>
            )}
          </h1>
        </div>
      </div>

      <div className="header-actions">
        {isAdmin ? (
          <button
            onClick={() => onNavigate("/")}
            className="tab-btn back-btn"
          >
            ← Back to Assistant
          </button>
        ) : (
          <button
            onClick={onNewChat}
            className="tab-btn new-chat-btn"
            title="Start a fresh conversation"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
            </svg>
            New Chat
          </button>
        )}

        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>{isAdmin ? "Admin Console" : "Connected"}</span>
        </div>
      </div>
    </header>
  );
}

