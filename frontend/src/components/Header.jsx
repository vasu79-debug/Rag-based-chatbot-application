import React, { useState } from "react";
import { killAgent } from "../api";
import { applyTheme, getInitialTheme } from "../theme";

export default function Header({ activeTab, onNavigate, onNewChat, healthData, sessionId }) {
  const isAdmin = activeTab === "admin";
  const [theme, setTheme] = useState(getInitialTheme);
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  };

  const handleKill = async () => {
    try {
      await killAgent(sessionId);
      alert("Emergency Stop Triggered: Browser closed and agent halted.");
    } catch (e) {
      alert("Error triggering Emergency Stop: " + e.message);
    }
  };

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/>
          </svg>
        </div>
        <div>
          <h1 className="brand-title">
            {isAdmin ? (
              "Travel Administration"
            ) : (
              <>
                Travel & Booking Assistant
                <span className="brand-subtitle">
                  · Intelligent Trip Planning & Autonomous Booking
                </span>
              </>
            )}
          </h1>
        </div>
      </div>

      <div className="header-actions">
        {!isAdmin && (
          <button
            onClick={handleKill}
            className="tab-btn btn-danger"
            title="Immediately stop the agent and close browser"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2" /></svg>
            <span className="btn-label">Emergency Stop</span>
          </button>
        )}

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
              <path d="M12 5v14M5 12h14"></path>
            </svg>
            <span className="btn-label">New Chat</span>
          </button>
        )}

        <button
          onClick={toggleTheme}
          className="tab-btn theme-toggle"
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>

        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>{isAdmin ? "Admin Console" : "Connected"}</span>
        </div>
      </div>
    </header>
  );
}

