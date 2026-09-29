import React from "react";

export default function Sidebar({ sessions, activeSessionId, onSelectSession, onNewSession, onDeleteSession }) {
  return (
    <div className="sidebar">
      <button className="new-chat-btn sidebar-btn" onClick={onNewSession}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 5v14M5 12h14"></path>
        </svg>
        New Chat
      </button>
      
      <div className="sessions-list">
        <div className="sessions-header">Recent Chats</div>
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === activeSessionId ? "active" : ""}`}
            onClick={() => onSelectSession(session.id)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <span className="session-title">{session.title}</span>
            <button 
              className="delete-session-btn" 
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(session.id);
              }}
              title="Delete Chat"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
