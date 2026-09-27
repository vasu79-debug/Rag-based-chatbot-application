import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatView({ messages, onSendMessage, loading, currentStage, onSelectCitation }) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, currentStage]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const text = input.trim();
    setInput("");
    onSendMessage(text);
  };

  return (
    <div className="chat-wrapper">
      <div className="messages-list">
        {messages.length === 0 ? (
          <div className="welcome-hero">
            <h2 className="hero-title">Ask anything about Krify</h2>
          </div>
        ) : (
          messages.map((msg, index) => {
            const isUser = msg.role === "user";

            if (isUser) {
              return (
                <div key={index} className="chat-row user">
                  <div className="user-bubble">{msg.content}</div>
                  <div className="avatar user">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                      <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                  </div>
                </div>
              );
            }

            // Assistant Response
            const payload = msg.payload || {};
            const content = payload.content || payload.answer || payload.org_section?.content || payload.general_section?.content || msg.content || "";
            const citations = payload.citations || payload.org_section?.citations || [];

            return (
              <div key={index} className="chat-row bot">
                <div className="avatar bot">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                  </svg>
                </div>
                <div className="bot-response-container">
                  <div className="knowledge-card unified">
                    {citations && citations.length > 0 && (
                      <div className="card-header unified">
                        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                          </svg>
                          Krify Knowledge Base
                        </span>
                        <span style={{ fontSize: "11.5px", color: "var(--org-emerald)" }}>
                          ✓ Verified Sources ({citations.length})
                        </span>
                      </div>
                    )}
                    <div className="card-body markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {content}
                      </ReactMarkdown>
                    </div>

                    {/* Interactive Citation Pills */}
                    {citations && citations.length > 0 && (
                      <div className="citations-bar">
                        <span style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "700" }}>SOURCES:</span>
                        {citations.map((cit, cIdx) => (
                          <button
                            key={cIdx}
                            className="citation-pill"
                            onClick={() => onSelectCitation(cit)}
                            title="Click to view reference passage"
                          >
                            📄 {cit.source} {cit.page > 0 ? `· P.${cit.page}` : ""}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}

        {loading && (
          <div className="chatgpt-status-row">
            <span className="chatgpt-status-icon">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
              </svg>
            </span>
            <span className="chatgpt-status-text">
              {currentStage?.label || "Searching Krify knowledge base..."}
            </span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <form onSubmit={handleSubmit} className="input-box-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Ask a question..."
            className="chat-input"
            rows={1}
          />
          <button type="submit" disabled={!input.trim() || loading} className="btn-send">
            Send ➔
          </button>
        </form>
      </div>
    </div>
  );
}

