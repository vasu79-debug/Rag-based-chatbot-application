import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatView({ messages, onSendMessage, loading, onSelectCitation }) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const text = input.trim();
    setInput("");
    onSendMessage(text);
  };

  const samplePrompts = [
    {
      type: "hyb",
      tag: "Hybrid Dual-Stream",
      text: "Compare our annual leave carryover policy with standard UK statutory law and draft an email requesting 4 days.",
    },
    {
      type: "org",
      tag: "Organisational RAG",
      text: "What is the maximum allowed annual leave carryover into the next year?",
    },
    {
      type: "gen",
      tag: "General Intelligence",
      text: "Write a high-performance Python script to parse CSV files and compute aggregates.",
    },
  ];

  return (
    <div className="chat-wrapper">
      <div className="messages-list">
        {messages.length === 0 ? (
          <div className="welcome-hero">
            <div className="hero-icon">⚡</div>
            <h2 className="hero-title">Hybrid Knowledge AI Assistant</h2>
            <p className="hero-desc">
              Experience the power of dual-knowledge intelligence. One unified response constructed from your <strong>private organizational documents</strong> and <strong>general world intelligence</strong>, clearly labeled.
            </p>

            <div className="example-grid">
              {samplePrompts.map((sample, idx) => (
                <div
                  key={idx}
                  className="example-card"
                  onClick={() => onSendMessage(sample.text)}
                >
                  <span className={`example-tag ${sample.type}`}>{sample.tag}</span>
                  <div className="example-text">"{sample.text}"</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, index) => {
            const isUser = msg.role === "user";

            if (isUser) {
              return (
                <div key={index} className="chat-row user">
                  <div className="user-bubble">{msg.content}</div>
                  <div className="avatar user">👤</div>
                </div>
              );
            }

            // Assistant Bot Dual Response
            const payload = msg.payload || {};
            const orgSec = payload.org_section;
            const genSec = payload.general_section;

            return (
              <div key={index} className="chat-row bot">
                <div className="avatar bot">🤖</div>
                <div className="bot-response-container">
                  {/* 1. Organisational Knowledge Card (if present) */}
                  {orgSec && (
                    <div className="knowledge-card org">
                      <div className="card-header">
                        <span>{orgSec.label}</span>
                        {orgSec.citations?.length > 0 && (
                          <span style={{ fontSize: "11.5px", color: "var(--org-emerald)" }}>
                            ✓ Verified Sources ({orgSec.citations.length})
                          </span>
                        )}
                      </div>
                      <div className="card-body markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {orgSec.content}
                        </ReactMarkdown>
                      </div>

                      {/* Interactive Citation Pills */}
                      {orgSec.citations && orgSec.citations.length > 0 && (
                        <div className="citations-bar">
                          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "700" }}>CITATIONS:</span>
                          {orgSec.citations.map((cit, cIdx) => (
                            <button
                              key={cIdx}
                              className="citation-pill"
                              onClick={() => onSelectCitation(cit)}
                              title="Click to inspect exact passage and score"
                            >
                              📄 {cit.source} · P.{cit.page} ({Math.round(cit.score * 100)}%)
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* 2. General Knowledge & AI Intelligence Card (if present) */}
                  {genSec && (
                    <div className="knowledge-card general">
                      <div className="card-header">
                        <span>{genSec.label}</span>
                        <span style={{ fontSize: "11.5px", color: "var(--gen-purple)" }}>
                          ⚡ World Knowledge / Actions
                        </span>
                      </div>
                      <div className="card-body markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {genSec.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}

        {loading && (
          <div className="chat-row bot">
            <div className="avatar bot">🤖</div>
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
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
            placeholder="Ask a question (e.g. Compare our leave with UK law and draft an email)..."
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
