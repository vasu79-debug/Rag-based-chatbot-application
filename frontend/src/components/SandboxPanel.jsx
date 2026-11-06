import React, { useEffect, useRef, useState } from "react";
import { getSandboxWsUrl, killAgent } from "../api";

/**
 * The "sandbox window": a persistent live view of the agent's headless
 * Playwright browser, fed over a WebSocket (/ws/sandbox/{session_id}) as a
 * rolling sequence of JPEG screenshots - the same idea as ChatGPT's agent
 * mode browser pane, just polling-based instead of a CDP screencast for now.
 */
export default function SandboxPanel({ sessionId, open, onToggle }) {
  const [frame, setFrame] = useState(null);
  const [pageUrl, setPageUrl] = useState("");
  const [connected, setConnected] = useState(false);
  const [stopping, setStopping] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!open || !sessionId) {
      setFrame(null);
      setConnected(false);
      return undefined;
    }

    let cancelled = false;
    const ws = new WebSocket(getSandboxWsUrl(sessionId));
    wsRef.current = ws;

    ws.onopen = () => {
      if (!cancelled) setConnected(true);
    };
    ws.onclose = () => {
      if (!cancelled) setConnected(false);
    };
    ws.onerror = () => {
      if (!cancelled) setConnected(false);
    };
    ws.onmessage = (evt) => {
      if (cancelled) return;
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "frame") {
          if (data.image) setFrame(data.image);
          if (typeof data.url === "string") setPageUrl(data.url);
          if (!data.active) setFrame(null);
        }
      } catch (e) {
        // ignore malformed frames
      }
    };

    return () => {
      cancelled = true;
      ws.close();
      wsRef.current = null;
    };
  }, [open, sessionId]);

  const handleStop = async () => {
    setStopping(true);
    try {
      await killAgent(sessionId);
      setFrame(null);
      setPageUrl("");
    } catch (e) {
      alert("Error stopping sandbox: " + e.message);
    } finally {
      setStopping(false);
    }
  };

  if (!open) return null;

  const isLive = connected && !!frame;

  return (
    <div className="sandbox-panel">
      <div className="sandbox-topbar">
        <span className={`sandbox-dot ${isLive ? "live" : "idle"}`} title={isLive ? "Live" : "Idle"} />
        <div className="sandbox-url" title={pageUrl || ""}>
          {pageUrl || "Sandbox idle — ask the agent to do something on the web"}
        </div>
        <button
          className="sandbox-stop-btn"
          onClick={handleStop}
          disabled={stopping}
          title="Stop the agent and close this sandbox"
        >
          ⏹ Stop
        </button>
        {onToggle && (
          <button className="sandbox-close-btn" onClick={onToggle} title="Hide sandbox panel">
            ✕
          </button>
        )}
      </div>
      <div className="sandbox-viewport">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Live sandbox view"
            className="sandbox-frame"
          />
        ) : (
          <div className="sandbox-placeholder">
            <div className="sandbox-placeholder-icon">🖥️</div>
            <p>No browser session yet.</p>
            <p className="sandbox-placeholder-sub">
              When the agent uses the browser tools, its sandboxed Chromium
              session will stream live here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
