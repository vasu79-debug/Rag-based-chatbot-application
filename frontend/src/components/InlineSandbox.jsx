import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  getSandboxWsUrl,
  killAgent,
  sendSandboxClick,
  sendSandboxKey,
  sendSandboxScroll,
  resumeSandboxControl,
} from "../api";

/**
 * The ChatGPT-agent-style "sandbox window" - inline inside the chat area.
 * It only opens while the agent is actively working on a turn (`active`),
 * connects to the live /ws/sandbox/{session_id} feed, and renders nothing at
 * all until the agent actually opens a browser page.
 *
 * When the backend hands control to the human (browser_tools.request_human_
 * action), the feed carries manual_control: true. In that mode this
 * component captures the user's own clicks/keystrokes/scrolls on the live
 * image and forwards them to the backend, which replays them on the real
 * sandboxed page - so the user can log in, enter an OTP, etc. directly,
 * without their credentials ever passing through the agent.
 */
export default function InlineSandbox({ sessionId, active, onStop }) {
  const [frame, setFrame] = useState(null);
  const [pageUrl, setPageUrl] = useState("");
  const [connected, setConnected] = useState(false);
  const [manualControl, setManualControl] = useState(false);
  const [manualReason, setManualReason] = useState("");
  const [manualDetails, setManualDetails] = useState("");
  const [resuming, setResuming] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const wsRef = useRef(null);
  const imgRef = useRef(null);

  // Esc always exits full screen, and full screen auto-exits if the
  // sandbox itself closes (agent finished, or the user hit Stop).
  useEffect(() => {
    if (!fullscreen) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  useEffect(() => {
    if (!active || !frame) setFullscreen(false);
  }, [active, frame]);

  useEffect(() => {
    if (!active || !sessionId) {
      setFrame(null);
      setConnected(false);
      setManualControl(false);
      return undefined;
    }

    let cancelled = false;
    const ws = new WebSocket(getSandboxWsUrl(sessionId));
    wsRef.current = ws;

    ws.onopen = () => !cancelled && setConnected(true);
    ws.onclose = () => !cancelled && setConnected(false);
    ws.onerror = () => !cancelled && setConnected(false);
    ws.onmessage = (evt) => {
      if (cancelled) return;
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "frame") {
          if (data.image) setFrame(data.image);
          if (typeof data.url === "string") setPageUrl(data.url);
          if (!data.active) setFrame(null);
          setManualControl(!!data.manual_control);
          setManualReason(data.manual_control_reason || "");
          setManualDetails(data.manual_control_details || "");
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
  }, [active, sessionId]);

  // Focus the sandbox for keyboard capture as soon as it's our turn to act.
  useEffect(() => {
    if (manualControl && imgRef.current) {
      imgRef.current.focus();
    }
  }, [manualControl]);

  const toImageCoords = useCallback((clientX, clientY) => {
    const el = imgRef.current;
    if (!el || !el.naturalWidth || !el.naturalHeight) return null;
    const rect = el.getBoundingClientRect();
    const scaleX = el.naturalWidth / rect.width;
    const scaleY = el.naturalHeight / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }, []);

  const handleClick = (e) => {
    if (!manualControl) return;
    e.preventDefault();
    const pt = toImageCoords(e.clientX, e.clientY);
    if (pt) sendSandboxClick(sessionId, pt.x, pt.y);
    imgRef.current && imgRef.current.focus();
  };

  const handleKeyDown = (e) => {
    if (!manualControl) return;
    // Let browser-level shortcuts (refresh, devtools, etc.) through; forward
    // everything else, including Tab/Backspace/Enter, which need
    // preventDefault so they don't move focus or navigate this page instead.
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    sendSandboxKey(sessionId, e.key);
  };

  const handleWheel = (e) => {
    if (!manualControl) return;
    e.preventDefault();
    sendSandboxScroll(sessionId, e.deltaX, e.deltaY);
  };

  const handleResume = async () => {
    setResuming(true);
    try {
      await resumeSandboxControl(sessionId);
    } catch (err) {
      alert("Error resuming the agent: " + err.message);
    } finally {
      setResuming(false);
    }
  };

  const handleStop = async (e) => {
    e.stopPropagation();
    setFrame(null);
    setFullscreen(false);
    if (onStop) {
      // Aborts the in-flight chat request immediately and flips the
      // parent's `loading` state off, instead of just closing the browser
      // and leaving the agent (and this component's `active` prop) running.
      onStop();
      return;
    }
    try {
      await killAgent(sessionId);
    } catch (err) {
      alert("Error stopping sandbox: " + err.message);
    }
  };

  const toggleFullscreen = (e) => {
    e.stopPropagation();
    setFullscreen((f) => !f);
  };

  if (!active || !frame) return null;

  return (
    <div className={`inline-sandbox ${manualControl ? "manual" : ""} ${fullscreen ? "fullscreen" : ""}`}>
      <div className="inline-sandbox-bar">
        <span className={`sandbox-dot ${connected ? "live" : "idle"}`} />
        <span className="inline-sandbox-label">
          {manualControl ? "Your turn" : "Agent's browser"}
        </span>
        <span className="inline-sandbox-url" title={pageUrl || ""}>
          {pageUrl || ""}
        </span>
        <button
          className="inline-sandbox-expand"
          onClick={toggleFullscreen}
          title={fullscreen ? "Exit full screen (Esc)" : "Full screen"}
        >
          {fullscreen ? "⤡ Exit full screen" : "⤢ Full screen"}
        </button>
        <button
          className="inline-sandbox-stop"
          onClick={handleStop}
          title="Stop the agent and close this sandbox"
        >
          ⏹ Stop
        </button>
      </div>

      {manualControl && (
        <div className="inline-sandbox-handoff">
          <div className="inline-sandbox-handoff-text">
            <strong>{manualReason || "The agent needs you to take over."}</strong>
            {manualDetails && <div className="inline-sandbox-handoff-sub">{manualDetails}</div>}
            <div className="inline-sandbox-handoff-sub">
              Click and type directly on the screen below - nothing you type goes through the agent.
            </div>
          </div>
          <button className="inline-sandbox-resume-btn" onClick={handleResume} disabled={resuming}>
            {resuming ? "Resuming…" : "Resume agent →"}
          </button>
        </div>
      )}

      <img
        ref={imgRef}
        src={`data:image/jpeg;base64,${frame}`}
        alt="Agent's live sandbox view"
        className={`inline-sandbox-frame ${manualControl ? "interactive" : ""}`}
        tabIndex={manualControl ? 0 : -1}
        draggable={false}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onWheel={handleWheel}
        onContextMenu={(e) => manualControl && e.preventDefault()}
      />
    </div>
  );
}
