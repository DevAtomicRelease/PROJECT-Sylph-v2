/**
 * DebugPanel — Development overlay showing WebSocket messages and status
 *
 * Displays:
 * - Sidecar connection status
 * - Recording state
 * - Current mood
 * - Recent WebSocket messages (scrolling log)
 *
 * Toggle with Ctrl+D or from the debug store.
 */

import { useEffect } from "react";
import {
  useDebugStore,
  useConnectionStore,
  useRecordingStore,
  useMoodStore,
} from "../stores";
import "./DebugPanel.css";

export function DebugPanel() {
  const { showDebug, messages, toggleDebug, clearMessages } = useDebugStore();
  const { status, reconnectAttempts } = useConnectionStore();
  const { isRecording } = useRecordingStore();
  const { mood } = useMoodStore();

  // Ctrl+D toggle
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "d") {
        e.preventDefault();
        toggleDebug();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [toggleDebug]);

  if (!showDebug) return null;

  const statusColor =
    status === "connected"
      ? "#4caf50"
      : status === "connecting"
        ? "#ff9800"
        : "#f44336";

  return (
    <div className="debug-panel" id="debug-panel">
      <div className="debug-header">
        <span className="debug-title">Sylph Debug</span>
        <button className="debug-close" onClick={toggleDebug}>
          ×
        </button>
      </div>

      <div className="debug-status-row">
        <span className="debug-status-item">
          <span
            className="debug-dot"
            style={{ background: statusColor }}
          />
          {status}
          {reconnectAttempts > 0 && ` (retry ${reconnectAttempts})`}
        </span>
        <span className="debug-status-item">
          🎤 {isRecording ? "REC" : "OFF"}
        </span>
        <span className="debug-status-item">😊 {mood}</span>
      </div>

      <div className="debug-messages">
        {messages.map((msg) => (
          <div key={msg.id} className="debug-message">
            <span className="debug-time">{msg.timestamp}</span>
            <span className="debug-type">[{msg.type}]</span>
            <span className="debug-text">{msg.text}</span>
          </div>
        ))}
      </div>

      <button className="debug-clear" onClick={clearMessages}>
        Clear
      </button>
    </div>
  );
}
