/**
 * StatusIndicator — Minimal recording + connection status overlay
 *
 * Shows:
 * - Red pulsing dot when recording (push-to-talk active)
 * - Connection status dot (green/yellow/red)
 * - Positioned at bottom of avatar overlay, click-through enabled
 */

import { useRecordingStore, useConnectionStore } from "../stores";
import "./StatusIndicator.css";

export function StatusIndicator() {
  const { isRecording } = useRecordingStore();
  const { status } = useConnectionStore();

  const connColor =
    status === "connected"
      ? "#4caf50"
      : status === "connecting"
        ? "#ff9800"
        : "#f44336";

  return (
    <div className="status-indicator" id="status-indicator">
      {/* Recording indicator */}
      {isRecording && (
        <div className="status-recording">
          <span className="status-rec-dot" />
          <span className="status-rec-text">REC</span>
        </div>
      )}

      {/* Connection indicator */}
      <div className="status-connection">
        <span
          className="status-conn-dot"
          style={{ background: connColor }}
        />
      </div>
    </div>
  );
}
