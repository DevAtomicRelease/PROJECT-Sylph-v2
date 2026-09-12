/**
 * Dashboard — the normal (non-overlay) Sylph window.
 * Phase 1 (agent UI): a text chat panel with an optional "speak replies" mode.
 *
 * Runs in its own Tauri window ("dashboard" label) with its own SidecarSocket
 * connection (the sidecar broadcasts to all clients). It deliberately does NOT
 * register a binary/audio handler, so spoken replies play through the avatar
 * overlay window, never doubled here. Future phases add capability/permission
 * controls, schedules, and an audit log as tabs alongside chat.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { sidecarSocket } from "./services/SidecarSocket";
import { useConnectionStore } from "./stores";
import "./Dashboard.css";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
}

let idCounter = 0;

export function Dashboard() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [speakReplies, setSpeakReplies] = useState(false);
  const [thinking, setThinking] = useState(false);
  const status = useConnectionStore((s) => s.status);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Connect this window's socket and wire chat responses.
  useEffect(() => {
    sidecarSocket.connect();

    const unsubTurn = sidecarSocket.onMessage("turn_complete", (payload) => {
      const response = (payload.response as string) ?? "";
      setThinking(false);
      if (response) {
        setMessages((m) => [...m, { id: ++idCounter, role: "assistant", text: response }]);
      }
    });

    // Voice turns (global push-to-talk) also surface here so the transcript is
    // visible in the dashboard, not just spoken.
    const unsubTranscript = sidecarSocket.onMessage("transcript", (payload) => {
      const t = (payload.text as string) ?? "";
      if (t) setMessages((m) => [...m, { id: ++idCounter, role: "user", text: t }]);
    });

    const unsubError = sidecarSocket.onMessage("error", (payload) => {
      setThinking(false);
      const msg = (payload.message as string) ?? "Something went wrong.";
      setMessages((m) => [...m, { id: ++idCounter, role: "assistant", text: `⚠️ ${msg}` }]);
    });

    return () => {
      unsubTurn();
      unsubTranscript();
      unsubError();
    };
  }, []);

  // Auto-scroll to the newest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, thinking]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || thinking) return;
    setMessages((m) => [...m, { id: ++idCounter, role: "user", text }]);
    setInput("");
    setThinking(true);
    sidecarSocket.send("chat", { text, speak: speakReplies });
  }, [input, thinking, speakReplies]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="dash">
      <header className="dash-header">
        <div className="dash-title">
          <span className="dash-logo">🜂</span> Sylph
          <span className={`dash-status dash-status--${status}`}>{status}</span>
        </div>
        <label className="dash-speak">
          <input
            type="checkbox"
            checked={speakReplies}
            onChange={(e) => setSpeakReplies(e.target.checked)}
          />
          🔊 Speak replies
        </label>
      </header>

      <div className="dash-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="dash-empty">
            Ask Sylph anything, or hold <kbd>Ctrl+Shift+Space</kbd> to talk.
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`dash-msg dash-msg--${m.role}`}>
            <div className="dash-bubble">{m.text}</div>
          </div>
        ))}
        {thinking && (
          <div className="dash-msg dash-msg--assistant">
            <div className="dash-bubble dash-bubble--thinking">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
      </div>

      <div className="dash-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Message Sylph…  (Enter to send, Shift+Enter for newline)"
          rows={1}
        />
        <button onClick={send} disabled={!input.trim() || thinking} aria-label="Send">
          ➤
        </button>
      </div>
    </div>
  );
}
