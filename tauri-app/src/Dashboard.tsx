/**
 * Dashboard — the normal (non-overlay) Sylph window.
 * Phase 1: text/speech chat.  Phase 2: a Controls tab to toggle what the agent
 * is allowed to do (capability domains) and restrict file access.
 *
 * Runs in its own Tauri window ("dashboard") with its own SidecarSocket. It
 * registers no audio handler, so spoken replies play through the avatar overlay,
 * never doubled here.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { sidecarSocket } from "./services/SidecarSocket";
import { useConnectionStore } from "./stores";
import "./Dashboard.css";

const SIDECAR_API = "http://127.0.0.1:8420";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
}

let idCounter = 0;

export function Dashboard() {
  const [tab, setTab] = useState<"chat" | "controls">("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [speakReplies, setSpeakReplies] = useState(false);
  const [thinking, setThinking] = useState(false);
  const status = useConnectionStore((s) => s.status);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sidecarSocket.connect();
    const unsubTurn = sidecarSocket.onMessage("turn_complete", (payload) => {
      setThinking(false);
      const response = (payload.response as string) ?? "";
      if (response) setMessages((m) => [...m, { id: ++idCounter, role: "assistant", text: response }]);
    });
    const unsubTranscript = sidecarSocket.onMessage("transcript", (payload) => {
      const t = (payload.text as string) ?? "";
      if (t) setMessages((m) => [...m, { id: ++idCounter, role: "user", text: t }]);
    });
    const unsubError = sidecarSocket.onMessage("error", (payload) => {
      setThinking(false);
      const msg = (payload.message as string) ?? "Something went wrong.";
      setMessages((m) => [...m, { id: ++idCounter, role: "assistant", text: `⚠️ ${msg}` }]);
    });
    return () => { unsubTurn(); unsubTranscript(); unsubError(); };
  }, []);

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
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="dash">
      <header className="dash-header">
        <div className="dash-title">
          <span className="dash-logo">🜂</span> Sylph
          <span className={`dash-status dash-status--${status}`}>{status}</span>
        </div>
        <nav className="dash-tabs">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
          <button className={tab === "controls" ? "active" : ""} onClick={() => setTab("controls")}>Controls</button>
        </nav>
      </header>

      {/* Chat stays mounted so messages + socket handlers persist across tabs */}
      <div className="dash-body" hidden={tab !== "chat"}>
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
              <div className="dash-bubble dash-bubble--thinking"><span></span><span></span><span></span></div>
            </div>
          )}
        </div>
        <div className="dash-input">
          <label className="dash-speak">
            <input type="checkbox" checked={speakReplies} onChange={(e) => setSpeakReplies(e.target.checked)} />
            🔊
          </label>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Message Sylph…  (Enter to send, Shift+Enter for newline)"
            rows={1}
          />
          <button onClick={send} disabled={!input.trim() || thinking} aria-label="Send">➤</button>
        </div>
      </div>

      {tab === "controls" && <ControlsPanel />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Controls tab — capability toggles + file-access allowlist (Phase 2)
// ---------------------------------------------------------------------------

interface PermConfig {
  version: number;
  domains: Record<string, { enabled: boolean; allowed_dirs?: string[] }>;
}
type DomainMeta = Record<string, { label: string; tools: string }>;

function ControlsPanel() {
  const [config, setConfig] = useState<PermConfig | null>(null);
  const [meta, setMeta] = useState<DomainMeta>({});
  const [saved, setSaved] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [newDir, setNewDir] = useState("");

  useEffect(() => {
    fetch(`${SIDECAR_API}/api/permissions`)
      .then((r) => r.json())
      .then((d) => { setConfig(d.config); setMeta(d.meta || {}); })
      .catch(() => setSaved("error"));
  }, []);

  const save = useCallback(async (next: PermConfig) => {
    setConfig(next);
    setSaved("saving");
    try {
      const r = await fetch(`${SIDECAR_API}/api/permissions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next),
      });
      const d = await r.json();
      setConfig(d.config);
      setSaved("done");
      setTimeout(() => setSaved("idle"), 1500);
    } catch {
      setSaved("error");
    }
  }, []);

  if (!config) {
    return <div className="dash-body dash-controls"><p className="dash-empty">Loading capabilities…</p></div>;
  }

  const toggle = (dom: string) => {
    const d = config.domains[dom];
    save({ ...config, domains: { ...config.domains, [dom]: { ...d, enabled: !d.enabled } } });
  };
  const addDir = () => {
    const dir = newDir.trim();
    if (!dir) return;
    const files = config.domains.files || { enabled: true, allowed_dirs: [] };
    const dirs = [...(files.allowed_dirs || []), dir];
    setNewDir("");
    save({ ...config, domains: { ...config.domains, files: { ...files, allowed_dirs: dirs } } });
  };
  const removeDir = (dir: string) => {
    const files = config.domains.files;
    const dirs = (files.allowed_dirs || []).filter((d) => d !== dir);
    save({ ...config, domains: { ...config.domains, files: { ...files, allowed_dirs: dirs } } });
  };

  const fileDirs = config.domains.files?.allowed_dirs || [];

  return (
    <div className="dash-body dash-controls">
      <div className="dash-controls-head">
        <h2>What Sylph is allowed to do</h2>
        <span className={`dash-save dash-save--${saved}`}>
          {saved === "saving" ? "Saving…" : saved === "done" ? "Saved ✓" : saved === "error" ? "Save failed" : ""}
        </span>
      </div>

      {Object.entries(config.domains).map(([dom, val]) => (
        <div className="dash-cap" key={dom}>
          <div className="dash-cap-info">
            <div className="dash-cap-label">{meta[dom]?.label || dom}</div>
            <div className="dash-cap-tools">{meta[dom]?.tools || ""}</div>
          </div>
          <button
            className={`dash-switch ${val.enabled ? "on" : "off"}`}
            onClick={() => toggle(dom)}
            aria-label={`Toggle ${dom}`}
          >
            <span className="dash-knob" />
          </button>
        </div>
      ))}

      {config.domains.files?.enabled && (
        <div className="dash-allowlist">
          <div className="dash-cap-label">Files — allowed folders</div>
          <div className="dash-cap-tools">
            {fileDirs.length === 0 ? "Empty = all folders reachable. Add paths to restrict." : "Only these folders are reachable:"}
          </div>
          {fileDirs.map((d) => (
            <div className="dash-dir" key={d}>
              <code>{d}</code>
              <button onClick={() => removeDir(d)} aria-label="Remove">✕</button>
            </div>
          ))}
          <div className="dash-dir-add">
            <input
              value={newDir}
              onChange={(e) => setNewDir(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addDir()}
              placeholder="C:\Users\you\Documents"
            />
            <button onClick={addDir} disabled={!newDir.trim()}>Add</button>
          </div>
        </div>
      )}
    </div>
  );
}
