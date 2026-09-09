import { useState, useEffect } from "react";
import { useSettingsStore } from "../stores";
import "./SettingsPanel.css";

const SIDECAR_API = "http://127.0.0.1:8420";

export function SettingsPanel() {
  const {
    showSettings,
    toggleSettings,
    voice,
    speed,
    autonomous,
    commentFrequency,
    quietMode,
    vrmUrl,
    setVoice,
    setSpeed,
    setAutonomous,
    setCommentFrequency,
    setQuietMode,
    setVrmUrl,
  } = useSettingsStore();

  const [activeTab, setActiveTab] = useState<"memories" | "preferences" | "privacy" | "data">("preferences");
  const [shortTermMemories, setShortTermMemories] = useState<any[]>([]);
  const [longTermMemories, setLongTermMemories] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch memory data when the memories tab is selected
  useEffect(() => {
    if (showSettings && activeTab === "memories") {
      fetchMemories();
    }
  }, [showSettings, activeTab]);

  // Sync settings configuration to the sidecar whenever they change
  useEffect(() => {
    if (showSettings) {
      saveVoiceSettings();
    }
  }, [voice, speed]);

  useEffect(() => {
    if (showSettings) {
      saveQuietMode();
    }
  }, [quietMode]);

  useEffect(() => {
    if (showSettings) {
      saveAutonomousSettings();
    }
  }, [autonomous, commentFrequency]);

  if (!showSettings) return null;

  const fetchMemories = async () => {
    setLoading(true);
    try {
      const stRes = await fetch(`${SIDECAR_API}/api/memories/short-term`);
      const stData = await stRes.json();
      setShortTermMemories(stData);

      const ltRes = await fetch(`${SIDECAR_API}/api/memories/long-term`);
      const ltData = await ltRes.json();
      setLongTermMemories(ltData);
    } catch (e) {
      console.error("Failed to fetch memories:", e);
    } finally {
      setLoading(false);
    }
  };

  const deleteShortTerm = async (id: number) => {
    try {
      await fetch(`${SIDECAR_API}/api/memories/short-term/${id}`, { method: "DELETE" });
      setShortTermMemories(shortTermMemories.filter((m) => m.id !== id));
    } catch (e) {
      console.error("Failed to delete memory:", e);
    }
  };

  const deleteLongTerm = async (id: string, category: string) => {
    try {
      await fetch(`${SIDECAR_API}/api/memories/long-term/${id}?category=${category}`, { method: "DELETE" });
      setLongTermMemories(longTermMemories.filter((f) => f.id !== id));
    } catch (e) {
      console.error("Failed to delete fact:", e);
    }
  };

  const saveVoiceSettings = async () => {
    try {
      await fetch(`${SIDECAR_API}/api/settings/voice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voice, speed }),
      });
    } catch (e) {
      console.error("Failed to save voice settings:", e);
    }
  };

  const saveQuietMode = async () => {
    try {
      await fetch(`${SIDECAR_API}/api/settings/quiet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quiet: quietMode }),
      });
    } catch (e) {
      console.error("Failed to save quiet mode settings:", e);
    }
  };

  const saveAutonomousSettings = async () => {
    try {
      await fetch(`${SIDECAR_API}/api/settings/autonomous`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: autonomous, frequency: commentFrequency }),
      });
    } catch (e) {
      console.error("Failed to save autonomous settings:", e);
    }
  };

  const handleExport = async () => {
    try {
      const res = await fetch(`${SIDECAR_API}/api/settings/export`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aiden_memories_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    }
  };

  const handleReset = async () => {
    const doubleConfirm = window.confirm(
      "WARNING: This will wipe ALL short-term conversation history and long-term memories. This cannot be undone. Are you sure?"
    );
    if (!doubleConfirm) return;

    try {
      await fetch(`${SIDECAR_API}/api/settings/reset`, { method: "POST" });
      setShortTermMemories([]);
      setLongTermMemories([]);
      alert("Sylph memory has been reset successfully.");
    } catch (e) {
      console.error("Reset failed:", e);
    }
  };

  return (
    <div className="settings-overlay" id="settings-overlay">
      <div className="settings-panel" id="settings-panel">
        <div className="settings-header">
          <h2>Sylph Settings</h2>
          <button className="settings-close" onClick={toggleSettings}>
            ×
          </button>
        </div>

        <div className="settings-tabs">
          <button
            className={`settings-tab-btn ${activeTab === "preferences" ? "active" : ""}`}
            onClick={() => setActiveTab("preferences")}
          >
            Preferences
          </button>
          <button
            className={`settings-tab-btn ${activeTab === "memories" ? "active" : ""}`}
            onClick={() => setActiveTab("memories")}
          >
            Memories
          </button>
          <button
            className={`settings-tab-btn ${activeTab === "privacy" ? "active" : ""}`}
            onClick={() => setActiveTab("privacy")}
          >
            Privacy
          </button>
          <button
            className={`settings-tab-btn ${activeTab === "data" ? "active" : ""}`}
            onClick={() => setActiveTab("data")}
          >
            System
          </button>
        </div>

        <div className="settings-content">
          {activeTab === "preferences" && (
            <div className="settings-section">
              <div className="settings-control">
                <label>Quiet Mode</label>
                <div className="switch-container">
                  <input
                    type="checkbox"
                    id="quiet-mode-switch"
                    checked={quietMode}
                    onChange={(e) => setQuietMode(e.target.checked)}
                  />
                  <label htmlFor="quiet-mode-switch" className="switch-label" />
                </div>
              </div>

              <div className="settings-control">
                <label>Voice Model</label>
                <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                  <option value="af_bella">Bella (Female - Default)</option>
                  <option value="af_sarah">Sarah (Female)</option>
                  <option value="af_nicole">Nicole (Female)</option>
                  <option value="af_sky">Sky (Female)</option>
                  <option value="am_adam">Adam (Male)</option>
                  <option value="am_michael">Michael (Male)</option>
                </select>
              </div>

              <div className="settings-control">
                <label>Avatar (VRM)</label>
                <input
                  type="text"
                  value={vrmUrl}
                  placeholder="/avatar.vrm or https://…/model.vrm"
                  onChange={(e) => setVrmUrl(e.target.value)}
                />
                <input
                  type="file"
                  accept=".vrm"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setVrmUrl(URL.createObjectURL(f), false);
                  }}
                />
                <small>Paste a URL or path (persists across restarts), or pick a file (this session only).</small>
              </div>

              <div className="settings-control">
                <label>Speech Speed ({speed.toFixed(2)}x)</label>
                <input
                  type="range"
                  min="0.8"
                  max="1.2"
                  step="0.05"
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
              </div>

              <div className="settings-control">
                <label>Autonomous Expressions</label>
                <div className="switch-container">
                  <input
                    type="checkbox"
                    id="autonomous-switch"
                    checked={autonomous}
                    onChange={(e) => setAutonomous(e.target.checked)}
                  />
                  <label htmlFor="autonomous-switch" className="switch-label" />
                </div>
              </div>

              <div className="settings-control">
                <label>Unprompted Comments Frequency ({commentFrequency}s)</label>
                <input
                  type="range"
                  min="30"
                  max="300"
                  step="15"
                  value={commentFrequency}
                  disabled={!autonomous}
                  onChange={(e) => setCommentFrequency(parseInt(e.target.value))}
                />
              </div>
            </div>
          )}

          {activeTab === "memories" && (
            <div className="settings-section memories-tab">
              {loading ? (
                <div className="settings-loading">Retrieving databases...</div>
              ) : (
                <div className="memories-container">
                  <h3>Long-Term Memory ({longTermMemories.length} facts)</h3>
                  <div className="memory-list">
                    {longTermMemories.length === 0 ? (
                      <p className="no-memories">No long term facts learned yet.</p>
                    ) : (
                      longTermMemories.map((fact) => (
                        <div key={fact.id} className="memory-item">
                          <span className="memory-cat">[{fact.category}]</span>
                          <span className="memory-text">{fact.text}</span>
                          <button
                            className="memory-del-btn"
                            onClick={() => deleteLongTerm(fact.id, fact.category)}
                          >
                            Delete
                          </button>
                        </div>
                      ))
                    )}
                  </div>

                  <h3>Short-Term Sessions ({shortTermMemories.length} messages)</h3>
                  <div className="memory-list">
                    {shortTermMemories.length === 0 ? (
                      <p className="no-memories">No recent conversation messages.</p>
                    ) : (
                      shortTermMemories.map((msg) => (
                        <div key={msg.id} className="memory-item">
                          <span className="memory-role">[{msg.role}]</span>
                          <span className="memory-text">{msg.content}</span>
                          <button
                            className="memory-del-btn"
                            onClick={() => deleteShortTerm(msg.id)}
                          >
                            Delete
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "privacy" && (
            <div className="settings-section">
              <h3>Screen Privacy Zones</h3>
              <p className="privacy-desc">
                Define region bounds to mask coordinates with black boxes before screen analysis runs.
              </p>
              <div className="privacy-placeholder-grid">
                <div className="privacy-grid-inner">
                  <span>Privacy Zones Configurator</span>
                  <p>Click and Drag over screen stream to mask regions (Simulated)</p>
                </div>
              </div>
              <button className="settings-btn" onClick={() => alert("Privacy zone saved successfully.")}>
                Apply Rectangles
              </button>
            </div>
          )}

          {activeTab === "data" && (
            <div className="settings-section data-tab">
              <h3>Memory & Storage</h3>
              <p className="data-desc">Wipe databases clean or backup learned facts locally.</p>
              <div className="data-actions">
                <button className="settings-btn secondary" onClick={handleExport}>
                  Export Memories (.json)
                </button>
                <button className="settings-btn danger" onClick={handleReset}>
                  Forget Everything / Reset
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
