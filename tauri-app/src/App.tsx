/**
 * Sylph — Main Application
 *
 * Wires together:
 * - Phase 1.4: AvatarCanvas (VRM rendering)
 * - Phase 1.5: Idle animations (via AvatarCanvas)
 * - Phase 1.6: Expression driver (via AvatarCanvas)
 * - Phase 2.3: SidecarSocket connection
 * - Phase 1.3: Tauri event listeners (recording state, tray events)
 * - Phase 4.5: Dual-payload parsing (audio + viseme from sidecar)
 * - Phase 4.6: VisemeScheduler lip-sync playback
 */

import { useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { AvatarCanvas } from "./components/AvatarCanvas";
import { DebugPanel } from "./components/DebugPanel";
import { StatusIndicator } from "./components/StatusIndicator";
import { SettingsPanel } from "./components/SettingsPanel";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { sidecarSocket } from "./services/SidecarSocket";
import { audioRecorder } from "./services/AudioRecorder";
import { parseDualPayload } from "./services/DualPayloadParser";
import { VisemeScheduler } from "./services/VisemeScheduler";
import { audioPlayer } from "./services/AudioPlayer";
import { avatarMotion } from "./services/AvatarMotion";
import {
  useRecordingStore,
  useDebugStore,
  useMoodStore,
  useConfirmStore,
  useSettingsStore,
} from "./stores";
import type { MoodLabel, MoodValues } from "./stores";
import { VRM } from "@pixiv/three-vrm";
import "./App.css";

// Path to the VRM avatar file — served from the public directory
const AVATAR_URL = "/avatar.vrm";

function App() {
  const visemeSchedulerRef = useRef<VisemeScheduler | null>(null);
  const wasStoppedManuallyRef = useRef<boolean>(true);
  const isFirstMountRef = useRef<boolean>(true);

  // -----------------------------------------------------------------------
  // Phase 2.3: Connect to the Python sidecar on mount
  // -----------------------------------------------------------------------
  useEffect(() => {
    sidecarSocket.connect();
    return () => sidecarSocket.disconnect();
  }, []);

  // -----------------------------------------------------------------------
  // Phase 7: Sync Quiet Mode to Rust
  // -----------------------------------------------------------------------
  const quietMode = useSettingsStore((state) => state.quietMode);
  useEffect(() => {
    invoke("set_quiet_mode", { quiet: quietMode }).catch((err) =>
      console.error("[App] Failed to sync quiet mode to Rust:", err)
    );
  }, [quietMode]);

  // -----------------------------------------------------------------------
  // Sync Recording State to Rust and Manage Audio Recorder Lifecycle
  // -----------------------------------------------------------------------
  const isRecording = useRecordingStore((state) => state.isRecording);
  useEffect(() => {
    if (isFirstMountRef.current) {
      isFirstMountRef.current = false;
      return;
    }

    // Sync state to Rust backend
    invoke("set_recording_state", { recording: isRecording }).catch((err) =>
      console.error("[App] Failed to sync recording state to Rust:", err)
    );

    // Sync state to debug store
    useDebugStore
      .getState()
      .addMessage(
        "recording",
        isRecording ? "Started" : "Stopped"
      );

    if (isRecording) {
      audioPlayer.stop();
      sidecarSocket.send("recording_start", {});
      audioRecorder.start().catch((err: any) => {
        console.error("[App] Failed to start recording:", err);
        useRecordingStore.getState().setRecording(false);
        useDebugStore
          .getState()
          .addMessage("error", `Mic error: ${err?.name || "Error"} - ${err?.message || String(err)}`);
      });
    } else {
      audioRecorder.stop();
      if (sidecarSocket.isConnected && wasStoppedManuallyRef.current) {
        sidecarSocket.send("vad_flush", {});
      }
    }
  }, [isRecording]);

  // -----------------------------------------------------------------------
  // Phase 4.5: Handle binary dual-payload frames (audio + viseme)
  // -----------------------------------------------------------------------
  useEffect(() => {
    // Set up playback complete callback on AudioPlayer
    audioPlayer.setOnPlaybackComplete(() => {
      useDebugStore.getState().addMessage("tts", "Speech playback complete");
      if (!wasStoppedManuallyRef.current) {
        useDebugStore.getState().addMessage("recording", "Auto-resuming recording...");
        useRecordingStore.getState().setRecording(true);
      }
    });

    const unsubBinary = sidecarSocket.onBinary((data: ArrayBuffer) => {
      try {
        const payload = parseDualPayload(data);

        useDebugStore
          .getState()
          .addMessage(
            "tts_audio",
            `${payload.audio.length} samples, ${payload.timeline.length} visemes`
          );

        // Play audio and sync visemes through AudioPlayer (turn-aware:
        // chunks from a superseded turn are dropped, never played)
        audioPlayer.play(payload.audio, payload.sampleRate, payload.timeline, payload.turnId);
      } catch (err) {
        console.error("[App] Failed to parse dual payload:", err);
        sidecarSocket.send("log", {
          level: "error",
          message: `[App] Failed to handle dual payload: ${err instanceof Error ? err.stack || err.message : String(err)}`
        });
      }
    });

    // Listen for TTS lifecycle events
    const unsubStart = sidecarSocket.onMessage("tts_start", (payload) => {
      // Auto-toggle recording off since the agent is starting to speak
      useRecordingStore.getState().setRecording(false);

      useDebugStore
        .getState()
        .addMessage("tts", `Speaking: "${(payload.text as string)?.slice(0, 50)}..."`);
    });

    const unsubThinking = sidecarSocket.onMessage("thinking", () => {
      // Auto-toggle recording off since the agent is now generating a response
      useRecordingStore.getState().setRecording(false);

      useDebugStore
        .getState()
        .addMessage("thinking", `Thinking...`);
    });

    const unsubComplete = sidecarSocket.onMessage("tts_complete", () => {
      useDebugStore.getState().addMessage("tts", "Speech complete");
      if (!wasStoppedManuallyRef.current) {
        useDebugStore.getState().addMessage("recording", "Auto-resuming recording...");
        useRecordingStore.getState().setRecording(true);
      }
    });

    const unsubStopAudio = sidecarSocket.onMessage("stop_audio", (payload) => {
      // min_turn_id makes the stop "sticky": any chunk still in flight from
      // a cancelled synthesis job is dropped on arrival instead of playing.
      audioPlayer.stop(payload.min_turn_id as number | undefined);
      useDebugStore.getState().addMessage("tts", "Speech stopped (interrupted)");
    });

    // Spatial avatar commands (deterministic intent router / move_avatar tool)
    const unsubAvatarCmd = sidecarSocket.onMessage("avatar_command", (payload) => {
      const command = payload.command as string;
      useDebugStore.getState().addMessage("avatar", `Command: ${command}`);
      avatarMotion.handleCommand(command, payload);
    });

    // Phase 8: Listen for mood updates from personality system
    const unsubMood = sidecarSocket.onMessage("mood_update", (payload) => {
      const mood = payload.mood as MoodLabel;
      const values = payload.values as MoodValues;
      useMoodStore.getState().setMood(mood, values);
      useDebugStore
        .getState()
        .addMessage("mood", `Mood: ${mood}`);
    });

    // Phase 8: Autonomous avatar events (glance, posture)
    const unsubGlance = sidecarSocket.onMessage("avatar_glance", (payload) => {
      useDebugStore
        .getState()
        .addMessage("avatar", `Glance: (${payload.target_x}, ${payload.target_y})`);
    });

    const unsubPosture = sidecarSocket.onMessage("avatar_posture", (payload) => {
      useDebugStore
        .getState()
        .addMessage("avatar", `Posture: ${payload.shift_type}`);
    });

    const unsubConfirm = sidecarSocket.onMessage("confirm_action", (payload) => {
      const id = payload.id as string;
      const action = payload.action as string;
      const details = payload.details as any;
      useConfirmStore.getState().requestConfirmation(id, action, details);
    });

    const unsubQuietUpdate = sidecarSocket.onMessage("quiet_mode_update", (payload) => {
      const quiet = payload.quiet as boolean;
      useSettingsStore.getState().setQuietMode(quiet);
    });

    // Capture screen on-demand when requested by sidecar
    const unsubCaptureRequest = sidecarSocket.onMessage("request_screen_capture", async (payload) => {
      const id = payload.id as string;
      console.log("[App] Screen capture requested on-demand, id:", id);
      try {
        const captureRes = await invoke<{ base64_jpeg: string }>("capture_screen");
        const activeWindow = await invoke<string>("get_active_window_title");
        
        useDebugStore
          .getState()
          .addMessage("vision", `Capturing on-demand: ${activeWindow}`);

        sidecarSocket.send("screen_capture_response", {
          id,
          image: captureRes.base64_jpeg,
          window_name: activeWindow,
        });
      } catch (err) {
        console.error("[App] Failed to handle screen capture request:", err);
        useDebugStore
          .getState()
          .addMessage("error", `Capture fail: ${err instanceof Error ? err.message : String(err)}`);
        
        sidecarSocket.send("screen_capture_response", {
          id,
          image: "",
          window_name: "Error",
        });
      }
    });

    const unsubTurnComplete = sidecarSocket.onMessage("turn_complete", () => {
      useDebugStore.getState().addMessage("system", "Turn complete");
      setTimeout(() => {
        if (!audioPlayer.playing && !wasStoppedManuallyRef.current) {
          useDebugStore.getState().addMessage("recording", "Auto-resuming recording (silent turn)...");
          useRecordingStore.getState().setRecording(true);
        }
      }, 500);
    });

    return () => {
      audioPlayer.setOnPlaybackComplete(() => {});
      unsubBinary();
      unsubStart();
      unsubThinking();
      unsubComplete();
      unsubStopAudio();
      unsubAvatarCmd();
      unsubMood();
      unsubGlance();
      unsubPosture();
      unsubConfirm();
      unsubQuietUpdate();
      unsubCaptureRequest();
      unsubTurnComplete();
    };
  }, []);

  // -----------------------------------------------------------------------
  // Phase 1.3: Listen for Tauri events (recording state, tray actions)
  // -----------------------------------------------------------------------
  useEffect(() => {
    const unlisteners: (() => void)[] = [];

    // Recording state from Rust push-to-talk hotkey
    listen<{ is_recording: boolean }>("recording_state", (event) => {
      const isRecording = event.payload.is_recording;
      wasStoppedManuallyRef.current = !isRecording;
      useRecordingStore.getState().setRecording(isRecording);
    }).then((unlisten) => unlisteners.push(unlisten));

    // Quiet mode toggle from system tray
    listen("quiet_mode_toggle", () => {
      useDebugStore.getState().addMessage("system", "Quiet mode toggled");
      const currentQuiet = useSettingsStore.getState().quietMode;
      const nextQuiet = !currentQuiet;
      useSettingsStore.getState().setQuietMode(nextQuiet);
      fetch("http://127.0.0.1:8420/api/settings/quiet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quiet: nextQuiet }),
      }).catch((err) => console.error("Failed to update quiet mode:", err));
    }).then((unlisten) => unlisteners.push(unlisten));

    // Settings request from system tray
    listen("open_settings", () => {
      useDebugStore.getState().addMessage("system", "Settings requested");
      useSettingsStore.getState().toggleSettings();
    }).then((unlisten) => unlisteners.push(unlisten));

    return () => {
      unlisteners.forEach((fn) => fn());
      audioRecorder.stop();
    };
  }, []);

  // -----------------------------------------------------------------------
  // Phase 1.4: VRM loaded callback
  // -----------------------------------------------------------------------
  const handleVRMLoaded = useCallback((vrm: VRM) => {
    console.log(
      "[App] VRM avatar loaded, expressions available:",
      vrm.expressionManager?.expressions.map((e) => e.expressionName)
    );
    useDebugStore.getState().addMessage("system", "VRM avatar loaded");
  }, []);

  // -----------------------------------------------------------------------
  // Phase 4.6: VisemeScheduler ready callback
  // -----------------------------------------------------------------------
  const handleVisemeSchedulerReady = useCallback(
    (scheduler: VisemeScheduler) => {
      visemeSchedulerRef.current = scheduler;
      audioPlayer.setVisemeScheduler(scheduler);
      console.log("[App] VisemeScheduler ready");
    },
    []
  );

  return (
    <div className="app-root" id="app-root">
      {/* Phase 1.4: 3D Avatar (transparent WebGL canvas) */}
      <AvatarCanvas
        vrmUrl={AVATAR_URL}
        onVRMLoaded={handleVRMLoaded}
        onVisemeSchedulerReady={handleVisemeSchedulerReady}
      />

      {/* Status indicator (recording + connection) */}
      <StatusIndicator />

      {/* Debug panel (Ctrl+D to toggle) */}
      <DebugPanel />

      {/* Settings Panel Overlay */}
      <SettingsPanel />

      {/* User Confirmation Dialog Modal */}
      <ConfirmDialog />
    </div>
  );
}

export default App;
