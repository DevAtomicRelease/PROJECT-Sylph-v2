/**
 * Sylph Zustand Stores
 *
 * Lightweight state management for:
 * - Mood state (drives avatar expression blending)
 * - Connection state (sidecar WebSocket status)
 * - Recording state (push-to-talk)
 * - Debug state (transcript display)
 */

import { create } from "zustand";

// ---------------------------------------------------------------------------
// Mood Store — Phase 1.6 / Phase 8
// ---------------------------------------------------------------------------

export interface MoodValues {
  playful: number;
  focused: number;
  bored: number;
  curious: number;
  annoyed: number;
  enthusiastic: number;
  tired: number;
}

export type MoodLabel =
  | "happy"
  | "sad"
  | "angry"
  | "relaxed"
  | "surprised"
  | "neutral"
  | "shy"
  | "bored";

interface MoodState {
  /** Current dominant mood label (maps to VRM expressions) */
  mood: MoodLabel;
  /** Full mood vector from sidecar */
  values: MoodValues;
  /** Update mood from sidecar WebSocket message */
  setMood: (mood: MoodLabel, values: MoodValues) => void;
}

export const useMoodStore = create<MoodState>((set) => ({
  mood: "neutral",
  values: {
    playful: 0.3,
    focused: 0.5,
    bored: 0.1,
    curious: 0.4,
    annoyed: 0.0,
    enthusiastic: 0.3,
    tired: 0.1,
  },
  setMood: (mood, values) => set({ mood, values }),
}));

// ---------------------------------------------------------------------------
// Connection Store — Phase 2.3
// ---------------------------------------------------------------------------

interface ConnectionState {
  /** WebSocket connection status */
  status: "connecting" | "connected" | "disconnected" | "error";
  /** Number of reconnect attempts */
  reconnectAttempts: number;
  setStatus: (status: ConnectionState["status"]) => void;
  incrementReconnect: () => void;
  resetReconnect: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: "disconnected",
  reconnectAttempts: 0,
  setStatus: (status) => set({ status }),
  incrementReconnect: () =>
    set((s) => ({ reconnectAttempts: s.reconnectAttempts + 1 })),
  resetReconnect: () => set({ reconnectAttempts: 0 }),
}));

// ---------------------------------------------------------------------------
// Recording Store — Phase 1.3
// ---------------------------------------------------------------------------

interface RecordingState {
  /** Whether push-to-talk is active */
  isRecording: boolean;
  setRecording: (isRecording: boolean) => void;
}

export const useRecordingStore = create<RecordingState>((set) => ({
  isRecording: false,
  setRecording: (isRecording) => set({ isRecording }),
}));

// ---------------------------------------------------------------------------
// Debug Store — development transcript display
// ---------------------------------------------------------------------------

interface DebugMessage {
  id: number;
  timestamp: string;
  type: string;
  text: string;
}

interface DebugState {
  /** Whether the debug panel is visible */
  showDebug: boolean;
  /** Recent messages for the debug overlay */
  messages: DebugMessage[];
  toggleDebug: () => void;
  addMessage: (type: string, text: string) => void;
  clearMessages: () => void;
}

let msgCounter = 0;

export const useDebugStore = create<DebugState>((set) => ({
  showDebug: false,
  messages: [],
  toggleDebug: () => set((s) => ({ showDebug: !s.showDebug })),
  addMessage: (type, text) =>
    set((s) => ({
      messages: [
        ...s.messages.slice(-49), // Keep last 50 messages
        {
          id: ++msgCounter,
          timestamp: new Date().toLocaleTimeString(),
          type,
          text,
        },
      ],
    })),
  clearMessages: () => set({ messages: [] }),
}));

// ---------------------------------------------------------------------------
// Confirmation Store — Phase 9
// ---------------------------------------------------------------------------

interface ConfirmState {
  showConfirm: boolean;
  confirmId: string;
  actionName: string;
  details: Record<string, any>;
  requestConfirmation: (id: string, action: string, details: any) => void;
  respond: (approved: boolean) => void;
}

export const useConfirmStore = create<ConfirmState>((set) => ({
  showConfirm: false,
  confirmId: "",
  actionName: "",
  details: {},
  requestConfirmation: (id, action, details) =>
    set({
      showConfirm: true,
      confirmId: id,
      actionName: action,
      details: details || {},
    }),
  respond: (approved) => {
    // Send response back via sidecar socket
    import("../services/SidecarSocket").then(({ sidecarSocket }) => {
      const id = useConfirmStore.getState().confirmId;
      sidecarSocket.send("confirm_response", { id, approved });
      set({ showConfirm: false, confirmId: "", actionName: "", details: {} });
    });
  },
}));

// ---------------------------------------------------------------------------
// Settings Store — Phase 10
// ---------------------------------------------------------------------------

// VRM avatar selection (Phase C). Persisted per-viewer in localStorage so the
// chosen model survives restarts. A file chosen via the picker becomes a
// session-only object URL; a pasted URL/path persists.
const VRM_KEY = "sylph.vrmUrl";
const DEFAULT_VRM = "/avatar.vrm";
function loadVrmUrl(): string {
  try {
    return localStorage.getItem(VRM_KEY) || DEFAULT_VRM;
  } catch {
    return DEFAULT_VRM;
  }
}

interface SettingsState {
  showSettings: boolean;
  voice: string;
  speed: number;
  autonomous: boolean;
  commentFrequency: number;
  quietMode: boolean;
  vrmUrl: string;
  toggleSettings: () => void;
  setVoice: (voice: string) => void;
  setSpeed: (speed: number) => void;
  setAutonomous: (autonomous: boolean) => void;
  setCommentFrequency: (frequency: number) => void;
  setQuietMode: (quiet: boolean) => void;
  /** Set the avatar VRM. `persist` false for session-only object URLs. */
  setVrmUrl: (url: string, persist?: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  showSettings: false,
  voice: "af_bella",
  speed: 1.0,
  autonomous: true,
  commentFrequency: 60,
  quietMode: false,
  vrmUrl: loadVrmUrl(),
  toggleSettings: () => set((s) => ({ showSettings: !s.showSettings })),
  setVoice: (voice) => set({ voice }),
  setSpeed: (speed) => set({ speed }),
  setAutonomous: (autonomous) => set({ autonomous }),
  setCommentFrequency: (commentFrequency) => set({ commentFrequency }),
  setQuietMode: (quietMode) => set({ quietMode }),
  setVrmUrl: (vrmUrl, persist = true) => {
    try {
      if (persist) localStorage.setItem(VRM_KEY, vrmUrl);
    } catch {
      /* private mode / storage disabled — keep it in memory only */
    }
    set({ vrmUrl });
  },
}));

