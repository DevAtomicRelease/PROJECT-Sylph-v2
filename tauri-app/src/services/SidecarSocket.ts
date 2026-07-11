/**
 * SidecarSocket — WebSocket client for the Python brain sidecar
 * Phase 2.3: Auto-reconnect with exponential backoff
 *
 * Message protocol (JSON text frames):
 * {
 *   "type": "ping" | "transcript" | "mood_update" | "audio_chunk" | ...,
 *   "payload": { ... }
 * }
 *
 * Binary frames are used for dual-payload audio (Phase 4).
 */

import { useConnectionStore, useDebugStore, useMoodStore, type MoodLabel, type MoodValues } from "../stores";

type MessageHandler = (payload: Record<string, unknown>) => void;
type BinaryHandler = (data: ArrayBuffer) => void;

const SIDECAR_URL = "ws://127.0.0.1:8420/ws";
const MAX_RECONNECT_DELAY_MS = 10_000;
const INITIAL_RECONNECT_DELAY_MS = 1_000;

class SidecarSocket {
  private ws: WebSocket | null = null;
  private handlers: Map<string, MessageHandler[]> = new Map();
  private binaryHandlers: BinaryHandler[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = INITIAL_RECONNECT_DELAY_MS;
  private intentionalClose = false;

  // -----------------------------------------------------------------------
  // Connection lifecycle
  // -----------------------------------------------------------------------

  connect(): void {
    if (this.ws) {
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        return;
      }
    }

    this.intentionalClose = false;
    useConnectionStore.getState().setStatus("connecting");

    try {
      const socket = new WebSocket(SIDECAR_URL);
      socket.binaryType = "arraybuffer";
      this.ws = socket;

      socket.onopen = () => {
        if (this.ws !== socket) return;
        console.log("[SidecarSocket] Connected to", SIDECAR_URL);
        useConnectionStore.getState().setStatus("connected");
        useConnectionStore.getState().resetReconnect();
        useDebugStore.getState().addMessage("system", "Connected to sidecar");
        this.reconnectDelay = INITIAL_RECONNECT_DELAY_MS;

        // Send initial ping
        this.send("ping", { hello: "from Sylph frontend" });
      };

      socket.onclose = (event) => {
        if (this.ws !== socket) return;
        console.log("[SidecarSocket] Disconnected:", event.code, event.reason);
        useConnectionStore.getState().setStatus("disconnected");

        if (!this.intentionalClose) {
          this.scheduleReconnect();
        }
      };

      socket.onerror = (error) => {
        if (this.ws !== socket) return;
        console.error("[SidecarSocket] Error:", error);
        useConnectionStore.getState().setStatus("error");
      };

      socket.onmessage = (event) => {
        if (this.ws !== socket) return;
        if (event.data instanceof ArrayBuffer) {
          // Binary frame (dual-payload audio + viseme timeline in Phase 4)
          this.binaryHandlers.forEach((handler) => handler(event.data));
          return;
        }

        // Text frame — JSON message
        try {
          const msg = JSON.parse(event.data as string);
          const type: string = msg.type || "unknown";
          const payload = msg.payload || {};

          // Debug logging
          useDebugStore
            .getState()
            .addMessage(type, JSON.stringify(payload).slice(0, 200));

          // Route to registered handlers
          const typeHandlers = this.handlers.get(type);
          if (typeHandlers) {
            typeHandlers.forEach((handler) => handler(payload));
          }

          // Built-in handlers
          this.handleBuiltInMessage(type, payload);
        } catch (err) {
          console.error("[SidecarSocket] Failed to parse message:", err);
        }
      };
    } catch (err) {
      console.error("[SidecarSocket] Failed to create WebSocket:", err);
      useConnectionStore.getState().setStatus("error");
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    this.intentionalClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  // -----------------------------------------------------------------------
  // Send messages
  // -----------------------------------------------------------------------

  send(type: string, payload: Record<string, unknown> = {}): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn("[SidecarSocket] Cannot send — not connected");
      return;
    }
    this.ws.send(JSON.stringify({ type, payload }));
  }

  sendBinary(data: ArrayBuffer | Uint8Array): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn("[SidecarSocket] Cannot send binary — not connected");
      return;
    }
    this.ws.send(data);
  }

  // -----------------------------------------------------------------------
  // Message handlers
  // -----------------------------------------------------------------------

  /** Register a handler for a specific message type */
  onMessage(type: string, handler: MessageHandler): () => void {
    const existing = this.handlers.get(type) || [];
    existing.push(handler);
    this.handlers.set(type, existing);

    // Return unsubscribe function
    return () => {
      const list = this.handlers.get(type);
      if (list) {
        const idx = list.indexOf(handler);
        if (idx >= 0) list.splice(idx, 1);
      }
    };
  }

  /** Register a handler for binary frames */
  onBinary(handler: BinaryHandler): () => void {
    this.binaryHandlers.push(handler);
    return () => {
      const idx = this.binaryHandlers.indexOf(handler);
      if (idx >= 0) this.binaryHandlers.splice(idx, 1);
    };
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  private handleBuiltInMessage(
    type: string,
    payload: Record<string, unknown>
  ): void {
    switch (type) {
      case "mood_update":
        // Update mood store from sidecar
        if (payload.mood && payload.values) {
          useMoodStore
            .getState()
            .setMood(
              payload.mood as MoodLabel,
              payload.values as MoodValues
            );
        }
        break;

      case "pong":
        console.log("[SidecarSocket] Pong received:", payload);
        break;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;

    const store = useConnectionStore.getState();
    store.incrementReconnect();

    const delay = Math.min(this.reconnectDelay, MAX_RECONNECT_DELAY_MS);
    console.log(
      `[SidecarSocket] Reconnecting in ${delay}ms (attempt ${store.reconnectAttempts + 1})`
    );
    useDebugStore
      .getState()
      .addMessage("system", `Reconnecting in ${delay / 1000}s...`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);

    // Exponential backoff: 1s → 2s → 4s → 8s → 10s (cap)
    this.reconnectDelay = Math.min(
      this.reconnectDelay * 2,
      MAX_RECONNECT_DELAY_MS
    );
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const sidecarSocket = new SidecarSocket();
