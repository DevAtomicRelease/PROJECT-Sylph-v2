/**
 * AudioPlayer — Web Audio API playback for TTS output
 * Phase 4.5: Plays float32 PCM audio from dual-payload WebSocket frames
 *
 * Uses AudioContext with BufferSource nodes for gapless playback.
 * Queues multiple audio chunks for seamless multi-sentence speech.
 */

import type { VisemeKeyframe } from "./DualPayloadParser";
import type { VisemeScheduler } from "./VisemeScheduler";

interface PlayQueueItem {
  buffer: AudioBuffer;
  timeline: VisemeKeyframe[];
  turnId: number;
  onEnd?: () => void;
}

export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private queue: PlayQueueItem[] = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private visemeScheduler: VisemeScheduler | null = null;
  private onPlaybackCompleteCallback?: () => void;
  /** Chunks stamped with a turnId below this are stale and dropped. */
  private minTurnId = 0;

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.gainNode = this.ctx.createGain();
      this.gainNode.connect(this.ctx.destination);
      this.gainNode.gain.value = 1.0;
    }
    // Resume if suspended (browsers require user gesture)
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    return this.ctx;
  }

  /**
   * Create + resume the AudioContext from within a real user gesture.
   * Browsers/webviews start the context "suspended" and only a gesture
   * (pointerdown/keydown) can resume it. Push-to-talk is a GLOBAL OS hotkey and
   * does NOT count as a webview gesture, so without this a voice turn generates
   * audio that never plays. Call this from a gesture listener once per session.
   */
  unlock(): void {
    try {
      const ctx = this.ensureContext();
      if (ctx.state === "suspended") void ctx.resume();
    } catch {
      /* ignore — will retry on next gesture */
    }
  }

  setVisemeScheduler(scheduler: VisemeScheduler): void {
    this.visemeScheduler = scheduler;
  }

  setOnPlaybackComplete(callback: () => void): void {
    this.onPlaybackCompleteCallback = callback;
  }

  /**
   * Play a float32 audio buffer immediately or queue it.
   * @param pcm Float32Array of audio samples
   * @param sampleRate Sample rate (default 24000 for Kokoro)
   * @param timeline Viseme keyframes for lip-sync
   * @param onEnd Callback when this chunk finishes playing
   */
  play(
    pcm: Float32Array,
    sampleRate = 24000,
    timeline: VisemeKeyframe[] = [],
    turnId = 0,
    onEnd?: () => void
  ): void {
    // Drop audio from a superseded turn. This is the frontend half of the
    // anti-zombie-audio mechanism: backend task cancellation cannot stop a
    // synthesis job already running inside an executor thread, so its output
    // may still arrive over the socket after an interruption — stamped with
    // the old turn id, it dies here instead of playing over the new response.
    if (turnId < this.minTurnId) {
      console.log(`[AudioPlayer] Dropping stale chunk (turn ${turnId} < ${this.minTurnId})`);
      return;
    }
    const ctx = this.ensureContext();

    // Create AudioBuffer from PCM
    const audioBuffer = ctx.createBuffer(1, pcm.length, sampleRate);
    audioBuffer.getChannelData(0).set(pcm);

    if (this.isPlaying) {
      // Queue for gapless playback
      this.queue.push({ buffer: audioBuffer, timeline, turnId, onEnd });
    } else {
      this.playBuffer(audioBuffer, timeline, onEnd);
    }
  }

  /**
   * Stop all playback and clear the queue.
   * @param minTurnId If provided, also drop any future chunk below this turn.
   */
  stop(minTurnId?: number): void {
    if (typeof minTurnId === "number" && minTurnId > this.minTurnId) {
      this.minTurnId = minTurnId;
    }
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch {
        // Already stopped
      }
      this.currentSource = null;
    }
    this.queue = [];
    this.isPlaying = false;
    this.visemeScheduler?.stop();
  }

  /**
   * Set playback volume (0.0 to 1.0).
   */
  setVolume(volume: number): void {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  private playBuffer(
    buffer: AudioBuffer,
    timeline: VisemeKeyframe[],
    onEnd?: () => void
  ): void {
    const ctx = this.ensureContext();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gainNode!);

    // Start viseme lip-sync at the exact same frame
    this.visemeScheduler?.play(timeline);

    source.onended = () => {
      this.currentSource = null;
      onEnd?.();

      // Play next queued buffer, skipping any that went stale while queued
      let next = this.queue.shift();
      while (next && next.turnId < this.minTurnId) {
        next = this.queue.shift();
      }
      if (next) {
        this.playBuffer(next.buffer, next.timeline, next.onEnd);
      } else {
        this.isPlaying = false;
        // Stop visemes when playback finishes completely
        this.visemeScheduler?.stop();
        this.onPlaybackCompleteCallback?.();
      }
    };

    this.currentSource = source;
    this.isPlaying = true;
    source.start();
  }

  get playing(): boolean {
    return this.isPlaying;
  }
}

// Singleton instance
export const audioPlayer = new AudioPlayer();
