/**
 * VisemeScheduler — Frame-accurate lip-sync playback
 * Phase 4.6: Replays viseme keyframes in sync with audio playback
 *
 * Design:
 * - Receives a viseme timeline from AudioPlayer
 * - Steps through keyframes using a high-resolution timer
 * - Linearly interpolates VRM blendshape weights between keyframes
 */

import { VRM } from "@pixiv/three-vrm";
import type { VisemeKeyframe } from "./DualPayloadParser";

const VISEME_CHANNELS = ["aa", "ee", "ih", "oh", "ou"] as const;
type VisemeChannel = (typeof VISEME_CHANNELS)[number];

export class VisemeScheduler {
  private vrm: VRM | null = null;
  private timeline: VisemeKeyframe[] = [];
  private startTime = 0;
  private isPlaying = false;
  private currentIndex = 0;

  /** Current blendshape weights (updated every frame) */
  private currentWeights: Record<VisemeChannel, number> = {
    aa: 0,
    ee: 0,
    ih: 0,
    oh: 0,
    ou: 0,
  };

  setVRM(vrm: VRM): void {
    this.vrm = vrm;
  }

  /**
   * Start playing a viseme timeline immediately.
   */
  play(timeline: VisemeKeyframe[]): void {
    this.timeline = timeline;
    this.currentIndex = 0;
    this.startTime = performance.now();
    this.isPlaying = true;
  }

  /**
   * Stop playback and reset mouth to neutral.
   */
  stop(): void {
    this.isPlaying = false;
    this.timeline = [];
    this.currentIndex = 0;
    this.resetWeights();
    this.applyWeights();
  }

  /**
   * Call every frame from the render loop (in AvatarCanvas).
   * Interpolates between keyframes and applies to VRM.
   */
  update(_delta: number): void {
    if (!this.isPlaying || !this.vrm || this.timeline.length === 0) {
      return;
    }

    const elapsed = performance.now() - this.startTime;

    // Find the current keyframe
    while (
      this.currentIndex < this.timeline.length - 1 &&
      elapsed >= this.timeline[this.currentIndex + 1].time_ms
    ) {
      this.currentIndex++;
    }

    const current = this.timeline[this.currentIndex];
    const next =
      this.currentIndex < this.timeline.length - 1
        ? this.timeline[this.currentIndex + 1]
        : null;

    if (next) {
      // Interpolate between current and next keyframe
      const segmentStart = current.time_ms;
      const segmentEnd = next.time_ms;
      const segmentDuration = segmentEnd - segmentStart;

      let t = 0;
      if (segmentDuration > 0) {
        t = Math.max(0, Math.min(1, (elapsed - segmentStart) / segmentDuration));
      }

      // Lerp each viseme channel
      for (let i = 0; i < VISEME_CHANNELS.length; i++) {
        const ch = VISEME_CHANNELS[i];
        const from = current.weights[i];
        const to = next.weights[i];
        this.currentWeights[ch] = from + (to - from) * t;
      }
    } else {
      // At or past the last keyframe
      if (elapsed > current.time_ms + current.duration_ms) {
        // Timeline complete: close the mouth and stop driving the channels
        // entirely. Continuing to "play" with stale state was one of the
        // freeze-open paths — a later interrupted chunk could leave the last
        // keyframe weights latched on the mesh.
        this.resetWeights();
        this.applyWeights();
        this.isPlaying = false;
        this.timeline = [];
        this.currentIndex = 0;
        return;
      } else {
        // Apply last keyframe weights
        for (let i = 0; i < VISEME_CHANNELS.length; i++) {
          this.currentWeights[VISEME_CHANNELS[i]] = current.weights[i];
        }
      }
    }

    this.applyWeights();
  }

  private applyWeights(): void {
    if (!this.vrm?.expressionManager) return;

    for (const ch of VISEME_CHANNELS) {
      this.vrm.expressionManager.setValue(ch, this.currentWeights[ch]);
    }
  }

  private resetWeights(): void {
    for (const ch of VISEME_CHANNELS) {
      this.currentWeights[ch] = 0;
    }
  }

  get playing(): boolean {
    return this.isPlaying;
  }
}
