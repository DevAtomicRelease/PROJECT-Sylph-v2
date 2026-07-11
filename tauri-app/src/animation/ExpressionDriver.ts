/**
 * ExpressionDriver — Mood-to-VRM expression blending
 * Phase 1.6: Maps mood state to VRM expressionManager weights
 *
 * Reads mood from the Zustand store and smoothly interpolates
 * VRM expression weights using lerp with a configurable blend time.
 *
 * VRM 1.0 standard expressions used:
 * - happy, angry, sad, relaxed, surprised, neutral
 *
 * Lip-sync viseme weights (aa, ee, ih, oh, ou) occupy separate channels
 * and are controlled by the VisemeScheduler in Phase 4 — no conflict.
 *
 * Features:
 * - flash() method for short micro-expression bursts (reactive emotions)
 * - shy and bored mood support
 * - Faster blend duration (0.15s) for snappier, more alive expressions
 * - Autonomous micro-expression scheduler (eyebrow flickers, smile twitches)
 */

import { VRM } from "@pixiv/three-vrm";
import * as THREE from "three";
import { useMoodStore, MoodLabel } from "../stores";

/** Maps our mood labels to VRM expression blends */
const MOOD_TO_EXPRESSION: Record<
  MoodLabel,
  Record<string, number>
> = {
  happy:     { happy: 0.7, relaxed: 0.3 },
  sad:       { sad: 0.8, relaxed: 0.1 },
  angry:     { angry: 0.7, surprised: 0.1 },
  relaxed:   { relaxed: 0.6, happy: 0.2 },
  surprised: { surprised: 0.8 },
  neutral:   { neutral: 0.5, relaxed: 0.2 },
  shy:       { neutral: 0.3, relaxed: 0.4, happy: 0.2 },
  bored:     { sad: 0.4, relaxed: 0.3, neutral: 0.2 },
};

/** All expression names we control (don't touch viseme channels) */
const ALL_EXPRESSIONS = [
  "happy",
  "angry",
  "sad",
  "relaxed",
  "surprised",
  "neutral",
];

/** Blend time in seconds — snappy enough to feel reactive */
const BLEND_DURATION = 0.15;

/** Seconds an expression holds at full strength before decaying */
const EXPRESSION_HOLD_SECONDS = 6.0;
/** Seconds over which it relaxes toward the neutral baseline */
const EXPRESSION_DECAY_SECONDS = 10.0;
/** Floor the decay leaves in place — a hint of mood, never a frozen mask */
const DECAY_FLOOR = 0.35;

// Micro-expression scheduler constants
const MICRO_MIN_INTERVAL = 4.0;  // minimum seconds between micro-expressions
const MICRO_MAX_INTERVAL = 12.0; // maximum seconds between micro-expressions

/** Palette of subtle micro-expressions to fire autonomously */
const MICRO_EXPRESSIONS: Array<{
  expressions: Record<string, number>;
  holdDuration: number;
  fadeDuration: number;
}> = [
  // Eyebrow raise (subtle curious flicker)
  { expressions: { surprised: 0.12 }, holdDuration: 0.2, fadeDuration: 0.4 },
  // Quick smile twitch
  { expressions: { happy: 0.10 }, holdDuration: 0.15, fadeDuration: 0.3 },
  // Thoughtful furrow + slight relax
  { expressions: { neutral: 0.08, relaxed: 0.06 }, holdDuration: 0.25, fadeDuration: 0.45 },
  // Brief wide-eye (micro-surprise)
  { expressions: { surprised: 0.08, happy: 0.05 }, holdDuration: 0.18, fadeDuration: 0.35 },
  // Subtle concern/sadness flicker
  { expressions: { sad: 0.08 }, holdDuration: 0.2, fadeDuration: 0.4 },
];

/** A temporary micro-expression flash (overrides mood blend briefly) */
interface Flash {
  expressions: Record<string, number>;
  duration: number;       // seconds the flash is held at peak
  elapsed: number;        // seconds since flash started
  fadeDuration: number;   // seconds to fade back out
}

export class ExpressionDriver {
  private vrm: VRM;
  private currentWeights: Map<string, number> = new Map();
  private targetWeights: Map<string, number> = new Map();
  private activeFlash: Flash | null = null;
  private flashWeights: Map<string, number> = new Map();

  // Organic-decay state: expressions relax toward neutral instead of
  // freezing at full strength when the mood label stops changing.
  private lastMood: MoodLabel | null = null;
  private moodAge = 0;
  // Slow pseudo-random phase for breathing-like intensity drift
  private noisePhase = Math.random() * 100;

  // Autonomous micro-expression scheduler
  private microTimer = 0;
  private nextMicroInterval: number;

  constructor(vrm: VRM) {
    this.vrm = vrm;
    this.nextMicroInterval = this.randomMicroInterval();

    // Initialize all expression weights to 0
    for (const name of ALL_EXPRESSIONS) {
      this.currentWeights.set(name, 0);
      this.targetWeights.set(name, 0);
      this.flashWeights.set(name, 0);
    }
  }

  /**
   * Called every frame from the render loop.
   * Reads the current mood from the store and smoothly
   * blends expression weights toward targets.
   * If a flash is active, it overrides the mood blend temporarily.
   */
  update(deltaTime: number): void {
    const { mood } = useMoodStore.getState();

    // Track how long this mood has been held; reset the decay clock when it
    // changes (or when a flash fires, which signals fresh emotional activity).
    if (mood !== this.lastMood) {
      this.lastMood = mood;
      this.moodAge = 0;
    } else {
      this.moodAge += deltaTime;
    }
    this.noisePhase += deltaTime;

    // Autonomous micro-expression scheduler (Item 6)
    // Only fire when no flash is already active (avoid conflicts with
    // lip-sync-triggered or event-triggered flashes)
    this.microTimer += deltaTime;
    if (this.microTimer >= this.nextMicroInterval && !this.activeFlash) {
      this.fireMicroExpression();
      this.microTimer = 0;
      this.nextMicroInterval = this.randomMicroInterval();
    }

    // Compute target weights from current mood
    this.computeTargets(mood);

    // Organic decay: after the hold window, relax the expression toward a
    // soft baseline so the face never locks into a frozen mask post-speech.
    // A slow sinusoidal drift (±4%) keeps even the held expression subtly
    // alive, like real facial muscle tone.
    const decayT = Math.min(
      Math.max(this.moodAge - EXPRESSION_HOLD_SECONDS, 0) / EXPRESSION_DECAY_SECONDS,
      1.0
    );
    const decayScale = 1.0 - decayT * (1.0 - DECAY_FLOOR);
    const drift =
      1.0 +
      0.04 * Math.sin(this.noisePhase * 0.7) * Math.sin(this.noisePhase * 0.23 + 1.3);
    for (const name of ALL_EXPRESSIONS) {
      const t = this.targetWeights.get(name) ?? 0;
      this.targetWeights.set(name, t * decayScale * drift);
    }

    // Lerp current weights toward targets
    const lerpFactor = Math.min(deltaTime / BLEND_DURATION, 1.0);

    for (const name of ALL_EXPRESSIONS) {
      const current = this.currentWeights.get(name) ?? 0;
      const target = this.targetWeights.get(name) ?? 0;
      const blended = THREE.MathUtils.lerp(current, target, lerpFactor);
      this.currentWeights.set(name, blended);
    }

    // Handle active flash — overrides base mood weights temporarily
    if (this.activeFlash) {
      this.activeFlash.elapsed += deltaTime;
      const { expressions, duration, elapsed, fadeDuration } = this.activeFlash;

      const flashLerpFactor = Math.min(deltaTime / 0.08, 1.0); // fast rise

      if (elapsed <= duration) {
        // Hold phase: lerp flashWeights toward flash expressions
        for (const name of ALL_EXPRESSIONS) {
          const fw = this.flashWeights.get(name) ?? 0;
          const target = expressions[name] ?? 0;
          this.flashWeights.set(name, THREE.MathUtils.lerp(fw, target, flashLerpFactor));
        }
      } else {
        // Fade-out phase
        const fadeElapsed = elapsed - duration;
        const fadeProgress = Math.min(fadeElapsed / fadeDuration, 1.0);
        for (const name of ALL_EXPRESSIONS) {
          const fw = this.flashWeights.get(name) ?? 0;
          this.flashWeights.set(name, fw * (1.0 - fadeProgress));
        }
        if (fadeProgress >= 1.0) {
          this.activeFlash = null;
          for (const name of ALL_EXPRESSIONS) {
            this.flashWeights.set(name, 0);
          }
        }
      }
    }

    // Apply: blend between mood weights and flash weights
    for (const name of ALL_EXPRESSIONS) {
      const moodW = this.currentWeights.get(name) ?? 0;
      const flashW = this.flashWeights.get(name) ?? 0;
      // Flash additively overlays on mood, clamped to 1.0
      const final = Math.min(moodW + flashW * 0.6, 1.0);
      this.vrm.expressionManager?.setValue(name, final);
    }
  }

  /**
   * Trigger a short micro-expression burst.
   * Great for reactive moments: avatar looks surprised when user asks something
   * unexpected, flashes happy when starting to speak, etc.
   *
   * @param expressions  Map of VRM expression name → peak weight (0–1)
   * @param holdDuration Seconds to hold at peak (default 0.4s)
   * @param fadeDuration Seconds to fade back out (default 0.5s)
   */
  flash(
    expressions: Record<string, number>,
    holdDuration = 0.4,
    fadeDuration = 0.5
  ): void {
    this.moodAge = 0; // fresh emotional activity resets the decay clock
    this.activeFlash = {
      expressions,
      duration: holdDuration,
      elapsed: 0,
      fadeDuration,
    };
  }

  /**
   * Fire a random subtle micro-expression from the palette.
   * Intensity varies ±30% for organic feel.
   */
  private fireMicroExpression(): void {
    const template = MICRO_EXPRESSIONS[Math.floor(Math.random() * MICRO_EXPRESSIONS.length)];

    // Randomize intensity ±30%
    const intensityScale = 0.7 + Math.random() * 0.6;
    const scaled: Record<string, number> = {};
    for (const [name, weight] of Object.entries(template.expressions)) {
      scaled[name] = weight * intensityScale;
    }

    this.flash(scaled, template.holdDuration, template.fadeDuration);
  }

  private randomMicroInterval(): number {
    return MICRO_MIN_INTERVAL + Math.random() * (MICRO_MAX_INTERVAL - MICRO_MIN_INTERVAL);
  }

  private computeTargets(mood: MoodLabel): void {
    // Reset all targets to 0
    for (const name of ALL_EXPRESSIONS) {
      this.targetWeights.set(name, 0);
    }

    // Apply mood mapping
    const mapping = MOOD_TO_EXPRESSION[mood];
    if (mapping) {
      for (const [name, weight] of Object.entries(mapping)) {
        if (ALL_EXPRESSIONS.includes(name)) {
          this.targetWeights.set(name, weight);
        }
      }
    }
  }

  /** Force an immediate expression (e.g., reaction to user input) */
  setImmediate(expressions: Record<string, number>): void {
    for (const [name, weight] of Object.entries(expressions)) {
      if (ALL_EXPRESSIONS.includes(name)) {
        this.currentWeights.set(name, weight);
        this.targetWeights.set(name, weight);
        this.vrm.expressionManager?.setValue(name, weight);
      }
    }
  }
}

