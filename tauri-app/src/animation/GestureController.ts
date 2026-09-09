/**
 * GestureController — mood- and speech-tied body movement
 *
 * Fills the gap between IdleAnimationController (breathing, blinks, gaze,
 * generic posture shifts) and ExpressionDriver (face): the *body* reacting to
 * what Sylph is saying and feeling.
 *
 * Three layers, all additive on top of the idle controller's pose (this
 * updates AFTER IdleAnimationController and BEFORE vrm.update, so offsets
 * compose with rest pose + breathing + gaze):
 *
 * 1. One-shot emotion gestures — GSAP timelines fired on mood changes and
 *    speech cues (happy bounce, surprised recoil, sad slump, angry lean,
 *    shy turn-away, hum sway, sigh, yawn). These move enough mass (hips,
 *    chest, head, arms) to visibly excite the VRM's spring bones
 *    (bust/hair/skirt/sleeves) — which is what makes them read as physical.
 * 2. Talk gesticulation — while TTS audio is actually playing, layered
 *    low-frequency beats on head/chest/arms, scaled by mood energy, eased
 *    in/out by an envelope so speech never starts or stops with a jerk.
 * 3. Idle fidgets + window-ride lean — small unscripted adjustments every
 *    8–20 s of quiet, plus a lean proportional to the window's glide
 *    velocity (via MotionBus) so relocations feel like riding, not sliding.
 */

import { VRM } from "@pixiv/three-vrm";
import { gsap } from "gsap";
import { useMoodStore, MoodLabel } from "../stores";
import { motionBus } from "./MotionBus";

const DEG = Math.PI / 180;

/** How energetic talk gesticulation is per mood (0..1). */
const MOOD_TALK_ENERGY: Record<MoodLabel, number> = {
  happy: 1.0,
  surprised: 0.9,
  angry: 0.85,
  relaxed: 0.45,
  neutral: 0.55,
  shy: 0.35,
  sad: 0.25,
  bored: 0.3,
};

/** Additive pose scratchpad the GSAP timelines write into. */
interface GesturePose {
  hipsBounce: number;     // vertical dip/spring (m, negative = crouch)
  leanForward: number;    // chest/spine pitch (rad·unit)
  leanSide: number;       // spine roll (rad·unit, + = lean right)
  twist: number;          // spine yaw (rad·unit)
  shoulderLift: number;   // both shoulders up (+) or slump down (−)
  armRaise: number;       // upper arms away from body (+) / clamp in (−)
  headNod: number;        // head pitch offset
  headTiltZ: number;      // head roll offset
  headBack: number;       // head pitch back (surprise/yawn)
  chestSwell: number;     // chest pitch open (sigh inhale / yawn)
}

function zeroPose(): GesturePose {
  return {
    hipsBounce: 0, leanForward: 0, leanSide: 0, twist: 0,
    shoulderLift: 0, armRaise: 0, headNod: 0, headTiltZ: 0,
    headBack: 0, chestSwell: 0,
  };
}

export class GestureController {
  private vrm: VRM;
  private pose: GesturePose = zeroPose();
  private activeTimeline: gsap.core.Timeline | null = null;

  // Talk gesticulation state
  private talking = false;
  private talkEnvelope = 0;          // eased 0..1
  private talkPhase = Math.random() * 10;
  private beatPhase = Math.random() * 10;

  // Mood tracking for one-shot triggers
  private lastMood: MoodLabel | null = null;

  // Idle fidget scheduling
  private fidgetTimer = 0;
  private nextFidgetIn = this.randomFidgetInterval();

  // Window-ride lean smoothing
  private rideLean = 0;

  // Captured rest height for hips — nothing else manages hips.position, so
  // we must SET (base + offset), never +=, or the bounce would accumulate
  // into a runaway sink at 60 fps.
  private baseHipsY: number | null = null;

  private elapsed = 0;

  constructor(vrm: VRM) {
    this.vrm = vrm;
  }

  // -----------------------------------------------------------------------
  // External triggers
  // -----------------------------------------------------------------------

  /** Called when TTS playback state changes (true while audio is audible). */
  setTalking(talking: boolean): void {
    this.talking = talking;
  }

  /**
   * Inspect an utterance about to be spoken and fire a matching gesture.
   * Mumble lines carry stage directions ("*hums quietly*", "*sighs*") —
   * these deserve motion, not just audio.
   */
  onSpeechStart(text: string): void {
    const t = (text || "").toLowerCase();
    if (t.includes("*hum")) return this.playHum();
    if (t.includes("*sigh")) return this.playSigh();
    if (t.includes("*yawn")) return this.playYawn();
    if (t.includes("*wiggles")) return this.playHappyBounce(0.6);
    // No stage direction: small conversational lead-in (weight gather)
    this.playOneShot((tl) => {
      tl.to(this.pose, { leanForward: 0.35, duration: 0.35, ease: "power2.out" })
        .to(this.pose, { leanForward: 0, duration: 0.9, ease: "sine.inOut" });
    });
  }

  /** Fire a gesture when the dominant mood label changes. */
  private onMoodChange(mood: MoodLabel): void {
    switch (mood) {
      case "happy":     return this.playHappyBounce(1.0);
      case "surprised": return this.playSurprised();
      case "sad":       return this.playSlump(1.0);
      case "bored":     return this.playSlump(0.6);
      case "angry":     return this.playAngryLean();
      case "shy":       return this.playShy();
      case "relaxed":   return this.playRelaxedSettle();
      case "neutral":   return; // no theatrical gesture for neutral
    }
  }

  // -----------------------------------------------------------------------
  // One-shot gesture vocabulary (GSAP timelines onto this.pose)
  // -----------------------------------------------------------------------

  private playOneShot(build: (tl: gsap.core.Timeline) => void): void {
    // A new gesture supersedes the previous one; ease whatever is left back
    // to zero quickly so limbs never jump.
    this.activeTimeline?.kill();
    const tl = gsap.timeline({
      onComplete: () => { this.activeTimeline = null; },
    });
    tl.to(this.pose, { ...zeroPose(), duration: 0.18, ease: "power2.out" });
    build(tl);
    this.activeTimeline = tl;
  }

  /** Springy vertical bounce — the signature "happy" move. Excites every spring chain. */
  playHappyBounce(scale = 1.0): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { hipsBounce: -0.028 * scale, leanForward: 0.3 * scale, duration: 0.16, ease: "power2.in" })
        .to(this.pose, { hipsBounce: 0.016 * scale, leanForward: -0.15 * scale, shoulderLift: 0.5 * scale, armRaise: 0.4 * scale, duration: 0.22, ease: "back.out(2.5)" })
        .to(this.pose, { hipsBounce: -0.006 * scale, shoulderLift: 0.15 * scale, duration: 0.18, ease: "sine.inOut" })
        .to(this.pose, { hipsBounce: 0, leanForward: 0, shoulderLift: 0, armRaise: 0, duration: 0.5, ease: "elastic.out(1, 0.45)" });
    });
  }

  /** Quick recoil: head back, shoulders up, slight crouch. */
  playSurprised(): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { headBack: 1.0, shoulderLift: 0.9, hipsBounce: -0.012, leanForward: -0.4, duration: 0.12, ease: "power3.out" })
        .to(this.pose, { headBack: 0.35, shoulderLift: 0.4, duration: 0.5, ease: "sine.inOut" })
        .to(this.pose, { headBack: 0, shoulderLift: 0, hipsBounce: 0, leanForward: 0, duration: 0.8, ease: "power2.inOut" });
    });
  }

  /** Slow deflate: shoulders drop, head down, spine curls. */
  playSlump(scale = 1.0): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { shoulderLift: -0.7 * scale, headNod: 0.8 * scale, leanForward: 0.5 * scale, chestSwell: -0.3 * scale, duration: 1.1, ease: "power2.inOut" })
        .to(this.pose, { shoulderLift: -0.35 * scale, headNod: 0.4 * scale, leanForward: 0.25 * scale, chestSwell: 0, duration: 1.6, ease: "sine.inOut" })
        .to(this.pose, { shoulderLift: 0, headNod: 0, leanForward: 0, duration: 1.4, ease: "sine.inOut" });
    });
  }

  /** Forward press: chin down, lean in, arms clamp slightly. */
  playAngryLean(): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { leanForward: 0.9, headNod: 0.5, armRaise: -0.5, shoulderLift: 0.3, duration: 0.25, ease: "power3.out" })
        .to(this.pose, { leanForward: 0.55, headNod: 0.3, armRaise: -0.3, duration: 1.2, ease: "sine.inOut" })
        .to(this.pose, { leanForward: 0, headNod: 0, armRaise: 0, shoulderLift: 0, duration: 0.9, ease: "power2.inOut" });
    });
  }

  /**
   * Shy / flustered ("blushing" proxy — this VRM ships no blush morph, so the
   * body carries it): turn away, chin down, shoulders in, arms clamp.
   */
  playShy(): void {
    const side = Math.random() > 0.5 ? 1 : -1;
    this.playOneShot((tl) => {
      tl.to(this.pose, { twist: 0.6 * side, headTiltZ: 0.5 * side, headNod: 0.55, armRaise: -0.6, shoulderLift: 0.35, duration: 0.5, ease: "power2.out" })
        .to(this.pose, { twist: 0.4 * side, headNod: 0.35, duration: 1.5, ease: "sine.inOut" })
        .to(this.pose, { twist: 0, headTiltZ: 0, headNod: 0, armRaise: 0, shoulderLift: 0, duration: 1.0, ease: "power2.inOut" });
    });
  }

  /** Settle: one soft weight release, like sitting back into comfort. */
  playRelaxedSettle(): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { hipsBounce: -0.008, shoulderLift: -0.25, chestSwell: 0.2, duration: 0.9, ease: "sine.inOut" })
        .to(this.pose, { hipsBounce: 0, shoulderLift: 0, chestSwell: 0, duration: 1.3, ease: "sine.inOut" });
    });
  }

  /** Humming: gentle metronome sway with a little head roll. */
  playHum(): void {
    this.playOneShot((tl) => {
      const beat = 0.62; // ~97 bpm, relaxed humming tempo
      tl.to(this.pose, { leanSide: 0.5, headTiltZ: 0.45, duration: beat, ease: "sine.inOut" });
      for (let i = 0; i < 3; i++) {
        tl.to(this.pose, { leanSide: -0.5, headTiltZ: -0.45, duration: beat, ease: "sine.inOut" })
          .to(this.pose, { leanSide: 0.5, headTiltZ: 0.45, duration: beat, ease: "sine.inOut" });
      }
      tl.to(this.pose, { leanSide: 0, headTiltZ: 0, duration: 0.8, ease: "sine.inOut" });
    });
  }

  /** Sigh: big inhale (chest opens, shoulders rise) then a dropped exhale. */
  playSigh(): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { chestSwell: 0.8, shoulderLift: 0.6, headBack: 0.25, duration: 0.9, ease: "sine.in" })
        .to(this.pose, { chestSwell: -0.3, shoulderLift: -0.5, headNod: 0.5, headBack: 0, duration: 0.55, ease: "power2.out" })
        .to(this.pose, { chestSwell: 0, shoulderLift: 0, headNod: 0, duration: 1.4, ease: "sine.inOut" });
    });
  }

  /** Yawn: head back + chest open, then a heavy-lidded slump. */
  playYawn(): void {
    this.playOneShot((tl) => {
      tl.to(this.pose, { headBack: 0.9, chestSwell: 0.7, shoulderLift: 0.5, armRaise: 0.25, duration: 1.1, ease: "sine.inOut" })
        .to(this.pose, { headBack: 0.9, chestSwell: 0.7, duration: 0.5 })
        .to(this.pose, { headBack: -0.2, chestSwell: -0.2, shoulderLift: -0.4, armRaise: 0, headNod: 0.4, duration: 0.8, ease: "power2.out" })
        .to(this.pose, { headBack: 0, chestSwell: 0, shoulderLift: 0, headNod: 0, duration: 1.2, ease: "sine.inOut" });
    });
  }

  /** Small unscripted idle adjustment — quieter than the backend posture shifts. */
  private playFidget(): void {
    const variant = Math.floor(Math.random() * 4);
    this.playOneShot((tl) => {
      switch (variant) {
        case 0: // micro weight rock
          tl.to(this.pose, { leanSide: 0.3, hipsBounce: -0.004, duration: 0.9, ease: "sine.inOut" })
            .to(this.pose, { leanSide: 0, hipsBounce: 0, duration: 1.2, ease: "sine.inOut" });
          break;
        case 1: // shoulder resettle
          tl.to(this.pose, { shoulderLift: 0.35, duration: 0.4, ease: "power2.out" })
            .to(this.pose, { shoulderLift: -0.1, duration: 0.5, ease: "power2.inOut" })
            .to(this.pose, { shoulderLift: 0, duration: 0.6, ease: "sine.inOut" });
          break;
        case 2: // slow attention drift (torso twist)
          tl.to(this.pose, { twist: 0.35, duration: 1.4, ease: "sine.inOut" })
            .to(this.pose, { twist: 0, duration: 1.6, ease: "sine.inOut" });
          break;
        case 3: // chin lift, as if a thought surfaced
          tl.to(this.pose, { headBack: 0.3, duration: 0.7, ease: "sine.inOut" })
            .to(this.pose, { headBack: 0, headNod: 0.15, duration: 0.9, ease: "sine.inOut" })
            .to(this.pose, { headNod: 0, duration: 0.6, ease: "sine.inOut" });
          break;
      }
    });
  }

  private randomFidgetInterval(): number {
    return 8 + Math.random() * 12; // 8–20 s
  }

  // -----------------------------------------------------------------------
  // Per-frame update
  // -----------------------------------------------------------------------

  update(delta: number): void {
    this.elapsed += delta;

    // --- Mood-change one-shots ---
    const { mood } = useMoodStore.getState();
    if (mood !== this.lastMood) {
      if (this.lastMood !== null) this.onMoodChange(mood);
      this.lastMood = mood;
    }

    // --- Talk envelope (ease in 0.25 s, out 0.6 s) ---
    const target = this.talking ? 1 : 0;
    const tau = this.talking ? 0.25 : 0.6;
    this.talkEnvelope += (target - this.talkEnvelope) * Math.min(delta / tau, 1);

    // --- Idle fidget scheduling (suppressed while talking or mid-gesture) ---
    if (!this.talking && !this.activeTimeline) {
      this.fidgetTimer += delta;
      if (this.fidgetTimer >= this.nextFidgetIn) {
        this.playFidget();
        this.fidgetTimer = 0;
        this.nextFidgetIn = this.randomFidgetInterval();
      }
    } else {
      this.fidgetTimer = 0;
    }

    // --- Window-ride lean (spring-bone excitation from window glides) ---
    // ~1200 px/s full glide ⇒ ≈4.5° lean into the motion.
    const targetLean = Math.max(-1, Math.min(1, motionBus.windowVelX / 1200)) * 0.08;
    this.rideLean += (targetLean - this.rideLean) * Math.min(delta / 0.12, 1);

    this.apply(delta);
  }

  private apply(delta: number): void {
    const humanoid = this.vrm.humanoid;
    if (!humanoid) return;

    const hips = humanoid.getNormalizedBoneNode("hips");
    const spine = humanoid.getNormalizedBoneNode("spine");
    const chest = humanoid.getNormalizedBoneNode("chest");
    const neck = humanoid.getNormalizedBoneNode("neck");
    const head = humanoid.getNormalizedBoneNode("head");
    const lShoulder = humanoid.getNormalizedBoneNode("leftShoulder");
    const rShoulder = humanoid.getNormalizedBoneNode("rightShoulder");
    const lUpperArm = humanoid.getNormalizedBoneNode("leftUpperArm");
    const rUpperArm = humanoid.getNormalizedBoneNode("rightUpperArm");
    const p = this.pose;

    // --- One-shot pose ---
    if (hips) {
      if (this.baseHipsY === null) this.baseHipsY = hips.position.y;
      hips.position.y = this.baseHipsY + p.hipsBounce;
    }
    if (spine) {
      spine.rotation.x += p.leanForward * 0.05;
      spine.rotation.z += (p.leanSide * 0.04) + this.rideLean * 0.5;
      spine.rotation.y += p.twist * 0.06;
    }
    if (chest) {
      chest.rotation.x += p.leanForward * 0.07 - p.chestSwell * 0.06;
      chest.rotation.z += (p.leanSide * 0.05) + this.rideLean * 0.35;
      chest.rotation.y += p.twist * 0.08;
    }
    if (neck) neck.rotation.x += p.headNod * 0.10 - p.headBack * 0.10;
    if (head) {
      head.rotation.x += p.headNod * 0.14 - p.headBack * 0.16;
      head.rotation.z += p.headTiltZ * 0.10 - this.rideLean * 0.4;
    }
    if (lShoulder) {
      lShoulder.position.y += p.shoulderLift * 0.008;
      lShoulder.rotation.z -= p.shoulderLift * 0.05;
    }
    if (rShoulder) {
      rShoulder.position.y += p.shoulderLift * 0.008;
      rShoulder.rotation.z += p.shoulderLift * 0.05;
    }
    if (lUpperArm) lUpperArm.rotation.z += p.armRaise * 8 * DEG;
    if (rUpperArm) rUpperArm.rotation.z -= p.armRaise * 8 * DEG;

    // --- Talk gesticulation (rhythmic, mood-scaled, envelope-gated) ---
    if (this.talkEnvelope > 0.01) {
      const { mood } = useMoodStore.getState();
      const energy = (MOOD_TALK_ENERGY[mood] ?? 0.55) * this.talkEnvelope;

      // Two incommensurate frequencies ⇒ non-looping feel
      this.talkPhase += delta * 2.1;
      this.beatPhase += delta * 3.37;
      const a = Math.sin(this.talkPhase);
      const b = Math.sin(this.beatPhase + 1.2);
      const c = Math.sin(this.talkPhase * 0.53 + 2.0);

      if (head) {
        head.rotation.x += (a * 0.5 + b * 0.5) * 0.022 * energy; // emphasis nods
        head.rotation.y += c * 0.016 * energy;                   // address shifts
        head.rotation.z += b * 0.010 * energy;
      }
      if (chest) {
        chest.rotation.x += a * 0.006 * energy;
        chest.rotation.y += c * 0.008 * energy;
      }
      // Forearm-less "hand talk": tiny upper-arm beats
      if (lUpperArm) lUpperArm.rotation.z += Math.max(0, a) * 2.4 * DEG * energy;
      if (rUpperArm) rUpperArm.rotation.z -= Math.max(0, b) * 2.4 * DEG * energy;
    }
  }

  dispose(): void {
    this.activeTimeline?.kill();
    this.activeTimeline = null;
    gsap.killTweensOf(this.pose);
  }
}
