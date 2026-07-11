/**
 * IdleAnimationController — Procedural idle animations for the VRM avatar using GSAP
 * Phase 10.4 / Refactored with GSAP
 *
 * Implements:
 * - Breathing: subtle spine/chest bone oscillation
 * - Blinking: VRM blink expression at natural intervals (3-6s)
 * - Head look: GSAP-tweened saccadic look-around, gaze aversion, and eye contact
 * - Posture shifts: GSAP timelines for weight shifts, head tilts, and shoulder rolls
 */

import { VRM } from "@pixiv/three-vrm";
import * as THREE from "three";
import { gsap } from "gsap";

export class IdleAnimationController {
  private vrm: VRM;

  // Blink timing
  private nextBlinkTime = 0;
  private blinkPhase: "idle" | "closing" | "opening" = "idle";
  private blinkTimer = 0;
  private readonly BLINK_CLOSE_DURATION = 0.06; // seconds
  private readonly BLINK_OPEN_DURATION = 0.1;

  // Breathing & continuous micro-sway phase
  private breathPhase = 0;
  private elapsed = 0;

  // GSAP animated properties
  private gaze = { x: 0, y: 0 };
  private posture = {
    headTilt: 0,
    shoulderRoll: 0,
    subtleStretch: 0,
    weightShift: 0,
  };

  // Auto-gaze management
  private isAutoGazeActive = true;
  private randomLookTimeout: ReturnType<typeof setTimeout> | null = null;

  // VRM Target Object
  private lookTargetObj: THREE.Object3D;
  private lookTargetX = 0;
  private lookTargetY = 0;

  // Base shoulder positions to prevent accumulation
  private baseLeftShoulderY: number | null = null;
  private baseRightShoulderY: number | null = null;

  constructor(vrm: VRM) {
    this.vrm = vrm;
    this.nextBlinkTime = this.randomBlinkInterval();

    // Create and add look-at target to vrm scene for proper world matrix calculation
    this.lookTargetObj = new THREE.Object3D();
    this.vrm.scene.add(this.lookTargetObj);
    if (this.vrm.lookAt) {
      this.vrm.lookAt.target = this.lookTargetObj;
    }

    // Initialize auto-gaze saccades
    this.triggerRandomLook();
  }

  /**
   * Triggers a random look-at target to simulate natural human eye saccades.
   */
  private triggerRandomLook = (): void => {
    if (!this.isAutoGazeActive) return;

    // Comfort-cone gaze limits for human idle looking
    const targetX = (Math.random() - 0.5) * 0.28; // -0.14 to 0.14
    const targetY = Math.random() * 0.12 - 0.02;  // -0.02 to 0.10
    
    // Snappy transitions (saccades are quick)
    const duration = 0.22 + Math.random() * 0.28;
    // Hold gaze at that direction for a natural pause
    const holdTime = 2.0 + Math.random() * 3.5;

    gsap.killTweensOf(this.gaze);
    gsap.to(this.gaze, {
      x: targetX,
      y: targetY,
      duration: duration,
      ease: "power2.out",
      onComplete: () => {
        // Schedule next random saccade
        this.randomLookTimeout = setTimeout(this.triggerRandomLook, holdTime * 1000);
      }
    });
  };

  /**
   * Focuses gaze on a specific coordinate target (e.g. conversational target).
   */
  setGlanceTarget(x: number, y: number, duration: number): void {
    // Disable auto-gaze behavior
    this.isAutoGazeActive = false;
    if (this.randomLookTimeout) {
      clearTimeout(this.randomLookTimeout);
      this.randomLookTimeout = null;
    }

    gsap.killTweensOf(this.gaze);

    // Eye contact / gaze transitions are slightly deliberate
    const transitionTime = Math.min(duration * 0.25, 0.7);

    gsap.to(this.gaze, {
      x: x,
      y: y,
      duration: transitionTime,
      ease: "power2.out",
      onComplete: () => {
        // Hold for the remaining duration before returning to auto-gaze
        const holdTime = duration - transitionTime;
        this.randomLookTimeout = setTimeout(() => {
          this.isAutoGazeActive = true;
          this.triggerRandomLook();
        }, Math.max(0, holdTime) * 1000);
      }
    });
  }

  /**
   * Initiates a body posture shift using a GSAP timeline that peaks in the middle and resolves back.
   */
  setPostureShift(shiftType: string, duration: number): void {
    const totalDuration = duration || 2.0;
    
    // Terminate any running posture tweens
    gsap.killTweensOf(this.posture);
    
    const tl = gsap.timeline();
    
    // Properties to reset to 0 in parallel
    const resets: Record<string, number> = {
      headTilt: 0,
      shoulderRoll: 0,
      subtleStretch: 0,
      weightShift: 0
    };
    
    let targetProp = "";
    let targetVal = 1.0;
    
    if (shiftType === "head_tilt") {
      targetProp = "headTilt";
    } else if (shiftType === "shoulder_roll") {
      targetProp = "shoulderRoll";
    } else if (shiftType === "subtle_stretch") {
      targetProp = "subtleStretch";
    } else if (shiftType === "weight_shift_left") {
      targetProp = "weightShift";
      targetVal = -1.0;
    } else if (shiftType === "weight_shift_right") {
      targetProp = "weightShift";
      targetVal = 1.0;
    }
    
    if (targetProp) {
      delete resets[targetProp];
      
      const halfDur = Math.max(0.2, (totalDuration - 0.2) / 2);
      
      // Phase 1: Reset active shifts, ease-in new target posture
      tl.to(this.posture, {
        ...resets,
        [targetProp]: targetVal,
        duration: halfDur,
        ease: "power2.out"
      });
      
      // Phase 2: Ease-out target posture back to zero
      tl.to(this.posture, {
        [targetProp]: 0.0,
        duration: halfDur,
        ease: "power2.inOut"
      });
    } else {
      // Fallback: reset all postures smoothly
      tl.to(this.posture, {
        ...resets,
        duration: 0.5,
        ease: "power2.out"
      });
    }
  }

  /**
   * Call every frame from the render loop.
   */
  update(delta: number): void {
    this.elapsed += delta;

    this.updateRestPose();
    this.updateBlink(delta);
    this.updateBreathing(delta);
    this.updateHeadLook();
    this.updatePostureShift();
  }

  // -----------------------------------------------------------------------
  // Blinking — natural random intervals with quick close/open
  // -----------------------------------------------------------------------

  private updateBlink(delta: number): void {
    const em = this.vrm.expressionManager;
    if (!em) return;

    switch (this.blinkPhase) {
      case "idle":
        this.nextBlinkTime -= delta;
        if (this.nextBlinkTime <= 0) {
          this.blinkPhase = "closing";
          this.blinkTimer = 0;
        }
        break;

      case "closing":
        this.blinkTimer += delta;
        const closeProgress = Math.min(
          this.blinkTimer / this.BLINK_CLOSE_DURATION,
          1
        );
        em.setValue("blink", closeProgress);
        if (closeProgress >= 1) {
          this.blinkPhase = "opening";
          this.blinkTimer = 0;
        }
        break;

      case "opening":
        this.blinkTimer += delta;
        const openProgress = Math.min(
          this.blinkTimer / this.BLINK_OPEN_DURATION,
          1
        );
        em.setValue("blink", 1 - openProgress);
        if (openProgress >= 1) {
          this.blinkPhase = "idle";
          this.nextBlinkTime = this.randomBlinkInterval();
          // Occasional double-blink
          if (Math.random() < 0.15) {
            this.nextBlinkTime = 0.2;
          }
        }
        break;
    }
  }

  private randomBlinkInterval(): number {
    return 3.0 + Math.random() * 3.0;
  }

  // -----------------------------------------------------------------------
  // Rest Pose — Relaxed A-pose rotation for arms and reset of spine bones
  // -----------------------------------------------------------------------

  private updateRestPose(): void {
    const humanoid = this.vrm.humanoid;
    if (!humanoid) return;

    const DEG = Math.PI / 180;
    const zAxis = new THREE.Vector3(0, 0, 1);

    // Reset spine, chest, neck, head rotations to prevent accumulation
    const spine = humanoid.getNormalizedBoneNode("spine");
    const chest = humanoid.getNormalizedBoneNode("chest");
    const neck = humanoid.getNormalizedBoneNode("neck");
    const head = humanoid.getNormalizedBoneNode("head");
    if (spine) spine.rotation.set(0, 0, 0);
    if (chest) chest.rotation.set(0, 0, 0);
    if (neck) neck.rotation.set(0, 0, 0);
    if (head) head.rotation.set(0, 0, 0);

    // Shoulders: rotate down slightly (~10°) to start the arm drop
    const leftShoulder = humanoid.getNormalizedBoneNode("leftShoulder");
    const rightShoulder = humanoid.getNormalizedBoneNode("rightShoulder");
    if (leftShoulder) {
      leftShoulder.quaternion.setFromAxisAngle(zAxis, -10 * DEG);
    }
    if (rightShoulder) {
      rightShoulder.quaternion.setFromAxisAngle(zAxis, 10 * DEG);
    }

    // Upper arms: rotate down from T-pose (~60° toward body)
    const leftUpperArm = humanoid.getNormalizedBoneNode("leftUpperArm");
    const rightUpperArm = humanoid.getNormalizedBoneNode("rightUpperArm");
    if (leftUpperArm) {
      leftUpperArm.quaternion.setFromAxisAngle(zAxis, -60 * DEG);
    }
    if (rightUpperArm) {
      rightUpperArm.quaternion.setFromAxisAngle(zAxis, 60 * DEG);
    }

    // Lower arms: slight bend inward for natural resting pose
    const leftLowerArm = humanoid.getNormalizedBoneNode("leftLowerArm");
    const rightLowerArm = humanoid.getNormalizedBoneNode("rightLowerArm");
    if (leftLowerArm) {
      leftLowerArm.quaternion.setFromAxisAngle(zAxis, -15 * DEG);
    }
    if (rightLowerArm) {
      rightLowerArm.quaternion.setFromAxisAngle(zAxis, 15 * DEG);
    }
  }

  // -----------------------------------------------------------------------
  // Breathing — subtle spine oscillation via bone rotation
  // -----------------------------------------------------------------------

  private updateBreathing(delta: number): void {
    this.breathPhase += delta * 0.75; // ~0.75 Hz breathing rate
    const breathSin = Math.sin(this.breathPhase);

    // Pronounced chest/spine breathing movements
    const chestBreath = breathSin * 0.024;
    const spineBreath = breathSin * 0.012;

    // Organic postural micro-sway (muscle balance noise) using overlapping frequencies
    const t = this.elapsed;
    const swayX = Math.sin(t * 0.35) * 0.020 + Math.cos(t * 0.15) * 0.010;
    const swayZ = Math.cos(t * 0.40) * 0.015 + Math.sin(t * 0.25) * 0.008;

    const humanoid = this.vrm.humanoid;
    if (humanoid) {
      const spine = humanoid.getNormalizedBoneNode("spine");
      const chest = humanoid.getNormalizedBoneNode("chest");
      const head = humanoid.getNormalizedBoneNode("head");

      if (spine) {
        spine.rotation.x += spineBreath;
        spine.rotation.y += swayX * 0.4;
        spine.rotation.z += swayZ * 0.4;
      }
      if (chest) {
        chest.rotation.x += chestBreath;
        chest.rotation.y += swayX * 0.6;
        chest.rotation.z += swayZ * 0.6;
      }
      if (head) {
        // Stabilize head rotation in opposition to breathing to maintain focus
        head.rotation.x -= (chestBreath + spineBreath) * 0.7;
        head.rotation.y -= swayX * 0.4;
        head.rotation.z -= swayZ * 0.3;
      }

      // Add breathing and sway movement to upper arms for a natural, soft posture
      const leftUpperArm = humanoid.getNormalizedBoneNode("leftUpperArm");
      const rightUpperArm = humanoid.getNormalizedBoneNode("rightUpperArm");
      if (leftUpperArm) {
        leftUpperArm.rotation.z += breathSin * 0.015;
        leftUpperArm.rotation.x += swayX * 0.3;
      }
      if (rightUpperArm) {
        rightUpperArm.rotation.z -= breathSin * 0.015;
        rightUpperArm.rotation.x += swayX * 0.3;
      }
    }

    // Subtle shoulder rise on inhale
    const leftShoulder =
      this.vrm.humanoid?.getNormalizedBoneNode("leftShoulder");
    const rightShoulder =
      this.vrm.humanoid?.getNormalizedBoneNode("rightShoulder");

    if (leftShoulder && rightShoulder) {
      if (this.baseLeftShoulderY === null) {
        this.baseLeftShoulderY = leftShoulder.position.y;
      }
      if (this.baseRightShoulderY === null) {
        this.baseRightShoulderY = rightShoulder.position.y;
      }

      const shoulderRise = Math.max(0, breathSin) * 0.005;
      leftShoulder.position.y = this.baseLeftShoulderY + shoulderRise;
      rightShoulder.position.y = this.baseRightShoulderY + shoulderRise;
    }
  }

  // -----------------------------------------------------------------------
  // Head Look — Set target eye coordinate & distribute bone rotations
  // -----------------------------------------------------------------------

  private updateHeadLook(): void {
    this.lookTargetX = this.gaze.x;
    this.lookTargetY = this.gaze.y;

    // Set position of the eye lookAt target Object3D
    this.lookTargetObj.position.set(this.lookTargetX, 1.25 + this.lookTargetY, 0.8);
    this.lookTargetObj.updateWorldMatrix(true, false);

    // Distribute yaw/pitch/roll rotations across multiple bones to create organic posture support
    const humanoid = this.vrm.humanoid;
    if (humanoid) {
      const spine = humanoid.getNormalizedBoneNode("spine");
      const chest = humanoid.getNormalizedBoneNode("chest");
      const neck = humanoid.getNormalizedBoneNode("neck");
      const head = humanoid.getNormalizedBoneNode("head");

      if (spine) {
        spine.rotation.y += this.lookTargetX * 0.08;
        spine.rotation.x += this.lookTargetY * 0.04;
      }
      if (chest) {
        chest.rotation.y += this.lookTargetX * 0.12;
        chest.rotation.x += this.lookTargetY * 0.06;
      }
      if (neck) {
        neck.rotation.y += this.lookTargetX * 0.45;
        neck.rotation.x += this.lookTargetY * 0.25;
        // Natural roll tilt when looking left/right
        neck.rotation.z += -this.lookTargetX * 0.08;
      }
      if (head) {
        head.rotation.y += this.lookTargetX * 0.20;
        head.rotation.x += this.lookTargetY * 0.15;
        // Secondary roll tilt
        head.rotation.z += -this.lookTargetX * 0.04;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Posture Shifts — Apply GSAP-timeline-driven joint rotation offsets
  // -----------------------------------------------------------------------

  private updatePostureShift(): void {
    const humanoid = this.vrm.humanoid;
    if (!humanoid) return;

    const DEG = Math.PI / 180;

    if (this.posture.headTilt !== 0) {
      const head = humanoid.getNormalizedBoneNode("head");
      if (head) {
        head.rotation.z += 8 * DEG * this.posture.headTilt;
      }
    }

    if (this.posture.shoulderRoll !== 0) {
      const leftShoulder = humanoid.getNormalizedBoneNode("leftShoulder");
      const rightShoulder = humanoid.getNormalizedBoneNode("rightShoulder");
      if (leftShoulder) {
        leftShoulder.rotation.x += 6 * DEG * this.posture.shoulderRoll;
        leftShoulder.rotation.y += 4 * DEG * this.posture.shoulderRoll;
      }
      if (rightShoulder) {
        rightShoulder.rotation.x += 6 * DEG * this.posture.shoulderRoll;
        rightShoulder.rotation.y -= 4 * DEG * this.posture.shoulderRoll;
      }
    }

    if (this.posture.subtleStretch !== 0) {
      const spine = humanoid.getNormalizedBoneNode("spine");
      const chest = humanoid.getNormalizedBoneNode("chest");
      if (spine) {
        spine.rotation.x -= 3 * DEG * this.posture.subtleStretch;
      }
      if (chest) {
        chest.rotation.x -= 3 * DEG * this.posture.subtleStretch;
      }
    }

    if (this.posture.weightShift !== 0) {
      const spine = humanoid.getNormalizedBoneNode("spine");
      const chest = humanoid.getNormalizedBoneNode("chest");
      if (spine) {
        spine.rotation.z += 4 * DEG * this.posture.weightShift;
      }
      if (chest) {
        chest.rotation.z += 2 * DEG * this.posture.weightShift;
      }
    }
  }

  /**
   * Dispose method to clear active timeouts and GSAP tweens/timelines.
   */
  dispose(): void {
    this.isAutoGazeActive = false;
    if (this.randomLookTimeout) {
      clearTimeout(this.randomLookTimeout);
      this.randomLookTimeout = null;
    }
    gsap.killTweensOf(this.gaze);
    gsap.killTweensOf(this.posture);
  }
}
