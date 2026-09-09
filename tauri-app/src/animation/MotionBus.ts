/**
 * MotionBus — tiny frame-rate signal bus between services and the render loop.
 *
 * AvatarMotion tweens the OS window across the monitor, but the three.js
 * scene knows nothing about it — so the spring bones (hair/skirt/sleeves)
 * never react to the ride. AvatarMotion publishes its horizontal velocity
 * here; GestureController turns it into a body lean each frame, and the
 * spring bones trail naturally from the resulting bone movement.
 *
 * Kept as a plain module singleton (not a Zustand store) because it's
 * written/read every frame — no subscriptions, no React involvement.
 */

class MotionBus {
  /** Window horizontal velocity, px/s (positive = moving right). */
  windowVelX = 0;
  /** Window vertical velocity, px/s (positive = moving down). */
  windowVelY = 0;

  private decayTimer: ReturnType<typeof setTimeout> | null = null;

  /** Publish instantaneous window velocity (called from AvatarMotion's tween). */
  setWindowVelocity(vx: number, vy: number): void {
    this.windowVelX = vx;
    this.windowVelY = vy;

    // Safety: if the tween stops publishing (killed / completed), zero out
    // shortly after so a stale velocity can't hold the avatar leaning.
    if (this.decayTimer) clearTimeout(this.decayTimer);
    this.decayTimer = setTimeout(() => {
      this.windowVelX = 0;
      this.windowVelY = 0;
    }, 120);
  }
}

export const motionBus = new MotionBus();
