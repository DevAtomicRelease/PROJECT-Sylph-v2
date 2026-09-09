/**
 * AvatarMotion — Smooth spatial transitions of the avatar window
 *
 * Handles `avatar_command` messages from the sidecar's deterministic intent
 * router (or the LLM's `move_avatar` tool) and tweens the actual Tauri
 * window position across the monitor with GSAP easing, so "move to the left"
 * is a smooth glide rather than a teleport. Hide/show fades the window.
 *
 * The conversation flow is never blocked: the sidecar already spoke its
 * acknowledgement before this runs, and the tween is fire-and-forget.
 */

import { getCurrentWindow, PhysicalPosition, currentMonitor } from "@tauri-apps/api/window";
import { gsap } from "gsap";
import { motionBus } from "../animation/MotionBus";

type NamedPosition =
  | "left"
  | "right"
  | "center"
  | "bottom_left"
  | "bottom_right";

const EDGE_MARGIN = 24; // px breathing room from screen edges
const MOVE_DURATION = 1.1; // seconds — deliberate but responsive

class AvatarMotion {
  private activeTween: gsap.core.Tween | null = null;
  private hidden = false;

  /** Entry point for `avatar_command` WebSocket messages. */
  async handleCommand(command: string, params: Record<string, unknown>): Promise<void> {
    try {
      switch (command) {
        case "avatar_move":
          await this.moveTo((params.position as NamedPosition) ?? "center");
          break;
        case "avatar_hide":
          await this.hide();
          break;
        case "avatar_show":
          await this.show();
          break;
        default:
          console.warn("[AvatarMotion] Unknown command:", command);
      }
    } catch (err) {
      console.error("[AvatarMotion] Command failed:", command, err);
    }
  }

  /** Glide the window to a named screen position. */
  async moveTo(position: NamedPosition): Promise<void> {
    const win = getCurrentWindow();
    const monitor = await currentMonitor();
    if (!monitor) return;

    const size = await win.outerSize();
    const cur = await win.outerPosition();

    const mx = monitor.position.x;
    const my = monitor.position.y;
    const mw = monitor.size.width;
    const mh = monitor.size.height;

    // Bottom-anchored (the avatar lives above the taskbar)
    const bottomY = my + mh - size.height - EDGE_MARGIN;
    const targets: Record<NamedPosition, { x: number; y: number }> = {
      left:         { x: mx + EDGE_MARGIN,                          y: cur.y },
      right:        { x: mx + mw - size.width - EDGE_MARGIN,        y: cur.y },
      center:       { x: mx + Math.round((mw - size.width) / 2),    y: cur.y },
      bottom_left:  { x: mx + EDGE_MARGIN,                          y: bottomY },
      bottom_right: { x: mx + mw - size.width - EDGE_MARGIN,        y: bottomY },
    };

    const target = targets[position];
    if (!target) return;

    // Tween a proxy object; apply integer positions per frame.
    this.activeTween?.kill();
    const proxy = { x: cur.x, y: cur.y };
    let lastX = cur.x;
    let lastY = cur.y;
    let lastT = performance.now();
    this.activeTween = gsap.to(proxy, {
      x: target.x,
      y: target.y,
      duration: MOVE_DURATION,
      ease: "power3.inOut",
      onUpdate: () => {
        // Publish glide velocity so the render loop can lean the body into
        // the motion and let the spring bones (hair/skirt) trail the ride.
        const now = performance.now();
        const dt = Math.max((now - lastT) / 1000, 1e-3);
        motionBus.setWindowVelocity((proxy.x - lastX) / dt, (proxy.y - lastY) / dt);
        lastX = proxy.x;
        lastY = proxy.y;
        lastT = now;
        win
          .setPosition(new PhysicalPosition(Math.round(proxy.x), Math.round(proxy.y)))
          .catch(() => {});
      },
      onComplete: () => {
        this.activeTween = null;
        motionBus.setWindowVelocity(0, 0);
      },
    });
  }

  async hide(): Promise<void> {
    if (this.hidden) return;
    this.hidden = true;
    const win = getCurrentWindow();
    // Brief delay so the spoken ack ("Poof. Gone.") starts before the window vanishes
    setTimeout(() => win.hide().catch(() => {}), 900);
  }

  async show(): Promise<void> {
    this.hidden = false;
    const win = getCurrentWindow();
    await win.show();
    await win.setFocus().catch(() => {});
  }
}

export const avatarMotion = new AvatarMotion();
