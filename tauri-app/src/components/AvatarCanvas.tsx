/**
 * AvatarCanvas — three.js + @pixiv/three-vrm renderer
 * Phase 1.4: Load and render the VRM avatar in a transparent WebGL canvas
 *
 * Features:
 * - Transparent background (alpha: true) for desktop overlay
 * - PerspectiveCamera framing head-to-waist
 * - VRM 1.0 loading with VRMLoaderPlugin
 * - SpringBone physics for hair/cloth
 * - 60fps render loop with delta time
 * - Exposes VRM instance via ref for animation and expression control
 */

import { useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { ExpressionDriver } from "../animation/ExpressionDriver";
import { IdleAnimationController } from "../animation/IdleAnimationController";
import { GestureController } from "../animation/GestureController";
import { VisemeScheduler } from "../services/VisemeScheduler";
import { sidecarSocket } from "../services/SidecarSocket";
import { audioPlayer } from "../services/AudioPlayer";

interface AvatarCanvasProps {
  vrmUrl: string;
  onVRMLoaded?: (vrm: VRM) => void;
  onVisemeSchedulerReady?: (scheduler: VisemeScheduler) => void;
}

export function AvatarCanvas({ vrmUrl, onVRMLoaded, onVisemeSchedulerReady }: AvatarCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const vrmRef = useRef<VRM | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const clockRef = useRef(new THREE.Clock());
  const frameIdRef = useRef<number>(0);
  const expressionDriverRef = useRef<ExpressionDriver | null>(null);
  const idleControllerRef = useRef<IdleAnimationController | null>(null);
  const gestureControllerRef = useRef<GestureController | null>(null);
  const visemeSchedulerRef = useRef<VisemeScheduler | null>(null);

  const setupScene = useCallback(() => {
    if (!containerRef.current) return null;

    // Scene
    const scene = new THREE.Scene();

    // Camera — frame head to hips, zoomed in and shifted lower
    const camera = new THREE.PerspectiveCamera(
      22, // Zoomed in from 28 to 22 (about 21% zoom increase)
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      20
    );
    camera.position.set(0, 1.35, 2.0); // Positioned at eye level (1.35m) and brought closer (2.3m)
    camera.lookAt(0, 1.25, 0); // Look at neck level (1.25m) to center upper body and prevent head clip

    // Renderer — transparent background for overlay
    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // Clear any existing canvas (React 18 Strict Mode remount safety)
    while (containerRef.current.firstChild) {
      containerRef.current.removeChild(containerRef.current.firstChild);
    }
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(1, 2, 3);
    scene.add(directionalLight);

    // Subtle fill light from below for avatar
    const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
    fillLight.position.set(-1, -0.5, 1);
    scene.add(fillLight);

    return { scene, camera, renderer };
  }, []);

  const loadVRM = useCallback(
    async (scene: THREE.Scene) => {
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));

      return new Promise<VRM>((resolve, reject) => {
        loader.load(
          vrmUrl,
          (gltf) => {
            const vrm = gltf.userData.vrm as VRM;
            if (!vrm) {
              reject(new Error("No VRM data found in GLTF"));
              return;
            }

            // Optimize draw calls. combineSkeletons replaces the deprecated
            // removeUnnecessaryJoints (three-vrm v3) — same goal, and it
            // leaves the spring-bone joint hierarchy untouched.
            VRMUtils.removeUnnecessaryVertices(gltf.scene);
            VRMUtils.combineSkeletons(gltf.scene);

            // Rotate VRM to face camera (VRM convention: +Z forward)
            VRMUtils.rotateVRM0(vrm);

            // Shift avatar down so the head is not cut off by the top of the canvas
            vrm.scene.position.y = -0.16;

            scene.add(vrm.scene);
            vrmRef.current = vrm;

            const expressions = vrm.expressionManager?.expressions.map((e) => e.expressionName) || [];
            console.log("[AvatarCanvas] VRM loaded successfully. Expressions:", expressions);

            // Verify spring-bone physics is actually live, not just present
            // in the file: count the joints the manager will simulate.
            const springJoints = vrm.springBoneManager?.joints.size ?? 0;
            const springColliders = vrm.springBoneManager?.colliderGroups.length ?? 0;
            if (springJoints === 0) {
              console.warn("[AvatarCanvas] No spring-bone joints active — secondary motion (hair/skirt/bust) will be dead!");
            } else {
              console.log(`[AvatarCanvas] SpringBone physics active: ${springJoints} joints, ${springColliders} collider groups`);
            }
            // Inspectable from devtools / automated checks
            (window as any).__sylphSpringBones = { joints: springJoints, colliderGroups: springColliders };
            sidecarSocket.send("log", {
              level: springJoints === 0 ? "warn" : "info",
              message: `[AvatarCanvas] VRM loaded. Expressions: ${expressions.join(", ")}; springBoneJoints=${springJoints}`
            });

            resolve(vrm);
          },
          (progress) => {
            const pct = ((progress.loaded / progress.total) * 100).toFixed(0);
            console.log(`[AvatarCanvas] Loading VRM: ${pct}%`);
          },
          (error) => {
            console.error("[AvatarCanvas] Failed to load VRM:", error);
            reject(error);
          }
        );
      });
    },
    [vrmUrl]
  );

  useEffect(() => {
    const sceneSetup = setupScene();
    if (!sceneSetup) return;

    const { scene, camera, renderer } = sceneSetup;

    // Load VRM and start render loop
    loadVRM(scene)
      .then((vrm) => {
        // Initialize expression driver (Phase 1.6)
        expressionDriverRef.current = new ExpressionDriver(vrm);

        // Initialize idle animations (Phase 1.5)
        idleControllerRef.current = new IdleAnimationController(vrm);

        // Initialize gesture layer (mood/speech-tied body movement)
        gestureControllerRef.current = new GestureController(vrm);

        // Initialize viseme scheduler (Phase 4.6)
        visemeSchedulerRef.current = new VisemeScheduler();
        visemeSchedulerRef.current.setVRM(vrm);
        onVisemeSchedulerReady?.(visemeSchedulerRef.current);

        // Notify parent
        onVRMLoaded?.(vrm);

        // Start render loop
        const animate = () => {
          frameIdRef.current = requestAnimationFrame(animate);
          const delta = clockRef.current.getDelta();

          // IMPORTANT: Modifications MUST happen BEFORE vrm.update()
          // three-vrm v3 propagates normalized bones → raw bones
          // and applies expressions → mesh blendshapes inside update()

          // 1. Set bone rotations (rest pose, breathing, head look)
          idleControllerRef.current?.update(delta);

          // 2. Additive gesture layer (mood/speech body movement) — after
          //    idle so its offsets compose onto rest pose + breathing
          gestureControllerRef.current?.setTalking(audioPlayer.playing);
          gestureControllerRef.current?.update(delta);

          // 3. Set mood expression weights
          expressionDriverRef.current?.update(delta);

          // 4. Set viseme lip-sync weights
          visemeSchedulerRef.current?.update(delta);

          // 5. Propagate all changes to the mesh (incl. spring-bone physics)
          vrm.update(delta);

          renderer.render(scene, camera);
        };
        animate();
      })
      .catch((err) => {
        console.error("[AvatarCanvas] Initialization failed:", err);
      });

    // Listen for autonomous events (Phase 8/9/10)
    const unsubGlance = sidecarSocket.onMessage("avatar_glance", (payload) => {
      const x = payload.target_x as number;
      const y = payload.target_y as number;
      const duration = payload.duration as number;
      idleControllerRef.current?.setGlanceTarget(x, y, duration);
    });

    const unsubPosture = sidecarSocket.onMessage("avatar_posture", (payload) => {
      const shiftType = payload.shift_type as string;
      const duration = payload.duration as number;
      idleControllerRef.current?.setPostureShift(shiftType, duration);
    });

    // Conversational gaze aversion and posture shifts during thinking state
    const unsubThinking = sidecarSocket.onMessage("thinking", () => {
      // Look up and to the side randomly (cognitive retrieval look)
      const targetX = Math.random() > 0.5 ? -0.22 : 0.22;
      const targetY = 0.15;
      idleControllerRef.current?.setGlanceTarget(targetX, targetY, 2.0);

      // Trigger a thinking posture shift
      const shifts = ["head_tilt", "subtle_stretch"];
      const chosen = shifts[Math.floor(Math.random() * shifts.length)];
      idleControllerRef.current?.setPostureShift(chosen, 2.5);

      // Micro-expression: brief curious/surprised look while thinking
      expressionDriverRef.current?.flash({ surprised: 0.25, relaxed: 0.1 }, 0.5, 0.7);
    });

    // Engage user with eye contact when speaking starts + flash a subtle happy/alive expression
    const unsubTTSStart = sidecarSocket.onMessage("tts_start", (payload) => {
      idleControllerRef.current?.setGlanceTarget(0.0, 0.04, 1.2);
      // Micro-expression: brief happy burst when starting to speak
      expressionDriverRef.current?.flash({ happy: 0.35, relaxed: 0.2 }, 0.35, 0.55);
      // Body gesture cued by the utterance itself (hum/sigh/yawn stage
      // directions get dedicated motion; everything else a subtle lead-in)
      gestureControllerRef.current?.onSpeechStart((payload.text as string) ?? "");
    });

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup — thorough teardown for React 18 Strict Mode
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(frameIdRef.current);
      unsubGlance();
      unsubPosture();
      unsubThinking();
      unsubTTSStart();
      gestureControllerRef.current?.dispose();
      idleControllerRef.current?.dispose();
      renderer.forceContextLoss();
      renderer.dispose();
      if (containerRef.current) {
        while (containerRef.current.firstChild) {
          containerRef.current.removeChild(containerRef.current.firstChild);
        }
      }
    };
  }, [setupScene, loadVRM, onVRMLoaded]);

  return (
    <div
      ref={containerRef}
      id="avatar-canvas"
      style={{
        width: "100%",
        height: "100%",
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    />
  );
}
