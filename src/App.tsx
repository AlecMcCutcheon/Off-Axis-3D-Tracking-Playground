import React, { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { EffectComposer, Vignette } from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import { FaceLandmarker, HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { FaceLandmarkerResult, HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import * as THREE from "three";

interface HeadPose {
  x: number;
  y: number;
  z: number;
  hasFace: boolean;
}

interface HandPos {
  x: number;
  y: number;
  z: number;
  hasHand: boolean;
}

interface PinchData {
  thumbNX: number;  // thumb tip, camera-normalized (0–1)
  thumbNY: number;
  indexNX: number;  // index tip, camera-normalized (0–1)
  indexNY: number;
  pinchNZ: number;  // hand-average depth in camera-normalized space
  handScaleN: number; // apparent hand size in camera-normalized space
  faceScaleN: number; // face eye-distance reference scale
  isPinching: boolean;
  hasHand: boolean;
}

interface TrackingState {
  headPose: HeadPose;
  handPos: HandPos;
  pinchData: PinchData;
  loading: boolean;
  error: string | null;
}

interface SceneParams {
  sensitivity: number;
  depthCalibration: number;
  cameraOffsetMode: CameraOffsetMode;
  cameraOffsetN: number;
}

type CameraOffsetMode = "auto" | "center" | "left" | "right";
type ObjectCollectionId = "starter" | "dense" | "chaos";

interface PersistedAppSettings {
  sensitivity: number;
  depthCalibration: number;
  cameraOffsetMode: CameraOffsetMode;
  cameraOffsetN: number;
  fingerGrab: boolean;
  roomPhysicsEnabled: boolean;
  gravityForceEnabled: boolean;
  gravityStrength: number;
  restitution: number;
  throwInertiaEnabled: boolean;
  objectCollection: ObjectCollectionId;
  settingsCollapsed: boolean;
  advancedSettingsExpanded: boolean;
}

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 360;

const SCREEN_WIDTH_CM = 30.0;
const SCREEN_HEIGHT_CM = 20.0;
const BOX_DEPTH_CM = 80.0;
const DEFAULT_HEAD_Z_CM = 60.0;
const NEAR = 0.5;
const FAR = 1000.0;
const LIMIT_X = 50;
const LIMIT_Y = 50;
const MIN_HEAD_Z_CM = 15;
const MAX_HEAD_Z_CM = 120;

const initialHeadPose: HeadPose = { x: 0, y: 0, z: DEFAULT_HEAD_Z_CM, hasFace: false };
const defaultParams: SceneParams = {
  sensitivity: 1.3,
  depthCalibration: 4.5,
  cameraOffsetMode: "auto",
  cameraOffsetN: 0,
};
const DEFAULT_FINGER_GRAB = false;
const DEFAULT_ROOM_PHYSICS_ENABLED = false;
const DEFAULT_GRAVITY_FORCE_ENABLED = false;
const DEFAULT_GRAVITY_STRENGTH = 98;
const DEFAULT_RESTITUTION = 0.62;
const DEFAULT_THROW_INERTIA_ENABLED = false;
const DEFAULT_OBJECT_COLLECTION: ObjectCollectionId = "starter";
const DEFAULT_SETTINGS_COLLAPSED = true;
const DEFAULT_ADVANCED_EXPANDED = false;
const SETTINGS_STORAGE_KEY = "off-axis-3d-playground.settings.v2";
const SETTINGS_STORAGE_KEY_LEGACY_V1 = "off-axis-3d-playground.settings.v1";

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
const CAMERA_MODE_BASE_OFFSET: Record<CameraOffsetMode, number> = {
  auto: 0,
  center: 0,
  left: 0.09,
  right: -0.09,
};
const getCameraCenterNX = (mode: CameraOffsetMode, offsetN: number) =>
  clamp(0.5 + CAMERA_MODE_BASE_OFFSET[mode] + offsetN, 0.12, 0.88);

const getDefaultPersistedSettings = (): PersistedAppSettings => ({
  sensitivity: defaultParams.sensitivity,
  depthCalibration: defaultParams.depthCalibration,
  cameraOffsetMode: defaultParams.cameraOffsetMode,
  cameraOffsetN: defaultParams.cameraOffsetN,
  fingerGrab: DEFAULT_FINGER_GRAB,
  roomPhysicsEnabled: DEFAULT_ROOM_PHYSICS_ENABLED,
  gravityForceEnabled: DEFAULT_GRAVITY_FORCE_ENABLED,
  gravityStrength: DEFAULT_GRAVITY_STRENGTH,
  restitution: DEFAULT_RESTITUTION,
  throwInertiaEnabled: DEFAULT_THROW_INERTIA_ENABLED,
  objectCollection: DEFAULT_OBJECT_COLLECTION,
  settingsCollapsed: DEFAULT_SETTINGS_COLLAPSED,
  advancedSettingsExpanded: DEFAULT_ADVANCED_EXPANDED,
});

/** v1 used `gravityEnabled` for the whole room-physics toggle. */
interface LegacyV1Settings {
  sensitivity?: number;
  depthCalibration?: number;
  cameraOffsetMode?: CameraOffsetMode;
  cameraOffsetN?: number;
  fingerGrab?: boolean;
  gravityEnabled?: boolean;
  gravityStrength?: number;
  restitution?: number;
  objectCollection?: ObjectCollectionId;
  settingsCollapsed?: boolean;
}

const readPersistedSettings = (): PersistedAppSettings => {
  const defaults = getDefaultPersistedSettings();
  if (typeof window === "undefined") return defaults;
  try {
    let raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    let legacyV1: LegacyV1Settings | null = null;
    if (!raw) {
      raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY_LEGACY_V1);
      if (raw) {
        legacyV1 = JSON.parse(raw) as LegacyV1Settings;
      } else {
        return defaults;
      }
    }
    const parsed = legacyV1 ?? (JSON.parse(raw!) as Partial<PersistedAppSettings>);
    const cameraOffsetMode =
      parsed.cameraOffsetMode === "auto" ||
      parsed.cameraOffsetMode === "center" ||
      parsed.cameraOffsetMode === "left" ||
      parsed.cameraOffsetMode === "right"
        ? parsed.cameraOffsetMode
        : defaults.cameraOffsetMode;
    const objectCollection =
      parsed.objectCollection === "starter" ||
      parsed.objectCollection === "dense" ||
      parsed.objectCollection === "chaos"
        ? parsed.objectCollection
        : defaults.objectCollection;

    const legacyGravity =
      legacyV1 && typeof legacyV1.gravityEnabled === "boolean"
        ? legacyV1.gravityEnabled
        : undefined;
    const legacyStrength =
      legacyV1 && typeof legacyV1.gravityStrength === "number" && Number.isFinite(legacyV1.gravityStrength)
        ? legacyV1.gravityStrength
        : undefined;

    const roomPhysicsEnabled =
      typeof parsed.roomPhysicsEnabled === "boolean"
        ? parsed.roomPhysicsEnabled
        : legacyGravity !== undefined
          ? legacyGravity
          : defaults.roomPhysicsEnabled;

    const gravityForceEnabled =
      typeof parsed.gravityForceEnabled === "boolean"
        ? parsed.gravityForceEnabled
        : legacyGravity !== undefined
          ? legacyGravity && (legacyStrength ?? 0) > 0
          : defaults.gravityForceEnabled;

    const throwInertiaEnabled =
      typeof parsed.throwInertiaEnabled === "boolean"
        ? parsed.throwInertiaEnabled
        : legacyV1
          ? true
          : defaults.throwInertiaEnabled;

    const advancedSettingsExpanded =
      typeof parsed.advancedSettingsExpanded === "boolean"
        ? parsed.advancedSettingsExpanded
        : defaults.advancedSettingsExpanded;

    return {
      sensitivity:
        typeof parsed.sensitivity === "number" && Number.isFinite(parsed.sensitivity)
          ? clamp(parsed.sensitivity, 0.5, 5)
          : defaults.sensitivity,
      depthCalibration:
        typeof parsed.depthCalibration === "number" && Number.isFinite(parsed.depthCalibration)
          ? clamp(parsed.depthCalibration, 1, 12)
          : defaults.depthCalibration,
      cameraOffsetMode,
      cameraOffsetN:
        typeof parsed.cameraOffsetN === "number" && Number.isFinite(parsed.cameraOffsetN)
          ? clamp(parsed.cameraOffsetN, -0.3, 0.3)
          : defaults.cameraOffsetN,
      fingerGrab:
        typeof parsed.fingerGrab === "boolean" ? parsed.fingerGrab : defaults.fingerGrab,
      roomPhysicsEnabled,
      gravityForceEnabled,
      gravityStrength:
        typeof parsed.gravityStrength === "number" && Number.isFinite(parsed.gravityStrength)
          ? clamp(parsed.gravityStrength, 0, 220)
          : defaults.gravityStrength,
      restitution:
        typeof parsed.restitution === "number" && Number.isFinite(parsed.restitution)
          ? clamp(parsed.restitution, 0, 0.95)
          : defaults.restitution,
      throwInertiaEnabled,
      objectCollection,
      settingsCollapsed:
        typeof parsed.settingsCollapsed === "boolean"
          ? parsed.settingsCollapsed
          : defaults.settingsCollapsed,
      advancedSettingsExpanded,
    };
  } catch {
    return defaults;
  }
};

/** Thumb-index tip distance (normalized); hysteresis keeps pinch and overlay stable. */
const PINCH_ON = 0.076;
const PINCH_OFF = 0.095;

/** MediaPipe 0–1 → virtual screen (cm); (0.5−n) matches mirrored selfie preview. */
function cameraNormToScreenCm(nx: number, ny: number, gain: number): { x: number; y: number } {
  const x = clamp((0.5 - nx) * SCREEN_WIDTH_CM * gain, -LIMIT_X, LIMIT_X);
  const y = clamp((0.5 - ny) * SCREEN_HEIGHT_CM * gain, -LIMIT_Y, LIMIT_Y);
  return { x, y };
}

/** Average hand depth from all landmarks for a steadier near/far signal. */
function getHandAverageDepthN(handLandmarks: NormalizedLandmark[]): number {
  if (!handLandmarks.length) return 0;
  let weightedSum = 0;
  let totalWeight = 0;
  for (let idx = 0; idx < handLandmarks.length; idx++) {
    let weight = 1.0;
    // Downweight thumb/index finger joints so pinch articulation does not
    // dominate depth estimation.
    if ((idx >= 1 && idx <= 4) || (idx >= 6 && idx <= 8)) weight = 0.25;
    if (idx === 5) weight = 0.5;
    weightedSum += handLandmarks[idx].z * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

// ── Overlay drawing ──────────────────────────────────────────────────────────
// Canvas has NO CSS transform; x is mirrored by (W - lm.x*W) to match the CSS-mirrored video.
function drawOverlay(
  canvas: HTMLCanvasElement,
  faceLandmarks: NormalizedLandmark[] | undefined,
  iris: { x: number; y: number } | null,
  handLandmarks: NormalizedLandmark[] | undefined,
  pinchHint: { isPinching: boolean; pinchDist: number } | null,
  cameraCenterNX: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // mirror x so overlay matches CSS scaleX(-1) on the video
  const px = (x: number) => (1 - x) * W;
  const py = (y: number) => y * H;
  const guideX = px(cameraCenterNX);

  // Dotted vertical guide for camera-center alignment calibration.
  ctx.save();
  ctx.strokeStyle = "rgba(120, 190, 255, 0.75)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 6]);
  ctx.beginPath();
  ctx.moveTo(guideX, 0);
  ctx.lineTo(guideX, H);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(120, 190, 255, 0.8)";
  ctx.beginPath();
  ctx.arc(guideX, H * 0.5, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (faceLandmarks) {
    ctx.fillStyle = "rgba(80,180,255,0.75)";
    for (const lm of faceLandmarks) {
      ctx.beginPath();
      ctx.arc(px(lm.x), py(lm.y), 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (iris) {
    ctx.fillStyle = "#30ff84";
    ctx.beginPath();
    ctx.arc(px(iris.x), py(iris.y), 5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (handLandmarks && handLandmarks.length >= 9) {
    // Pass 1 — all joints as yellow dots; capture thumb (idx 4) and index (idx 8)
    // during the same iteration so bracket-access issues can't hide them.
    let t:  NormalizedLandmark | null = null;
    let i8: NormalizedLandmark | null = null;
    let lmIdx = 0;
    ctx.fillStyle = "rgba(255,210,40,0.85)";
    for (const lm of handLandmarks) {
      ctx.beginPath();
      ctx.arc(px(lm.x), py(lm.y), 3, 0, Math.PI * 2);
      ctx.fill();
      if (lmIdx === 4) t  = lm;
      if (lmIdx === 8) i8 = lm;
      lmIdx++;
    }

    // Pass 2 — thumb and index drawn larger on top with line between them
    if (
      t &&
      i8 &&
      Number.isFinite(t.x) &&
      Number.isFinite(t.y) &&
      Number.isFinite(i8.x) &&
      Number.isFinite(i8.y)
    ) {
      const tx = px(t.x), ty = py(t.y);
      const ix = px(i8.x), iy = py(i8.y);
      const midX = (tx + ix) / 2;
      const midY = (ty + iy) / 2;
      const pinching = pinchHint?.isPinching ?? false;
      const lineCol  = pinching ? "#ff2255" : "#ffe040";

      // Line thumb → index
      ctx.strokeStyle = lineCol;
      ctx.lineWidth   = pinching ? 5 : 3;
      ctx.lineCap     = "round";
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(ix, iy);
      ctx.stroke();

      // Thumb — large orange dot with white ring
      ctx.fillStyle   = "#ff8800";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 3;
      ctx.beginPath();
      ctx.arc(tx, ty, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Index — large cyan dot with white ring
      ctx.fillStyle   = "#00ccff";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 3;
      ctx.beginPath();
      ctx.arc(ix, iy, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Midpoint ring (changes color on pinch)
      ctx.strokeStyle = lineCol;
      ctx.fillStyle   = pinching ? "rgba(255,30,80,0.30)" : "rgba(255,224,64,0.15)";
      ctx.lineWidth   = 3;
      ctx.beginPath();
      ctx.arc(midX, midY, pinching ? 15 : 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Status label at bottom of preview
      ctx.fillStyle = pinching ? "#ff5599" : "#c8d8ff";
      ctx.font      = "bold 12px ui-sans-serif, system-ui, sans-serif";
      const dist = pinchHint?.pinchDist ?? 0;
      ctx.fillText(
        pinching ? "● PINCH detected" : `○ open  d=${dist.toFixed(3)}`,
        6, H - 8
      );
    }
  }
}

// ── Iris extraction ──────────────────────────────────────────────────────────
const extractIris = (lms: NormalizedLandmark[] | undefined) => {
  if (!lms?.length) return { cx: 0.5, cy: 0.5, eyeDist: 0, ok: false };
  const l = lms[468] ?? lms[33];
  const r = lms[473] ?? lms[263];
  if (!l || !r) return { cx: 0.5, cy: 0.5, eyeDist: 0, ok: false };
  return {
    cx: (l.x + r.x) * 0.5,
    cy: (l.y + r.y) * 0.5,
    eyeDist: Math.hypot(l.x - r.x, l.y - r.y),
    ok: true
  };
};

const initialHandPos: HandPos = { x: 0, y: 0, z: 0, hasHand: false };
const initialPinch: PinchData = {
  thumbNX: 0.5, thumbNY: 0.5,
  indexNX: 0.5, indexNY: 0.5,
  pinchNZ: 0,
  handScaleN: 0,
  faceScaleN: 0,
  isPinching: false, hasHand: false,
};

// ── Face + Hand tracking hook ────────────────────────────────────────────────
const useFaceTracking = (params: SceneParams, autoRecalibrateToken: number): {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  trackingState: TrackingState;
  autoCalibrationLocked: boolean;
  autoCalibrationProgress: number;
} => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [autoCalibrationLocked, setAutoCalibrationLocked] = useState(false);
  const [autoCalibrationProgress, setAutoCalibrationProgress] = useState(0);
  const [trackingState, setTrackingState] = useState<TrackingState>({
    headPose: initialHeadPose, handPos: initialHandPos, pinchData: initialPinch, loading: true, error: null
  });
  const currentHead = useRef<HeadPose>(initialHeadPose);
  const currentHand = useRef<HandPos>(initialHandPos);
  const currentPinch = useRef<PinchData>(initialPinch);
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const pinchLatch = useRef(false);
  const HAND_POS_LERP = 0.24;
  const PINCH_LERP = 0.26;
  const PINCH_Z_LERP = 0.18;
  const PINCH_HAND_SCALE_LERP = 0.22;
  const FACE_SCALE_LERP = 0.16;
  const AUTO_CALIBRATION_DELAY_MS = 3000;
  const AUTO_STABLE_ENTER_N = 0.0012;
  const AUTO_STABLE_EXIT_N = 0.0018;
  const AUTO_STABLE_HOLD_MS = 2600;
  const faceScaleRef = useRef<{ value: number; valid: boolean }>({ value: 0.065, valid: false });
  const autoCenterNXRef = useRef(0.5);
  const autoLockedRef = useRef(false);
  const autoStartDelayUntilMsRef = useRef<number | null>(null);
  const autoStableSinceMsRef = useRef<number | null>(null);
  const autoLastIrisRef = useRef<{ x: number; y: number } | null>(null);
  const autoStillLatchRef = useRef(false);
  const autoRecalibrateTokenRef = useRef(autoRecalibrateToken);
  const autoRecalibrateInputRef = useRef(autoRecalibrateToken);
  const autoProgressRef = useRef(0);
  autoRecalibrateInputRef.current = autoRecalibrateToken;

  useEffect(() => {
    let alive = true;
    let stream: MediaStream | null = null;
    let faceLandmarker: FaceLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let raf = 0;

    (async () => {
      try {
        // Try preferred constraints first; fall back to bare video if the
        // device doesn't match facingMode or resolution ideals.
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: VIDEO_WIDTH }, height: { ideal: VIDEO_HEIGHT }, facingMode: "user" },
            audio: false,
          });
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        if (!alive || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        [faceLandmarker, handLandmarker] = await Promise.all([
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.65,
            minTrackingConfidence: 0.65
          })
        ]);
        if (alive) setTrackingState(s => ({ ...s, loading: false }));

        const tick = () => {
          if (!alive || !videoRef.current || !faceLandmarker || !handLandmarker) return;
          const video = videoRef.current;
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) { raf = requestAnimationFrame(tick); return; }

          const canv = canvasRef.current;
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          if (canv && vw > 0 && vh > 0 && (canv.width !== vw || canv.height !== vh)) {
            canv.width = vw;
            canv.height = vh;
          }

          const now = performance.now();
          const faceResult: FaceLandmarkerResult = faceLandmarker.detectForVideo(video, now);
          const handResult: HandLandmarkerResult = handLandmarker.detectForVideo(video, now);

          const lms = faceResult.faceLandmarks[0];
          const iris = extractIris(lms);
          const handLms = handResult.landmarks[0];
          const { sensitivity, depthCalibration, cameraOffsetMode, cameraOffsetN } = paramsRef.current;
          if (autoRecalibrateTokenRef.current !== autoRecalibrateInputRef.current) {
            autoRecalibrateTokenRef.current = autoRecalibrateInputRef.current;
            autoCenterNXRef.current = 0.5;
            autoLockedRef.current = false;
            autoStartDelayUntilMsRef.current = now + AUTO_CALIBRATION_DELAY_MS;
            autoStableSinceMsRef.current = null;
            autoLastIrisRef.current = null;
            autoStillLatchRef.current = false;
            setAutoCalibrationLocked(false);
            setAutoCalibrationProgress(0);
            autoProgressRef.current = 0;
          }
          let cameraCenterBaseNX = getCameraCenterNX(cameraOffsetMode, 0);
          if (cameraOffsetMode === "auto") {
            if (autoStartDelayUntilMsRef.current == null) {
              autoStartDelayUntilMsRef.current = now + AUTO_CALIBRATION_DELAY_MS;
            }
            cameraCenterBaseNX = autoLockedRef.current ? autoCenterNXRef.current : 0.5;
          } else {
            autoStartDelayUntilMsRef.current = null;
            if (autoProgressRef.current !== 0) {
              autoProgressRef.current = 0;
              setAutoCalibrationProgress(0);
            }
            if (autoCalibrationLocked) {
              setAutoCalibrationLocked(false);
            }
          }
          const cameraCenterNX = clamp(cameraCenterBaseNX + cameraOffsetN, 0.12, 0.88);

          let pinchOverlay: { isPinching: boolean; pinchDist: number } | null = null;
          let pinch: PinchData = { ...initialPinch, hasHand: false };
          if (!handLms || handLms.length < 9) {
            pinchLatch.current = false;
            currentHand.current = { ...currentHand.current, hasHand: false };
            currentPinch.current = { ...currentPinch.current, isPinching: false, hasHand: false };
          } else {
            const palmLm = handLms[9];
            const { x: hx, y: hy } = cameraNormToScreenCm(palmLm.x, palmLm.y, sensitivity);
            currentHand.current = {
              x: THREE.MathUtils.lerp(currentHand.current.x, hx, HAND_POS_LERP),
              y: THREE.MathUtils.lerp(currentHand.current.y, hy, HAND_POS_LERP),
              z: 0,
              hasHand: true,
            };

            const thumb = handLms[4];
            const index = handLms[8];
            const indexMcp = handLms[5];
            const pinkyMcp = handLms[17];
            const wrist = handLms[0];
            const middleMcp = handLms[9];
            const handAverageDepthN = getHandAverageDepthN(handLms);
            const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
            const palmWidth = Math.hypot(indexMcp.x - pinkyMcp.x, indexMcp.y - pinkyMcp.y);
            const palmHeight = Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y);
            const handScaleN = (palmWidth + palmHeight) * 0.5;
            if (!pinchLatch.current) {
              if (pinchDist < PINCH_ON) pinchLatch.current = true;
            } else if (pinchDist > PINCH_OFF) {
              pinchLatch.current = false;
            }
            const rawPinch: PinchData = {
              thumbNX: thumb.x,
              thumbNY: thumb.y,
              indexNX: index.x,
              indexNY: index.y,
              pinchNZ: handAverageDepthN,
              handScaleN,
              faceScaleN: faceScaleRef.current.value,
              isPinching: pinchLatch.current,
              hasHand: true,
            };
            if (!currentPinch.current.hasHand) {
              currentPinch.current = rawPinch;
            } else {
              currentPinch.current = {
                thumbNX: THREE.MathUtils.lerp(currentPinch.current.thumbNX, rawPinch.thumbNX, PINCH_LERP),
                thumbNY: THREE.MathUtils.lerp(currentPinch.current.thumbNY, rawPinch.thumbNY, PINCH_LERP),
                indexNX: THREE.MathUtils.lerp(currentPinch.current.indexNX, rawPinch.indexNX, PINCH_LERP),
                indexNY: THREE.MathUtils.lerp(currentPinch.current.indexNY, rawPinch.indexNY, PINCH_LERP),
                pinchNZ: THREE.MathUtils.lerp(currentPinch.current.pinchNZ, rawPinch.pinchNZ, PINCH_Z_LERP),
                handScaleN: THREE.MathUtils.lerp(currentPinch.current.handScaleN, rawPinch.handScaleN, PINCH_HAND_SCALE_LERP),
                faceScaleN: THREE.MathUtils.lerp(currentPinch.current.faceScaleN, rawPinch.faceScaleN, FACE_SCALE_LERP),
                isPinching: rawPinch.isPinching,
                hasHand: true,
              };
            }
            pinch = currentPinch.current;
            pinchOverlay = { isPinching: pinchLatch.current, pinchDist };
          }

          if (canvasRef.current) {
            drawOverlay(
              canvasRef.current,
              lms,
              iris.ok ? { x: iris.cx, y: iris.cy } : null,
              handLms,
              pinchOverlay,
              cameraCenterNX
            );
          }

          if (!iris.ok || iris.eyeDist < 0.00001) {
            currentHead.current = { ...currentHead.current, hasFace: false };
            if (cameraOffsetMode === "auto" && !autoLockedRef.current) {
              autoStableSinceMsRef.current = null;
              autoLastIrisRef.current = null;
              autoStillLatchRef.current = false;
              if (autoProgressRef.current !== 0) {
                autoProgressRef.current = 0;
                setAutoCalibrationProgress(0);
              }
            }
          } else {
            if (cameraOffsetMode === "auto" && !autoLockedRef.current) {
              const calibrationDelayRemainingMs =
                (autoStartDelayUntilMsRef.current ?? now) - now;
              if (calibrationDelayRemainingMs > 0) {
                autoStableSinceMsRef.current = null;
                autoLastIrisRef.current = { x: iris.cx, y: iris.cy };
                autoStillLatchRef.current = false;
                if (autoProgressRef.current !== 0) {
                  autoProgressRef.current = 0;
                  setAutoCalibrationProgress(0);
                }
                faceScaleRef.current.value = THREE.MathUtils.lerp(
                  faceScaleRef.current.value,
                  iris.eyeDist,
                  FACE_SCALE_LERP
                );
                faceScaleRef.current.valid = true;
                const tx = clamp((cameraCenterNX - iris.cx) * SCREEN_WIDTH_CM * sensitivity, -LIMIT_X, LIMIT_X);
                const ty = clamp((0.5 - iris.cy) * SCREEN_HEIGHT_CM * sensitivity * 0.8, -LIMIT_Y, LIMIT_Y);
                const tz = clamp(depthCalibration / iris.eyeDist, MIN_HEAD_Z_CM, MAX_HEAD_Z_CM);
                currentHead.current = { x: tx, y: ty, z: tz, hasFace: true };
                setTrackingState({ headPose: currentHead.current, handPos: currentHand.current, pinchData: pinch, loading: false, error: null });
                raf = requestAnimationFrame(tick);
                return;
              }
              const prev = autoLastIrisRef.current ?? { x: iris.cx, y: iris.cy };
              const motion = Math.hypot(iris.cx - prev.x, iris.cy - prev.y);
              const stableGate = autoStillLatchRef.current
                ? AUTO_STABLE_EXIT_N
                : AUTO_STABLE_ENTER_N;
              const withinStableDeadzone = motion <= stableGate;
              autoStillLatchRef.current = withinStableDeadzone;
              autoLastIrisRef.current = { x: iris.cx, y: iris.cy };
              if (withinStableDeadzone) {
                autoStableSinceMsRef.current = autoStableSinceMsRef.current ?? now;
              } else {
                autoStableSinceMsRef.current = null;
              }

              const stableForMs =
                autoStableSinceMsRef.current == null ? 0 : now - autoStableSinceMsRef.current;
              const progress = clamp(stableForMs / AUTO_STABLE_HOLD_MS, 0, 1);
              const quantizedProgress = Math.round(progress * 100) / 100;
              if (Math.abs(autoProgressRef.current - quantizedProgress) >= 0.01) {
                autoProgressRef.current = quantizedProgress;
                setAutoCalibrationProgress(quantizedProgress);
              }

              if (stableForMs >= AUTO_STABLE_HOLD_MS) {
                autoCenterNXRef.current = clamp(iris.cx, 0.12, 0.88);
                autoLockedRef.current = true;
                setAutoCalibrationLocked(true);
                autoProgressRef.current = 1;
                setAutoCalibrationProgress(1);
              }
            }
            faceScaleRef.current.value = THREE.MathUtils.lerp(
              faceScaleRef.current.value,
              iris.eyeDist,
              FACE_SCALE_LERP
            );
            faceScaleRef.current.valid = true;
            const tx = clamp((cameraCenterNX - iris.cx) * SCREEN_WIDTH_CM * sensitivity, -LIMIT_X, LIMIT_X);
            const ty = clamp((0.5 - iris.cy) * SCREEN_HEIGHT_CM * sensitivity * 0.8, -LIMIT_Y, LIMIT_Y);
            const tz = clamp(depthCalibration / iris.eyeDist, MIN_HEAD_Z_CM, MAX_HEAD_Z_CM);
            currentHead.current = { x: tx, y: ty, z: tz, hasFace: true };
          }

          setTrackingState({ headPose: currentHead.current, handPos: currentHand.current, pinchData: pinch, loading: false, error: null });
          raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      } catch (err) {
        if (!alive) return;
        setTrackingState({ headPose: initialHeadPose, handPos: initialHandPos, pinchData: initialPinch, loading: false,
          error: err instanceof Error ? err.message : "Camera error" });
      }
    })();

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      faceLandmarker?.close();
      handLandmarker?.close();
      stream?.getTracks().forEach(t => t.stop());
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    trackingState,
    autoCalibrationLocked,
    autoCalibrationProgress,
  };
};

// ── Off-axis camera rig ──────────────────────────────────────────────────────
const CameraRig = ({ headPose }: { headPose: HeadPose }) => {
  const { camera } = useThree();
  const headRef = useRef(headPose);
  headRef.current = headPose;

  const filtX = useRef(0);
  const filtY = useRef(0);
  const filtZ = useRef(DEFAULT_HEAD_Z_CM);
  const MAX_DELTA_XY = 2.5;
  const MAX_DELTA_Z  = 5;
  const XY_LERP = 0.22;
  const Z_LERP = 0.08;
  const XY_DEADZONE_CM = 0.6;
  const Z_DEADZONE_CM = 2.2;

  // Ring buffer: always read oldest slot. On tracking loss the buffer
  // freezes so we hold a value from ~6 frames before the flail began.
  const BUFFER_SIZE = 6;
  const posBuffer = useRef(
    Array.from({ length: BUFFER_SIZE }, () => ({ x: 0, y: 0, z: DEFAULT_HEAD_Z_CM }))
  );

  useEffect(() => {
    (camera as THREE.PerspectiveCamera).updateProjectionMatrix = () => {};
  }, [camera]);

  useFrame(() => {
    const h = headRef.current;

    if (h.hasFace) {
      const dx = h.x - filtX.current;
      const dy = h.y - filtY.current;
      const dz = h.z - filtZ.current;

      const targetX = Math.abs(dx) <= XY_DEADZONE_CM
        ? filtX.current
        : filtX.current + clamp(dx, -MAX_DELTA_XY, MAX_DELTA_XY);
      const targetY = Math.abs(dy) <= XY_DEADZONE_CM
        ? filtY.current
        : filtY.current + clamp(dy, -MAX_DELTA_XY, MAX_DELTA_XY);
      const targetZ = Math.abs(dz) <= Z_DEADZONE_CM
        ? filtZ.current
        : filtZ.current + clamp(dz, -MAX_DELTA_Z, MAX_DELTA_Z);

      filtX.current = THREE.MathUtils.lerp(filtX.current, targetX, XY_LERP);
      filtY.current = THREE.MathUtils.lerp(filtY.current, targetY, XY_LERP);
      filtZ.current = THREE.MathUtils.lerp(filtZ.current, targetZ, Z_LERP);
      posBuffer.current.push({ x: filtX.current, y: filtY.current, z: filtZ.current });
      posBuffer.current.shift();
    }

    const dp = posBuffer.current[0];
    const dist = Math.max(dp.z, 0.1);

    camera.position.set(dp.x, dp.y, dist);
    camera.quaternion.identity();
    const hw = SCREEN_WIDTH_CM / 2;
    const hh = SCREEN_HEIGHT_CM / 2;
    const s = NEAR / dist;
    camera.projectionMatrix.makePerspective(
      (-hw - dp.x) * s, (hw - dp.x) * s,
      (hh - dp.y) * s, (-hh - dp.y) * s,
      NEAR, FAR
    );
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  });

  return null;
};

// ── Global registry of draggable meshes (for finger-grab raycasting) ─────────
const draggableRegistry = new Set<THREE.Mesh>();
const fingerGrabbedRegistry = new Set<THREE.Mesh>();
const ROOM_HALF_WIDTH = SCREEN_WIDTH_CM / 2;
const ROOM_HALF_HEIGHT = SCREEN_HEIGHT_CM / 2;
// Push the front collision wall far enough that front bounces happen out of
// view for normal head/camera positions.
const ROOM_FRONT_Z = 140;
const ROOM_BACK_Z = -BOX_DEPTH_CM;
const ROOM_COLLISION_EPS = 0.1;
const VIEWPLANE_GUARD_Z = -0.8;
const VIEWPLANE_SIDE_INSET = 0.7;

interface MeshWorldExtrema {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
  minXPoint: THREE.Vector3;
  maxXPoint: THREE.Vector3;
  minYPoint: THREE.Vector3;
  maxYPoint: THREE.Vector3;
  minZPoint: THREE.Vector3;
  maxZPoint: THREE.Vector3;
  center: THREE.Vector3;
}

const getMeshWorldExtrema = (mesh: THREE.Mesh): MeshWorldExtrema => {
  mesh.updateMatrixWorld(true);
  const center = new THREE.Vector3();
  mesh.getWorldPosition(center);
  const pos = mesh.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;

  if (!pos || pos.count === 0) {
    const bounds = new THREE.Box3().setFromObject(mesh);
    return {
      minX: bounds.min.x,
      maxX: bounds.max.x,
      minY: bounds.min.y,
      maxY: bounds.max.y,
      minZ: bounds.min.z,
      maxZ: bounds.max.z,
      minXPoint: new THREE.Vector3(bounds.min.x, center.y, center.z),
      maxXPoint: new THREE.Vector3(bounds.max.x, center.y, center.z),
      minYPoint: new THREE.Vector3(center.x, bounds.min.y, center.z),
      maxYPoint: new THREE.Vector3(center.x, bounds.max.y, center.z),
      minZPoint: new THREE.Vector3(center.x, center.y, bounds.min.z),
      maxZPoint: new THREE.Vector3(center.x, center.y, bounds.max.z),
      center,
    };
  }

  const p = new THREE.Vector3();
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  const minXPoint = new THREE.Vector3();
  const maxXPoint = new THREE.Vector3();
  const minYPoint = new THREE.Vector3();
  const maxYPoint = new THREE.Vector3();
  const minZPoint = new THREE.Vector3();
  const maxZPoint = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    p.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
    if (p.x < minX) { minX = p.x; minXPoint.copy(p); }
    if (p.x > maxX) { maxX = p.x; maxXPoint.copy(p); }
    if (p.y < minY) { minY = p.y; minYPoint.copy(p); }
    if (p.y > maxY) { maxY = p.y; maxYPoint.copy(p); }
    if (p.z < minZ) { minZ = p.z; minZPoint.copy(p); }
    if (p.z > maxZ) { maxZ = p.z; maxZPoint.copy(p); }
  }

  return {
    minX, maxX, minY, maxY, minZ, maxZ,
    minXPoint, maxXPoint, minYPoint, maxYPoint, minZPoint, maxZPoint,
    center,
  };
};

const clampMeshToRoom = (
  mesh: THREE.Mesh,
  frontZLimit = ROOM_FRONT_Z - ROOM_COLLISION_EPS,
  backZLimit = ROOM_BACK_Z + ROOM_COLLISION_EPS
) => {
  const ext = getMeshWorldExtrema(mesh);
  const minX = -ROOM_HALF_WIDTH;
  const maxX = ROOM_HALF_WIDTH;
  const minY = -ROOM_HALF_HEIGHT;
  const maxY = ROOM_HALF_HEIGHT;
  const minZ = backZLimit;
  const maxZ = frontZLimit;
  if (ext.minX < minX) mesh.position.x += minX - ext.minX;
  if (ext.maxX > maxX) mesh.position.x -= ext.maxX - maxX;
  if (ext.minY < minY) mesh.position.y += minY - ext.minY;
  if (ext.maxY > maxY) mesh.position.y -= ext.maxY - maxY;
  if (ext.minZ < minZ) mesh.position.z += minZ - ext.minZ;
  if (ext.maxZ > maxZ) mesh.position.z -= ext.maxZ - maxZ;
};

const getMeshVelocity = (mesh: THREE.Mesh): THREE.Vector3 => {
  const bag = mesh.userData as { __velocity?: THREE.Vector3 };
  if (!bag.__velocity) bag.__velocity = new THREE.Vector3();
  return bag.__velocity;
};

const getMeshAngularVelocity = (mesh: THREE.Mesh): THREE.Vector3 => {
  const bag = mesh.userData as { __angularVelocity?: THREE.Vector3 };
  if (!bag.__angularVelocity) bag.__angularVelocity = new THREE.Vector3();
  return bag.__angularVelocity;
};

interface MeshPhysics {
  mass: number;
  linearDamping: number;
  angularDamping: number;
  surfaceFriction: number;
}

const getMeshPhysics = (mesh: THREE.Mesh): MeshPhysics => {
  const bag = mesh.userData as { __physics?: MeshPhysics };
  return bag.__physics ?? {
    mass: 1,
    linearDamping: 0.22,
    angularDamping: 0.82,
    surfaceFriction: 0.45,
  };
};

const getMeshSleepState = (mesh: THREE.Mesh): { asleep: boolean; restFrames: number } => {
  const bag = mesh.userData as { __sleep?: { asleep: boolean; restFrames: number } };
  if (!bag.__sleep) bag.__sleep = { asleep: false, restFrames: 0 };
  return bag.__sleep;
};

const getMeshOscillationState = (
  mesh: THREE.Mesh
): {
  prevLinear: THREE.Vector3;
  prevAngular: THREE.Vector3;
  prevEnergy: number;
  chaosScore: number;
  dampFrames: number;
} => {
  const bag = mesh.userData as {
    __oscillation?: {
      prevLinear: THREE.Vector3;
      prevAngular: THREE.Vector3;
      prevEnergy: number;
      chaosScore: number;
      dampFrames: number;
    };
  };
  if (!bag.__oscillation) {
    bag.__oscillation = {
      prevLinear: new THREE.Vector3(),
      prevAngular: new THREE.Vector3(),
      prevEnergy: 0,
      chaosScore: 0,
      dampFrames: 0,
    };
  }
  return bag.__oscillation;
};

const wakeMesh = (mesh: THREE.Mesh) => {
  const sleep = getMeshSleepState(mesh);
  const osc = getMeshOscillationState(mesh);
  sleep.asleep = false;
  sleep.restFrames = 0;
  osc.prevLinear.set(0, 0, 0);
  osc.prevAngular.set(0, 0, 0);
  osc.prevEnergy = 0;
  osc.chaosScore = 0;
  osc.dampFrames = 0;
};

const getMeshLocalCenterOfMass = (mesh: THREE.Mesh): THREE.Vector3 => {
  const bag = mesh.userData as { __comLocal?: THREE.Vector3 };
  if (bag.__comLocal) return bag.__comLocal;
  const pos = mesh.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (!pos || pos.count === 0) {
    bag.__comLocal = new THREE.Vector3();
    return bag.__comLocal;
  }
  const com = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    com.x += pos.getX(i);
    com.y += pos.getY(i);
    com.z += pos.getZ(i);
  }
  com.multiplyScalar(1 / pos.count);
  bag.__comLocal = com;
  return com;
};

const getMeshWorldCenterOfMass = (mesh: THREE.Mesh): THREE.Vector3 => {
  return getMeshLocalCenterOfMass(mesh).clone().applyMatrix4(mesh.matrixWorld);
};

const getFloorSupportPoint = (mesh: THREE.Mesh, floorY: number): THREE.Vector3 => {
  const pos = mesh.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (!pos || pos.count === 0) return new THREE.Vector3(mesh.position.x, floorY, mesh.position.z);
  const p = new THREE.Vector3();
  const sum = new THREE.Vector3();
  let count = 0;
  const band = 0.18;
  for (let i = 0; i < pos.count; i++) {
    p.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
    if (p.y <= floorY + band) {
      sum.x += p.x;
      sum.z += p.z;
      count++;
    }
  }
  if (count > 0) return new THREE.Vector3(sum.x / count, floorY, sum.z / count);
  const ext = getMeshWorldExtrema(mesh);
  return new THREE.Vector3(ext.minYPoint.x, floorY, ext.minYPoint.z);
};

const getApproxInertiaFromExtrema = (ext: MeshWorldExtrema, mass: number): number => {
  const dx = Math.max(ext.maxX - ext.minX, 0.01);
  const dy = Math.max(ext.maxY - ext.minY, 0.01);
  const dz = Math.max(ext.maxZ - ext.minZ, 0.01);
  const radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
  // Sphere-like approximation: I = 2/5 * m * r^2
  return Math.max(0.4 * Math.max(mass, 0.2) * radius * radius, 0.02);
};

const getMeshApproxInertia = (mesh: THREE.Mesh, mass: number): number => {
  const ext = getMeshWorldExtrema(mesh);
  return getApproxInertiaFromExtrema(ext, mass);
};

const addAngularImpulse = (
  angularVelocity: THREE.Vector3 | undefined,
  leverArm: THREE.Vector3,
  linearImpulse: THREE.Vector3,
  impulseScale: number,
  inverseInertia = 1,
  maxSpin = 7
) => {
  if (!angularVelocity) return;
  const torque = new THREE.Vector3()
    .copy(leverArm)
    .cross(linearImpulse)
    .multiplyScalar(impulseScale * inverseInertia);
  angularVelocity.x = clamp(angularVelocity.x + torque.x, -maxSpin, maxSpin);
  angularVelocity.y = clamp(angularVelocity.y + torque.y, -maxSpin, maxSpin);
  angularVelocity.z = clamp(angularVelocity.z + torque.z, -maxSpin, maxSpin);
};

const resolveMeshRoomCollision = (
  mesh: THREE.Mesh,
  velocity: THREE.Vector3,
  restitution: number,
  surfaceFriction: number,
  mass: number,
  angularVelocity?: THREE.Vector3,
  frontZLimit = ROOM_FRONT_Z - ROOM_COLLISION_EPS,
  backZLimit = ROOM_BACK_Z + ROOM_COLLISION_EPS
) => {
  let ext = getMeshWorldExtrema(mesh);
  const minX = -ROOM_HALF_WIDTH;
  const maxX = ROOM_HALF_WIDTH;
  const minY = -ROOM_HALF_HEIGHT;
  const maxY = ROOM_HALF_HEIGHT;
  const minZ = backZLimit;
  const maxZ = frontZLimit;
  const tangentDamping = clamp(1 - surfaceFriction * 0.22, 0.45, 0.96);
  const collisionSpinScale = 0.22;
  const minBounceSpeed = 2.8;
  const effectiveRestitution = clamp(0.06 + restitution * 0.52, 0.06, 0.58);
  const lowSpeedRestitution = clamp(0.04 + restitution * 0.22, 0.04, 0.26);
  const reflectNormalVelocity = (normalVelocity: number) => {
    const speed = Math.abs(normalVelocity);
    const rest = speed < minBounceSpeed ? lowSpeedRestitution : effectiveRestitution;
    return -Math.sign(normalVelocity || 1) * speed * rest;
  };
  const floorSnapEps = 0.08;
  let touchingBoundary = false;
  let onFloor = false;

  if (ext.minX < minX) {
    touchingBoundary = true;
    mesh.position.x += minX - ext.minX;
    if (velocity.x < 0) {
      const preImpact = velocity.clone();
      velocity.x = reflectNormalVelocity(preImpact.x);
      velocity.y *= tangentDamping;
      velocity.z *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.minXPoint.clone();
      contact.x = minX;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  if (ext.maxX > maxX) {
    touchingBoundary = true;
    mesh.position.x -= ext.maxX - maxX;
    if (velocity.x > 0) {
      const preImpact = velocity.clone();
      velocity.x = reflectNormalVelocity(preImpact.x);
      velocity.y *= tangentDamping;
      velocity.z *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.maxXPoint.clone();
      contact.x = maxX;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  if (ext.minY < minY) {
    onFloor = true;
    touchingBoundary = true;
    mesh.position.y += minY - ext.minY;
    if (velocity.y < 0) {
      const preImpact = velocity.clone();
      velocity.y = reflectNormalVelocity(preImpact.y);
      velocity.x *= tangentDamping;
      velocity.z *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.minYPoint.clone();
      contact.y = minY;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  if (ext.maxY > maxY) {
    touchingBoundary = true;
    mesh.position.y -= ext.maxY - maxY;
    if (velocity.y > 0) {
      const preImpact = velocity.clone();
      velocity.y = reflectNormalVelocity(preImpact.y);
      velocity.x *= tangentDamping;
      velocity.z *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.maxYPoint.clone();
      contact.y = maxY;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  if (ext.minZ < minZ) {
    touchingBoundary = true;
    mesh.position.z += minZ - ext.minZ;
    if (velocity.z < 0) {
      const preImpact = velocity.clone();
      velocity.z = reflectNormalVelocity(preImpact.z);
      velocity.x *= tangentDamping;
      velocity.y *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.minZPoint.clone();
      contact.z = minZ;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  if (ext.maxZ > maxZ) {
    touchingBoundary = true;
    mesh.position.z -= ext.maxZ - maxZ;
    if (velocity.z > 0) {
      const preImpact = velocity.clone();
      velocity.z = reflectNormalVelocity(preImpact.z);
      velocity.x *= tangentDamping;
      velocity.y *= tangentDamping;
      const inertia = getApproxInertiaFromExtrema(ext, mass);
      const contact = ext.maxZPoint.clone();
      contact.z = maxZ;
      const worldCom = getMeshWorldCenterOfMass(mesh);
      addAngularImpulse(
        angularVelocity,
        contact.sub(worldCom),
        new THREE.Vector3().copy(velocity).sub(preImpact),
        collisionSpinScale
        ,1 / inertia
      );
    }
  }
  ext = getMeshWorldExtrema(mesh);

  // Near the view plane, apply side guard rails so objects cannot slip around
  // the room opening and bounce out through edge cases.
  if (ext.maxZ > VIEWPLANE_GUARD_Z) {
    const guardMinX = minX + VIEWPLANE_SIDE_INSET;
    const guardMaxX = maxX - VIEWPLANE_SIDE_INSET;
    const guardTangentDamping = clamp(tangentDamping * 0.95, 0.42, 0.95);

    if (ext.minX < guardMinX) {
      touchingBoundary = true;
      mesh.position.x += guardMinX - ext.minX;
      if (velocity.x < 0) {
        const preImpact = velocity.clone();
        velocity.x = reflectNormalVelocity(preImpact.x);
        velocity.y *= guardTangentDamping;
        velocity.z *= guardTangentDamping;
      }
    }
    ext = getMeshWorldExtrema(mesh);

    if (ext.maxX > guardMaxX) {
      touchingBoundary = true;
      mesh.position.x -= ext.maxX - guardMaxX;
      if (velocity.x > 0) {
        const preImpact = velocity.clone();
        velocity.x = reflectNormalVelocity(preImpact.x);
        velocity.y *= guardTangentDamping;
        velocity.z *= guardTangentDamping;
      }
    }
  }

  // Keep objects stably supported by the floor while spinning/settling.
  ext = getMeshWorldExtrema(mesh);
  if (ext.minY > minY && ext.minY <= minY + floorSnapEps && Math.abs(velocity.y) < minBounceSpeed) {
    onFloor = true;
    touchingBoundary = true;
    mesh.position.y -= ext.minY - minY;
    velocity.y = 0;
  }
  return { touchingBoundary, onFloor };
};

// ── Draggable object ──────────────────────────────────────────────────────────
const DraggableObject = ({
  position,
  roomPhysicsEnabled,
  gravityForceEnabled,
  gravityStrength,
  restitution,
  throwInertiaEnabled,
  mass,
  linearDamping,
  angularDamping,
  surfaceFriction,
  children,
}: {
  position: [number, number, number];
  roomPhysicsEnabled: boolean;
  gravityForceEnabled: boolean;
  gravityStrength: number;
  restitution: number;
  throwInertiaEnabled: boolean;
  mass: number;
  linearDamping: number;
  angularDamping: number;
  surfaceFriction: number;
  children: React.ReactNode;
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, raycaster, gl } = useThree();
  const isDragging = useRef(false);
  const isHovered = useRef(false);
  const dragPlane = useRef(new THREE.Plane());
  const offset = useRef(new THREE.Vector3());
  const intersect = useRef(new THREE.Vector3());
  const mouse = useRef(new THREE.Vector2());
  const lastDragPos = useRef(new THREE.Vector3());
  const lastDragTimeMs = useRef(0);
  const grabLeverArm = useRef(new THREE.Vector3());
  const MAX_GRAVITY_DT = 1 / 30;
  const ANGULAR_DAMPING = 0.992;
  const SLEEP_LINEAR_SPEED = 0.26;
  const SLEEP_ANGULAR_SPEED = 0.36;
  const WAKE_LINEAR_SPEED = 0.9;
  const WAKE_ANGULAR_SPEED = 1.25;
  const SLEEP_FRAMES_REQUIRED = 14;
  const OSC_LINEAR_AXIS_LIMIT = 0.95;
  const OSC_ANGULAR_AXIS_LIMIT = 0.9;
  const OSC_ENERGY_LIMIT = 2.4;
  const OSC_CHAOS_TRIGGER = 4.5;

  useEffect(() => {
    const mesh = meshRef.current;
    if (mesh) {
      draggableRegistry.add(mesh);
      (mesh.userData as { __physics?: MeshPhysics }).__physics = {
        mass,
        linearDamping,
        angularDamping,
        surfaceFriction,
      };
    }
    return () => { if (mesh) draggableRegistry.delete(mesh); };
  }, [mass, linearDamping, angularDamping, surfaceFriction]);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const velocity = getMeshVelocity(mesh);
    const angularVelocity = getMeshAngularVelocity(mesh);
    const physics = getMeshPhysics(mesh);
    const sleep = getMeshSleepState(mesh);
    const osc = getMeshOscillationState(mesh);

    if (isDragging.current || fingerGrabbedRegistry.has(mesh)) {
      return;
    }
    if (!roomPhysicsEnabled) {
      velocity.set(0, 0, 0);
      angularVelocity.set(0, 0, 0);
      return;
    }
    if (sleep.asleep) {
      const linearSpeed = velocity.length();
      const angularSpeed = angularVelocity.length();
      if (linearSpeed > WAKE_LINEAR_SPEED || angularSpeed > WAKE_ANGULAR_SPEED) {
        sleep.asleep = false;
        sleep.restFrames = 0;
      } else {
        velocity.set(0, 0, 0);
        angularVelocity.set(0, 0, 0);
        return;
      }
    }
    if (sleep.asleep) {
      return;
    }

    const dt = Math.min(delta, MAX_GRAVITY_DT);
    if (gravityForceEnabled && gravityStrength > 0) {
      velocity.y -= gravityStrength * dt;
    }
    const linDampFactor = Math.exp(-physics.linearDamping * dt);
    const angDampFactor = Math.exp(-physics.angularDamping * dt);
    velocity.multiplyScalar(linDampFactor);
    mesh.position.addScaledVector(velocity, dt);
    mesh.rotation.x += angularVelocity.x * dt;
    mesh.rotation.y += angularVelocity.y * dt;
    mesh.rotation.z += angularVelocity.z * dt;
    const collision = resolveMeshRoomCollision(
      mesh,
      velocity,
      restitution,
      physics.surfaceFriction,
      physics.mass,
      angularVelocity
    );
    if (gravityForceEnabled && gravityStrength > 0 && collision.onFloor) {
      const worldCom = getMeshWorldCenterOfMass(mesh);
      const support = getFloorSupportPoint(mesh, -ROOM_HALF_HEIGHT);
      const lever = new THREE.Vector3(worldCom.x - support.x, 0, worldCom.z - support.z);
      const gravityForce = new THREE.Vector3(0, -gravityStrength * physics.mass, 0);
      const torque = new THREE.Vector3().copy(lever).cross(gravityForce);
      const inertia = getMeshApproxInertia(mesh, physics.mass);
      angularVelocity.x += (torque.x / inertia) * dt * 0.14;
      angularVelocity.z += (torque.z / inertia) * dt * 0.14;
      // Multi-axis oscillation detector:
      // detect chaotic small-amplitude sign-flip loops across linear+angular axes.
      const countAxisFlips = (curr: THREE.Vector3, prev: THREE.Vector3, lim: number) => {
        let flips = 0;
        if (Math.sign(curr.x) !== 0 && Math.sign(prev.x) !== 0 && Math.sign(curr.x) !== Math.sign(prev.x) && Math.abs(curr.x) < lim && Math.abs(prev.x) < lim) flips++;
        if (Math.sign(curr.y) !== 0 && Math.sign(prev.y) !== 0 && Math.sign(curr.y) !== Math.sign(prev.y) && Math.abs(curr.y) < lim && Math.abs(prev.y) < lim) flips++;
        if (Math.sign(curr.z) !== 0 && Math.sign(prev.z) !== 0 && Math.sign(curr.z) !== Math.sign(prev.z) && Math.abs(curr.z) < lim && Math.abs(prev.z) < lim) flips++;
        return flips;
      };

      const linearFlips = countAxisFlips(velocity, osc.prevLinear, OSC_LINEAR_AXIS_LIMIT);
      const angularFlips = countAxisFlips(angularVelocity, osc.prevAngular, OSC_ANGULAR_AXIS_LIMIT);
      const totalFlips = linearFlips + angularFlips;
      const energy = velocity.lengthSq() + angularVelocity.lengthSq() * 0.28;
      const deltaEnergy = Math.abs(energy - osc.prevEnergy);
      const nearRestBand = energy < OSC_ENERGY_LIMIT;
      const chaoticLoop = nearRestBand && totalFlips >= 2;

      if (chaoticLoop) {
        const decayFlat = deltaEnergy < 0.12 ? 0.4 : 0;
        osc.chaosScore = Math.min(12, osc.chaosScore + 1 + totalFlips * 0.25 + decayFlat);
      } else {
        const decay = nearRestBand ? 0.35 : 1.2;
        osc.chaosScore = Math.max(0, osc.chaosScore - decay);
      }

      if (osc.chaosScore >= OSC_CHAOS_TRIGGER) {
        osc.dampFrames = Math.min(osc.dampFrames + 1, 10);
        const linExtra = clamp(0.9 - osc.dampFrames * 0.02, 0.62, 0.9);
        const angExtra = clamp(0.86 - osc.dampFrames * 0.024, 0.54, 0.86);
        velocity.multiplyScalar(linExtra);
        angularVelocity.multiplyScalar(angExtra);
      } else {
        osc.dampFrames = Math.max(0, osc.dampFrames - 1);
      }

      osc.prevLinear.copy(velocity);
      osc.prevAngular.copy(angularVelocity);
      osc.prevEnergy = energy;
    } else {
      osc.prevLinear.copy(velocity);
      osc.prevAngular.copy(angularVelocity);
      osc.prevEnergy = velocity.lengthSq() + angularVelocity.lengthSq() * 0.28;
      osc.chaosScore = Math.max(0, osc.chaosScore - 0.6);
      osc.dampFrames = Math.max(0, osc.dampFrames - 1);
    }
    angularVelocity.multiplyScalar(ANGULAR_DAMPING * angDampFactor);

    const linearSpeed = velocity.length();
    const angularSpeed = angularVelocity.length();
    const lowEnergy = linearSpeed < SLEEP_LINEAR_SPEED && angularSpeed < SLEEP_ANGULAR_SPEED;
    if (collision.touchingBoundary && lowEnergy) {
      sleep.restFrames += 1;
      if (sleep.restFrames >= SLEEP_FRAMES_REQUIRED) {
        velocity.set(0, 0, 0);
        angularVelocity.set(0, 0, 0);
        sleep.asleep = true;
      }
    } else {
      sleep.restFrames = 0;
      sleep.asleep = false;
    }
  });

  useEffect(() => {
    const canvas = gl.domElement;

    const onMove = (e: PointerEvent) => {
      if (!isDragging.current || !meshRef.current) return;
      const rect = canvas.getBoundingClientRect();
      mouse.current.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      raycaster.setFromCamera(mouse.current, camera);
      if (raycaster.ray.intersectPlane(dragPlane.current, intersect.current)) {
        meshRef.current.position.copy(intersect.current).sub(offset.current);
        wakeMesh(meshRef.current);
        clampMeshToRoom(meshRef.current);
        const applyInertia = roomPhysicsEnabled && throwInertiaEnabled;
        if (!applyInertia) {
          lastDragPos.current.copy(meshRef.current.position);
          lastDragTimeMs.current = performance.now();
          return;
        }
        const nowMs = performance.now();
        const dt = Math.max((nowMs - lastDragTimeMs.current) / 1000, 1 / 240);
        const velocity = getMeshVelocity(meshRef.current);
        const angularVelocity = getMeshAngularVelocity(meshRef.current);
        const physics = getMeshPhysics(meshRef.current);
        const sampleVel = new THREE.Vector3()
          .copy(meshRef.current.position)
          .sub(lastDragPos.current)
          .multiplyScalar(1 / dt);
        const response = 1 / (0.55 + 0.45 * physics.mass);
        velocity.lerp(sampleVel, 0.6 * response);
        const inertia = getMeshApproxInertia(meshRef.current, physics.mass);
        const torqueFromDrag = new THREE.Vector3()
          .copy(grabLeverArm.current)
          .cross(sampleVel)
          .multiplyScalar(0.18 / inertia);
        angularVelocity.lerp(torqueFromDrag, 0.42 * response);
        lastDragPos.current.copy(meshRef.current.position);
        lastDragTimeMs.current = nowMs;
      }
    };

    const onUp = () => {
      if (isDragging.current && meshRef.current) {
        const applyInertia = roomPhysicsEnabled && throwInertiaEnabled;
        if (applyInertia) {
          const nowMs = performance.now();
          const dt = Math.max((nowMs - lastDragTimeMs.current) / 1000, 1 / 240);
          const velocity = getMeshVelocity(meshRef.current);
          const angularVelocity = getMeshAngularVelocity(meshRef.current);
          const physics = getMeshPhysics(meshRef.current);
          const releaseVel = new THREE.Vector3()
            .copy(meshRef.current.position)
            .sub(lastDragPos.current)
            .multiplyScalar(1 / dt);
          const response = 1 / (0.55 + 0.45 * physics.mass);
          velocity.lerp(releaseVel, 0.7 * response);
          const inertia = getMeshApproxInertia(meshRef.current, physics.mass);
          const torqueFromRelease = new THREE.Vector3()
            .copy(grabLeverArm.current)
            .cross(releaseVel)
            .multiplyScalar(0.22 / inertia);
          angularVelocity.lerp(torqueFromRelease, 0.55 * response);
        } else {
          const velocity = getMeshVelocity(meshRef.current);
          const angularVelocity = getMeshAngularVelocity(meshRef.current);
          velocity.set(0, 0, 0);
          angularVelocity.set(0, 0, 0);
        }
      }
      isDragging.current = false;
      canvas.style.cursor = isHovered.current ? 'grab' : 'default';
    };

    const onWheel = (e: WheelEvent) => {
      if ((!isHovered.current && !isDragging.current) || !meshRef.current) return;
      e.preventDefault();
      // Move along z: scroll down = deeper into room (more negative z)
      meshRef.current.position.z -= e.deltaY * 0.05;
      wakeMesh(meshRef.current);
      clampMeshToRoom(meshRef.current);
      const velocity = getMeshVelocity(meshRef.current);
      const angularVelocity = getMeshAngularVelocity(meshRef.current);
      if (roomPhysicsEnabled && throwInertiaEnabled) {
        const physics = getMeshPhysics(meshRef.current);
        const response = 1 / (0.55 + 0.45 * physics.mass);
        velocity.z -= e.deltaY * 0.09 * response;
        angularVelocity.x += e.deltaY * 0.0025 * response;
      } else {
        velocity.set(0, 0, 0);
        angularVelocity.set(0, 0, 0);
      }
      // Keep drag plane in sync with new depth
      if (isDragging.current) {
        dragPlane.current.constant = -meshRef.current.position.z;
      }
    };

    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('wheel', onWheel);
    };
  }, [camera, gl, raycaster, roomPhysicsEnabled, throwInertiaEnabled]);

  return (
    <mesh
      ref={meshRef}
      position={position}
      castShadow
      receiveShadow
      onPointerDown={(e) => {
        e.stopPropagation();
        isDragging.current = true;
        const velocity = getMeshVelocity(meshRef.current!);
        const angularVelocity = getMeshAngularVelocity(meshRef.current!);
        wakeMesh(meshRef.current!);
        velocity.set(0, 0, 0);
        angularVelocity.set(0, 0, 0);
        lastDragPos.current.copy(meshRef.current!.position);
        lastDragTimeMs.current = performance.now();
        gl.domElement.style.cursor = 'grabbing';
        dragPlane.current.setFromNormalAndCoplanarPoint(
          new THREE.Vector3(0, 0, 1),
          meshRef.current!.position
        );
        offset.current.copy(e.point).sub(meshRef.current!.position);
        grabLeverArm.current.copy(offset.current);
      }}
      onPointerEnter={() => { isHovered.current = true; gl.domElement.style.cursor = 'grab'; }}
      onPointerLeave={() => { isHovered.current = false; if (!isDragging.current) gl.domElement.style.cursor = 'default'; }}
    >
      {children}
    </mesh>
  );
};

// ── Experimental finger dots ──────────────────────────────────────────────────
// Two small shadow-casting dots on the screen plane (z≈0), one per fingertip.
// Grab aims a world-space ray camera → pinch point on that plane. NDC + project()
// fights CameraRig’s off-axis projection; mouse drag still uses canvas NDC.
const FingerDots = ({
  pinchData,
  sensitivity,
  roomPhysicsEnabled,
  throwInertiaEnabled,
}: {
  pinchData: PinchData;
  sensitivity: number;
  roomPhysicsEnabled: boolean;
  throwInertiaEnabled: boolean;
}) => {
  const { camera, raycaster } = useThree();
  const pinchRef = useRef(pinchData);
  pinchRef.current = pinchData;
  const sensRef = useRef(sensitivity);
  sensRef.current = sensitivity;
  const applyInertia = roomPhysicsEnabled && throwInertiaEnabled;

  const thumbRef    = useRef<THREE.Mesh>(null!);
  const indexRef    = useRef<THREE.Mesh>(null!);
  const grabbedMesh = useRef<THREE.Mesh | null>(null);
  const grabOffset  = useRef(new THREE.Vector3());
  const dragPlane   = useRef(new THREE.Plane());
  const wasGrabbing = useRef(false);
  const releaseStreak = useRef(0);
  const grabStartPinchZ = useRef(0);
  const grabStartObjectZ = useRef(0);
  const fingerGrabLeverArm = useRef(new THREE.Vector3());
  const lastFingerDragPos = useRef(new THREE.Vector3());
  const lastFingerDragTimeMs = useRef(0);
  const dotDepthState = useRef({
    initialized: false,
    basePinchNZ: 0,
    baseHandScaleN: 0.12,
    baseFaceScaleN: 0,
    baseSceneZ: 0.5,
    currentSceneZ: 0.5,
  });
  const dotSpatialState = useRef({
    initialized: false,
    thumb: new THREE.Vector3(),
    index: new THREE.Vector3(),
    pinchWorld: new THREE.Vector3(),
    thumbTarget: new THREE.Vector3(),
    indexTarget: new THREE.Vector3(),
    pinchTarget: new THREE.Vector3(),
  });
  const intersect   = useRef(new THREE.Vector3());
  const camWorld    = useRef(new THREE.Vector3());
  const pinchWorld  = useRef(new THREE.Vector3());
  const aimDir      = useRef(new THREE.Vector3());
  const GRAB_XY_RADIUS = 5.4;
  const GRAB_PINCH_XY_RADIUS = 4.8;
  const GRAB_Z_TOLERANCE = 9.5;
  const GRAB_RELEASE_GRACE_FRAMES = 8;
  const DOT_FRONT_Z_LIMIT = ROOM_FRONT_Z - ROOM_COLLISION_EPS;
  const DRAG_FRONT_Z_LIMIT = ROOM_FRONT_Z - ROOM_COLLISION_EPS;
  const DRAG_BACK_Z_LIMIT = ROOM_BACK_Z + ROOM_COLLISION_EPS;
  const DOT_BACK_Z_LIMIT = ROOM_BACK_Z + ROOM_COLLISION_EPS;
  const DEPTH_DRAG_GAIN = BOX_DEPTH_CM * 4.1;
  const DOT_DEPTH_GAIN = BOX_DEPTH_CM * 5.2;
  const DOT_DEPTH_LERP = 0.24;
  const DOT_XY_LERP = 0.34;
  const PINCH_RAY_LERP = 0.28;
  const HAND_DEPTH_DEADZONE_N = 0.0055;
  const SCALE_DEPTH_CONTRIBUTION = 0.11;
  const BASELINE_RECALIBRATION_LERP = 0.015;
  const RECALIBRATION_DEPTH_GATE_N = 0.012;
  const RECALIBRATION_SCALE_GATE = 0.12;

  useFrame(() => {
    const p = pinchRef.current;
    if (!p.hasHand) {
      if (grabbedMesh.current) {
        wakeMesh(grabbedMesh.current);
        if (applyInertia) {
          const nowMs = performance.now();
          const dt = Math.max((nowMs - lastFingerDragTimeMs.current) / 1000, 1 / 240);
          const velocity = getMeshVelocity(grabbedMesh.current);
          const angularVelocity = getMeshAngularVelocity(grabbedMesh.current);
          const physics = getMeshPhysics(grabbedMesh.current);
          const releaseVel = new THREE.Vector3()
            .copy(grabbedMesh.current.position)
            .sub(lastFingerDragPos.current)
            .multiplyScalar(1 / dt);
          const response = 1 / (0.55 + 0.45 * physics.mass);
          velocity.lerp(releaseVel, 0.68 * response);
          const inertia = getMeshApproxInertia(grabbedMesh.current, physics.mass);
          const torqueFromRelease = new THREE.Vector3()
            .copy(fingerGrabLeverArm.current)
            .cross(releaseVel)
            .multiplyScalar(0.2 / inertia);
          angularVelocity.lerp(torqueFromRelease, 0.5 * response);
        } else {
          const velocity = getMeshVelocity(grabbedMesh.current);
          const angularVelocity = getMeshAngularVelocity(grabbedMesh.current);
          velocity.set(0, 0, 0);
          angularVelocity.set(0, 0, 0);
        }
        fingerGrabbedRegistry.delete(grabbedMesh.current);
        grabbedMesh.current = null;
      }
      dotDepthState.current.initialized = false;
      return;
    }

    if (!dotDepthState.current.initialized) {
      dotDepthState.current.initialized = true;
      dotDepthState.current.basePinchNZ = p.pinchNZ;
      dotDepthState.current.baseHandScaleN = Math.max(p.handScaleN, 0.02);
      dotDepthState.current.baseFaceScaleN = p.faceScaleN > 0 ? p.faceScaleN : 0;
      dotDepthState.current.baseSceneZ = 1.5;
      dotDepthState.current.currentSceneZ = 1.5;
    }

    const s = sensRef.current;
    const t = cameraNormToScreenCm(p.thumbNX, p.thumbNY, s);
    const i = cameraNormToScreenCm(p.indexNX, p.indexNY, s);
    const handOnlyRelativeScale = p.handScaleN / dotDepthState.current.baseHandScaleN;
    const canUseFaceReference =
      p.faceScaleN > 0.0001 && dotDepthState.current.baseFaceScaleN > 0.0001;
    const relativeHandScale = clamp(
      canUseFaceReference
        ? (p.handScaleN / p.faceScaleN) /
            (dotDepthState.current.baseHandScaleN / dotDepthState.current.baseFaceScaleN)
        : handOnlyRelativeScale,
      0.5,
      2.8
    );
    // Linear depth scaling around baseline hand size.
    const depthScaleMultiplier = clamp(0.9 + 0.35 * relativeHandScale, 0.9, 1.9);
    const rawDepthDelta = p.pinchNZ - dotDepthState.current.basePinchNZ;
    const depthDelta =
      Math.abs(rawDepthDelta) <= HAND_DEPTH_DEADZONE_N
        ? 0
        : Math.sign(rawDepthDelta) * (Math.abs(rawDepthDelta) - HAND_DEPTH_DEADZONE_N);
    // Face-referenced hand scale acts as a stable depth cue:
    // larger hand ratio (closer hand) pushes deeper into the room.
    const scaleDepthDelta = (1 - relativeHandScale) * SCALE_DEPTH_CONTRIBUTION;
    const combinedDepthDelta = depthDelta + scaleDepthDelta;
    const dotTargetZ = clamp(
      dotDepthState.current.baseSceneZ + combinedDepthDelta * DOT_DEPTH_GAIN * depthScaleMultiplier,
      DOT_BACK_Z_LIMIT,
      DOT_FRONT_Z_LIMIT
    );
    dotDepthState.current.currentSceneZ = THREE.MathUtils.lerp(
      dotDepthState.current.currentSceneZ,
      dotTargetZ,
      DOT_DEPTH_LERP
    );
    const dotZ = dotDepthState.current.currentSceneZ;

    if (!dotSpatialState.current.initialized) {
      dotSpatialState.current.initialized = true;
      dotSpatialState.current.thumb.set(t.x, t.y, dotZ);
      dotSpatialState.current.index.set(i.x, i.y, dotZ);
      dotSpatialState.current.pinchWorld.set((t.x + i.x) * 0.5, (t.y + i.y) * 0.5, dotZ);
    } else {
      dotSpatialState.current.thumbTarget.set(t.x, t.y, dotZ);
      dotSpatialState.current.indexTarget.set(i.x, i.y, dotZ);
      dotSpatialState.current.thumb.lerp(dotSpatialState.current.thumbTarget, DOT_XY_LERP);
      dotSpatialState.current.index.lerp(dotSpatialState.current.indexTarget, DOT_XY_LERP);
    }

    if (thumbRef.current) thumbRef.current.position.copy(dotSpatialState.current.thumb);
    if (indexRef.current) indexRef.current.position.copy(dotSpatialState.current.index);

    if (!p.isPinching) {
      // Recalibrate only near neutral; avoid fighting active depth movement.
      const nearDepthBaseline = Math.abs(rawDepthDelta) <= RECALIBRATION_DEPTH_GATE_N;
      const nearScaleBaseline = Math.abs(relativeHandScale - 1) <= RECALIBRATION_SCALE_GATE;
      if (nearDepthBaseline && nearScaleBaseline) {
        dotDepthState.current.basePinchNZ = THREE.MathUtils.lerp(
          dotDepthState.current.basePinchNZ,
          p.pinchNZ,
          BASELINE_RECALIBRATION_LERP
        );
        dotDepthState.current.baseHandScaleN = THREE.MathUtils.lerp(
          dotDepthState.current.baseHandScaleN,
          Math.max(p.handScaleN, 0.02),
          BASELINE_RECALIBRATION_LERP
        );
        if (p.faceScaleN > 0.0001) {
          dotDepthState.current.baseFaceScaleN = dotDepthState.current.baseFaceScaleN > 0.0001
            ? THREE.MathUtils.lerp(
                dotDepthState.current.baseFaceScaleN,
                p.faceScaleN,
                BASELINE_RECALIBRATION_LERP
              )
            : p.faceScaleN;
        }
      }
      if (grabbedMesh.current) {
        releaseStreak.current += 1;
        if (releaseStreak.current >= GRAB_RELEASE_GRACE_FRAMES) {
          wakeMesh(grabbedMesh.current);
          if (applyInertia) {
            const nowMs = performance.now();
            const dt = Math.max((nowMs - lastFingerDragTimeMs.current) / 1000, 1 / 240);
            const velocity = getMeshVelocity(grabbedMesh.current);
            const angularVelocity = getMeshAngularVelocity(grabbedMesh.current);
            const physics = getMeshPhysics(grabbedMesh.current);
            const releaseVel = new THREE.Vector3()
              .copy(grabbedMesh.current.position)
              .sub(lastFingerDragPos.current)
              .multiplyScalar(1 / dt);
            const response = 1 / (0.55 + 0.45 * physics.mass);
            velocity.lerp(releaseVel, 0.68 * response);
            const inertia = getMeshApproxInertia(grabbedMesh.current, physics.mass);
            const torqueFromRelease = new THREE.Vector3()
              .copy(fingerGrabLeverArm.current)
              .cross(releaseVel)
              .multiplyScalar(0.2 / inertia);
            angularVelocity.lerp(torqueFromRelease, 0.5 * response);
          } else {
            const velocity = getMeshVelocity(grabbedMesh.current);
            const angularVelocity = getMeshAngularVelocity(grabbedMesh.current);
            velocity.set(0, 0, 0);
            angularVelocity.set(0, 0, 0);
          }
          fingerGrabbedRegistry.delete(grabbedMesh.current);
          grabbedMesh.current = null;
          wasGrabbing.current = false;
          releaseStreak.current = 0;
        }
      } else {
        wasGrabbing.current = false;
        releaseStreak.current = 0;
      }
      return;
    }
    releaseStreak.current = 0;

    const midNX = (p.thumbNX + p.indexNX) / 2;
    const midNY = (p.thumbNY + p.indexNY) / 2;
    const m = cameraNormToScreenCm(midNX, midNY, s);
    dotSpatialState.current.pinchTarget.set(m.x, m.y, 0.5);
    dotSpatialState.current.pinchWorld.lerp(dotSpatialState.current.pinchTarget, PINCH_RAY_LERP);
    pinchWorld.current.copy(dotSpatialState.current.pinchWorld);
    camera.getWorldPosition(camWorld.current);
    aimDir.current.copy(pinchWorld.current).sub(camWorld.current);
    const len = aimDir.current.length();
    if (len < 1e-6) return;
    aimDir.current.multiplyScalar(1 / len);
    raycaster.ray.origin.copy(camWorld.current);
    raycaster.ray.direction.copy(aimDir.current);

    if (!wasGrabbing.current) {
      // Intersect against actual mesh geometry — same as the mouse drag path.
      const hits = raycaster.intersectObjects([...draggableRegistry], false);
      if (hits.length > 0) {
        const hit = hits[0];
        const mesh = hit.object as THREE.Mesh;
        const thumbDx = hit.point.x - dotSpatialState.current.thumb.x;
        const thumbDy = hit.point.y - dotSpatialState.current.thumb.y;
        const thumbDz = Math.abs(hit.point.z - dotSpatialState.current.thumb.z);
        const indexDx = hit.point.x - dotSpatialState.current.index.x;
        const indexDy = hit.point.y - dotSpatialState.current.index.y;
        const indexDz = Math.abs(hit.point.z - dotSpatialState.current.index.z);
        const pinchDx = hit.point.x - dotSpatialState.current.pinchWorld.x;
        const pinchDy = hit.point.y - dotSpatialState.current.pinchWorld.y;
        const pinchDz = Math.abs(hit.point.z - dotSpatialState.current.pinchWorld.z);

        const thumbWithin =
          Math.hypot(thumbDx, thumbDy) <= GRAB_XY_RADIUS && thumbDz <= GRAB_Z_TOLERANCE;
        const indexWithin =
          Math.hypot(indexDx, indexDy) <= GRAB_XY_RADIUS && indexDz <= GRAB_Z_TOLERANCE;
        const pinchWithin =
          Math.hypot(pinchDx, pinchDy) <= GRAB_PINCH_XY_RADIUS && pinchDz <= GRAB_Z_TOLERANCE;
        const isTouching = thumbWithin || indexWithin || pinchWithin;

        if (isTouching) {
          grabbedMesh.current = mesh;
          wakeMesh(mesh);
          fingerGrabbedRegistry.add(mesh);
          const velocity = getMeshVelocity(mesh);
          const angularVelocity = getMeshAngularVelocity(mesh);
          velocity.set(0, 0, 0);
          angularVelocity.set(0, 0, 0);
          grabStartPinchZ.current = p.pinchNZ;
          grabStartObjectZ.current = mesh.position.z;
          lastFingerDragPos.current.copy(mesh.position);
          lastFingerDragTimeMs.current = performance.now();
          dragPlane.current.setFromNormalAndCoplanarPoint(
            new THREE.Vector3(0, 0, 1),
            mesh.position
          );
          if (raycaster.ray.intersectPlane(dragPlane.current, intersect.current)) {
            grabOffset.current.copy(intersect.current).sub(mesh.position);
          }
          fingerGrabLeverArm.current.copy(hit.point).sub(mesh.position);
          wasGrabbing.current = true;
        }
      }
    }

    if (grabbedMesh.current) {
      if (raycaster.ray.intersectPlane(dragPlane.current, intersect.current)) {
        wakeMesh(grabbedMesh.current);
        grabbedMesh.current.position.copy(intersect.current).sub(grabOffset.current);
        const pinchDepthDelta = p.pinchNZ - grabStartPinchZ.current;
        const targetZ =
          grabStartObjectZ.current + (pinchDepthDelta + scaleDepthDelta) * DEPTH_DRAG_GAIN * depthScaleMultiplier;
        grabbedMesh.current.position.z = clamp(targetZ, DRAG_BACK_Z_LIMIT, DRAG_FRONT_Z_LIMIT);
        clampMeshToRoom(grabbedMesh.current, DRAG_FRONT_Z_LIMIT, DRAG_BACK_Z_LIMIT);
        if (applyInertia) {
          const nowMs = performance.now();
          const dt = Math.max((nowMs - lastFingerDragTimeMs.current) / 1000, 1 / 240);
          const velocity = getMeshVelocity(grabbedMesh.current);
          const physics = getMeshPhysics(grabbedMesh.current);
          const sampleVel = new THREE.Vector3()
            .copy(grabbedMesh.current.position)
            .sub(lastFingerDragPos.current)
            .multiplyScalar(1 / dt);
          const response = 1 / (0.55 + 0.45 * physics.mass);
          velocity.lerp(sampleVel, 0.5 * response);
          const angularVelocity = getMeshAngularVelocity(grabbedMesh.current);
          const inertia = getMeshApproxInertia(grabbedMesh.current, physics.mass);
          const torqueFromDrag = new THREE.Vector3()
            .copy(fingerGrabLeverArm.current)
            .cross(sampleVel)
            .multiplyScalar(0.16 / inertia);
          angularVelocity.lerp(torqueFromDrag, 0.38 * response);
          lastFingerDragPos.current.copy(grabbedMesh.current.position);
          lastFingerDragTimeMs.current = nowMs;
        } else {
          lastFingerDragPos.current.copy(grabbedMesh.current.position);
          lastFingerDragTimeMs.current = performance.now();
        }
      }
    }
  });

  return (
    <group visible={pinchData.hasHand}>
      <mesh ref={thumbRef} castShadow>
        <sphereGeometry args={[0.35, 8, 8]} />
        <meshStandardMaterial color="#ff8800" emissive="#ff4400" emissiveIntensity={0.6} />
      </mesh>
      <mesh ref={indexRef} castShadow>
        <sphereGeometry args={[0.35, 8, 8]} />
        <meshStandardMaterial color="#00ccff" emissive="#0088cc" emissiveIntensity={0.6} />
      </mesh>
    </group>
  );
};

// ── Room surfaces: floor + walls with subtle edge darkening ───────────────────
const useEdgeDarkenedTexture = (baseColor: string, edgeAlpha = 0.3, cornerAlpha = 0.25) =>
  useMemo(() => {
    const size = 512;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.fillStyle = baseColor;
    ctx.fillRect(0, 0, size, size);

    const band = Math.floor(size * 0.3);
    const gradients = [
      { from: [0, 0], to: [band, 0], rect: [0, 0, band, size] }, // left
      { from: [size, 0], to: [size - band, 0], rect: [size - band, 0, band, size] }, // right
      { from: [0, 0], to: [0, band], rect: [0, 0, size, band] }, // top
      { from: [0, size], to: [0, size - band], rect: [0, size - band, size, band] }, // bottom
    ] as const;

    for (const g of gradients) {
      const grad = ctx.createLinearGradient(g.from[0], g.from[1], g.to[0], g.to[1]);
      grad.addColorStop(0, `rgba(0,0,0,${edgeAlpha})`);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(g.rect[0], g.rect[1], g.rect[2], g.rect[3]);
    }

    const cornerRadius = Math.floor(size * 0.42);
    const corners: Array<[number, number]> = [
      [0, 0],
      [size, 0],
      [0, size],
      [size, size],
    ];

    for (const [cx, cy] of corners) {
      const cornerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, cornerRadius);
      cornerGrad.addColorStop(0, `rgba(0,0,0,${cornerAlpha})`);
      cornerGrad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = cornerGrad;
      ctx.fillRect(
        cx === 0 ? 0 : size - cornerRadius,
        cy === 0 ? 0 : size - cornerRadius,
        cornerRadius,
        cornerRadius
      );
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }, [baseColor, edgeAlpha, cornerAlpha]);

const Floor = () => {
  const w = SCREEN_WIDTH_CM;
  const d = BOX_DEPTH_CM;
  const floorTexture = useEdgeDarkenedTexture("#35383f", 0.45, 0.45);

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -SCREEN_HEIGHT_CM / 2, -d / 2]}
      receiveShadow
    >
      <planeGeometry args={[w, d]} />
      <meshStandardMaterial
        map={floorTexture ?? undefined}
        color="#ffffff"
        roughness={0.92}
        metalness={0.04}
        envMapIntensity={0.45}
      />
    </mesh>
  );
};

const RoomWalls = () => {
  const w = SCREEN_WIDTH_CM;
  const h = SCREEN_HEIGHT_CM;
  const d = BOX_DEPTH_CM;
  const wallTexture = useEdgeDarkenedTexture("#2f333b", 0.56, 0.58);

  return (
    <>
      {/* Back wall */}
      <mesh position={[0, 0, -d]} receiveShadow>
        <planeGeometry args={[w, h]} />
        <meshStandardMaterial
          map={wallTexture ?? undefined}
          color="#ffffff"
          roughness={0.95}
          metalness={0.02}
          envMapIntensity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Left wall */}
      <mesh position={[-w / 2, 0, -d / 2]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[d, h]} />
        <meshStandardMaterial
          map={wallTexture ?? undefined}
          color="#ffffff"
          roughness={0.95}
          metalness={0.02}
          envMapIntensity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Right wall */}
      <mesh position={[w / 2, 0, -d / 2]} rotation={[0, -Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[d, h]} />
        <meshStandardMaterial
          map={wallTexture ?? undefined}
          color="#ffffff"
          roughness={0.95}
          metalness={0.02}
          envMapIntensity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>
    </>
  );
};

const Roof = () => {
  const w = SCREEN_WIDTH_CM;
  const d = BOX_DEPTH_CM;
  const h = SCREEN_HEIGHT_CM;
  const roofTexture = useEdgeDarkenedTexture("#2b2f37", 0.56, 0.58);

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, h / 2, -d / 2]}
      receiveShadow
    >
      <planeGeometry args={[w, d]} />
      <meshStandardMaterial
        map={roofTexture ?? undefined}
        color="#ffffff"
        roughness={0.95}
        metalness={0.02}
        envMapIntensity={0.3}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

const RoomBoxOutline = () => {
  const w = SCREEN_WIDTH_CM;
  const h = SCREEN_HEIGHT_CM;
  const d = BOX_DEPTH_CM;
  const outlineGeometry = useMemo(() => {
    const hw = w / 2;
    const hh = h / 2;
    const zFront = 0;
    const zBack = -d;
    const inset = 0.06;
    const xL = -hw + inset;
    const xR = hw - inset;
    const yB = -hh + inset;
    const yT = hh - inset;
    const zF = zFront - inset;
    const zBk = zBack + inset;

    const points = [
      // Back wall border rectangle
      xL, yB, zBk,   xR, yB, zBk,
      xR, yB, zBk,   xR, yT, zBk,
      xR, yT, zBk,   xL, yT, zBk,
      xL, yT, zBk,   xL, yB, zBk,

      // Front view-window border rectangle
      xL, yB, zF,  xR, yB, zF,
      xR, yB, zF,  xR, yT, zF,
      xR, yT, zF,  xL, yT, zF,
      xL, yT, zF,  xL, yB, zF,

      // Connect back-wall corners to the front view-window corners
      xL, yB, zBk,   xL, yB, zF, // floor-left edge
      xR, yB, zBk,   xR, yB, zF, // floor-right edge
      xL, yT, zBk,   xL, yT, zF, // ceiling-left edge
      xR, yT, zBk,   xR, yT, zF, // ceiling-right edge
    ];

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    return geometry;
  }, [w, h, d]);

  return (
    <lineSegments geometry={outlineGeometry}>
      <lineBasicMaterial color="#7f8ea9" transparent opacity={0.33} depthTest depthWrite={false} />
    </lineSegments>
  );
};

// ── HDRI: image-based lighting only (no visible background sphere) ────────────
const HDRIEnvironment = ({ url }: { url: string }) => {
  const { scene, gl } = useThree();
  const hdr = useLoader(HDRLoader, url);

  const envMap = useMemo(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    pmrem.compileEquirectangularShader();
    const rt = pmrem.fromEquirectangular(hdr);
    pmrem.dispose();
    return rt.texture;
  }, [gl, hdr]);

  useEffect(() => {
    scene.environment = envMap;
    scene.environmentIntensity = 0.95;
    return () => { scene.environment = null; };
  }, [scene, envMap]);

  return null;
};

// ── Draggable scene objects ───────────────────────────────────────────────────
const SceneObjects = ({
  roomPhysicsEnabled,
  gravityForceEnabled,
  gravityStrength,
  restitution,
  throwInertiaEnabled,
  collection,
}: {
  roomPhysicsEnabled: boolean;
  gravityForceEnabled: boolean;
  gravityStrength: number;
  restitution: number;
  throwInertiaEnabled: boolean;
  collection: ObjectCollectionId;
}) => {
  if (collection === "dense") {
    return (
      <>
        <DraggableObject position={[5.6, -4.2, -12]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={0.8} linearDamping={0.2} angularDamping={0.66} surfaceFriction={0.35}>
          <icosahedronGeometry args={[2.1, 0]} />
          <meshStandardMaterial color="#f26ae6" roughness={0.2} metalness={0.6} />
        </DraggableObject>
        <DraggableObject position={[-5.8, -4.3, -16]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={2.7} linearDamping={0.38} angularDamping={1.12} surfaceFriction={0.66}>
          <boxGeometry args={[4.4, 4.4, 4.4]} />
          <meshStandardMaterial color="#6f86ff" roughness={0.42} metalness={0.68} />
        </DraggableObject>
        <DraggableObject position={[0.8, -3.5, -25]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={1.2} linearDamping={0.24} angularDamping={0.76} surfaceFriction={0.44}>
          <dodecahedronGeometry args={[2.6, 0]} />
          <meshStandardMaterial color="#44d2ff" roughness={0.32} metalness={0.4} />
        </DraggableObject>
        <DraggableObject position={[7.3, -4.4, -34]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={1.5} linearDamping={0.26} angularDamping={0.84} surfaceFriction={0.48}>
          <sphereGeometry args={[2.8, 48, 48]} />
          <meshStandardMaterial color="#ff9854" roughness={0.58} metalness={0.12} />
        </DraggableObject>
        <DraggableObject position={[-7.2, -3.9, -44]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={2} linearDamping={0.31} angularDamping={0.95} surfaceFriction={0.6}>
          <torusGeometry args={[2.4, 0.8, 28, 36]} />
          <meshStandardMaterial color="#7ce9aa" roughness={0.24} metalness={0.55} />
        </DraggableObject>
        <DraggableObject position={[0.2, 1.2, -59]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={2.2} linearDamping={0.32} angularDamping={1.03} surfaceFriction={0.58}>
          <torusKnotGeometry args={[2.2, 0.64, 120, 22]} />
          <meshStandardMaterial color="#66f4d6" roughness={0.08} metalness={0.84} />
        </DraggableObject>
      </>
    );
  }

  if (collection === "chaos") {
    return (
      <>
        <DraggableObject position={[0, -4.2, -10]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={4.4} linearDamping={0.46} angularDamping={1.22} surfaceFriction={0.82}>
          <cylinderGeometry args={[3.1, 3.1, 2.8, 36]} />
          <meshStandardMaterial color="#8d7aff" roughness={0.52} metalness={0.42} />
        </DraggableObject>
        <DraggableObject position={[0.2, 2.4, -21]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={0.6} linearDamping={0.14} angularDamping={0.58} surfaceFriction={0.22}>
          <tetrahedronGeometry args={[2.9, 0]} />
          <meshStandardMaterial color="#ff6d9e" roughness={0.26} metalness={0.35} />
        </DraggableObject>
        <DraggableObject position={[-6.8, -2.1, -28]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={1} linearDamping={0.21} angularDamping={0.72} surfaceFriction={0.36}>
          <coneGeometry args={[2.5, 5.4, 22]} />
          <meshStandardMaterial color="#7bd3ff" roughness={0.3} metalness={0.46} />
        </DraggableObject>
        <DraggableObject position={[6.8, -2.3, -34]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={2.8} linearDamping={0.35} angularDamping={1.08} surfaceFriction={0.72}>
          <boxGeometry args={[5.2, 2.2, 5.2]} />
          <meshStandardMaterial color="#90a8ff" roughness={0.48} metalness={0.62} />
        </DraggableObject>
        <DraggableObject position={[-2.2, 2, -47]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={1.6} linearDamping={0.28} angularDamping={0.88} surfaceFriction={0.54}>
          <torusKnotGeometry args={[1.8, 0.58, 128, 24]} />
          <meshStandardMaterial color="#53f0cb" roughness={0.14} metalness={0.82} />
        </DraggableObject>
        <DraggableObject position={[3.8, -3.8, -57]} roomPhysicsEnabled={roomPhysicsEnabled} gravityForceEnabled={gravityForceEnabled} gravityStrength={gravityStrength} restitution={restitution} throwInertiaEnabled={throwInertiaEnabled} mass={1.3} linearDamping={0.2} angularDamping={0.74} surfaceFriction={0.39}>
          <sphereGeometry args={[3.1, 56, 56]} />
          <meshStandardMaterial color="#ffb35f" roughness={0.63} metalness={0.1} />
        </DraggableObject>
      </>
    );
  }

  return (
    <>
      {/* Close — polished pink gem */}
      <DraggableObject
        position={[3, -2, -6]}
        roomPhysicsEnabled={roomPhysicsEnabled}
        gravityForceEnabled={gravityForceEnabled}
        gravityStrength={gravityStrength}
        restitution={restitution}
        throwInertiaEnabled={throwInertiaEnabled}
        mass={0.9}
        linearDamping={0.18}
        angularDamping={0.68}
        surfaceFriction={0.38}
      >
        <octahedronGeometry args={[2.8, 0]} />
        <meshStandardMaterial color="#ff4fcc" roughness={0.1} metalness={0.7} />
      </DraggableObject>

      {/* Mid — brushed blue metal box */}
      <DraggableObject
        position={[-6.2, -3.8, -22]}
        roomPhysicsEnabled={roomPhysicsEnabled}
        gravityForceEnabled={gravityForceEnabled}
        gravityStrength={gravityStrength}
        restitution={restitution}
        throwInertiaEnabled={throwInertiaEnabled}
        mass={2.4}
        linearDamping={0.34}
        angularDamping={1.08}
        surfaceFriction={0.62}
      >
        <boxGeometry args={[5, 5, 5]} />
        <meshStandardMaterial color="#6b8cff" roughness={0.35} metalness={0.75} />
      </DraggableObject>

      {/* Far — warm matte sphere */}
      <DraggableObject
        position={[6.5, -4.4, -38]}
        roomPhysicsEnabled={roomPhysicsEnabled}
        gravityForceEnabled={gravityForceEnabled}
        gravityStrength={gravityStrength}
        restitution={restitution}
        throwInertiaEnabled={throwInertiaEnabled}
        mass={1.4}
        linearDamping={0.24}
        angularDamping={0.78}
        surfaceFriction={0.45}
      >
        <sphereGeometry args={[3.2, 64, 64]} />
        <meshStandardMaterial color="#ff8f3a" roughness={0.55} metalness={0.1} />
      </DraggableObject>

      {/* Deepest — glassy torus knot with slight emissive */}
      <DraggableObject
        position={[0.4, 0.6, -56]}
        roomPhysicsEnabled={roomPhysicsEnabled}
        gravityForceEnabled={gravityForceEnabled}
        gravityStrength={gravityStrength}
        restitution={restitution}
        throwInertiaEnabled={throwInertiaEnabled}
        mass={2.1}
        linearDamping={0.3}
        angularDamping={0.96}
        surfaceFriction={0.58}
      >
        <torusKnotGeometry args={[2.1, 0.65, 128, 24]} />
        <meshStandardMaterial color="#67f5d2" roughness={0.05} metalness={0.85} />
      </DraggableObject>
    </>
  );
};

// ── UI shadow plane ───────────────────────────────────────────────────────────
// Invisible mesh that matches the overlay panel's screen rect so the scene
// light projects its silhouette as a shadow onto the room walls and objects.
const UIShadowPlane = ({ rect }: { rect: DOMRect | null }) => {
  const { size } = useThree();
  if (!rect) return null;

  const hw = SCREEN_WIDTH_CM / 2;
  const hh = SCREEN_HEIGHT_CM / 2;

  const left   = (rect.left   / size.width)  * 2 - 1;
  const right  = (rect.right  / size.width)  * 2 - 1;
  const top    = -((rect.top    / size.height) * 2 - 1);
  const bottom = -((rect.bottom / size.height) * 2 - 1);

  const cx = ((left + right)  / 2) * hw;
  const cy = ((top  + bottom) / 2) * hh;
  const w  = (right - left)   * hw;
  const h  = (top   - bottom) * hh;

  return (
    <mesh position={[cx, cy, 0.5]} castShadow>
      <planeGeometry args={[w, h]} />
      {/* colorWrite:false — invisible to camera, still appears in shadow map */}
      <meshStandardMaterial colorWrite={false} />
    </mesh>
  );
};

const DEFAULT_HDRI = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/small_empty_room_3_1k.hdr";

// ── Shadow-casting spot (HDRI handles all ambient/specular) ───────────────────
const Lights = () => {
  const keyRef = useRef<THREE.SpotLight>(null);

  useEffect(() => {
    keyRef.current?.target.position.set(0, 0, -BOX_DEPTH_CM / 2);
    keyRef.current?.target.updateMatrixWorld();
  }, []);

  return (
    <>
      {/* Primary ceiling key light — casts shadows */}
      <spotLight
        ref={keyRef}
        position={[0, SCREEN_HEIGHT_CM * 1.5, DEFAULT_HEAD_Z_CM * 0.9]}
        angle={Math.PI / 4}
        penumbra={0.3}
        intensity={18000}
        distance={0}
        decay={2}
        color="#fff5e0"
        castShadow
        shadow-mapSize={[4096, 4096]}
        shadow-camera-near={40}
        shadow-camera-far={170}
        shadow-bias={-0.0003}
        shadow-normalBias={0.04}
        shadow-blurSamples={25}
      />
      {/* Low-angle side fill — grazes surfaces so normal maps read */}
      <directionalLight
        position={[SCREEN_WIDTH_CM * 2, SCREEN_HEIGHT_CM * 0.1, -BOX_DEPTH_CM * 0.5]}
        intensity={1.5}
        color="#c8deff"
      />
      {/* Counter-fill from opposite side */}
      <directionalLight
        position={[-SCREEN_WIDTH_CM * 0.5, -SCREEN_HEIGHT_CM * 0.8, -BOX_DEPTH_CM * 0.2]}
        intensity={0.8}
        color="#ffe8c8"
      />
    </>
  );
};

// ── Scene ─────────────────────────────────────────────────────────────────────
const Scene = ({
  headPose,
  pinchData,
  overlayRect,
  fingerGrab,
  hdriUrl,
  fingerSensitivity,
  roomPhysicsEnabled,
  gravityForceEnabled,
  gravityStrength,
  restitution,
  throwInertiaEnabled,
  objectCollection,
  objectResetToken,
}: {
  headPose: HeadPose;
  pinchData: PinchData;
  overlayRect: DOMRect | null;
  fingerGrab: boolean;
  hdriUrl: string;
  fingerSensitivity: number;
  roomPhysicsEnabled: boolean;
  gravityForceEnabled: boolean;
  gravityStrength: number;
  restitution: number;
  throwInertiaEnabled: boolean;
  objectCollection: ObjectCollectionId;
  objectResetToken: number;
}) => (
  <Canvas shadows="variance" camera={{ position: [0, 0, DEFAULT_HEAD_Z_CM], near: NEAR, far: FAR }} dpr={[1, 1.6]}>
    <CameraRig headPose={headPose} />
    <Lights />
    <HDRIEnvironment url={hdriUrl} />
    <Floor />
    <RoomWalls />
    <Roof />
    <RoomBoxOutline />
    <SceneObjects
      key={`${objectCollection}-${objectResetToken}`}
      roomPhysicsEnabled={roomPhysicsEnabled}
      gravityForceEnabled={gravityForceEnabled}
      gravityStrength={gravityStrength}
      restitution={restitution}
      throwInertiaEnabled={throwInertiaEnabled}
      collection={objectCollection}
    />
    <UIShadowPlane rect={overlayRect} />
    {fingerGrab && (
      <FingerDots
        pinchData={pinchData}
        sensitivity={fingerSensitivity}
        roomPhysicsEnabled={roomPhysicsEnabled}
        throwInertiaEnabled={throwInertiaEnabled}
      />
    )}
    <OrbitControls enablePan={false} enableZoom={false} enableRotate={false} />
    <EffectComposer multisampling={4}>
      <Vignette
        offset={0.3}
        darkness={0.7}
        eskil={false}
        blendFunction={BlendFunction.NORMAL}
      />
    </EffectComposer>
  </Canvas>
);

// ── Slider helper ─────────────────────────────────────────────────────────────
const Slider = ({
  label, value, min, max, step, onChange
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) => {
  const decimals = Math.min(
    3,
    Math.max(0, (String(step).split(".")[1] ?? "").length)
  );
  return (
    <label className="slider-row">
      <span className="slider-label">{label}</span>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
      />
      <span className="slider-value">{value.toFixed(decimals)}</span>
    </label>
  );
};

// ── App ───────────────────────────────────────────────────────────────────────
const App = () => {
  const persistedSettings = useMemo(() => readPersistedSettings(), []);
  const [autoRecalibrateToken, setAutoRecalibrateToken] = useState(0);
  const [objectResetToken, setObjectResetToken] = useState(0);
  const [params, setParams] = useState<SceneParams>({
    sensitivity: persistedSettings.sensitivity,
    depthCalibration: persistedSettings.depthCalibration,
    cameraOffsetMode: persistedSettings.cameraOffsetMode,
    cameraOffsetN: persistedSettings.cameraOffsetN,
  });
  const [fingerGrab, setFingerGrab] = useState(persistedSettings.fingerGrab);
  const [roomPhysicsEnabled, setRoomPhysicsEnabled] = useState(persistedSettings.roomPhysicsEnabled);
  const [gravityForceEnabled, setGravityForceEnabled] = useState(persistedSettings.gravityForceEnabled);
  const [gravityStrength, setGravityStrength] = useState(persistedSettings.gravityStrength);
  const [restitution, setRestitution] = useState(persistedSettings.restitution);
  const [throwInertiaEnabled, setThrowInertiaEnabled] = useState(persistedSettings.throwInertiaEnabled);
  const [objectCollection, setObjectCollection] = useState<ObjectCollectionId>(
    persistedSettings.objectCollection
  );
  const [settingsCollapsed, setSettingsCollapsed] = useState(persistedSettings.settingsCollapsed);
  const [advancedSettingsExpanded, setAdvancedSettingsExpanded] = useState(
    persistedSettings.advancedSettingsExpanded
  );
  const [overlayRect, setOverlayRect] = useState<DOMRect | null>(null);
  const overlayRef = useRef<HTMLElement>(null);

  const [viewportNarrow, setViewportNarrow] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia("(max-width: 768px)").matches : false
  );
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 768px)");
    const onChange = () => setViewportNarrow(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const isMobileDevice = useMemo(
    () =>
      typeof navigator !== "undefined" &&
      (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || viewportNarrow),
    [viewportNarrow]
  );

  const allowHeavyFeatures = !isMobileDevice;

  useEffect(() => {
    if (!isMobileDevice) return;
    setFingerGrab(false);
    setRoomPhysicsEnabled(false);
    setGravityForceEnabled(false);
    setThrowInertiaEnabled(false);
  }, [isMobileDevice]);

  const {
    videoRef,
    canvasRef,
    trackingState,
    autoCalibrationLocked,
    autoCalibrationProgress,
  } = useFaceTracking(
    params,
    autoRecalibrateToken
  );

  useLayoutEffect(() => {
    const update = () => {
      if (overlayRef.current) setOverlayRect(overlayRef.current.getBoundingClientRect());
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  const set = (key: keyof SceneParams) => (v: number) =>
    setParams(p => ({ ...p, [key]: v }));

  useEffect(() => {
    if (typeof window === "undefined") return;
    const payload: PersistedAppSettings = {
      sensitivity: params.sensitivity,
      depthCalibration: params.depthCalibration,
      cameraOffsetMode: params.cameraOffsetMode,
      cameraOffsetN: params.cameraOffsetN,
      fingerGrab,
      roomPhysicsEnabled,
      gravityForceEnabled,
      gravityStrength,
      restitution,
      throwInertiaEnabled,
      objectCollection,
      settingsCollapsed,
      advancedSettingsExpanded,
    };
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(payload));
  }, [
    params.sensitivity,
    params.depthCalibration,
    params.cameraOffsetMode,
    params.cameraOffsetN,
    fingerGrab,
    roomPhysicsEnabled,
    gravityForceEnabled,
    gravityStrength,
    restitution,
    throwInertiaEnabled,
    objectCollection,
    settingsCollapsed,
    advancedSettingsExpanded,
  ]);

  const resetSettings = () => {
    const defaults = getDefaultPersistedSettings();
    setParams({
      sensitivity: defaults.sensitivity,
      depthCalibration: defaults.depthCalibration,
      cameraOffsetMode: defaults.cameraOffsetMode,
      cameraOffsetN: defaults.cameraOffsetN,
    });
    setFingerGrab(defaults.fingerGrab);
    setRoomPhysicsEnabled(defaults.roomPhysicsEnabled);
    setGravityForceEnabled(defaults.gravityForceEnabled);
    setGravityStrength(defaults.gravityStrength);
    setRestitution(defaults.restitution);
    setThrowInertiaEnabled(defaults.throwInertiaEnabled);
    setObjectCollection(defaults.objectCollection);
    setSettingsCollapsed(defaults.settingsCollapsed);
    setAdvancedSettingsExpanded(defaults.advancedSettingsExpanded);
    setAutoRecalibrateToken((v) => v + 1);
  };

  return (
    <main className="app-shell">
      <Scene
        headPose={trackingState.headPose}
        pinchData={trackingState.pinchData}
        overlayRect={overlayRect}
        fingerGrab={fingerGrab}
        hdriUrl={DEFAULT_HDRI}
        fingerSensitivity={params.sensitivity}
        roomPhysicsEnabled={roomPhysicsEnabled}
        gravityForceEnabled={gravityForceEnabled}
        gravityStrength={gravityStrength}
        restitution={restitution}
        throwInertiaEnabled={throwInertiaEnabled}
        objectCollection={objectCollection}
        objectResetToken={objectResetToken}
      />

      {params.cameraOffsetMode === "auto" && !autoCalibrationLocked && (
        <div className="auto-calibration-banner" role="status" aria-live="polite">
          <strong>Auto camera calibration</strong>
          <span>Move until you are centered with the screen/window.</span>
          <div className="auto-calibration-progress">
            <div
              className="auto-calibration-progress-fill"
              style={{ width: `${Math.round(autoCalibrationProgress * 100)}%` }}
            />
          </div>
        </div>
      )}

      <aside className="tracker-overlay" ref={overlayRef as React.RefObject<HTMLDivElement>}>
        <button
          type="button"
          className={`tracker-header tracker-header-btn ${settingsCollapsed ? "tracker-header-collapsed" : ""}`}
          aria-expanded={!settingsCollapsed}
          onClick={() => setSettingsCollapsed((v) => !v)}
        >
          <strong>Face Tracker</strong>
          <div className="tracker-header-right">
            <span className={trackingState.headPose.hasFace ? "status status-live" : "status status-idle"}>
              {trackingState.loading ? "loading" : trackingState.headPose.hasFace ? "live" : "no face"}
            </span>
            <span className="tracker-collapse-indicator">{settingsCollapsed ? "open" : "close"}</span>
          </div>
        </button>

        <div className={`tracker-video-wrapper ${settingsCollapsed ? "tracker-video-wrapper-hidden" : ""}`}>
          <video ref={videoRef} className="tracker-video" muted playsInline />
          <canvas ref={canvasRef} className="tracker-canvas" width={VIDEO_WIDTH} height={VIDEO_HEIGHT} />
        </div>

        {!settingsCollapsed && (
          <>
            <div className="sliders">
              <Slider label="Sensitivity"   value={params.sensitivity}      min={0.5} max={5}  step={0.1} onChange={set("sensitivity")} />
              <Slider label="Depth cal."    value={params.depthCalibration} min={1}   max={12} step={0.1} onChange={set("depthCalibration")} />
              <label className="slider-row toggle-row">
                <span className="slider-label">Cam side</span>
                <select
                  className="light-select"
                  value={params.cameraOffsetMode}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      cameraOffsetMode: e.target.value as CameraOffsetMode,
                    }))
                  }
                >
                  <option value="auto">Auto (calibrate once)</option>
                  <option value="center">Centered screen (static)</option>
                  <option value="left">Left screen</option>
                  <option value="right">Right screen</option>
                </select>
              </label>
              {params.cameraOffsetMode === "auto" && (
                <label className="slider-row toggle-row">
                  <span className="slider-label">Auto align</span>
                  <button
                    type="button"
                    className={`toggle-btn ${autoCalibrationLocked ? "toggle-btn-on" : ""}`}
                    onClick={() => setAutoRecalibrateToken((v) => v + 1)}
                  >
                    {autoCalibrationLocked ? "Locked — recalibrate" : "Calibrating... click to restart"}
                  </button>
                </label>
              )}
              <Slider
                label="Cam offset"
                value={params.cameraOffsetN}
                min={-0.3}
                max={0.3}
                step={0.005}
                onChange={set("cameraOffsetN")}
              />
            </div>

            <label className="slider-row toggle-row" style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(108,130,162,0.25)" }}>
              <span className="slider-label" style={{ color: "#d7c39d" }}>Objects</span>
              <select
                className="light-select"
                value={objectCollection}
                onChange={(e) => setObjectCollection(e.target.value as ObjectCollectionId)}
              >
                <option value="starter">Starter set</option>
                <option value="dense">Dense set</option>
                <option value="chaos">Chaos set</option>
              </select>
            </label>

            <button
              type="button"
              className="advanced-settings-toggle"
              aria-expanded={advancedSettingsExpanded}
              onClick={() => setAdvancedSettingsExpanded((v) => !v)}
            >
              Advanced settings
              <span className="tracker-collapse-indicator">{advancedSettingsExpanded ? "hide" : "show"}</span>
            </button>

            {advancedSettingsExpanded && (
              <div className="advanced-settings-block">
                {!allowHeavyFeatures && (
                  <p className="advanced-settings-mobile-hint">
                    Heavy options are disabled on small or touch devices.
                  </p>
                )}

                <label className="slider-row toggle-row">
                  <span className="slider-label" title={!allowHeavyFeatures ? "Not available on this device" : undefined}>
                    Finger grab <span className="settings-cpu-tag">CPU</span>
                  </span>
                  <button
                    type="button"
                    disabled={!allowHeavyFeatures}
                    className={`toggle-btn ${fingerGrab ? "toggle-btn-on" : ""}`}
                    title={!allowHeavyFeatures ? "Not available on mobile / narrow viewports" : undefined}
                    onClick={() => allowHeavyFeatures && setFingerGrab((v) => !v)}
                  >
                    {fingerGrab
                      ? trackingState.pinchData.isPinching
                        ? "ON — grabbing"
                        : trackingState.pinchData.hasHand
                          ? "ON — hand tracked"
                          : "ON — no hand"
                      : "OFF"}
                  </button>
                </label>

                <label className="slider-row toggle-row">
                  <span className="slider-label" title={!allowHeavyFeatures ? "Not available on this device" : undefined}>
                    Room physics <span className="settings-cpu-tag">CPU</span>
                  </span>
                  <button
                    type="button"
                    disabled={!allowHeavyFeatures}
                    className={`toggle-btn ${roomPhysicsEnabled ? "toggle-btn-on" : ""}`}
                    title={!allowHeavyFeatures ? "Not available on mobile / narrow viewports" : "Collision, bounce, damping, sleep"}
                    onClick={() => {
                      if (!allowHeavyFeatures) return;
                      setRoomPhysicsEnabled((v) => {
                        const next = !v;
                        if (!next) {
                          setGravityForceEnabled(false);
                          setThrowInertiaEnabled(false);
                        }
                        return next;
                      });
                    }}
                  >
                    {roomPhysicsEnabled ? "ON — walls + floor" : "OFF"}
                  </button>
                </label>

                <label className="slider-row toggle-row">
                  <span
                    className="slider-label"
                    title={
                      !allowHeavyFeatures
                        ? "Not available on this device"
                        : !roomPhysicsEnabled
                          ? "Turn on Room physics first"
                          : undefined
                    }
                  >
                    Downward gravity <span className="settings-cpu-tag">CPU</span>
                  </span>
                  <button
                    type="button"
                    disabled={!allowHeavyFeatures || !roomPhysicsEnabled}
                    className={`toggle-btn ${gravityForceEnabled ? "toggle-btn-on" : ""}`}
                    onClick={() =>
                      allowHeavyFeatures && roomPhysicsEnabled && setGravityForceEnabled((v) => !v)
                    }
                  >
                    {gravityForceEnabled
                      ? gravityStrength <= 0
                        ? "ON — strength 0"
                        : "ON"
                      : "OFF"}
                  </button>
                </label>

                <label className="slider-row toggle-row">
                  <span
                    className="slider-label"
                    title={
                      !allowHeavyFeatures
                        ? "Not available on this device"
                        : !roomPhysicsEnabled
                          ? "Turn on Room physics first"
                          : undefined
                    }
                  >
                    Throw inertia <span className="settings-cpu-tag">CPU</span>
                  </span>
                  <button
                    type="button"
                    disabled={!allowHeavyFeatures || !roomPhysicsEnabled}
                    className={`toggle-btn ${throwInertiaEnabled ? "toggle-btn-on" : ""}`}
                    onClick={() =>
                      allowHeavyFeatures && roomPhysicsEnabled && setThrowInertiaEnabled((v) => !v)
                    }
                  >
                    {throwInertiaEnabled ? "ON — mouse / hand" : "OFF"}
                  </button>
                </label>

                {roomPhysicsEnabled && (
                  <>
                    <Slider
                      label="Gravity strength"
                      value={gravityStrength}
                      min={0}
                      max={220}
                      step={2}
                      onChange={setGravityStrength}
                    />
                    <Slider
                      label="Bounce"
                      value={restitution}
                      min={0}
                      max={0.95}
                      step={0.01}
                      onChange={setRestitution}
                    />
                  </>
                )}
              </div>
            )}

            <div className="settings-actions">
              <button
                type="button"
                className="reset-objects-btn"
                onClick={() => setObjectResetToken((v) => v + 1)}
              >
                Reset objects
              </button>
              <button type="button" className="reset-btn" onClick={resetSettings}>
                Reset to defaults
              </button>
            </div>

            {trackingState.error && <p className="tracker-error">{trackingState.error}</p>}
          </>
        )}
      </aside>
    </main>
  );
};

export default App;
