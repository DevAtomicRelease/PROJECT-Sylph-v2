/**
 * DualPayloadParser — Parses binary WebSocket frames from the sidecar
 * Phase 4.5: Extracts audio PCM + viseme timeline from a single binary message
 *
 * Binary frame layout (from Python payload.py):
 * [Magic(2)] [Version(1)] [VisemeLen(4)] [Flags(1)] [VisemeJSON(N)] [AudioPCM(M)]
 */

export interface VisemeKeyframe {
  /** Start time in milliseconds */
  time_ms: number;
  /** Duration in milliseconds */
  duration_ms: number;
  /** Weights: [aa, ee, ih, oh, ou] */
  weights: [number, number, number, number, number];
}

export interface DualPayload {
  version: number;
  flags: number;
  isFinal: boolean;
  /** Turn this audio belongs to — chunks from superseded turns are dropped */
  turnId: number;
  sampleRate: number;
  timeline: VisemeKeyframe[];
  audio: Float32Array;
}

const MAGIC = "AV";
const FLAG_FINAL = 0x01;

export function parseDualPayload(buffer: ArrayBuffer): DualPayload {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Validate magic bytes
  const magic = String.fromCharCode(bytes[0], bytes[1]);
  if (magic !== MAGIC) {
    throw new Error(`Invalid magic: "${magic}", expected "${MAGIC}"`);
  }

  // Parse header (little-endian). v2 adds uint32 turn_id after flags.
  const version = view.getUint8(2);
  const visemeLen = view.getUint32(3, true); // little-endian
  const flags = view.getUint8(7);
  const isFinal = (flags & FLAG_FINAL) !== 0;
  const turnId = version >= 2 ? view.getUint32(8, true) : 0;

  // Parse viseme JSON
  const visemeStart = version >= 2 ? 12 : 8;
  const visemeEnd = visemeStart + visemeLen;
  const decoder = new TextDecoder("utf-8");
  const visemeJsonStr = decoder.decode(buffer.slice(visemeStart, visemeEnd));
  const visemeData = JSON.parse(visemeJsonStr);

  // Parse compact timeline: [[time_ms, dur_ms, aa, ee, ih, oh, ou], ...]
  const rawTimeline: number[][] = visemeData.timeline || [];
  const timeline: VisemeKeyframe[] = rawTimeline.map((kf) => ({
    time_ms: kf[0],
    duration_ms: kf[1],
    weights: [kf[2], kf[3], kf[4], kf[5], kf[6]] as [
      number,
      number,
      number,
      number,
      number,
    ],
  }));

  // Parse audio PCM (float32)
  const audioStart = visemeEnd;
  const audio = new Float32Array(buffer.slice(audioStart));

  return {
    version,
    flags,
    isFinal,
    turnId,
    sampleRate: visemeData.sample_rate || 24000,
    timeline,
    audio,
  };
}
