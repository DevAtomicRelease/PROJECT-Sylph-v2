/**
 * AudioRecorder — Microphone capture for STT input
 *
 * Optimized: uses an AudioWorklet (dedicated audio-render thread) instead of
 * the deprecated ScriptProcessorNode. ScriptProcessor runs its callback on
 * the MAIN thread — the same thread driving the three.js render loop — so
 * mic capture and avatar rendering used to fight each other, causing both
 * dropped audio frames and animation jank. The worklet removes that
 * contention entirely. ScriptProcessor is kept only as a fallback for
 * WebViews without worklet support.
 *
 * Output: 512-sample (32ms) float32 frames at 16kHz, streamed to the sidecar.
 */

import { sidecarSocket } from "./SidecarSocket";
import { audioPlayer } from "./AudioPlayer";

const FRAME_SIZE = 512;

/** Inline worklet processor: buffers 128-sample quanta into 512-sample frames. */
const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(${FRAME_SIZE});
    this.offset = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];
    let i = 0;
    while (i < samples.length) {
      const space = ${FRAME_SIZE} - this.offset;
      const n = Math.min(space, samples.length - i);
      this.buffer.set(samples.subarray(i, i + n), this.offset);
      this.offset += n;
      i += n;
      if (this.offset === ${FRAME_SIZE}) {
        const frame = this.buffer;
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.buffer = new Float32Array(${FRAME_SIZE});
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("sylph-capture", CaptureProcessor);
`;

export class AudioRecorder {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private processorNode: ScriptProcessorNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private sampleBuffer: number[] = [];
  private isRecordingActive = false;
  private workletUrl: string | null = null;

  async start(): Promise<void> {
    if (this.isRecordingActive) return;
    this.isRecordingActive = true;
    this.sampleBuffer = [];

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // 16kHz context: the browser resamples the mic input for us.
      this.audioContext = new AudioContext({ sampleRate: 16000 });
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      try {
        await this.startWorklet(this.audioContext, this.sourceNode);
        console.log("[AudioRecorder] Recording via AudioWorklet at 16kHz");
      } catch (workletErr) {
        console.warn("[AudioRecorder] AudioWorklet unavailable, falling back to ScriptProcessor:", workletErr);
        this.startScriptProcessorFallback(this.audioContext, this.sourceNode);
      }

      sidecarSocket.send("log", {
        level: "info",
        message: "[AudioRecorder] Recording started at 16kHz",
      });
    } catch (error) {
      console.error("[AudioRecorder] Failed to start audio recording:", error);
      sidecarSocket.send("log", {
        level: "error",
        message: `[AudioRecorder] Failed to start recording: ${
          error instanceof Error ? error.stack || error.message : String(error)
        }`,
      });
      this.stop();
      throw error;
    }
  }

  private async startWorklet(
    ctx: AudioContext,
    source: MediaStreamAudioSourceNode
  ): Promise<void> {
    if (!this.workletUrl) {
      const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
      this.workletUrl = URL.createObjectURL(blob);
    }
    await ctx.audioWorklet.addModule(this.workletUrl);

    this.workletNode = new AudioWorkletNode(ctx, "sylph-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    });

    this.workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      if (!this.isRecordingActive) return;
      // Half-duplex: don't feed Sylph's own voice back into the VAD
      if (audioPlayer.playing) return;
      sidecarSocket.sendBinary(event.data);
    };

    source.connect(this.workletNode);
  }

  /** Legacy path for WebViews without AudioWorklet support. */
  private startScriptProcessorFallback(
    ctx: AudioContext,
    source: MediaStreamAudioSourceNode
  ): void {
    this.processorNode = ctx.createScriptProcessor(2048, 1, 1);
    this.processorNode.onaudioprocess = (event) => {
      if (!this.isRecordingActive) return;
      if (audioPlayer.playing) {
        this.sampleBuffer = [];
        return;
      }
      const inputData = event.inputBuffer.getChannelData(0);
      for (let i = 0; i < inputData.length; i++) {
        this.sampleBuffer.push(inputData[i]);
      }
      while (this.sampleBuffer.length >= FRAME_SIZE) {
        const chunk = this.sampleBuffer.splice(0, FRAME_SIZE);
        sidecarSocket.sendBinary(new Float32Array(chunk));
      }
    };
    source.connect(this.processorNode);
    this.processorNode.connect(ctx.destination);
  }

  stop(): void {
    this.isRecordingActive = false;

    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch { /* already disconnected */ }
      this.workletNode = null;
    }

    if (this.processorNode) {
      try {
        this.processorNode.disconnect();
      } catch { /* already disconnected */ }
      this.processorNode.onaudioprocess = null;
      this.processorNode = null;
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch { /* already disconnected */ }
      this.sourceNode = null;
    }

    if (this.audioContext) {
      if (this.audioContext.state !== "closed") {
        this.audioContext.close().catch(() => {});
      }
      this.audioContext = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => {
        try { track.stop(); } catch { /* already stopped */ }
      });
      this.mediaStream = null;
    }

    this.sampleBuffer = [];
    console.log("[AudioRecorder] Recording stopped and resources cleaned up");
  }

  get isRecording(): boolean {
    return this.isRecordingActive;
  }
}

export const audioRecorder = new AudioRecorder();
