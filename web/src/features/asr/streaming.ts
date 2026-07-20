import type {
  AsrFormState,
  AsrCaptionEndpointingOptions,
  AsrStreamCaptionPartialEvent,
  AsrStreamEvent,
  AsrStreamInputFormat,
  AsrStreamOutputMode,
} from "./types";

export const asrStreamEndpointPath = "/v1/audio/transcriptions/stream";
export const asrCaptionVadFrameDurationMs = 30;
export const asrCaptionMaxCandidateMs = 60_000;
export const asrCaptionMaxDisplaySegments = 300;
export const asrStreamBufferedAmountHighWatermark = 1024 * 1024;
export const asrStreamBufferedAmountLowWatermark = 256 * 1024;

interface StoppableMediaStream {
  getTracks(): readonly { stop(): void }[];
}

interface BufferedSocket {
  readonly readyState: number;
  readonly bufferedAmount: number;
}

export type AsrCaptionEndpointingValidationError =
  | "invalid"
  | "invalid_candidate_window"
  | "invalid_rounded_window";

export interface AsrStreamStartInput {
  form: AsrFormState;
  inputAudioFormat: AsrStreamInputFormat;
  outputMode?: AsrStreamOutputMode;
  endpointing?: AsrCaptionEndpointingOptions;
  apiKey: string;
}

export type MicrophoneSupportIssue =
  | "insecure_context"
  | "media_devices_unavailable"
  | "media_recorder_unavailable";

export function buildAsrStreamUrl(path: string = asrStreamEndpointPath): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const baseUrl = new URL(normalizedPath, window.location.origin);
  baseUrl.protocol = baseUrl.protocol === "https:" ? "wss:" : "ws:";

  return baseUrl.toString();
}

export function buildAsrStreamStartMessage(input: AsrStreamStartInput): string {
  const message: Record<string, unknown> = {
    type: "start",
    model: input.form.model.trim(),
    response_format: streamResponseFormat(input.form.responseFormat),
    input_audio_format: input.inputAudioFormat,
  };
  if (input.outputMode === "caption") {
    message.mode = "caption";
    if (input.endpointing) {
      message.endpointing = input.endpointing;
    }
  }
  appendNonblank(message, "language", input.form.language);
  appendNonblank(message, "prompt", input.form.prompt);
  appendNonblank(message, "api_key", input.apiKey);

  return JSON.stringify(message);
}

export async function acquireAsrMicrophoneStream<T extends StoppableMediaStream>(
  getUserMedia: () => Promise<T>,
  isSessionActive: () => boolean,
): Promise<T | null> {
  const stream = await getUserMedia();
  if (isSessionActive()) {
    return stream;
  }

  stream.getTracks().forEach((track) => track.stop());
  return null;
}

export async function waitForAsrStreamWritable(
  socket: BufferedSocket,
  isSessionActive: () => boolean,
  wait: () => Promise<void> = waitForBufferedAmountPoll,
): Promise<boolean> {
  if (socket.readyState !== WebSocket.OPEN || !isSessionActive()) {
    return false;
  }
  if (socket.bufferedAmount <= asrStreamBufferedAmountHighWatermark) {
    return true;
  }

  while (socket.bufferedAmount > asrStreamBufferedAmountLowWatermark) {
    await wait();
    if (socket.readyState !== WebSocket.OPEN || !isSessionActive()) {
      return false;
    }
  }
  return true;
}

export function parseAsrStreamEvent(text: string): AsrStreamEvent {
  return JSON.parse(text) as AsrStreamEvent;
}

export function isAsrCaptionPartialEvent(event: AsrStreamEvent): event is AsrStreamCaptionPartialEvent {
  return event.type === "partial" && "segment_id" in event;
}

export function validateAsrCaptionEndpointingOptions(
  endpointing: AsrCaptionEndpointingOptions,
): AsrCaptionEndpointingValidationError | null {
  if (
    !isNonnegativeInteger(endpointing.speech_padding_ms) ||
    !isPositiveInteger(endpointing.min_speech_ms) ||
    !isPositiveInteger(endpointing.min_silence_ms)
  ) {
    return "invalid";
  }

  const candidateMs = endpointing.speech_padding_ms + endpointing.min_speech_ms;
  if (candidateMs > asrCaptionMaxCandidateMs) {
    return "invalid_candidate_window";
  }

  const roundedMinSpeechMs = Math.ceil(endpointing.min_speech_ms / asrCaptionVadFrameDurationMs) * asrCaptionVadFrameDurationMs;
  if (candidateMs < roundedMinSpeechMs) {
    return "invalid_rounded_window";
  }

  return null;
}

export function upsertBoundedAsrCaptionSegments<T extends { id: number }>(
  segments: readonly T[],
  segment: T,
  maxSegments: number = asrCaptionMaxDisplaySegments,
): T[] {
  const index = segments.findIndex((currentSegment) => currentSegment.id === segment.id);
  const nextSegments = index === -1
    ? [...segments, segment]
    : segments.map((currentSegment, currentIndex) => (currentIndex === index ? segment : currentSegment));

  if (nextSegments.length <= maxSegments) {
    return nextSegments;
  }
  return nextSegments.slice(nextSegments.length - maxSegments);
}

export function detectAsrStreamInputFormat(file: File): AsrStreamInputFormat | null {
  const name = file.name.toLowerCase();
  const mimeType = file.type.toLowerCase();

  if (name.endsWith(".wav") || mimeType === "audio/wav" || mimeType === "audio/x-wav") {
    return "wav";
  }
  if (name.endsWith(".mp3") || mimeType === "audio/mpeg" || mimeType === "audio/mp3") {
    return "mp3";
  }
  if (name.endsWith(".m4a") || mimeType === "audio/mp4" || mimeType === "audio/x-m4a") {
    return "m4a";
  }
  if (name.endsWith(".aac") || mimeType === "audio/aac") {
    return "aac";
  }
  if (name.endsWith(".flac") || mimeType === "audio/flac" || mimeType === "audio/x-flac") {
    return "flac";
  }
  if (name.endsWith(".ogg") || name.endsWith(".opus") || mimeType === "audio/ogg" || mimeType === "audio/opus") {
    return "ogg";
  }
  if (name.endsWith(".webm") || mimeType.includes("webm") || mimeType.includes("opus")) {
    return "webm_opus";
  }

  return null;
}

export function preferredMicrophoneMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") {
    return undefined;
  }
  if (typeof MediaRecorder.isTypeSupported !== "function") {
    return undefined;
  }
  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
    return "audio/webm;codecs=opus";
  }
  if (MediaRecorder.isTypeSupported("audio/webm")) {
    return "audio/webm";
  }

  return undefined;
}

export function detectMicrophoneSupportIssue(): MicrophoneSupportIssue | null {
  if (typeof window !== "undefined" && !window.isSecureContext) {
    return "insecure_context";
  }
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return "media_devices_unavailable";
  }
  if (typeof MediaRecorder === "undefined") {
    return "media_recorder_unavailable";
  }

  return null;
}

export function formatAsrStreamResult(event: AsrStreamEvent): string {
  return JSON.stringify(event, null, 2);
}

function appendNonblank(message: Record<string, unknown>, name: string, value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    message[name] = trimmedValue;
  }
}

function isPositiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function isNonnegativeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

function streamResponseFormat(_format: AsrFormState["responseFormat"]): "json" {
  return "json";
}

function waitForBufferedAmountPoll(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 25));
}
