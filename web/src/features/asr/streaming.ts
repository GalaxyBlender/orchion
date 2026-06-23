import type { AsrFormState, AsrStreamEvent, AsrStreamInputFormat } from "./types";

export const asrStreamEndpointPath = "/v1/audio/transcriptions/stream";

export interface AsrStreamStartInput {
  form: AsrFormState;
  inputAudioFormat: AsrStreamInputFormat;
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
  const message: Record<string, string> = {
    type: "start",
    model: input.form.model.trim(),
    response_format: streamResponseFormat(input.form.responseFormat),
    input_audio_format: input.inputAudioFormat,
  };
  appendNonblank(message, "language", input.form.language);
  appendNonblank(message, "prompt", input.form.prompt);
  appendNonblank(message, "temperature", input.form.temperature);
  appendNonblank(message, "api_key", input.apiKey);

  return JSON.stringify(message);
}

export function parseAsrStreamEvent(text: string): AsrStreamEvent {
  return JSON.parse(text) as AsrStreamEvent;
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

function appendNonblank(message: Record<string, string>, name: string, value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    message[name] = trimmedValue;
  }
}

function streamResponseFormat(_format: AsrFormState["responseFormat"]): "json" {
  return "json";
}
