export type AsrResponseFormat = "json" | "text" | "verbose_json" | "srt";
export type AsrMode = "file" | "stream";
export type AsrStreamInputMode = "microphone" | "file";
export type AsrStreamInputFormat = "auto" | "webm_opus" | "mp3" | "wav" | "m4a" | "aac" | "flac" | "ogg";

export interface AsrFormState {
  model: string;
  language: string;
  responseFormat: AsrResponseFormat;
  prompt: string;
  temperature: string;
  timestampGranularities: string[];
}

export interface AsrRequestInput extends AsrFormState {
  file: File;
}

export interface ParameterMetadata {
  name: string;
  label: string;
  defaultValue: string;
  description: string;
  required: boolean;
  supported: boolean;
  notice?: string;
  options?: readonly string[];
}

export interface AsrStreamReadyEvent {
  type: "ready";
}

export interface AsrStreamTranscriptEvent {
  type: "partial" | "final";
  text: string;
}

export interface AsrStreamErrorEvent {
  type: "error";
  error: {
    message: string;
    type?: string;
    code?: string | null;
    param?: string | null;
  };
}

export type AsrStreamEvent =
  | AsrStreamReadyEvent
  | AsrStreamTranscriptEvent
  | AsrStreamErrorEvent;
