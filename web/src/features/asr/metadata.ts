import type { ParameterMetadata } from "./types";

export const asrResponseFormats = ["json", "text", "verbose_json", "srt"] as const;
export const asrTimestampGranularityOptions = ["", "segment", "word"] as const;

export const asrLanguageOptions = [
  "",
  "English",
  "Chinese",
  "Japanese",
  "Korean",
  "German",
  "French",
  "Russian",
  "Portuguese",
  "Spanish",
  "Italian",
] as const;

export const asrParameterMetadata: ParameterMetadata[] = [
  {
    name: "file",
    label: "Audio file",
    defaultValue: "",
    description: "Audio file to transcribe.",
    required: true,
    supported: true,
  },
  {
    name: "model",
    label: "Model",
    defaultValue: "",
    description: "ASR model identifier to use for transcription.",
    required: true,
    supported: true,
  },
  {
    name: "language",
    label: "Language",
    defaultValue: "",
    description: "Optional spoken language hint. Leave empty for automatic detection.",
    required: false,
    supported: true,
    options: asrLanguageOptions,
  },
  {
    name: "response_format",
    label: "Response format",
    defaultValue: "json",
    description: "Transcription response shape returned by the backend.",
    required: false,
    supported: true,
    options: asrResponseFormats,
  },
  {
    name: "prompt",
    label: "Prompt",
    defaultValue: "",
    description: "Optional initial context for streaming transcription.",
    required: false,
    supported: true,
    notice: "Prompt is supported by streaming transcription only.",
  },
  {
    name: "temperature",
    label: "Temperature",
    defaultValue: "",
    description: "Optional OpenAI-compatible sampling temperature field.",
    required: false,
    supported: false,
    notice: "Temperature is not accepted because the ASR runtime does not expose sampling temperature.",
  },
  {
    name: "timestamp_granularities",
    label: "Timestamp granularities",
    defaultValue: "",
    description: "Optional timestamp granularity for transcription output. Segment is supported; word is not supported.",
    required: false,
    supported: true,
    options: asrTimestampGranularityOptions,
    notice: "Orchion segment timestamps are supported; word timestamps are not supported and are not sent.",
  },
];
