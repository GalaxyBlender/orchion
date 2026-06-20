export type TtsMode = "preset" | "clone" | "design";

export type TtsResponseFormat = "wav" | "mp3" | "aac" | "opus" | "flac" | "pcm";

export interface TtsFormState {
  mode: TtsMode;
  model: string;
  input: string;
  language: string;
  responseFormat: TtsResponseFormat;
  speaker: string;
  referenceAudioName: string;
  referenceText: string;
  voicePrompt: string;
  speed: string;
  seed: string;
  temperature: string;
  topK: string;
  topP: string;
  repetitionPenalty: string;
  maxLength: string;
}

export interface TtsRequestInput extends TtsFormState {
  referenceAudio?: File | null;
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

export interface TtsModeMetadata {
  value: TtsMode;
  label: string;
  description: string;
}

export interface TtsSpeakerOption {
  value: string;
  label: string;
}

export type TtsJsonBody = Record<string, string | number>;

export type TtsPayload =
  | {
      kind: "json";
      body: TtsJsonBody;
      headers: { "Content-Type": "application/json" };
    }
  | {
      kind: "multipart";
      formData: FormData;
      headers: Record<string, never>;
    };
