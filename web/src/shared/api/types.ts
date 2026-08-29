export type ModelType = "asr" | "tts" | "ocr";

export type ModelCapability =
  | "asr_transcription"
  | "asr_streaming"
  | "tts_voice_cloning"
  | "tts_preset_speakers"
  | "tts_voice_design"
  | "ocr_text"
  | "ocr_layout"
  | "ocr_vision_language"
  | "ocr_markdown"
  | "ocr_html";

export interface ApiSettings {
  serverBaseUrl: string;
  apiKey: string;
}

export interface ModelObject {
  id: string;
  type?: ModelType;
  name?: string;
  capabilities: ModelCapability[];
  object?: string;
  created?: number;
  owned_by?: string;
  [key: string]: unknown;
}

export interface ModelList {
  object?: string;
  data: ModelObject[];
}

export interface ApiErrorDetail {
  message: string;
  type?: string;
  code?: string | null;
  param?: string | null;
  status?: number;
}

export class ApiRequestError extends Error {
  readonly detail: ApiErrorDetail;

  constructor(detail: ApiErrorDetail) {
    super(detail.message);
    this.name = "ApiRequestError";
    this.detail = detail;
  }
}
