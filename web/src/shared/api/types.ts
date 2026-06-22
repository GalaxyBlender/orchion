export type ModelType = "asr" | "tts" | "ocr";

export type ModelSubtype =
  | "standard"
  | "vl"
  | "layout"
  | "preset_voice"
  | "voice_clone"
  | "voice_design";

export interface ApiSettings {
  serverBaseUrl: string;
  apiKey: string;
}

export interface ModelObject {
  id: string;
  type?: ModelType;
  subtype?: ModelSubtype;
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
