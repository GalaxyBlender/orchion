export type ModelType = "asr" | "tts" | "ocr" | "llm";

export type ModelCapability =
  | "asr_transcription"
  | "asr_streaming"
  | "tts_voice_cloning"
  | "tts_preset_speakers"
  | "tts_voice_design"
  | "ocr_text"
  | "ocr_layout"
  | "ocr_table_structure"
  | "ocr_vision_language"
  | "ocr_markdown"
  | "ocr_html"
  | "llm_chat"
  | "llm_responses"
  | "llm_completions"
  | "llm_input_tokens"
  | "llm_tools"
  | "llm_parallel_tools"
  | "llm_json_object"
  | "llm_json_schema"
  | "llm_logprobs"
  | "llm_logit_bias"
  | "llm_multiple_choices"
  | "llm_reasoning"
  | "llm_reasoning_control"
  | "llm_vision"
  | "llm_streaming"
  | "llm_embeddings"
  | "llm_resumable_streaming";

export interface ApiSettings {
  serverBaseUrl: string;
  apiKey: string;
}

export interface ModelObject {
  id: string;
  type?: ModelType;
  name?: string;
  capabilities: ModelCapability[];
  capability_details?: {
    max_choices: number;
    max_top_logprobs: number;
    legacy_max_logprobs: number;
    strict_json_schema: boolean;
    runtime_template_validation: boolean;
  };
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
