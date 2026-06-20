export type AsrResponseFormat = "json" | "text" | "verbose_json" | "srt";

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
