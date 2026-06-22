export type OcrResponseFormat = "json" | "text" | "markdown" | "html";

export type OcrTask = "ocr" | "table" | "formula" | "chart" | "spotting" | "seal";

export interface OcrFormState {
  model: string;
  responseFormat: OcrResponseFormat;
  task: OcrTask;
  layoutModel: string;
  maxTokens: string;
}

export interface OcrRequestInput extends OcrFormState {
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
