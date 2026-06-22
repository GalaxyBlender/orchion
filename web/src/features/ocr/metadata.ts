import type { OcrResponseFormat, OcrTask, ParameterMetadata } from "./types";

export const ocrResponseFormats: OcrResponseFormat[] = ["json", "text", "markdown", "html"];

export const ocrTaskOptions: OcrTask[] = ["ocr", "table", "formula", "chart", "spotting", "seal"];

export const ocrParameterMetadata: ParameterMetadata[] = [
  {
    name: "file",
    label: "Document file",
    defaultValue: "",
    description: "Image or PDF file to recognize.",
    required: true,
    supported: true,
  },
  {
    name: "model",
    label: "Model",
    defaultValue: "",
    description: "OCR or OCR-VL model identifier.",
    required: true,
    supported: true,
  },
  {
    name: "response_format",
    label: "Response format",
    defaultValue: "json",
    description: "Response shape returned by the backend.",
    required: false,
    supported: true,
    options: ocrResponseFormats,
  },
  {
    name: "task",
    label: "Task",
    defaultValue: "ocr",
    description: "OCR-VL task hint such as ocr, table, formula, chart, spotting, or seal.",
    required: false,
    supported: true,
    options: ocrTaskOptions,
    notice: "Traditional OCR accepts only the ocr task. Other tasks require OCR-VL models.",
  },
  {
    name: "layout_model",
    label: "Layout model",
    defaultValue: "",
    description: "Optional OCR-VL layout detector model.",
    required: false,
    supported: true,
    notice: "Only configured layout models returned by /v1/models can be selected.",
  },
  {
    name: "max_tokens",
    label: "Max tokens",
    defaultValue: "",
    description: "Optional OCR-VL generation token limit.",
    required: false,
    supported: true,
    notice: "This parameter is used by OCR-VL generation and is ignored by traditional OCR.",
  },
];
