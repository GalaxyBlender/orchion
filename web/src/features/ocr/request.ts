import { apiCurlUrl } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type { OcrRequestInput } from "./types";

const endpointPath = "/v1/ocr";

export function buildOcrFormData(input: OcrRequestInput): FormData {
  const formData = new FormData();

  formData.append("file", input.file);
  formData.append("model", input.model.trim());
  formData.append("response_format", input.responseFormat);
  appendNonblank(formData, "task", input.task);
  appendNonblank(formData, "layout_model", input.layoutModel);
  appendNonblank(formData, "max_tokens", input.maxTokens);

  return formData;
}

interface OcrSummaryText {
  model: (model: string) => string;
  file: (file: string) => string;
  responseFormat: (format: string) => string;
  task: (task: string) => string;
  layoutModel: (model: string) => string;
  maxTokens: (value: string) => string;
}

const defaultSummaryText: OcrSummaryText = {
  model: (model) => `Model: ${model}`,
  file: (file) => `File: ${file}`,
  responseFormat: (format) => `Response format: ${format}`,
  task: (task) => `Task: ${task}`,
  layoutModel: (model) => `Layout model: ${model}`,
  maxTokens: (value) => `Max tokens: ${value}`,
};

export function summarizeOcrRequest(input: OcrRequestInput, text: OcrSummaryText = defaultSummaryText): string[] {
  const lines = [
    text.model(input.model.trim()),
    text.file(input.file.name),
    text.responseFormat(input.responseFormat),
    text.task(input.task),
  ];
  const layoutModel = input.layoutModel.trim();
  const maxTokens = input.maxTokens.trim();

  if (layoutModel !== "") {
    lines.push(text.layoutModel(layoutModel));
  }
  if (maxTokens !== "") {
    lines.push(text.maxTokens(maxTokens));
  }

  return lines;
}

export function buildOcrCurl(settings: ApiSettings, input: OcrRequestInput): string {
  const parts = ["curl", "-X", "POST", quote(apiCurlUrl(settings, endpointPath))];
  const apiKey = settings.apiKey.trim();
  const fields: Array<[string, string]> = [
    ["model", input.model.trim()],
    ["response_format", input.responseFormat],
    ["task", input.task],
  ];

  if (apiKey !== "") {
    parts.push("-H", quote(`Authorization: Bearer ${apiKey}`));
  }

  parts.push("-F", quote(`file=@${input.file.name}`));
  pushOptionalField(fields, "layout_model", input.layoutModel);
  pushOptionalField(fields, "max_tokens", input.maxTokens);

  for (const [name, value] of fields) {
    parts.push("-F", quote(`${name}=${value}`));
  }

  return parts.join(" ");
}

function appendNonblank(formData: FormData, name: string, value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    formData.append(name, trimmedValue);
  }
}

function pushOptionalField(fields: Array<[string, string]>, name: string, value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    fields.push([name, trimmedValue]);
  }
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}
