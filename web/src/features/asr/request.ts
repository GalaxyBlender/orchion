import { apiCurlUrl } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type { AsrRequestInput } from "./types";

const endpointPath = "/v1/audio/transcriptions";

export function buildAsrFormData(input: AsrRequestInput): FormData {
  const formData = new FormData();

  formData.append("file", input.file);
  formData.append("model", input.model.trim());
  formData.append("response_format", input.responseFormat);
  appendNonblank(formData, "language", input.language);
  appendSupportedTimestampGranularities(formData, input.timestampGranularities);

  return formData;
}

interface AsrSummaryText {
  model: (model: string) => string;
  file: (file: string) => string;
  responseFormat: (format: string) => string;
  language: (language: string) => string;
  timestamp: (values: string) => string;
}

const defaultSummaryText: AsrSummaryText = {
  model: (model) => `Model: ${model}`,
  file: (file) => `File: ${file}`,
  responseFormat: (format) => `Response format: ${format}`,
  language: (language) => `Language: ${language}`,
  timestamp: (values) => `Timestamp granularities: ${values} sent for segment-level verbose output.`,
};

export function summarizeAsrRequest(input: AsrRequestInput, text: AsrSummaryText = defaultSummaryText): string[] {
  const lines = [
    text.model(input.model.trim()),
    text.file(input.file.name),
    text.responseFormat(input.responseFormat),
  ];
  const language = input.language.trim();
  const timestampGranularities = nonblankValues(input.timestampGranularities);

  if (language !== "") {
    lines.push(text.language(language));
  }
  if (timestampGranularities.length > 0) {
    lines.push(text.timestamp(timestampGranularities.join(", ")));
  }

  return lines;
}

export function buildAsrCurl(settings: ApiSettings, input: AsrRequestInput): string {
  const parts = ["curl", "-X", "POST", quote(apiCurlUrl(settings, endpointPath))];
  const apiKey = settings.apiKey.trim();
  const fields: Array<[string, string]> = [
    ["model", input.model.trim()],
    ["response_format", input.responseFormat],
  ];

  if (apiKey !== "") {
    parts.push("-H", quote(`Authorization: Bearer ${apiKey}`));
  }

  parts.push("-F", quote(`file=@${input.file.name}`));
  pushOptionalField(fields, "language", input.language);
  for (const value of supportedTimestampGranularities(input.timestampGranularities)) {
    fields.push(["timestamp_granularities[]", value]);
  }

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

function appendSupportedTimestampGranularities(formData: FormData, values: readonly string[]): void {
  for (const value of supportedTimestampGranularities(values)) {
    formData.append("timestamp_granularities[]", value);
  }
}

function supportedTimestampGranularities(values: readonly string[]): string[] {
  return values.map((value) => value.trim()).filter((value) => value === "segment");
}

function nonblankValues(values: readonly string[]): string[] {
  return values.map((value) => value.trim()).filter((value) => value !== "");
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}
