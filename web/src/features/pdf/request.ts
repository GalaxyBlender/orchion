import { apiCurlUrl } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type { PdfRequestInput } from "./types";

export const pdfImagesEndpoint = "/v1/pdf/images";

export function buildPdfFormData(input: PdfRequestInput): FormData {
  const formData = new FormData();

  formData.append("file", input.file);
  formData.append("response_format", input.responseFormat);
  appendNonblank(formData, "pages", input.pages);
  appendNonblank(formData, "scale", input.scale);

  return formData;
}

export function summarizePdfRequest(input: PdfRequestInput): string[] {
  const pages = input.pages?.trim() ?? "";
  const scale = input.scale?.trim() ?? "";

  return [
    `Endpoint: POST ${pdfImagesEndpoint}`,
    `Output format: ${input.responseFormat}`,
    `Page selection: ${pages === "" ? "all pages" : pages}`,
    `Scale: ${scale === "" ? "backend default" : scale}`,
  ];
}

export function buildPdfCurl(settings: ApiSettings, input: PdfRequestInput): string {
  const parts = ["curl", "-X", "POST", quote(apiCurlUrl(settings, pdfImagesEndpoint))];
  const apiKey = settings.apiKey.trim();
  const fields: Array<[string, string]> = [["response_format", input.responseFormat]];

  if (apiKey !== "") {
    parts.push("-H", quote(`Authorization: Bearer ${apiKey}`));
  }

  parts.push("-F", quote(`file=@${input.file.name}`));
  pushOptionalField(fields, "pages", input.pages);
  pushOptionalField(fields, "scale", input.scale);

  for (const [name, value] of fields) {
    parts.push("-F", quote(`${name}=${value}`));
  }

  return parts.join(" ");
}

function appendNonblank(formData: FormData, name: string, value: string | undefined): void {
  const trimmedValue = value?.trim() ?? "";
  if (trimmedValue !== "") {
    formData.append(name, trimmedValue);
  }
}

function pushOptionalField(fields: Array<[string, string]>, name: string, value: string | undefined): void {
  const trimmedValue = value?.trim() ?? "";
  if (trimmedValue !== "") {
    fields.push([name, trimmedValue]);
  }
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}
