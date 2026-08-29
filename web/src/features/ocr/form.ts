import type { ModelCapability } from "../../shared/api/types";
import type { PersistentOcrState } from "../../shared/storage/persistentState";
import type { OcrFormState, OcrResponseFormat } from "./types";

export function ocrStateToForm(state: PersistentOcrState): OcrFormState {
  return {
    model: state.model,
    responseFormat: state.responseFormat,
    task: state.task,
    maxTokens: state.maxTokens,
  };
}

export function formToOcrState(form: OcrFormState): PersistentOcrState {
  return {
    model: form.model,
    responseFormat: form.responseFormat,
    task: form.task,
    maxTokens: form.maxTokens,
  };
}

export function normalizeOcrFormForCapabilities(
  form: OcrFormState,
  capabilities: readonly ModelCapability[],
): OcrFormState {
  const responseFormats = ocrResponseFormatsForCapabilities(capabilities);
  return {
    ...form,
    responseFormat: responseFormats.includes(form.responseFormat) ? form.responseFormat : "json",
    task: capabilities.includes("ocr_vision_language") ? form.task : "ocr",
    maxTokens: capabilities.includes("ocr_vision_language") ? form.maxTokens : "",
  };
}

export function ocrResponseFormatsForCapabilities(
  capabilities: readonly ModelCapability[],
): OcrResponseFormat[] {
  const formats: OcrResponseFormat[] = ["json", "text"];
  if (capabilities.includes("ocr_markdown")) formats.push("markdown");
  if (capabilities.includes("ocr_html")) formats.push("html");
  return formats;
}
