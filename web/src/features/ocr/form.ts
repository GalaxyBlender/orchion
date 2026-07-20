import type { ModelSubtype } from "../../shared/api/types";
import type { PersistentOcrState } from "../../shared/storage/persistentState";
import type { OcrFormState } from "./types";

export function ocrStateToForm(state: PersistentOcrState): OcrFormState {
  return {
    model: state.model,
    responseFormat: state.responseFormat,
    task: state.task,
    layoutModel: state.layoutModel,
    maxTokens: state.maxTokens,
  };
}

export function formToOcrState(form: OcrFormState): PersistentOcrState {
  return {
    model: form.model,
    responseFormat: form.responseFormat,
    task: form.task,
    layoutModel: form.layoutModel,
    maxTokens: form.maxTokens,
  };
}

export function normalizeOcrFormForSubtype(
  form: OcrFormState,
  subtype: ModelSubtype | undefined,
): OcrFormState {
  if (subtype !== "standard") {
    return form;
  }
  return { ...form, task: "ocr", maxTokens: "" };
}
