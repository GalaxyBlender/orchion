import { describe, expect, test } from "bun:test";
import { normalizeOcrFormForCapabilities, ocrStateToForm } from "../features/ocr/form";
import type { OcrFormState } from "../features/ocr/types";

const form: OcrFormState = {
  model: "PaddlePaddle/PP-OCRv5_mobile",
  responseFormat: "json",
  task: "table",
  maxTokens: "2048",
};

describe("OCR form contract", () => {
  test("restores persisted OCR request state", () => {
    expect(ocrStateToForm(form)).toEqual(form);
  });

  test("removes OCR-VL-only parameters for traditional models", () => {
    expect(normalizeOcrFormForCapabilities(form, ["ocr_text"])).toEqual({
      ...form,
      task: "ocr",
      maxTokens: "",
    });
  });

  test("uses capabilities to constrain structured formats", () => {
    expect(normalizeOcrFormForCapabilities({ ...form, responseFormat: "html" }, ["ocr_text"])).toEqual({
      ...form,
      responseFormat: "json",
      task: "ocr",
      maxTokens: "",
    });
    expect(normalizeOcrFormForCapabilities({ ...form, responseFormat: "html" }, ["ocr_text", "ocr_html"]).responseFormat).toBe("html");
    expect(
      normalizeOcrFormForCapabilities(
        { ...form, responseFormat: "html" },
        ["ocr_text", "ocr_layout", "ocr_markdown"],
      ).responseFormat,
    ).toBe("json");
  });
});
