import { describe, expect, test } from "bun:test";
import { normalizeOcrFormForCapabilities, ocrResponseFormatsForCapabilities, ocrStateToForm } from "../features/ocr/form";
import type { OcrFormState } from "../features/ocr/types";

const form: OcrFormState = {
  model: "paddlepaddle/pp-ocrv5-mobile",
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

  test("table tasks support all formats without a document layout model", () => {
    const capabilities = ["ocr_text", "ocr_vision_language"] as const;
    expect(ocrResponseFormatsForCapabilities(capabilities, "table")).toEqual(["json", "text", "markdown", "html"]);
    expect(normalizeOcrFormForCapabilities({ ...form, responseFormat: "html" }, capabilities)).toEqual({
      ...form,
      responseFormat: "html",
    });
    expect(ocrResponseFormatsForCapabilities(capabilities, "ocr")).toEqual(["json", "text"]);
  });

  test("non-document VL tasks expose only JSON and text even with layout capabilities", () => {
    const capabilities = ["ocr_text", "ocr_vision_language", "ocr_markdown", "ocr_html"] as const;
    for (const task of ["formula", "chart", "spotting", "seal"] as const) {
      expect(ocrResponseFormatsForCapabilities(capabilities, task)).toEqual(["json", "text"]);
      expect(normalizeOcrFormForCapabilities({ ...form, task, responseFormat: "html" }, capabilities)).toEqual({
        ...form,
        task,
        responseFormat: "json",
      });
      expect(normalizeOcrFormForCapabilities({ ...form, task, responseFormat: "text" }, capabilities).responseFormat).toBe("text");
    }
  });

  test("traditional model normalization uses the OCR task before choosing formats", () => {
    expect(normalizeOcrFormForCapabilities(
      { ...form, task: "formula", responseFormat: "markdown" },
      ["ocr_text", "ocr_layout", "ocr_markdown"],
    )).toEqual({ ...form, task: "ocr", responseFormat: "markdown", maxTokens: "" });
  });
});
