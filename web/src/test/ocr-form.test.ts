import { describe, expect, test } from "bun:test";
import { normalizeOcrFormForSubtype, ocrStateToForm } from "../features/ocr/form";
import type { OcrFormState } from "../features/ocr/types";

const form: OcrFormState = {
  model: "PaddlePaddle/PP-OCRv5_mobile",
  responseFormat: "json",
  task: "table",
  layoutModel: "PaddlePaddle/PP-DocLayoutV3",
  maxTokens: "2048",
};

describe("OCR form contract", () => {
  test("restores the persisted layout model", () => {
    expect(ocrStateToForm(form).layoutModel).toBe("PaddlePaddle/PP-DocLayoutV3");
  });

  test("removes OCR-VL-only parameters for traditional models", () => {
    expect(normalizeOcrFormForSubtype(form, "standard")).toEqual({
      ...form,
      task: "ocr",
      maxTokens: "",
    });
  });
});
