import { describe, expect, mock, test } from "bun:test";
import type { OcrRequestInput } from "../features/ocr/types";

mock.module("@/shared/api/client", () => ({
  apiCurlUrl: () => "http://localhost/v1/ocr",
}));

const input: OcrRequestInput = {
  file: new File(["image"], "document.png", { type: "image/png" }),
  model: "paddlepaddle/pp-ocrv6-tiny",
  responseFormat: "markdown",
  task: "ocr",
  maxTokens: "",
};

describe("OCR request contract", () => {
  test("selects only the primary model and never sends layout_model", async () => {
    const { buildOcrCurl, buildOcrFormData } = await import("../features/ocr/request");
    const form = buildOcrFormData(input);

    expect(form.get("model")).toBe("paddlepaddle/pp-ocrv6-tiny");
    expect(form.get("response_format")).toBe("markdown");
    expect(form.has("layout_model")).toBe(false);
    expect(buildOcrCurl({ serverBaseUrl: "", apiKey: "" }, input)).not.toContain("layout_model");
  });
});
