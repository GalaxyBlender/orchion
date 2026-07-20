import { describe, expect, mock, test } from "bun:test";
import type { AsrRequestInput } from "../features/asr/types";

mock.module("@/shared/api/client", () => ({
  apiCurlUrl: () => "http://localhost/v1/audio/transcriptions",
}));

describe("ASR batch request contract", () => {
  test("omits parameters that the batch runtime does not support", async () => {
    const { buildAsrFormData } = await import("../features/asr/request");
    const input: AsrRequestInput = {
      model: "Qwen/Qwen3-ASR-0.6B",
      file: new File(["audio"], "audio.wav"),
      language: "English",
      responseFormat: "json",
      prompt: "previous context",
      temperature: "0.2",
      timestampGranularities: [],
    };

    const formData = buildAsrFormData(input);

    expect(formData.has("prompt")).toBeFalse();
    expect(formData.has("temperature")).toBeFalse();
  });
});
