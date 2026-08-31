import { describe, expect, mock, test } from "bun:test";
import type { TtsRequestInput } from "../features/tts/types";

mock.module("@/shared/api/client", () => ({
  apiCurlUrl: () => "http://localhost/v1/audio/speech",
}));

const baseInput: TtsRequestInput = {
  mode: "preset",
  model: "alibaba/qwen3-tts-12hz-0.6b-customvoice",
  input: "hello",
  language: "",
  responseFormat: "wav",
  speaker: "Serena",
  referenceAudioName: "",
  referenceText: "",
  voicePrompt: "",
  speed: "1.8",
  seed: "42",
  temperature: "0.7",
  topK: "20",
  topP: "0.8",
  repetitionPenalty: "1.05",
  maxLength: "2048",
};

describe("TTS request contract", () => {
  test("always sends the only server-supported speed", async () => {
    const { buildTtsPayload } = await import("../features/tts/request");
    const payload = buildTtsPayload(baseInput);

    expect(payload.kind).toBe("json");
    if (payload.kind === "json") {
      expect(payload.body.speed).toBe(1);
    }
  });
});
