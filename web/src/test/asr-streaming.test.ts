import { describe, expect, test } from "bun:test";
import {
  buildAsrStreamStartMessage,
  formatAsrStreamResult,
  parseAsrStreamEvent,
  upsertBoundedAsrCaptionSegments,
  validateAsrCaptionEndpointingOptions,
} from "../features/asr/streaming";
import type { AsrFormState } from "../features/asr/types";

const baseForm: AsrFormState = {
  model: "Qwen/Qwen3-ASR-0.6B",
  language: "",
  responseFormat: "json",
  prompt: "",
  temperature: "",
  timestampGranularities: [],
};

describe("ASR streaming protocol helpers", () => {
  test("sends caption mode only for caption output streams", () => {
    const captionStart = JSON.parse(
      buildAsrStreamStartMessage({
        form: baseForm,
        inputAudioFormat: "mp3",
        apiKey: "",
        outputMode: "caption",
        endpointing: {
          min_speech_ms: 300,
          min_silence_ms: 500,
          speech_padding_ms: 200,
        },
      }),
    );

    const liveStart = JSON.parse(
      buildAsrStreamStartMessage({
        form: baseForm,
        inputAudioFormat: "mp3",
        apiKey: "",
        outputMode: "transcript",
      }),
    );

    expect(captionStart).toMatchObject({
      type: "start",
      model: "Qwen/Qwen3-ASR-0.6B",
      input_audio_format: "mp3",
      mode: "caption",
      endpointing: {
        min_speech_ms: 300,
        min_silence_ms: 500,
        speech_padding_ms: 200,
      },
    });
    expect(liveStart).not.toHaveProperty("mode");
  });

  test("parses and formats caption stream events", () => {
    const partial = parseAsrStreamEvent('{"type":"partial","segment_id":2,"text":"hello"}');
    const final = parseAsrStreamEvent('{"type":"segment_final","segment_id":2,"text":"hello","start_ms":120,"end_ms":900}');
    const completed = parseAsrStreamEvent('{"type":"completed"}');

    expect(partial).toEqual({ type: "partial", segment_id: 2, text: "hello" });
    expect(final).toEqual({ type: "segment_final", segment_id: 2, text: "hello", start_ms: 120, end_ms: 900 });
    expect(completed).toEqual({ type: "completed" });
    expect(formatAsrStreamResult(completed)).toBe(JSON.stringify({ type: "completed" }, null, 2));
  });

  test("validates caption endpointing like the server and sdk", () => {
    expect(validateAsrCaptionEndpointingOptions({
      min_speech_ms: 300,
      min_silence_ms: 500,
      speech_padding_ms: 200,
    })).toBeNull();
    expect(validateAsrCaptionEndpointingOptions({
      min_speech_ms: 21,
      min_silence_ms: 350,
      speech_padding_ms: 0,
    })).toBe("invalid_rounded_window");
  });

  test("keeps caption segment display history bounded", () => {
    const segments = Array.from({ length: 300 }, (_value, id) => ({ id, text: String(id) }));
    const nextSegments = upsertBoundedAsrCaptionSegments(segments, { id: 300, text: "300" });

    expect(nextSegments).toHaveLength(300);
    expect(nextSegments[0].id).toBe(1);
    expect(nextSegments.at(-1)).toEqual({ id: 300, text: "300" });

    const updatedSegments = upsertBoundedAsrCaptionSegments(nextSegments, { id: 300, text: "updated" });
    expect(updatedSegments).toHaveLength(300);
    expect(updatedSegments.at(-1)).toEqual({ id: 300, text: "updated" });
  });
});
