import { describe, expect, test } from "bun:test";
import { ApiRequestError } from "../shared/api/types";
import { consumeChatCompletionStream } from "../features/llm/stream";

describe("LLM chat stream", () => {
  test("parses deltas and usage across arbitrary chunks", async () => {
    const chunks = [
      "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Hel",
      "lo\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
      "data: [DONE]\n\n",
    ];
    const deltas: string[] = [];
    let usage = null;

    await consumeChatCompletionStream(stream(chunks), {
      onDelta: (delta) => deltas.push(delta),
      onUsage: (value) => { usage = value; },
    });

    expect(deltas).toEqual(["Hello"]);
    expect(usage).toEqual({ promptTokens: 2, completionTokens: 1, totalTokens: 3 });
  });

  test("surfaces errors sent after the stream starts", async () => {
    const body = stream([
      'data: {"error":{"message":"capacity exhausted","type":"rate_limit_error","code":"resource_exhausted","param":null}}\n\n',
    ]);

    try {
      await consumeChatCompletionStream(body, { onDelta: () => {}, onUsage: () => {} });
      throw new Error("expected stream error");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiRequestError);
      expect((error as ApiRequestError).detail.code).toBe("resource_exhausted");
    }
  });

  test("rejects streams without the terminal marker", async () => {
    expect(consumeChatCompletionStream(stream(['data: {"choices":[]}\n\n']), {
      onDelta: () => {},
      onUsage: () => {},
    })).rejects.toThrow("before [DONE]");
  });
});

function stream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}
