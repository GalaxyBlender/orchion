import { describe, expect, mock, test } from "bun:test";
import type { LlmFormState, LlmMessage } from "../features/llm/types";

mock.module("@/shared/api/client", () => ({
  apiCurlUrl: () => "http://localhost/v1/chat/completions",
}));

const form: LlmFormState = {
  model: " local/llm ",
  systemPrompt: " Be concise. ",
  temperature: "0.7",
  topP: "0.9",
  maxCompletionTokens: "256",
  seed: "42",
};
const messages: LlmMessage[] = [
  { id: "1", role: "user", content: "Hello" },
  { id: "2", role: "assistant", content: "Hi" },
];

describe("LLM chat request contract", () => {
  test("builds a supported streaming chat request", async () => {
    const { buildLlmRequest } = await import("../features/llm/request");

    expect(buildLlmRequest(form, messages)).toEqual({
      model: "local/llm",
      messages: [
        { role: "system", content: "Be concise." },
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
      ],
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0.7,
      top_p: 0.9,
      max_completion_tokens: 256,
      seed: 42,
    });
  });

  test("omits blank optional parameters and system prompt", async () => {
    const { buildLlmRequest } = await import("../features/llm/request");
    const request = buildLlmRequest(
      {
        ...form,
        systemPrompt: " ",
        temperature: "",
        topP: "",
        maxCompletionTokens: "",
        seed: "",
      },
      messages.slice(0, 1),
    );

    expect(request.messages).toEqual([{ role: "user", content: "Hello" }]);
    expect("temperature" in request).toBeFalse();
    expect("top_p" in request).toBeFalse();
    expect("max_completion_tokens" in request).toBeFalse();
    expect("seed" in request).toBeFalse();
  });
});
