import { ApiRequestError } from "../../shared/api/types";
import type { LlmUsage } from "./types";

interface ChatStreamCallbacks {
  onDelta: (delta: string) => void;
  onUsage: (usage: LlmUsage) => void;
}

interface ChatStreamChoice {
  delta?: { content?: unknown };
}

interface ChatStreamPayload {
  choices?: ChatStreamChoice[];
  usage?: {
    prompt_tokens?: unknown;
    completion_tokens?: unknown;
    total_tokens?: unknown;
  } | null;
  error?: {
    message?: unknown;
    type?: unknown;
    code?: unknown;
    param?: unknown;
  };
}

export async function consumeChatCompletionStream(
  body: ReadableStream<Uint8Array> | null,
  callbacks: ChatStreamCallbacks,
): Promise<void> {
  if (!body) {
    throw new Error("Streaming response did not include a body.");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;

  try {
    while (!completed) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });

      const parsed = consumeFrames(buffer, callbacks);
      buffer = parsed.remainder;
      completed = parsed.completed;

      if (done) {
        if (!completed && buffer.trim() !== "") {
          completed = consumeFrame(buffer, callbacks);
        }
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!completed) {
    throw new Error("Streaming response ended before [DONE].");
  }
}

function consumeFrames(buffer: string, callbacks: ChatStreamCallbacks): { remainder: string; completed: boolean } {
  let remainder = buffer;

  while (true) {
    const separator = remainder.match(/\r?\n\r?\n/);
    if (!separator || separator.index === undefined) {
      return { remainder, completed: false };
    }

    const frame = remainder.slice(0, separator.index);
    remainder = remainder.slice(separator.index + separator[0].length);
    if (consumeFrame(frame, callbacks)) {
      return { remainder, completed: true };
    }
  }
}

function consumeFrame(frame: string, callbacks: ChatStreamCallbacks): boolean {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");

  if (data === "") {
    return false;
  }
  if (data === "[DONE]") {
    return true;
  }

  let payload: ChatStreamPayload;
  try {
    payload = JSON.parse(data) as ChatStreamPayload;
  } catch {
    throw new Error("Streaming response contained invalid JSON.");
  }

  if (payload.error) {
    throw streamApiError(payload.error);
  }

  for (const choice of payload.choices ?? []) {
    const content = choice.delta?.content;
    if (typeof content === "string" && content !== "") {
      callbacks.onDelta(content);
    }
  }

  const usage = payload.usage;
  if (
    usage
    && typeof usage.prompt_tokens === "number"
    && typeof usage.completion_tokens === "number"
    && typeof usage.total_tokens === "number"
  ) {
    callbacks.onUsage({
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
    });
  }

  return false;
}

function streamApiError(error: NonNullable<ChatStreamPayload["error"]>): ApiRequestError {
  return new ApiRequestError({
    message: typeof error.message === "string" ? error.message : "LLM generation failed.",
    type: typeof error.type === "string" ? error.type : undefined,
    code: typeof error.code === "string" || error.code === null ? error.code : undefined,
    param: typeof error.param === "string" || error.param === null ? error.param : undefined,
  });
}
