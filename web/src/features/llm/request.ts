import { apiCurlUrl } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type { ChatCompletionRequest, LlmFormState, LlmMessage } from "./types";

export const llmEndpointPath = "/v1/chat/completions";

export function buildLlmRequest(form: LlmFormState, messages: readonly LlmMessage[]): ChatCompletionRequest {
  const request: ChatCompletionRequest = {
    model: form.model.trim(),
    messages: [],
    stream: true,
    stream_options: { include_usage: true },
  };
  const systemPrompt = form.systemPrompt.trim();

  if (systemPrompt !== "") {
    request.messages.push({ role: "system", content: systemPrompt });
  }
  request.messages.push(
    ...messages.map(({ role, content }) => ({ role, content })),
  );

  appendOptionalNumber(request, "temperature", form.temperature);
  appendOptionalNumber(request, "top_p", form.topP);
  appendOptionalNumber(request, "max_completion_tokens", form.maxCompletionTokens);
  appendOptionalNumber(request, "seed", form.seed);

  return request;
}

export function buildLlmCurl(settings: ApiSettings, request: ChatCompletionRequest): string {
  const lines = [`curl -X POST ${quote(apiCurlUrl(settings, llmEndpointPath))}`];
  const apiKey = settings.apiKey.trim();

  if (apiKey !== "") {
    lines.push(`-H ${quote(`Authorization: Bearer ${apiKey}`)}`);
  }
  lines.push(`-H ${quote("Content-Type: application/json")}`);
  lines.push(`--data ${quote(JSON.stringify(request))}`);

  return lines.map((line, index) => (index === lines.length - 1 ? line : `${line} \\`)).join("\n");
}

function appendOptionalNumber<K extends "temperature" | "top_p" | "max_completion_tokens" | "seed">(
  request: ChatCompletionRequest,
  field: K,
  value: string,
): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    request[field] = Number(trimmedValue);
  }
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}
