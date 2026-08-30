export type LlmMessageRole = "user" | "assistant";

export interface LlmMessage {
  id: string;
  role: LlmMessageRole;
  content: string;
  includeInContext?: boolean;
  streamState?: "streaming" | "complete" | "stopped" | "error";
}

export interface LlmFormState {
  model: string;
  systemPrompt: string;
  temperature: string;
  topP: string;
  maxCompletionTokens: string;
  seed: string;
}

export interface ChatCompletionMessage {
  role: "system" | LlmMessageRole;
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatCompletionMessage[];
  stream: true;
  stream_options: {
    include_usage: true;
  };
  temperature?: number;
  top_p?: number;
  max_completion_tokens?: number;
  seed?: number;
}

export interface LlmUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}
