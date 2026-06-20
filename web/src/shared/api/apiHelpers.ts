import { ApiRequestError } from "./types";

export interface SubmissionError {
  type: "validation" | "network" | "api";
  message: string;
  detail?: any;
}

export function buildApiError(error: unknown): SubmissionError {
  if (error instanceof ApiRequestError) {
    return {
      type: "api",
      message: error.detail.message,
      detail: error.detail,
    };
  }
  
  if (error instanceof Error) {
    // If it's a network request failed error (from parseNetworkError)
    if (error.message.includes("Network") || error.message.includes("failed")) {
      return {
        type: "network",
        message: error.message,
      };
    }
    return {
      type: "network",
      message: error.message,
    };
  }

  return {
    type: "network",
    message: String(error),
  };
}

export async function readResponsePayload(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return text;
  }
}
