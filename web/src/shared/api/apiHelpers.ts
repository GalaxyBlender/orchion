import { ApiRequestError, type ApiErrorDetail } from "./types";

export interface SubmissionError {
  type: "validation" | "network" | "api";
  message: string;
  detail?: ApiErrorDetail;
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

export async function readResponsePayload(response: Response): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}
