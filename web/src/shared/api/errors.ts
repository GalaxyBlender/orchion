import { ApiRequestError, type ApiErrorDetail } from "./types";

export function parseApiError(response: Response, payload: unknown): ApiRequestError {
  const fallbackMessage = formatHttpError(response);
  const detail: ApiErrorDetail = {
    message: fallbackMessage,
    status: response.status,
  };

  if (isRecord(payload) && isRecord(payload.error)) {
    const { error } = payload;

    if (typeof error.message === "string" && error.message.trim() !== "") {
      detail.message = error.message;
    }

    if (typeof error.type === "string") {
      detail.type = error.type;
    }

    if (typeof error.code === "string" || error.code === null) {
      detail.code = error.code;
    }

    if (typeof error.param === "string" || error.param === null) {
      detail.param = error.param;
    }
  } else if (typeof payload === "string" && payload.trim() !== "") {
    detail.message = payload.trim();
  }

  return new ApiRequestError(detail);
}

export function parseNetworkError(error: unknown): ApiRequestError {
  if (error instanceof Error && error.message.trim() !== "") {
    return new ApiRequestError({ message: error.message });
  }

  return new ApiRequestError({ message: "Network request failed" });
}

function formatHttpError(response: Response): string {
  const status = response.status > 0 ? `HTTP ${response.status}` : "HTTP request failed";
  const statusText = response.statusText.trim();

  return statusText === "" ? status : `${status} ${statusText}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
