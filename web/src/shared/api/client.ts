import { parseApiError, parseNetworkError } from "./errors";
import type { ApiSettings, ModelList } from "./types";

export function apiUrl(_settings: ApiSettings, path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  return normalizedPath;
}

export function apiCurlUrl(settings: ApiSettings, path: string): string {
  const normalizedPath = apiUrl(settings, path);

  if (typeof window === "undefined") {
    return `http://127.0.0.1:9090${normalizedPath}`;
  }

  return new URL(normalizedPath, window.location.origin).toString();
}

export function authHeaders(settings: ApiSettings): HeadersInit {
  const apiKey = settings.apiKey.trim();

  return apiKey === "" ? {} : { Authorization: `Bearer ${apiKey}` };
}

export async function requestJson<T>(
  settings: ApiSettings,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await request(settings, path, init);

  return (await response.json()) as T;
}

export async function requestBlob(
  settings: ApiSettings,
  path: string,
  init: RequestInit = {},
): Promise<{ blob: Blob; headers: Headers }> {
  const response = await request(settings, path, init);

  return { blob: await response.blob(), headers: response.headers };
}

export function fetchModels(settings: ApiSettings): Promise<ModelList> {
  return requestJson<ModelList>(settings, "/v1/models", { method: "GET" });
}

async function request(settings: ApiSettings, path: string, init: RequestInit): Promise<Response> {
  try {
    const response = await fetch(apiUrl(settings, path), {
      ...init,
      headers: mergeHeaders(init.headers, authHeaders(settings)),
    });

    if (!response.ok) {
      throw await responseError(response);
    }

    return response;
  } catch (error) {
    if (error instanceof Error && error.name === "ApiRequestError") {
      throw error;
    }

    throw parseNetworkError(error);
  }
}

async function responseError(response: Response): Promise<Error> {
  const payload = await readErrorPayload(response);

  return parseApiError(response, payload);
}

async function readErrorPayload(response: Response): Promise<unknown> {
  try {
    return await response.clone().json();
  } catch {
  }

  try {
    return await response.text();
  } catch {
    return undefined;
  }
}

function mergeHeaders(...sources: Array<HeadersInit | undefined>): Headers {
  const headers = new Headers();

  for (const source of sources) {
    new Headers(source).forEach((value, key) => headers.set(key, value));
  }

  return headers;
}
