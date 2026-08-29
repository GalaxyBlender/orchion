import { requestJson, requestStream } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type {
  ActivityEventPayload,
  ActivityEventType,
  ActivityPage,
  ActivityQuery,
  ActivityStreamEvent,
} from "./types";
import { createSseParser } from "./sse";
import { observeActivityEvent, observeActivityPage } from "./timing";

const EVENT_TYPES = new Set<ActivityEventType>([
  "snapshot",
  "started",
  "updated",
  "completed",
  "reset",
]);

export function fetchActivity(
  settings: ApiSettings,
  query: ActivityQuery,
  signal?: AbortSignal,
): Promise<ActivityPage> {
  const search = new URLSearchParams();
  if (query.limit !== undefined) search.set("limit", String(query.limit));
  if (query.before) search.set("before", query.before);
  if (query.operation) search.set("operation", query.operation);
  if (query.outcome) search.set("outcome", query.outcome);
  if (query.model) search.set("model", query.model);
  const suffix = search.size > 0 ? `?${search.toString()}` : "";

  return requestJson<ActivityPage>(settings, `/api/activity${suffix}`, { method: "GET", signal })
    .then((page) => observeActivityPage(page));
}

export async function streamActivity(
  settings: ApiSettings,
  signal: AbortSignal,
  onEvent: (event: ActivityStreamEvent) => void,
): Promise<void> {
  const response = await requestStream(settings, "/api/activity/events", {
    method: "GET",
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!response.body) {
    throw new Error("Activity event stream has no response body");
  }

  const parser = createSseParser((eventType, data) => {
    if (!EVENT_TYPES.has(eventType as ActivityEventType)) return;
    const payload = JSON.parse(data) as ActivityEventPayload;
    if (typeof payload.cursor !== "string") return;
    onEvent(observeActivityEvent({ type: eventType as ActivityEventType, payload }));
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
  } finally {
    reader.releaseLock();
  }
}
