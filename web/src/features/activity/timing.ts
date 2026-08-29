import type { ActivityEntry, ActivityPage, ActivityStreamEvent } from "./types";

export function observeActivityEntry(
  entry: ActivityEntry,
  observedAtMs = monotonicNowMs(),
): ActivityEntry {
  if (entry.state !== "in_flight") return entry;
  return { ...entry, duration_observed_at_ms: observedAtMs };
}

export function observeActivityPage(
  page: ActivityPage,
  observedAtMs = monotonicNowMs(),
): ActivityPage {
  return {
    ...page,
    active: page.active.map((entry) => observeActivityEntry(entry, observedAtMs)),
  };
}

export function observeActivityEvent(
  event: ActivityStreamEvent,
  observedAtMs = monotonicNowMs(),
): ActivityStreamEvent {
  return {
    ...event,
    payload: {
      ...event.payload,
      entry: event.payload.entry
        ? observeActivityEntry(event.payload.entry, observedAtMs)
        : undefined,
      active: event.payload.active?.map((entry) => observeActivityEntry(entry, observedAtMs)),
    },
  };
}

export function activityDurationMs(entry: ActivityEntry, nowMs: number): number | undefined {
  if (entry.state === "completed") return entry.duration_ms;
  const sampledDuration = entry.duration_ms ?? 0;
  const observedAt = entry.duration_observed_at_ms ?? nowMs;
  return sampledDuration + Math.max(0, nowMs - observedAt);
}

export function formatActivityDuration(value?: number | null): string {
  if (value === undefined || value === null) return "-";
  if (value < 1_000) return `${value.toFixed(2)} ms`;
  if (value < 60_000) return `${(value / 1_000).toFixed(2)} s`;
  return `${Math.floor(value / 60_000)}m ${((value % 60_000) / 1_000).toFixed(2)}s`;
}

export function monotonicNowMs(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}
