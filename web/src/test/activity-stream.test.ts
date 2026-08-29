import { expect, test } from "bun:test";
import { createSseParser } from "../features/activity/sse";
import {
  reduceActivityEvent,
  replayActivityEvents,
  scrubCompletedActivityEvents,
} from "../features/activity/reducer";
import {
  activityDurationMs,
  formatActivityDuration,
  observeActivityEntry,
} from "../features/activity/timing";
import type { ActivityEntry, ActivityPage } from "../features/activity/types";

const entry: ActivityEntry = {
  id: "7",
  state: "in_flight",
  transport: "http",
  operation: "asr",
  method: "POST",
  route: "/v1/audio/transcriptions",
  started_at_ms: 1_000,
};

test("SSE parser handles CRLF and chunk boundaries", () => {
  const events: Array<{ type: string; data: string }> = [];
  const parser = createSseParser((type, data) => events.push({ type, data }));

  parser.push("event: snap");
  parser.push("shot\r\nid: 1\r\ndata: {\"cursor\":\"1\",\r\n");
  parser.push("data: \"active\":[]}\r\n\r\n: keep-alive\r\n\r\n");
  parser.finish();

  expect(events).toEqual([
    { type: "snapshot", data: "{\"cursor\":\"1\",\n\"active\":[]}" },
  ]);
});

test("activity reducer upserts live entries and awaits REST for completed history", () => {
  const page: ActivityPage = {
    enabled: true,
    cursor: "0",
    active: [],
    history: [],
    summary: { active: 0, retained: 0, success_rate: null, p95_duration_ms: null },
  };
  const started = reduceActivityEvent(page, {
    type: "started",
    payload: { cursor: "1", entry },
  })!;
  const completed = reduceActivityEvent(started, {
    type: "completed",
    payload: {
      cursor: "2",
      entry: { ...entry, state: "completed", outcome: "success", duration_ms: 250 },
      summary: { active: 0, retained: 1, success_rate: 100, p95_duration_ms: 250 },
    },
  })!;

  expect(started.active).toHaveLength(1);
  expect(completed.active).toHaveLength(0);
  expect(completed.history).toHaveLength(0);
  expect(completed.summary.retained).toBe(1);
});

test("activity reducer applies live model and client metadata updates", () => {
  const page: ActivityPage = {
    enabled: true,
    cursor: "1",
    active: [entry],
    history: [],
    summary: { active: 1, retained: 0, success_rate: null, p95_duration_ms: null },
  };

  const updated = reduceActivityEvent(page, {
    type: "updated",
    payload: {
      cursor: "2",
      entry: {
        ...entry,
        model: "Qwen/Qwen3-ASR-0.6B",
        address: "203.0.113.7",
        user_agent: "orchion-test-agent/1.0",
      },
    },
  })!;

  expect(updated.active[0]?.model).toBe("Qwen/Qwen3-ASR-0.6B");
  expect(updated.active[0]?.address).toBe("203.0.113.7");
  expect(updated.active[0]?.user_agent).toBe("orchion-test-agent/1.0");
});

test("activity reducer ignores events at or behind the current cursor", () => {
  const page: ActivityPage = {
    enabled: true,
    cursor: "4",
    active: [],
    history: [],
    summary: { active: 0, retained: 0, success_rate: null, p95_duration_ms: null },
  };

  const result = reduceActivityEvent(page, {
    type: "started",
    payload: { cursor: "3", entry },
  });

  expect(result).toBe(page);
});

test("REST snapshots replay newer buffered stream events", () => {
  const page: ActivityPage = {
    enabled: true,
    cursor: "4",
    active: [],
    history: [],
    summary: { active: 0, retained: 0, success_rate: null, p95_duration_ms: null },
  };

  const merged = replayActivityEvents(page, [
    {
      type: "started",
      payload: {
        cursor: "5",
        entry,
        summary: { active: 1, retained: 0, success_rate: null, p95_duration_ms: null },
      },
    },
  ]);

  expect(merged.cursor).toBe("5");
  expect(merged.active[0]?.id).toBe("7");
});

test("filtered rows do not replace the global activity summary", () => {
  const page: ActivityPage = {
    enabled: true,
    cursor: "0",
    active: [],
    history: [],
    summary: { active: 0, retained: 0, success_rate: null, p95_duration_ms: null },
  };

  const result = reduceActivityEvent(
    page,
    {
      type: "snapshot",
      payload: {
        cursor: "1",
        active: [entry],
        summary: { active: 3, retained: 9, success_rate: 80, p95_duration_ms: 700 },
      },
    },
    { model: "Other/Model" },
  )!;

  expect(result.active).toHaveLength(0);
  expect(result.summary.active).toBe(3);
  expect(result.summary.retained).toBe(9);
});

test("live duration advances from the server sample without comparing wall clocks", () => {
  const observed = observeActivityEntry({
    ...entry,
    started_at_ms: 99_999_999,
    duration_ms: 1_250,
  }, 5_000);

  expect(activityDurationMs(observed, 5_375)).toBe(1_625);
});

test("activity durations render with two decimal places", () => {
  expect(formatActivityDuration(42)).toBe("42.00 ms");
  expect(formatActivityDuration(1_234)).toBe("1.23 s");
  expect(formatActivityDuration(61_250)).toBe("1m 1.25s");
});

test("completed requests remove live client metadata from the replay buffer", () => {
  const liveEntry = {
    ...entry,
    address: "203.0.113.7",
    user_agent: "orchion-test-agent/1.0",
  };
  const buffered = scrubCompletedActivityEvents([
    { type: "started", payload: { cursor: "1", entry: liveEntry } },
    { type: "snapshot", payload: { cursor: "2", active: [liveEntry] } },
  ], entry.id);

  expect(buffered).toHaveLength(1);
  expect(buffered[0]?.payload.active).toEqual([]);
  expect(JSON.stringify(buffered)).not.toContain("203.0.113.7");
  expect(JSON.stringify(buffered)).not.toContain("orchion-test-agent/1.0");
});
