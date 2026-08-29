import type { ActivityEntry, ActivityPage, ActivityQuery, ActivityStreamEvent } from "./types";

export function reduceActivityEvent(
  page: ActivityPage | null,
  event: ActivityStreamEvent,
  query: ActivityQuery = {},
): ActivityPage | null {
  if (!page) return page;
  if (!isNewerActivityCursor(event.payload.cursor, page.cursor)) return page;

  const summary = event.payload.summary ?? page.summary;

  if (event.type === "snapshot") {
    const active = (event.payload.active ?? []).filter((entry) => matchesQuery(entry, query));
    return {
      ...page,
      cursor: event.payload.cursor,
      active,
      summary,
    };
  }

  const entry = event.payload.entry;
  if (!entry || event.type === "reset") {
    return { ...page, cursor: event.payload.cursor, summary };
  }

  const activeWithoutEntry = page.active.filter((item) => item.id !== entry.id);
  if (!matchesQuery(entry, query)) {
    return {
      ...page,
      cursor: event.payload.cursor,
      active: activeWithoutEntry,
      summary,
    };
  }

  if (event.type === "started" || event.type === "updated") {
    const active = [entry, ...activeWithoutEntry].sort(
      (left, right) => right.started_at_ms - left.started_at_ms,
    );
    return {
      ...page,
      cursor: event.payload.cursor,
      active,
      summary,
    };
  }

  return {
    ...page,
    cursor: event.payload.cursor,
    active: activeWithoutEntry,
    history: page.history,
    summary,
  };
}

export function isNewerActivityCursor(candidate: string, current: string): boolean {
  try {
    return BigInt(candidate) > BigInt(current);
  } catch {
    return false;
  }
}

export function replayActivityEvents(
  page: ActivityPage,
  events: ActivityStreamEvent[],
  query: ActivityQuery = {},
): ActivityPage {
  return events.reduce(
    (current, event) => reduceActivityEvent(current, event, query) ?? current,
    page,
  );
}

export function scrubCompletedActivityEvents(
  events: ActivityStreamEvent[],
  completedId: string,
): ActivityStreamEvent[] {
  return events.flatMap((event) => {
    if (event.payload.entry?.id === completedId) return [];
    if (!event.payload.active?.some((entry) => entry.id === completedId)) return [event];
    return [{
      ...event,
      payload: {
        ...event.payload,
        active: event.payload.active.filter((entry) => entry.id !== completedId),
      },
    }];
  });
}

function matchesQuery(entry: ActivityEntry, query: ActivityQuery): boolean {
  if (query.before && entry.state === "completed") {
    try {
      if (BigInt(entry.id) >= BigInt(query.before)) return false;
    } catch {
      return false;
    }
  }
  if (query.operation && entry.operation !== query.operation) return false;
  if (query.outcome && entry.outcome !== query.outcome) return false;
  if (query.model && entry.model !== query.model) return false;
  return true;
}
