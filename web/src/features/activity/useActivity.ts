import { useEffect, useRef, useState } from "react";
import { parseNetworkError } from "@/shared/api/errors";
import { ApiRequestError, type ApiSettings } from "@/shared/api/types";
import { fetchActivity, streamActivity } from "./client";
import {
  isNewerActivityCursor,
  reduceActivityEvent,
  replayActivityEvents,
  scrubCompletedActivityEvents,
} from "./reducer";
import type {
  ActivityConnection,
  ActivityPage,
  ActivityQuery,
  ActivityStreamEvent,
} from "./types";

const MAX_BUFFERED_EVENTS = 2_048;
const STABLE_CONNECTION_MS = 10_000;

export function useActivity(settings: ApiSettings, query: ActivityQuery) {
  const [page, setPage] = useState<ActivityPage | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<ApiRequestError | null>(null);
  const [connection, setConnection] = useState<ActivityConnection>("connecting");
  const [reloadToken, setReloadToken] = useState(0);
  const requestIdRef = useRef(0);
  const bufferedEventsRef = useRef<ActivityStreamEvent[]>([]);
  const queryKey = JSON.stringify([
    query.limit,
    query.before,
    query.operation,
    query.outcome,
    query.model,
  ]);
  const pageQueryKeyRef = useRef(queryKey);

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const controller = new AbortController();
    if (pageQueryKeyRef.current !== queryKey) {
      pageQueryKeyRef.current = queryKey;
      bufferedEventsRef.current = [];
      setPage(null);
    }
    setIsLoading(true);

    fetchActivity(settings, query, controller.signal)
      .then((nextPage) => {
        if (requestId !== requestIdRef.current) return;
        const bufferedEvents = bufferedEventsRef.current.filter((event) =>
          isNewerActivityCursor(event.payload.cursor, nextPage.cursor),
        );
        bufferedEventsRef.current = bufferedEvents;
        const mergedPage = replayActivityEvents(nextPage, bufferedEvents, query);
        setPage((current) =>
          current && isNewerActivityCursor(current.cursor, mergedPage.cursor) ? current : mergedPage,
        );
        setError(null);
        setIsLoading(false);
      })
      .catch((caughtError: unknown) => {
        if (controller.signal.aborted || requestId !== requestIdRef.current) return;
        setError(caughtError instanceof ApiRequestError ? caughtError : parseNetworkError(caughtError));
        setIsLoading(false);
      });

    return () => controller.abort();
  }, [
    settings.apiKey,
    queryKey,
    reloadToken,
  ]);

  useEffect(() => {
    const controller = new AbortController();
    let attempt = 0;
    let refreshTimer: ReturnType<typeof setTimeout> | undefined;

    const scheduleRefresh = () => {
      if (refreshTimer !== undefined) return;
      refreshTimer = setTimeout(() => {
        refreshTimer = undefined;
        setReloadToken((token) => token + 1);
      }, 200);
    };
    const run = async () => {
      while (!controller.signal.aborted) {
        setConnection(attempt === 0 ? "connecting" : attempt >= 3 ? "offline" : "reconnecting");
        let connectedAt: number | undefined;
        try {
          await streamActivity(settings, controller.signal, (event) => {
            connectedAt ??= Date.now();
            setConnection("live");
            const completedId = event.type === "completed" ? event.payload.entry?.id : undefined;
            if (completedId) {
              bufferedEventsRef.current = scrubCompletedActivityEvents(
                bufferedEventsRef.current,
                completedId,
              );
            }
            bufferedEventsRef.current.push(event);
            if (bufferedEventsRef.current.length > MAX_BUFFERED_EVENTS) {
              bufferedEventsRef.current.splice(
                0,
                bufferedEventsRef.current.length - MAX_BUFFERED_EVENTS,
              );
              scheduleRefresh();
            }
            setPage((current) => reduceActivityEvent(current, event, query));
            if (event.type === "snapshot" || event.type === "completed" || event.type === "reset") {
              scheduleRefresh();
            }
          });
          if (!controller.signal.aborted) throw new Error("Activity event stream closed");
        } catch {
          if (controller.signal.aborted) return;
          if (connectedAt !== undefined && Date.now() - connectedAt >= STABLE_CONNECTION_MS) {
            attempt = 0;
          }
          attempt += 1;
          setConnection(attempt >= 3 ? "offline" : "reconnecting");
          const delay = Math.min(5_000, 1_000 * 2 ** Math.min(attempt - 1, 2));
          await abortableDelay(delay, controller.signal);
        }
      }
    };

    void run();
    return () => {
      controller.abort();
      if (refreshTimer !== undefined) clearTimeout(refreshTimer);
    };
  }, [settings.apiKey, queryKey]);

  return {
    page,
    isLoading,
    error,
    connection,
    reload: () => setReloadToken((token) => token + 1),
  };
}

function abortableDelay(delay: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timeout);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timeout = setTimeout(finish, delay);
    signal.addEventListener("abort", finish, { once: true });
  });
}
