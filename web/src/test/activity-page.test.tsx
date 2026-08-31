import { expect, mock, test } from "bun:test";
import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as activityTiming from "../features/activity/timing";

function TestUi({ children }: { children?: ReactNode }) {
  return <div>{children}</div>;
}

mock.module("@/features/activity/timing", () => activityTiming);

mock.module("@/shared/ui", () => ({
  Alert: TestUi,
  Badge: TestUi,
  Button: TestUi,
  StateView: TestUi,
}));

mock.module("@/features/activity/useActivity", () => ({
  useActivity: () => ({
    connection: "live",
    error: undefined,
    isLoading: false,
    page: {
      enabled: true,
      cursor: "2",
      active: [{
        id: "active",
        state: "in_flight",
        transport: "http",
        operation: "chat",
        method: "POST",
        route: "/v1/chat/completions",
        started_at_ms: 1_000,
        prefill_tokens_per_second: 999.9,
        decode_tokens_per_second: 888.8,
      }],
      history: [
        {
          id: "llm",
          state: "completed",
          transport: "http",
          operation: "responses",
          method: "POST",
          route: "/v1/responses",
          started_at_ms: 1_000,
          outcome: "success",
          prefill_tokens_per_second: 123.456,
          decode_tokens_per_second: 45.678,
        },
        {
          id: "asr",
          state: "completed",
          transport: "http",
          operation: "asr",
          method: "POST",
          route: "/v1/audio/transcriptions",
          started_at_ms: 1_000,
          outcome: "success",
          prefill_tokens_per_second: 777.7,
          decode_tokens_per_second: 666.6,
        },
      ],
      summary: { active: 1, retained: 2, success_rate: 100, p95_duration_ms: 10 },
    },
    reload: () => {},
  }),
}));

mock.module("@/shared/storage/persistentState", () => ({
  loadPersistentState: () => ({ settings: {} }),
}));

mock.module("react-i18next", () => ({
  initReactI18next: { type: "3rdParty", init: () => {} },
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { resolvedLanguage: "en" },
  }),
}));

test("renders throughput only for completed LLM requests", async () => {
  const { ActivityPage } = await import("../pages/ActivityPage");
  const html = renderToStaticMarkup(<ActivityPage />);
  const historyHeading = html.indexOf("activity.history.title");
  const prefillHeading = html.indexOf("activity.columns.prefill");
  const decodeHeading = html.indexOf("activity.columns.decode");

  expect(historyHeading).toBeGreaterThanOrEqual(0);
  expect(prefillHeading).toBeGreaterThan(historyHeading);
  expect(decodeHeading).toBeGreaterThan(historyHeading);
  expect(html).toContain("123.5 tok/s");
  expect(html).toContain("45.7 tok/s");
  expect(html).not.toContain("999.9 tok/s");
  expect(html).not.toContain("888.8 tok/s");
  expect(html).not.toContain("777.7 tok/s");
  expect(html).not.toContain("666.6 tok/s");
});
