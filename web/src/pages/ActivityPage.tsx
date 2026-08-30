import { startTransition, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Activity as ActivityIcon, ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { useActivity } from "@/features/activity/useActivity";
import {
  activityDurationMs,
  formatActivityDuration,
  monotonicNowMs,
} from "@/features/activity/timing";
import type {
  ActivityConnection,
  ActivityEntry,
  ActivityOperation,
  ActivityOutcome,
} from "@/features/activity/types";
import { loadPersistentState } from "@/shared/storage/persistentState";
import { Alert, Badge, Button, StateView } from "@/shared/ui";

const OPERATIONS: ActivityOperation[] = ["chat", "responses", "asr", "asr_stream", "tts", "ocr", "pdf"];
const OUTCOMES: ActivityOutcome[] = [
  "success",
  "client_error",
  "server_error",
  "cancelled",
  "disconnected",
  "timeout",
  "resource_exhausted",
];

export function ActivityPage() {
  const { t, i18n } = useTranslation();
  const [settings] = useState(() => loadPersistentState().settings);
  const [operation, setOperation] = useState<ActivityOperation | "">("");
  const [outcome, setOutcome] = useState<ActivityOutcome | "">("");
  const [modelDraft, setModelDraft] = useState("");
  const [model, setModel] = useState("");
  const [before, setBefore] = useState<string | undefined>();
  const [newerCursors, setNewerCursors] = useState<Array<string | undefined>>([]);
  const activity = useActivity(settings, {
    limit: 50,
    before,
    operation: operation || undefined,
    outcome: outcome || undefined,
    model: model || undefined,
  });

  const resetCursor = () => {
    setBefore(undefined);
    setNewerCursors([]);
  };
  const handleOlder = () => {
    if (!activity.page?.next_before) return;
    setNewerCursors((cursors) => [...cursors, before]);
    setBefore(activity.page.next_before);
  };
  const handleNewer = () => {
    setNewerCursors((cursors) => {
      if (cursors.length === 0) return cursors;
      setBefore(cursors[cursors.length - 1]);
      return cursors.slice(0, -1);
    });
  };
  const handleReload = () => {
    startTransition(() => activity.reload());
  };

  return (
    <div className="page activity-page animate-fade-in">
      <header className="page-header">
        <div className="page-title-row">
          <div className="stack gap-xs">
            <p className="card-eyebrow">{t("activity.kicker")}</p>
            <div className="activity-title-line">
              <h2 className="page-title">{t("activity.title")}</h2>
              <ConnectionBadge connection={activity.connection} />
            </div>
          </div>
          <Button
            className="activity-reload"
            variant="secondary"
            loading={activity.isLoading}
            onClick={handleReload}
            icon={<RefreshCw size={16} />}
            aria-label={t("activity.reload")}
            title={t("activity.reload")}
          >
            {t("activity.reload")}
          </Button>
        </div>
        <p className="page-description">{t("activity.subtitle")}</p>
      </header>

      {activity.error && (
        <Alert variant="warning" title={t("activity.unavailable")}>{activity.error.message}</Alert>
      )}

      {activity.isLoading && !activity.page ? (
        <StateView type="loading" message={t("activity.loading")} />
      ) : activity.page && !activity.page.enabled ? (
        <StateView
          type="empty"
          title={t("activity.disabledTitle")}
          description={t("activity.disabledDescription")}
        />
      ) : activity.page ? (
        <>
          <section className="activity-stats" aria-label={t("activity.summaryLabel")}>
            <Stat label={t("activity.stats.active")} value={String(activity.page.summary.active)} />
            <Stat label={t("activity.stats.retained")} value={String(activity.page.summary.retained)} />
            <Stat
              label={t("activity.stats.success")}
              value={formatPercent(activity.page.summary.success_rate)}
            />
            <Stat
              label={t("activity.stats.p95")}
              value={formatActivityDuration(activity.page.summary.p95_duration_ms)}
            />
          </section>

          <section className="activity-section">
            <div className="activity-section-heading">
              <div>
                <p className="card-eyebrow">{t("activity.inFlight.kicker")}</p>
                <h3>{t("activity.inFlight.title")}</h3>
              </div>
              <Badge variant={activity.page.active.length > 0 ? "accent" : "default"}>
                {activity.page.active.length}
              </Badge>
            </div>
            {activity.page.active.length === 0 ? (
              <div className="activity-empty-line">{t("activity.inFlight.empty")}</div>
            ) : (
              <ActivityTable
                entries={activity.page.active}
                locale={i18n.resolvedLanguage}
                live
              />
            )}
          </section>

          <section className="activity-section">
            <div className="activity-section-heading activity-history-heading">
              <div>
                <p className="card-eyebrow">{t("activity.history.kicker")}</p>
                <h3>{t("activity.history.title")}</h3>
              </div>
              <form
                className="activity-filters"
                onSubmit={(event) => {
                  event.preventDefault();
                  setModel(modelDraft.trim());
                  resetCursor();
                }}
              >
                <select
                  className="select activity-filter-control"
                  aria-label={t("activity.filters.operation")}
                  value={operation}
                  onChange={(event) => {
                    setOperation(event.target.value as ActivityOperation | "");
                    resetCursor();
                  }}
                >
                  <option value="">{t("activity.filters.allOperations")}</option>
                  {OPERATIONS.map((value) => (
                    <option key={value} value={value}>{t(`activity.operations.${value}`)}</option>
                  ))}
                </select>
                <select
                  className="select activity-filter-control"
                  aria-label={t("activity.filters.outcome")}
                  value={outcome}
                  onChange={(event) => {
                    setOutcome(event.target.value as ActivityOutcome | "");
                    resetCursor();
                  }}
                >
                  <option value="">{t("activity.filters.allOutcomes")}</option>
                  {OUTCOMES.map((value) => (
                    <option key={value} value={value}>{t(`activity.outcomes.${value}`)}</option>
                  ))}
                </select>
                <input
                  className="input activity-model-filter"
                  value={modelDraft}
                  onChange={(event) => setModelDraft(event.target.value)}
                  placeholder={t("activity.filters.modelPlaceholder")}
                  aria-label={t("activity.filters.model")}
                />
                <Button size="sm" type="submit">{t("activity.filters.apply")}</Button>
              </form>
            </div>

            {activity.page.history.length === 0 ? (
              <div className="activity-empty-line">{t("activity.history.empty")}</div>
            ) : (
              <ActivityTable
                entries={activity.page.history}
                locale={i18n.resolvedLanguage}
              />
            )}

            <div className="activity-pagination">
              <Button
                variant="ghost"
                size="sm"
                disabled={newerCursors.length === 0}
                onClick={handleNewer}
                icon={<ChevronLeft size={15} />}
              >
                {t("activity.pagination.newer")}
              </Button>
              <span>{t("activity.pagination.retained", { count: activity.page.summary.retained })}</span>
              <Button
                variant="ghost"
                size="sm"
                disabled={!activity.page.next_before}
                onClick={handleOlder}
                icon={<ChevronRight size={15} />}
                iconPosition="right"
              >
                {t("activity.pagination.older")}
              </Button>
            </div>
          </section>
        </>
      ) : (
        <StateView
          type="offline"
          title={t("activity.unavailable")}
          description={activity.error?.message}
          action={<Button onClick={handleReload}>{t("activity.reload")}</Button>}
        />
      )}
    </div>
  );
}

function ConnectionBadge({ connection }: { connection: ActivityConnection }) {
  const { t } = useTranslation();
  const variant = connection === "live" ? "success" : connection === "offline" ? "danger" : "warning";
  return (
    <Badge variant={variant} dot pulse={connection === "live"}>
      {t(`activity.connection.${connection}`)}
    </Badge>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="activity-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ActivityTable({
  entries,
  locale,
  live = false,
}: {
  entries: ActivityEntry[];
  locale?: string;
  live?: boolean;
}) {
  const { t } = useTranslation();
  const [now, setNow] = useState(monotonicNowMs);
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setNow(monotonicNowMs()), 10);
    return () => window.clearInterval(timer);
  }, [live]);
  return (
    <div className="activity-table-wrap">
      <table className={`activity-table${live ? " activity-table-live" : ""}`}>
        <thead>
          <tr>
            <th scope="col" className="activity-col-started">{t("activity.columns.started")}</th>
            <th scope="col" className="activity-col-type">{t("activity.columns.type")}</th>
            <th scope="col" className="activity-col-model">{t("activity.columns.model")}</th>
            <th scope="col" className="activity-col-request">{t("activity.columns.request")}</th>
            {live && (
              <>
                <th scope="col" className="activity-col-address">{t("activity.columns.address")}</th>
                <th scope="col" className="activity-col-user-agent">{t("activity.columns.userAgent")}</th>
              </>
            )}
            <th scope="col" className="activity-col-result">{t("activity.columns.result")}</th>
            {!live && (
              <th scope="col" className="activity-col-status">{t("activity.columns.status")}</th>
            )}
            <th scope="col" className="activity-col-duration">{t("activity.columns.duration")}</th>
            <th scope="col" className="activity-col-input">{t("activity.columns.input")}</th>
            <th scope="col" className="activity-col-id">{t("activity.columns.id")}</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.id}>
              <td data-label={t("activity.columns.started")} className="activity-time">
                {new Intl.DateTimeFormat(locale, {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                }).format(entry.started_at_ms)}
              </td>
              <td data-label={t("activity.columns.type")}>
                <Badge variant={entry.transport === "websocket" ? "accent-blue" : "default"}>
                  {t(`activity.operations.${entry.operation}`)}
                </Badge>
              </td>
              <td data-label={t("activity.columns.model")} className="activity-model" title={entry.model}>
                {entry.model ?? "-"}
              </td>
              <td data-label={t("activity.columns.request")}>
                <div className="activity-request-cell">
                  <span>{entry.method}</span>
                  <code>{entry.route}</code>
                </div>
              </td>
              {live && (
                <>
                  <td
                    data-label={t("activity.columns.address")}
                    className="activity-address activity-mono"
                    title={entry.address}
                  >
                    {entry.address ?? "-"}
                  </td>
                  <td
                    data-label={t("activity.columns.userAgent")}
                    className="activity-user-agent"
                    title={entry.user_agent}
                  >
                    {entry.user_agent ?? "-"}
                  </td>
                </>
              )}
              <td data-label={t("activity.columns.result")}>
                {live ? (
                  <Badge variant="accent" pulse>{t("activity.outcomes.in_flight")}</Badge>
                ) : (
                  <OutcomeBadge outcome={entry.outcome} />
                )}
                {entry.error_code && <small className="activity-error-code">{entry.error_code}</small>}
                {entry.error_message && (
                  <small className="activity-error-message" title={entry.error_message}>
                    {entry.error_message}
                  </small>
                )}
              </td>
              {!live && (
                <td data-label={t("activity.columns.status")} className="activity-mono">
                  {entry.http_status ?? "-"}
                </td>
              )}
              <td data-label={t("activity.columns.duration")} className="activity-mono">
                {formatActivityDuration(live ? activityDurationMs(entry, now) : entry.duration_ms)}
              </td>
              <td data-label={t("activity.columns.input")} className="activity-mono">
                {formatBytes(entry.input_bytes)}
              </td>
              <td data-label={t("activity.columns.id")} className="activity-id">#{entry.id}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OutcomeBadge({ outcome }: { outcome?: ActivityOutcome }) {
  const { t } = useTranslation();
  const variant = outcome === "success"
    ? "success"
    : outcome === "client_error" || outcome === "resource_exhausted"
      ? "warning"
      : "danger";
  return <Badge variant={variant}>{t(`activity.outcomes.${outcome ?? "server_error"}`)}</Badge>;
}

function formatPercent(value: number | null): string {
  return value === null ? "-" : `${value.toFixed(1)}%`;
}

function formatBytes(value?: number): string {
  if (value === undefined) return "-";
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KiB`;
  return `${(value / 1_048_576).toFixed(1)} MiB`;
}
