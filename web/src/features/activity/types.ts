export type ActivityState = "in_flight" | "completed";
export type ActivityTransport = "http" | "websocket";
export type ActivityOperation = "asr" | "asr_stream" | "tts" | "ocr" | "pdf";
export type ActivityOutcome =
  | "success"
  | "client_error"
  | "server_error"
  | "cancelled"
  | "disconnected"
  | "timeout"
  | "resource_exhausted";

export interface ActivityEntry {
  id: string;
  state: ActivityState;
  transport: ActivityTransport;
  operation: ActivityOperation;
  method: string;
  route: string;
  model?: string;
  address?: string;
  user_agent?: string;
  started_at_ms: number;
  duration_ms?: number;
  duration_observed_at_ms?: number;
  http_status?: number;
  outcome?: ActivityOutcome;
  input_bytes?: number;
  error_code?: string;
  error_message?: string;
}

export interface ActivitySummary {
  active: number;
  retained: number;
  success_rate: number | null;
  p95_duration_ms: number | null;
}

export interface ActivityPage {
  enabled: boolean;
  cursor: string;
  active: ActivityEntry[];
  history: ActivityEntry[];
  summary: ActivitySummary;
  next_before?: string;
}

export interface ActivityQuery {
  limit?: number;
  before?: string;
  operation?: ActivityOperation;
  outcome?: ActivityOutcome;
  model?: string;
}

export type ActivityConnection = "connecting" | "live" | "reconnecting" | "offline";
export type ActivityEventType = "snapshot" | "started" | "updated" | "completed" | "reset";

export interface ActivityEventPayload {
  cursor: string;
  entry?: ActivityEntry;
  active?: ActivityEntry[];
  summary?: ActivitySummary;
}

export interface ActivityStreamEvent {
  type: ActivityEventType;
  payload: ActivityEventPayload;
}
