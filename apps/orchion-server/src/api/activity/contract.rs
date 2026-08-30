use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    InFlight,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActivityTransport {
    Http,
    Websocket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActivityOperation {
    Asr,
    AsrStream,
    Tts,
    Ocr,
    Pdf,
    Chat,
    Responses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActivityOutcome {
    Success,
    ClientError,
    ServerError,
    Cancelled,
    Disconnected,
    Timeout,
    ResourceExhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, ToSchema)]
pub struct ActivityEntry {
    pub id: String,
    pub state: ActivityState,
    pub transport: ActivityTransport,
    pub operation: ActivityOperation,
    pub method: String,
    pub route: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    pub started_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ActivityOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ActivityError {
    pub(crate) code: Option<String>,
    pub(crate) message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
pub struct ActivitySummary {
    pub active: usize,
    pub retained: usize,
    pub success_rate: Option<f64>,
    pub p95_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
pub struct ActivityPage {
    pub enabled: bool,
    pub cursor: String,
    pub active: Vec<ActivityEntry>,
    pub history: Vec<ActivityEntry>,
    pub summary: ActivitySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_before: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivityEventPayload {
    pub cursor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<ActivityEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<Vec<ActivityEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ActivitySummary>,
}
