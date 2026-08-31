use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    InFlight,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ActivityTransport {
    Http,
    Websocket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
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
    pub prefill_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ActivitySummary {
    pub active: usize,
    pub retained: usize,
    pub success_rate: Option<f64>,
    pub p95_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ActivityPage {
    pub enabled: bool,
    pub cursor: String,
    pub active: Vec<ActivityEntry>,
    pub history: Vec<ActivityEntry>,
    pub summary: ActivitySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_before: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ActivityEventPayload {
    pub cursor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<ActivityEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<Vec<ActivityEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ActivitySummary>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> ActivityEntry {
        ActivityEntry {
            id: "42".to_string(),
            state: ActivityState::Completed,
            transport: ActivityTransport::Http,
            operation: ActivityOperation::Responses,
            method: "POST".to_string(),
            route: "/v1/responses".to_string(),
            model: Some("model-id".to_string()),
            address: Some("127.0.0.1".to_string()),
            user_agent: Some("test-agent".to_string()),
            started_at_ms: 100,
            duration_ms: Some(25),
            http_status: Some(200),
            outcome: Some(ActivityOutcome::Success),
            input_bytes: Some(128),
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            queue_time_ms: Some(2),
            eval_time_ms: Some(18),
            prefill_tokens_per_second: Some(5.0),
            decode_tokens_per_second: Some(10.0),
            error_code: Some("none".to_string()),
            error_message: Some("none".to_string()),
        }
    }

    fn round_trip<T>(value: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let encoded = serde_json::to_string(value).unwrap();
        assert_eq!(&serde_json::from_str::<T>(&encoded).unwrap(), value);
    }

    #[test]
    fn activity_types_round_trip() {
        for state in [ActivityState::InFlight, ActivityState::Completed] {
            round_trip(&state);
        }
        for transport in [ActivityTransport::Http, ActivityTransport::Websocket] {
            round_trip(&transport);
        }
        for operation in [
            ActivityOperation::Asr,
            ActivityOperation::AsrStream,
            ActivityOperation::Tts,
            ActivityOperation::Ocr,
            ActivityOperation::Pdf,
            ActivityOperation::Chat,
            ActivityOperation::Responses,
        ] {
            round_trip(&operation);
        }
        for outcome in [
            ActivityOutcome::Success,
            ActivityOutcome::ClientError,
            ActivityOutcome::ServerError,
            ActivityOutcome::Cancelled,
            ActivityOutcome::Disconnected,
            ActivityOutcome::Timeout,
            ActivityOutcome::ResourceExhausted,
        ] {
            round_trip(&outcome);
        }

        let entry = entry();
        let summary = ActivitySummary {
            active: 1,
            retained: 2,
            success_rate: Some(0.5),
            p95_duration_ms: Some(25),
        };
        let page = ActivityPage {
            enabled: true,
            cursor: "43".to_string(),
            active: vec![entry.clone()],
            history: vec![entry.clone()],
            summary: summary.clone(),
            next_before: Some("41".to_string()),
        };
        let event = ActivityEventPayload {
            cursor: "43".to_string(),
            entry: Some(entry.clone()),
            active: Some(vec![entry.clone()]),
            summary: Some(summary.clone()),
        };

        round_trip(&entry);
        round_trip(&summary);
        round_trip(&page);
        round_trip(&event);
    }

    #[test]
    fn optional_activity_fields_retain_existing_omission_rules() {
        let mut entry = entry();
        entry.model = None;
        entry.address = None;
        entry.user_agent = None;
        entry.duration_ms = None;
        entry.http_status = None;
        entry.outcome = None;
        entry.input_bytes = None;
        entry.prompt_tokens = None;
        entry.completion_tokens = None;
        entry.queue_time_ms = None;
        entry.eval_time_ms = None;
        entry.prefill_tokens_per_second = None;
        entry.decode_tokens_per_second = None;
        entry.error_code = None;
        entry.error_message = None;

        let entry_json = serde_json::to_value(&entry).unwrap();
        for field in [
            "model",
            "address",
            "user_agent",
            "duration_ms",
            "http_status",
            "outcome",
            "input_bytes",
            "prompt_tokens",
            "completion_tokens",
            "queue_time_ms",
            "eval_time_ms",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "error_code",
            "error_message",
        ] {
            assert!(entry_json.get(field).is_none(), "{field} must be omitted");
        }
        round_trip(&entry);

        let summary = ActivitySummary {
            active: 0,
            retained: 0,
            success_rate: None,
            p95_duration_ms: None,
        };
        let summary_json = serde_json::to_value(&summary).unwrap();
        assert!(summary_json.get("success_rate").unwrap().is_null());
        assert!(summary_json.get("p95_duration_ms").unwrap().is_null());

        let page = ActivityPage {
            enabled: false,
            cursor: "0".to_string(),
            active: Vec::new(),
            history: Vec::new(),
            summary,
            next_before: None,
        };
        let page_json = serde_json::to_value(&page).unwrap();
        assert!(page_json.get("next_before").is_none());
        round_trip(&page);

        let event = ActivityEventPayload {
            cursor: "0".to_string(),
            entry: None,
            active: None,
            summary: None,
        };
        let event_json = serde_json::to_value(&event).unwrap();
        assert_eq!(event_json.as_object().unwrap().len(), 1);
        round_trip(&event);
    }
}
