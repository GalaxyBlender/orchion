use crate::api::activity::{ActivityEventPayload, ActivityFilter, ActivityHub};
use crate::api::http::ServerShutdown;
use crate::api::http_shared::authorize;
use crate::api::openai::ApiError;
use crate::application::ServerApplication;
use async_stream::stream;
use axum::Json;
use axum::extract::{Extension, RawQuery, State};
use axum::http::{HeaderMap, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;

#[derive(Debug, Default)]
pub(crate) struct ActivityQuery {
    limit: Option<usize>,
    before: Option<u64>,
    operation: Option<crate::api::activity::ActivityOperation>,
    outcome: Option<crate::api::activity::ActivityOutcome>,
    model: Option<String>,
}

pub(crate) async fn list_activity<S>(
    State(state): State<Arc<S>>,
    Extension(hub): Extension<ActivityHub>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
) -> Result<Json<crate::api::activity::ActivityPage>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let query = ActivityQuery::from_raw(raw_query.as_deref())?;
    Ok(Json(hub.page(&query.into_filter())))
}

pub(crate) async fn activity_events<S>(
    State(state): State<Arc<S>>,
    Extension(hub): Extension<ActivityHub>,
    Extension(shutdown): Extension<ServerShutdown>,
    headers: HeaderMap,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let mut receiver = hub.subscribe();
    let snapshot = hub.snapshot_event();
    let event_stream = stream! {
        yield Ok::<Event, Infallible>(sse_event("snapshot", &snapshot));
        loop {
            let event = tokio::select! {
                () = shutdown.cancelled() => break,
                event = receiver.recv() => event,
            };
            match event {
                Ok(event) => yield Ok(sse_event(event.kind.as_str(), &event.payload)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    let (next_receiver, reset) = hub.reset_subscription();
                    receiver = next_receiver;
                    yield Ok(sse_event("reset", &reset));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    let mut response = Sse::new(event_stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}

impl ActivityQuery {
    fn from_raw(raw: Option<&str>) -> Result<Self, ApiError> {
        let mut query = Self::default();
        for (key, value) in url::form_urlencoded::parse(raw.unwrap_or_default().as_bytes()) {
            match key.as_ref() {
                "limit" => {
                    let limit = value.parse::<usize>().map_err(|_| invalid_query("limit"))?;
                    if !(1..=MAX_LIMIT).contains(&limit) {
                        return Err(ApiError::invalid_request(
                            format!("`limit` must be between 1 and {MAX_LIMIT}"),
                            Some("limit"),
                            Some("invalid_activity_query"),
                        ));
                    }
                    query.limit = Some(limit);
                }
                "before" => {
                    query.before = Some(value.parse().map_err(|_| invalid_query("before"))?);
                }
                "operation" => query.operation = Some(parse_operation(&value)?),
                "outcome" => query.outcome = Some(parse_outcome(&value)?),
                "model" => query.model = Some(value.into_owned()),
                _ => {}
            }
        }
        Ok(query)
    }

    fn into_filter(self) -> ActivityFilter {
        ActivityFilter {
            limit: self.limit.unwrap_or(DEFAULT_LIMIT),
            before: self.before,
            operation: self.operation,
            outcome: self.outcome,
            model: self.model.filter(|model| !model.is_empty()),
        }
    }
}

fn parse_operation(value: &str) -> Result<crate::api::activity::ActivityOperation, ApiError> {
    use crate::api::activity::ActivityOperation;
    match value {
        "asr" => Ok(ActivityOperation::Asr),
        "asr_stream" => Ok(ActivityOperation::AsrStream),
        "tts" => Ok(ActivityOperation::Tts),
        "ocr" => Ok(ActivityOperation::Ocr),
        "pdf" => Ok(ActivityOperation::Pdf),
        "chat" => Ok(ActivityOperation::Chat),
        "responses" => Ok(ActivityOperation::Responses),
        _ => Err(invalid_query("operation")),
    }
}

fn parse_outcome(value: &str) -> Result<crate::api::activity::ActivityOutcome, ApiError> {
    use crate::api::activity::ActivityOutcome;
    match value {
        "success" => Ok(ActivityOutcome::Success),
        "client_error" => Ok(ActivityOutcome::ClientError),
        "server_error" => Ok(ActivityOutcome::ServerError),
        "cancelled" => Ok(ActivityOutcome::Cancelled),
        "disconnected" => Ok(ActivityOutcome::Disconnected),
        "timeout" => Ok(ActivityOutcome::Timeout),
        "resource_exhausted" => Ok(ActivityOutcome::ResourceExhausted),
        _ => Err(invalid_query("outcome")),
    }
}

fn invalid_query(param: &'static str) -> ApiError {
    ApiError::invalid_request(
        format!("invalid `{param}` activity query value"),
        Some(param),
        Some("invalid_activity_query"),
    )
}

fn sse_event(event_type: &'static str, payload: &ActivityEventPayload) -> Event {
    Event::default()
        .event(event_type)
        .id(payload.cursor.clone())
        .data(serde_json::to_string(payload).expect("activity event payload must serialize"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::activity::ActivityOperation;

    #[test]
    fn operation_query_accepts_llm_operations() {
        assert_eq!(parse_operation("chat").unwrap(), ActivityOperation::Chat);
        assert_eq!(
            parse_operation("responses").unwrap(),
            ActivityOperation::Responses
        );
    }
}
