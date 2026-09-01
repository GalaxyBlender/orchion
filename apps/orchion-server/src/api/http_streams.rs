use crate::api::http_shared::authorize;
use crate::api::llm_streams::{AccessError, LlmStreams, valid_stream_id};
use crate::api::openai::ApiError;
use crate::api::sse;
use crate::application::ServerApplication;
use axum::Json;
use axum::extract::{Extension, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use std::sync::Arc;

pub(crate) type StreamLookupRequest = orchion_protocol::LlmStreamLookupRequest;
pub(crate) type StreamLookupResponse = orchion_protocol::LlmStreamLookupResponse;

pub(crate) async fn get_stream<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    RawQuery(query): RawQuery,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    let (stream_id, follow) = parse_query(query.as_deref(), true)?;
    let after = match headers.get("last-event-id") {
        None => 0,
        Some(value) => value
            .to_str()
            .ok()
            .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| malformed("Last-Event-ID"))?,
    };
    let body = streams
        .attach(&stream_id, principal, after, follow)
        .map_err(map_access_error)?;
    Ok(sse::response(sse::keepalive(
        body,
        state.api_policy().streaming.keepalive_interval,
    )))
}

pub(crate) async fn lookup_streams<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    Json(request): Json<StreamLookupRequest>,
) -> Result<Json<StreamLookupResponse>, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    if request.stream_ids.len() > streams.lookup_max() {
        return Err(ApiError::invalid_request(
            format!(
                "stream_ids may contain at most {} entries",
                streams.lookup_max()
            ),
            Some("stream_ids"),
            Some("invalid_stream_lookup"),
        ));
    }
    Ok(Json(StreamLookupResponse {
        streams: streams.lookup(principal, &request.stream_ids),
    }))
}

pub(crate) async fn delete_stream<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    RawQuery(query): RawQuery,
) -> Result<StatusCode, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    let (stream_id, _) = parse_query(query.as_deref(), false)?;
    streams.delete(&stream_id, principal);
    Ok(StatusCode::NO_CONTENT)
}

fn parse_query(raw: Option<&str>, allow_follow: bool) -> Result<(String, bool), ApiError> {
    let mut stream_id = None;
    let mut follow = true;
    for (key, value) in url::form_urlencoded::parse(raw.unwrap_or_default().as_bytes()) {
        match key.as_ref() {
            "stream_id" if stream_id.is_none() => stream_id = Some(value.into_owned()),
            "follow" if allow_follow => {
                follow = match value.as_ref() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(malformed("follow")),
                };
            }
            _ => {}
        }
    }
    let stream_id = stream_id.ok_or_else(|| malformed("stream_id"))?;
    if !valid_stream_id(&stream_id) {
        return Err(malformed("stream_id"));
    }
    Ok((stream_id, follow))
}

fn malformed(param: &'static str) -> ApiError {
    ApiError::invalid_request(
        format!("invalid `{param}`"),
        Some(param),
        Some("invalid_resumable_stream"),
    )
}

fn map_access_error(error: AccessError) -> ApiError {
    match error {
        AccessError::NotFound => ApiError::stream_not_found(),
        AccessError::ReplayLost => ApiError::replay_lost(),
        AccessError::FollowersExhausted => ApiError::stream_capacity("stream follower"),
        AccessError::ShuttingDown => ApiError::shutting_down(),
        AccessError::InvalidCursor => malformed("Last-Event-ID"),
    }
}
