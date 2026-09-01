use crate::api::activity::{ActivityHub, track_activity};
use crate::api::http_activity::{activity_events, list_activity};
use crate::api::http_audio::{create_speech, create_transcription, create_transcription_ws};
use crate::api::http_llm::{
    control_chat_completion, count_response_input_tokens, create_chat_completion,
    create_completion, create_embeddings, create_response,
};
use crate::api::http_models::{
    list_model_statuses, list_models, load_model, retrieve_model, unload_model,
};
use crate::api::http_ocr::create_ocr;
use crate::api::http_pdf_images::create_pdf_images;
use crate::api::http_shared::authorize;
use crate::api::http_shared::origin_is_allowed;
use crate::api::http_streams::{delete_stream, get_stream, lookup_streams};
use crate::api::llm_streams::LlmStreams;
use crate::api::{docs, ui};
use crate::application::ServerApplication;
use crate::application::metrics::{
    HttpMethod, HttpRoute, Metrics, OPENMETRICS_CONTENT_TYPE, Outcome,
};
use crate::application::model_cache::ModelResidencyStatus;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, LOCATION};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Router, middleware};
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::sync::Arc;
use std::time::Instant;
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::watch;
use tower_http::cors::{AllowHeaders, AllowOrigin, Any, CorsLayer};
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

#[derive(Clone, Debug)]
pub struct ServerShutdown {
    signal: watch::Sender<bool>,
}

impl ServerShutdown {
    #[must_use]
    pub fn new() -> Self {
        let (signal, _) = watch::channel(false);
        Self { signal }
    }

    pub fn trigger(&self) {
        self.signal.send_replace(true);
    }

    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.signal.subscribe();
        if *receiver.borrow() {
            return;
        }
        let _ = receiver.changed().await;
    }

    pub(crate) fn is_triggered(&self) -> bool {
        *self.signal.borrow()
    }
}

impl Default for ServerShutdown {
    fn default() -> Self {
        Self::new()
    }
}

pub fn router<S>(state: Arc<S>) -> Router
where
    S: ServerApplication,
{
    router_with_shutdown(state, ServerShutdown::new())
}

pub fn router_with_shutdown<S>(state: Arc<S>, shutdown: ServerShutdown) -> Router
where
    S: ServerApplication,
{
    router_with_ui_routes_and_shutdown(state, ui::routes::<S>(), shutdown)
}

pub fn router_with_ui_routes<S>(state: Arc<S>, ui_routes: Router<Arc<S>>) -> Router
where
    S: ServerApplication,
{
    router_with_ui_routes_and_shutdown(state, ui_routes, ServerShutdown::new())
}

fn router_with_ui_routes_and_shutdown<S>(
    state: Arc<S>,
    ui_routes: Router<Arc<S>>,
    shutdown: ServerShutdown,
) -> Router
where
    S: ServerApplication,
{
    let policy = state.api_policy();
    let max_upload_size = policy.max_upload_size;
    let cors = cors_layer(&policy.cors_allowed_origins);
    let tts_enabled = policy.tts_models.is_some();
    let asr_enabled = policy.asr.is_some();
    let ocr_enabled = policy.ocr_enabled;
    let llm_enabled = policy.llm_enabled;
    let activity = ActivityHub::new(policy.activity);
    let metrics_store = state.metrics().clone();
    let llm_streams = LlmStreams::new(policy.streaming, shutdown.clone(), metrics_store.clone());
    let chat_controls = llm_streams.controls();
    let mut router = Router::new()
        .route("/", get(root_redirect))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz::<S>))
        .route("/metrics", get(metrics_endpoint::<S>))
        .route("/v1/models", get(list_models::<S>))
        .route(
            "/v1/models/{model}",
            get(retrieve_model::<S>).fallback(|| async { StatusCode::NOT_FOUND }),
        )
        .route("/api/models/status", get(list_model_statuses::<S>))
        .route("/api/models/load", post(load_model::<S>))
        .route("/api/models/unload", post(unload_model::<S>))
        .route("/v1/pdf/images", post(create_pdf_images::<S>))
        .route("/api/activity", get(list_activity::<S>))
        .route("/api/activity/events", get(activity_events::<S>));

    if tts_enabled {
        router = router.route("/v1/audio/speech", post(create_speech::<S>));
    }
    if asr_enabled {
        router = router
            .route("/v1/audio/transcriptions", post(create_transcription::<S>))
            .route(
                "/v1/audio/transcriptions/stream",
                get(create_transcription_ws::<S>),
            );
    }
    if ocr_enabled {
        router = router.route("/v1/ocr", post(create_ocr::<S>));
    }
    if llm_enabled {
        router = router
            .route("/v1/chat/completions", post(create_chat_completion::<S>))
            .route(
                "/v1/chat/completions/control",
                post(control_chat_completion::<S>),
            )
            .route("/v1/completions", post(create_completion::<S>))
            .route("/v1/responses", post(create_response::<S>))
            .route(
                "/v1/responses/input_tokens",
                post(count_response_input_tokens::<S>),
            )
            .route("/v1/embeddings", post(create_embeddings::<S>));
        router = router
            .route(
                "/v1/stream",
                get(get_stream::<S>).delete(delete_stream::<S>),
            )
            .route("/v1/streams/lookup", post(lookup_streams::<S>));
    }

    router
        .merge(ui_routes)
        .merge(docs::swagger_ui())
        .layer(DefaultBodyLimit::max(max_upload_size))
        .layer(cors)
        .with_state(state)
        .layer(middleware::from_fn_with_state(metrics_store, track_metrics))
        .layer(middleware::from_fn_with_state(
            activity.clone(),
            track_activity,
        ))
        .layer(Extension(shutdown))
        .layer(Extension(llm_streams))
        .layer(Extension(chat_controls))
        .layer(Extension(activity))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::DEBUG))
                .on_request(DefaultOnRequest::new().level(Level::DEBUG))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
}

fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(Any);
    if allowed_origins == ["*"] {
        layer.allow_origin(Any)
    } else {
        let allowed_origins = allowed_origins.to_vec();
        layer.allow_origin(AllowOrigin::predicate(move |origin, _| {
            origin_is_allowed(&allowed_origins, origin)
        }))
    }
}

async fn root_redirect() -> impl IntoResponse {
    (
        StatusCode::FOUND,
        [(LOCATION, HeaderValue::from_static("/ui"))],
    )
}

async fn healthz() -> &'static str {
    "ok"
}

#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub(crate) struct ReadinessResponse {
    status: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    reasons: Vec<ReadinessReason>,
}

#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub(crate) struct ReadinessReason {
    code: ReadinessReasonCode,
}

#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReadinessReasonCode {
    Shutdown,
    ResidentWorkerUnhealthy,
    RequiredModelLoadFailed,
}

async fn readyz<S>(
    State(state): State<Arc<S>>,
    Extension(shutdown): Extension<ServerShutdown>,
) -> impl IntoResponse
where
    S: ServerApplication,
{
    let snapshot = state.observability_snapshot().await;
    let mut reasons = Vec::new();
    if shutdown.is_triggered() || snapshot.shutdown {
        reasons.push(ReadinessReason {
            code: ReadinessReasonCode::Shutdown,
        });
    }
    for model in &snapshot.models {
        if model.residency == ModelResidencyStatus::Loaded && !model.worker_healthy {
            reasons.push(ReadinessReason {
                code: ReadinessReasonCode::ResidentWorkerUnhealthy,
            });
        }
        if model.required && model.last_load_failure.is_some() {
            reasons.push(ReadinessReason {
                code: ReadinessReasonCode::RequiredModelLoadFailed,
            });
        }
    }
    let ready = reasons.is_empty();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        [(CACHE_CONTROL, HeaderValue::from_static("no-store"))],
        axum::Json(ReadinessResponse {
            status: if ready { "ready" } else { "not_ready" },
            reasons,
        }),
    )
}

async fn metrics_endpoint<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
) -> impl IntoResponse
where
    S: ServerApplication,
{
    if let Err(error) = authorize(state.as_ref(), &headers) {
        return error.into_response();
    }
    let snapshot = state.observability_snapshot().await;
    match state
        .metrics()
        .scrape(&snapshot.models, streams.observation())
    {
        Ok(body) => (
            StatusCode::OK,
            [(
                CONTENT_TYPE,
                HeaderValue::from_static(OPENMETRICS_CONTENT_TYPE),
            )],
            body,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn track_metrics(
    State(metrics): State<Metrics>,
    request: Request<Body>,
    next: Next,
) -> axum::response::Response {
    let method = http_method(request.method());
    let route = http_route(request.uri().path());
    metrics.begin_http();
    let mut completion = Some(HttpMetricsCompletion {
        metrics,
        method,
        route,
        started: Instant::now(),
        outcome: None,
        completed: false,
    });
    let response = next.run(request).await;
    track_metrics_body(
        response,
        completion
            .take()
            .expect("HTTP metrics completion remains armed"),
    )
}

fn track_metrics_body(
    response: axum::response::Response,
    mut completion: HttpMetricsCompletion,
) -> axum::response::Response {
    completion.outcome = Some(http_outcome(response.status()));
    if response.body().is_end_stream() {
        completion.complete();
        return response;
    }
    let (parts, body) = response.into_parts();
    axum::response::Response::from_parts(
        parts,
        Body::new(MetricsBody {
            inner: body,
            completion: Some(completion),
        }),
    )
}

fn http_outcome(status: StatusCode) -> Outcome {
    match status.as_u16() {
        200..=399 => Outcome::Success,
        408 => Outcome::Timeout,
        429 => Outcome::ResourceExhausted,
        400..=499 => Outcome::ClientError,
        _ => Outcome::ServerError,
    }
}

struct MetricsBody {
    inner: Body,
    completion: Option<HttpMetricsCompletion>,
}

impl HttpBody for MetricsBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.inner).poll_frame(context) {
            Poll::Ready(None) => {
                if let Some(completion) = self.completion.take() {
                    completion.complete();
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(frame))) => {
                let finished = frame.is_trailers() || self.inner.is_end_stream();
                if finished && let Some(completion) = self.completion.take() {
                    completion.complete();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(completion) = self.completion.take() {
                    completion.fail();
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.completion.is_none() && self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

struct HttpMetricsCompletion {
    metrics: Metrics,
    method: HttpMethod,
    route: HttpRoute,
    started: Instant,
    outcome: Option<Outcome>,
    completed: bool,
}

impl HttpMetricsCompletion {
    fn complete(mut self) {
        self.metrics.finish_http(
            self.method.clone(),
            self.route.clone(),
            self.outcome.clone().unwrap_or(Outcome::Cancelled),
            self.started.elapsed(),
        );
        self.completed = true;
    }

    fn fail(mut self) {
        self.outcome = Some(Outcome::ServerError);
        self.metrics.finish_http(
            self.method.clone(),
            self.route.clone(),
            Outcome::ServerError,
            self.started.elapsed(),
        );
        self.completed = true;
    }
}

impl Drop for HttpMetricsCompletion {
    fn drop(&mut self) {
        if !self.completed {
            self.metrics.finish_http(
                self.method.clone(),
                self.route.clone(),
                Outcome::Cancelled,
                self.started.elapsed(),
            );
        }
    }
}

fn http_method(method: &axum::http::Method) -> HttpMethod {
    match *method {
        axum::http::Method::GET => HttpMethod::Get,
        axum::http::Method::POST => HttpMethod::Post,
        axum::http::Method::DELETE => HttpMethod::Delete,
        axum::http::Method::OPTIONS => HttpMethod::Options,
        _ => HttpMethod::Other,
    }
}

fn http_route(path: &str) -> HttpRoute {
    match path {
        "/" => HttpRoute::Root,
        "/healthz" => HttpRoute::Healthz,
        "/readyz" => HttpRoute::Readyz,
        "/metrics" => HttpRoute::Metrics,
        "/v1/models" => HttpRoute::Models,
        value if value.starts_with("/v1/models/") => HttpRoute::Model,
        "/api/models/status" => HttpRoute::ModelStatus,
        "/api/models/load" => HttpRoute::ModelLoad,
        "/api/models/unload" => HttpRoute::ModelUnload,
        "/api/activity" => HttpRoute::Activity,
        "/api/activity/events" => HttpRoute::ActivityEvents,
        "/v1/audio/speech" => HttpRoute::Speech,
        "/v1/audio/transcriptions" => HttpRoute::Transcriptions,
        "/v1/audio/transcriptions/stream" => HttpRoute::TranscriptionsStream,
        "/v1/ocr" => HttpRoute::Ocr,
        "/v1/pdf/images" => HttpRoute::PdfImages,
        "/v1/chat/completions" => HttpRoute::ChatCompletions,
        "/v1/chat/completions/control" => HttpRoute::ChatCompletionsControl,
        "/v1/completions" => HttpRoute::Completions,
        "/v1/responses" => HttpRoute::Responses,
        "/v1/responses/input_tokens" => HttpRoute::ResponsesInputTokens,
        "/v1/embeddings" => HttpRoute::Embeddings,
        "/v1/stream" => HttpRoute::Stream,
        "/v1/streams/lookup" => HttpRoute::StreamsLookup,
        "/openapi/v1.json" => HttpRoute::Openapi,
        value if value.starts_with("/docs") => HttpRoute::Docs,
        value if value.starts_with("/ui") => HttpRoute::Ui,
        _ => HttpRoute::Unmatched,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::http_audio::parse_timestamp_granularities;
    use crate::api::http_shared::multipart_file_suffix;
    use crate::api::openai::ApiError;
    use crate::infrastructure::orchion::AppState;
    use crate::settings::ServerConfig;
    use crate::settings::{
        ChatTemplateConfig, LlmGenerationConfig, LlmModelDeployment, LlmRuntimeConfig,
        ModelDeployment, OcrModelDeployment,
    };
    use axum::body::Body;
    use axum::extract::connect_info::ConnectInfo;
    use axum::http::header::{
        ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
        ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, AUTHORIZATION, CONTENT_TYPE,
        ORIGIN, USER_AGENT,
    };
    use axum::http::{Method, Request, StatusCode};
    use axum::response::Response;
    use futures_util::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use http_body_util::StreamBody;
    use orchion::llm_test_support::{
        scripted_context_limit_llm_engine, scripted_embedding_llm_engine, scripted_llm_engine,
        scripted_reasoning_llm_engine,
    };
    use orchion::{
        AsrModel, GenerationEvent, GenerationFinishReason, LlmTimings, LlmUsage, ModelCategory,
        ModelDownloader, ModelId, ModelSpec, ModelUrl, ModelUrlSource, OcrModel, OcrModelKind,
        TtsModel,
    };
    use serde_json::{Value, json};
    use std::net::SocketAddr;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    use tower::ServiceExt;

    #[test]
    fn http_metrics_classify_timeout_and_capacity_statuses() {
        assert_eq!(http_outcome(StatusCode::REQUEST_TIMEOUT), Outcome::Timeout);
        assert_eq!(
            http_outcome(StatusCode::TOO_MANY_REQUESTS),
            Outcome::ResourceExhausted
        );
        assert_eq!(http_outcome(StatusCode::BAD_REQUEST), Outcome::ClientError);
        assert_eq!(
            http_outcome(StatusCode::INTERNAL_SERVER_ERROR),
            Outcome::ServerError
        );
    }

    fn test_metrics_body(
        metrics: &Metrics,
        frames: impl futures_util::Stream<Item = Result<Frame<Bytes>, std::io::Error>>
        + Send
        + Sync
        + 'static,
    ) -> MetricsBody {
        metrics.begin_http();
        MetricsBody {
            inner: Body::new(StreamBody::new(frames)),
            completion: Some(HttpMetricsCompletion {
                metrics: metrics.clone(),
                method: HttpMethod::Get,
                route: HttpRoute::Healthz,
                started: Instant::now(),
                outcome: Some(Outcome::Success),
                completed: false,
            }),
        }
    }

    fn http_metric(metrics: &Metrics, outcome: Outcome) -> u64 {
        let outcome = match outcome {
            Outcome::Success => "success",
            Outcome::ClientError => "client_error",
            Outcome::ServerError => "server_error",
            Outcome::Cancelled => "cancelled",
            Outcome::Timeout => "timeout",
            Outcome::ResourceExhausted => "resource_exhausted",
        };
        metrics
            .encode()
            .unwrap()
            .lines()
            .find(|line| {
                line.starts_with("orchion_http_requests_total{")
                    && line.contains("method=\"GET\"")
                    && line.contains("route=\"/healthz\"")
                    && line.contains(&format!("outcome=\"{outcome}\""))
            })
            .and_then(|line| line.rsplit_once(' '))
            .and_then(|(_, value)| value.parse().ok())
            .unwrap_or(0)
    }

    fn http_active(metrics: &Metrics) -> i64 {
        metrics
            .encode()
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("orchion_http_active ")
                    .and_then(|value| value.parse().ok())
            })
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn metrics_body_waits_for_trailers_before_success() {
        let metrics = Metrics::new();
        let mut trailers = HeaderMap::new();
        trailers.insert("x-complete", HeaderValue::from_static("true"));
        let mut body = test_metrics_body(
            &metrics,
            futures_util::stream::iter([
                Ok(Frame::data(Bytes::from_static(b"data"))),
                Ok(Frame::trailers(trailers)),
            ]),
        );
        assert!(body.frame().await.unwrap().unwrap().is_data());
        assert_eq!(http_active(&metrics), 1);
        assert_eq!(http_metric(&metrics, Outcome::Success), 0);
        assert!(body.frame().await.unwrap().unwrap().is_trailers());
        assert_eq!(http_active(&metrics), 0);
        assert_eq!(http_metric(&metrics, Outcome::Success), 1);
    }

    #[tokio::test]
    async fn metrics_body_records_data_then_error_as_server_error() {
        let metrics = Metrics::new();
        let mut body = test_metrics_body(
            &metrics,
            futures_util::stream::iter([
                Ok(Frame::data(Bytes::from_static(b"data"))),
                Err(std::io::Error::other("body failed")),
            ]),
        );
        assert!(body.frame().await.unwrap().unwrap().is_data());
        assert!(body.frame().await.unwrap().is_err());
        assert_eq!(http_active(&metrics), 0);
        assert_eq!(http_metric(&metrics, Outcome::ServerError), 1);
        assert_eq!(http_metric(&metrics, Outcome::Cancelled), 0);
    }

    #[test]
    fn dropping_metrics_body_before_poll_records_cancellation() {
        let metrics = Metrics::new();
        let body = test_metrics_body(
            &metrics,
            futures_util::stream::pending::<Result<Frame<Bytes>, std::io::Error>>(),
        );
        drop(body);
        assert_eq!(http_active(&metrics), 0);
        assert_eq!(http_metric(&metrics, Outcome::Cancelled), 1);
    }

    #[test]
    fn parse_timestamp_granularities_accepts_segment() {
        let values = vec!["segment".to_string()];

        assert!(parse_timestamp_granularities(&values).unwrap());
    }

    #[test]
    fn parse_timestamp_granularities_rejects_word() {
        let values = vec!["segment".to_string(), "word".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_timestamp_granularity")
        );
    }

    #[test]
    fn parse_timestamp_granularities_rejects_unknown_value() {
        let values = vec!["sentence".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("timestamp_granularities")
        );
    }

    #[test]
    fn multipart_file_suffix_uses_supported_mime_type() {
        assert_eq!(multipart_file_suffix(Some("image/png")), ".png");
        assert_eq!(multipart_file_suffix(Some("image/jpeg")), ".jpg");
        assert_eq!(multipart_file_suffix(Some("application/pdf")), ".pdf");
        assert_eq!(multipart_file_suffix(Some("video/mp4")), ".mp4");
        assert_eq!(multipart_file_suffix(Some("text/plain")), "");
        assert_eq!(multipart_file_suffix(None), "");
    }

    #[tokio::test]
    async fn cors_allows_every_origin_by_default() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
    }

    #[tokio::test]
    async fn readiness_is_public_no_store_and_turns_false_on_shutdown() {
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("ready-secret-sentinel".to_string());
        });
        let shutdown = ServerShutdown::new();
        let app = router_with_ui_routes_and_shutdown(state, Router::new(), shutdown.clone());

        let ready = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        assert_eq!(ready.headers()[CACHE_CONTROL], "no-store");
        assert_eq!(json_body(ready).await, json!({"status":"ready"}));

        shutdown.trigger();
        let unavailable = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(unavailable).await;
        assert_eq!(body["status"], "not_ready");
        assert_eq!(body["reasons"][0]["code"], "shutdown");
        assert!(!body.to_string().contains("ready-secret-sentinel"));
    }

    #[tokio::test]
    async fn readiness_latches_only_required_default_load_failures() {
        let optional = AsrModel::parse("alibaba/qwen3-asr-1.7b").unwrap();
        let default = AsrModel::parse("alibaba/qwen3-asr-0.6b").unwrap();
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config
                .services
                .asr
                .models
                .push(ModelDeployment::from_asr_runtime(optional.clone()));
        });

        let Err(optional_error) = state.asr(optional).await else {
            panic!("optional fixture must fail runtime loading");
        };
        let optional_ready = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(optional_ready.status(), StatusCode::OK);

        let Err(default_error) = state.asr(default).await else {
            panic!("default fixture must fail runtime loading");
        };
        let required_failed = router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(required_failed.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_body(required_failed).await;
        assert_eq!(body["reasons"].as_array().unwrap().len(), 1);
        assert_eq!(body["reasons"][0]["code"], "required_model_load_failed");
        assert!(body["reasons"][0].get("service").is_none());
        assert!(body["reasons"][0].get("model").is_none());
        let serialized = body.to_string();
        assert!(!serialized.contains("alibaba/qwen3-asr-0.6b"));
        assert!(!serialized.contains(&optional_error.to_string()));
        assert!(!serialized.contains(&default_error.to_string()));
    }

    #[tokio::test]
    async fn metrics_require_global_auth_and_expose_bounded_openmetrics() {
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("metrics-secret-sentinel".to_string());
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        denied.into_body().collect().await.unwrap();

        let unmatched = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/private/path-sentinel?prompt=prompt-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unmatched.status(), StatusCode::NOT_FOUND);
        unmatched.into_body().collect().await.unwrap();

        let scrape = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(AUTHORIZATION, "Bearer metrics-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(scrape.status(), StatusCode::OK);
        assert_eq!(scrape.headers()[CONTENT_TYPE], OPENMETRICS_CONTENT_TYPE);
        let body = String::from_utf8(
            scrape
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("# HELP orchion_http_requests"));
        assert!(body.contains("# TYPE orchion_http_requests counter"));
        assert!(body.ends_with("# EOF\n"));
        assert!(body.contains("route=\"unmatched\""));
        assert!(!body.contains("path-sentinel"));
        assert!(!body.contains("prompt-secret-sentinel"));
        assert!(!body.contains("metrics-secret-sentinel"));

        let residency = body
            .lines()
            .filter(|line| {
                line.starts_with("orchion_model_residency{")
                    && line.contains("model=\"alibaba/qwen3-asr-0.6b\"")
            })
            .collect::<Vec<_>>();
        assert_eq!(residency.len(), 4);
        assert_eq!(
            residency.iter().filter(|line| line.ends_with(" 1")).count(),
            1
        );
    }

    #[tokio::test]
    async fn cors_only_echoes_configured_origins() {
        let state = test_state_with_config(false, false, |config| {
            config.server.cors_allowed_origins = vec!["https://app.example.com".to_string()];
        });
        let allowed = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let denied = router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://other.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            allowed.headers()[ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://app.example.com"
        );
        assert!(!denied.headers().contains_key(ACCESS_CONTROL_ALLOW_ORIGIN));
    }

    #[tokio::test]
    async fn cors_handles_authorization_preflight_requests() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .header(ACCESS_CONTROL_REQUEST_METHOD, "GET")
                    .header(ACCESS_CONTROL_REQUEST_HEADERS, "authorization")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        assert!(
            response
                .headers()
                .contains_key(ACCESS_CONTROL_ALLOW_METHODS)
        );
        assert_eq!(
            response.headers()[ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization"
        );
    }

    #[tokio::test]
    async fn activity_excludes_model_and_control_requests() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        for route in ["/v1/models", "/api/models/status"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(route).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        for route in ["/api/models/load", "/api/models/unload"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri(route)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(r#"{"model":"Private/Control-Sentinel"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["summary"]["retained"], 0);
        assert!(activity["history"].as_array().unwrap().is_empty());
        assert!(!activity.to_string().contains("Private/Control-Sentinel"));
    }

    #[tokio::test]
    async fn activity_excludes_unmatched_v1_routes() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/not-a-route?secret=sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["summary"]["retained"], 0);
        assert!(!activity.to_string().contains("sentinel"));
    }

    #[tokio::test]
    async fn activity_records_auth_errors_without_leaking_credentials() {
        let boundary = "activity-auth";
        let body = multipart_body(
            boundary,
            &[("model", "alibaba/qwen3-asr-0.6b")],
            Some(("file", "audio.wav", b"audio")),
        );
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("activity-secret-sentinel".to_string());
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(AUTHORIZATION, "Bearer wrong-secret-sentinel")
                    .header(USER_AGENT, "orchion-test-agent/1.0")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .extension(ConnectInfo(
                        "203.0.113.7:4242".parse::<SocketAddr>().unwrap(),
                    ))
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let live = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .header(AUTHORIZATION, "Bearer activity-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let live = json_body(live).await;
        assert_eq!(live["active"][0]["address"], "203.0.113.7");
        assert_eq!(live["active"][0]["user_agent"], "orchion-test-agent/1.0");
        denied.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .header(AUTHORIZATION, "Bearer activity-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;
        let serialized = activity.to_string();

        assert_eq!(activity["history"][0]["http_status"], 401);
        assert_eq!(activity["history"][0]["error_code"], "invalid_api_key");
        assert!(activity["history"][0].get("address").is_none());
        assert!(activity["history"][0].get("user_agent").is_none());
        assert!(!serialized.contains("activity-secret-sentinel"));
        assert!(!serialized.contains("wrong-secret-sentinel"));
    }

    #[tokio::test]
    async fn activity_sanitizes_internal_error_details_for_server_errors() {
        let app = router_with_ui_routes(
            test_state(false, false),
            Router::new().route(
                "/v1/audio/speech",
                post(|| async {
                    Err::<(), ApiError>(ApiError::internal(
                        "model load failed at /tmp/private/model.gguf: sentinel",
                    ))
                }),
            ),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/speech")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let response_body = json_body(response).await;
        assert_eq!(response_body["error"]["message"], "internal server error");
        assert!(
            !response_body
                .to_string()
                .contains("/tmp/private/model.gguf")
        );

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["history"][0]["error_code"], "internal_error");
        assert_eq!(
            activity["history"][0]["error_message"],
            "internal server error"
        );
        assert!(!activity.to_string().contains("/tmp/private/model.gguf"));
        assert!(!activity.to_string().contains("sentinel"));
    }

    #[tokio::test]
    async fn activity_query_auth_precedes_validation() {
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("activity-query-secret".to_string());
        });
        let app = router_with_ui_routes(state, Router::new());
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/activity?limit=invalid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let invalid = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity?limit=invalid")
                    .header(AUTHORIZATION, "Bearer activity-query-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(invalid).await["error"]["code"],
            "invalid_activity_query"
        );
    }

    #[tokio::test]
    async fn activity_does_not_record_unavailable_client_model_values() {
        let boundary = "activity-model-privacy";
        let body = multipart_body(
            boundary,
            &[("model", "Private/Client-Sentinel")],
            Some(("file", "audio.wav", b"audio")),
        );
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert!(activity["history"][0].get("model").is_none());
        assert!(!activity.to_string().contains("Private/Client-Sentinel"));
    }

    #[tokio::test]
    async fn llm_unknown_model_is_not_recorded_before_allowlist_validation() {
        let root = tempfile::tempdir().unwrap();
        let gguf = root.path().join("main.gguf");
        let mmproj = root.path().join("mmproj.gguf");
        std::fs::write(&gguf, "test gguf fixture").unwrap();
        std::fs::write(&mmproj, "test mmproj fixture").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        let models_dir = config.models.dir.clone();
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let id = ModelId::parse("qwen/test").unwrap();
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: None,
            model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
            mmproj_model: Some(
                orchion::ModelUrl::parse(&format!("file://{}", mmproj.display())).unwrap(),
            ),
            runtime: LlmRuntimeConfig::default(),
            chat_template: ChatTemplateConfig::default(),
            prompt_cache: crate::settings::PromptCacheConfig::default(),
            generation: LlmGenerationConfig::default(),
            kind: crate::settings::LlmDeploymentKind::Generation,
            vision: crate::settings::LlmVisionLimits::default(),
        }];
        let state = AppState::load(config).await.unwrap();
        let manifest = serde_json::from_str::<Value>(
            &std::fs::read_to_string(find_deployment_manifest(&models_dir).unwrap()).unwrap(),
        )
        .unwrap();
        let manifest_text = manifest.to_string();
        assert!(manifest_text.contains("llm_model"));
        assert!(manifest_text.contains("llm_mmproj"));
        assert!(manifest_text.contains("sha256"));
        assert!(manifest_text.contains("size"));
        assert!(!manifest_text.contains("runtime="));
        assert!(!manifest_text.contains("generation="));
        assert!(
            !manifest["source_intent"]
                .as_str()
                .unwrap()
                .contains("qwen/test")
        );
        assert!(manifest_text.contains(&gguf.to_string_lossy().to_string()));
        assert!(manifest_text.contains(&mmproj.to_string_lossy().to_string()));
        let lock = std::fs::read_to_string(models_dir.join("orchion-models.lock")).unwrap();
        assert!(lock.contains("llm_model"));
        assert!(lock.contains("llm_mmproj"));
        assert!(!models_dir.join(".orchion/blobs").exists());
        let app = router_with_ui_routes(state, Router::new());
        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let listed = json_body(listed).await;
        assert_eq!(listed["data"][0]["id"], "qwen/test");
        assert_eq!(listed["data"][0]["type"], "llm");
        assert_eq!(
            listed["data"][0]["capabilities"],
            serde_json::json!([
                "llm_chat",
                "llm_responses",
                "llm_streaming",
                "llm_completions",
                "llm_input_tokens",
                "llm_tools",
                "llm_parallel_tools",
                "llm_json_object",
                "llm_json_schema",
                "llm_logprobs",
                "llm_logit_bias",
                "llm_vision",
                "llm_resumable_streaming"
            ])
        );
        assert_eq!(listed["data"][0]["capability_details"]["max_choices"], 1);
        assert!(!listed.to_string().contains("main.gguf"));
        let response = app.clone().oneshot(Request::builder()
            .method(Method::POST).uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"Private/Prompt-Sentinel","messages":[{"role":"user","content":"secret prompt"}]}"#)).unwrap())
            .await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();
        let activity = wait_for_activity_history(app).await;
        assert!(activity["history"][0].get("model").is_none());
        let serialized = activity.to_string();
        assert!(!serialized.contains("Private/Prompt-Sentinel"));
        assert!(!serialized.contains("secret prompt"));
    }

    #[tokio::test]
    async fn configured_vision_limit_rejects_before_cold_model_load() {
        let root = tempfile::tempdir().unwrap();
        let gguf = root.path().join("main.gguf");
        let mmproj = root.path().join("mmproj.gguf");
        std::fs::write(&gguf, "scripted generation fixture").unwrap();
        std::fs::write(&mmproj, "scripted projector fixture").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let id = ModelId::parse("qwen/vision").unwrap();
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: None,
            model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
            mmproj_model: Some(
                orchion::ModelUrl::parse(&format!("file://{}", mmproj.display())).unwrap(),
            ),
            runtime: LlmRuntimeConfig::default(),
            chat_template: ChatTemplateConfig::default(),
            prompt_cache: crate::settings::PromptCacheConfig::default(),
            generation: LlmGenerationConfig::default(),
            kind: crate::settings::LlmDeploymentKind::Generation,
            vision: crate::settings::LlmVisionLimits {
                max_bytes_per_image: 1024 * 1024,
                max_total_bytes: 2 * 1024 * 1024,
                ..crate::settings::LlmVisionLimits::default()
            },
        }];
        let (engine, control) = scripted_llm_engine(Vec::new());
        let state = AppState::load_with_test_llm_engine(config, engine)
            .await
            .unwrap();
        let app = router_with_ui_routes(state.clone(), Router::new());
        let payload = "A".repeat(1_398_104);
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model":"qwen/vision",
                            "messages":[{"role":"user","content":[{
                                "type":"image_url",
                                "image_url":{"url":format!("data:image/png;base64,{payload}")}
                            }]}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "messages.content.image_url.url");
        assert!(!control.has_started());
        tokio::time::timeout(std::time::Duration::from_secs(1), state.shutdown())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn embeddings_endpoint_normalizes_reduces_encodes_and_advertises_only_embeddings() {
        let root = tempfile::tempdir().unwrap();
        let gguf = root.path().join("embedding.gguf");
        std::fs::write(&gguf, "scripted embedding fixture").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let id = ModelId::parse("qwen/embed").unwrap();
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: Some("Embedding test".to_string()),
            model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
            mmproj_model: None,
            runtime: LlmRuntimeConfig::default(),
            chat_template: ChatTemplateConfig::default(),
            prompt_cache: crate::settings::PromptCacheConfig::default(),
            generation: LlmGenerationConfig::default(),
            kind: crate::settings::LlmDeploymentKind::Embeddings(
                crate::settings::LlmEmbeddingConfig {
                    pooling: crate::settings::LlmEmbeddingPooling::Last,
                    min_dimensions: 1,
                    max_input_tokens: 8192,
                },
            ),
            vision: crate::settings::LlmVisionLimits::default(),
        }];
        let (engine, control) =
            scripted_embedding_llm_engine(vec![vec![3.0, 4.0, 100.0], vec![0.0, 0.0, 1.0]], 5);
        let state = AppState::load_with_test_llm_engine(config, engine)
            .await
            .unwrap();
        let app = router_with_ui_routes(state.clone(), Router::new());
        let models = json_body(
            app.clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            models["data"][0]["capabilities"],
            serde_json::json!(["llm_embeddings"])
        );

        let activity_app = app.clone();
        let response = tokio::spawn(
            app.oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/embeddings")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model":"qwen/embed",
                            "input":["first","second"],
                            "dimensions":2,
                            "encoding_format":"float",
                            "user":"accepted-and-ignored"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            ),
        );
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let preparation = control.clone();
        tokio::task::spawn_blocking(move || preparation.wait_preparation_started())
            .await
            .unwrap();
        control.release_cleanup();
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(body["data"][1]["index"], 1);
        assert_eq!(body["data"][0]["embedding"], serde_json::json!([0.6, 0.8]));
        assert_eq!(body["data"][1]["embedding"], serde_json::json!([0.0, 0.0]));
        assert_eq!(
            body["usage"],
            serde_json::json!({"prompt_tokens":5,"total_tokens":5})
        );
        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["operation"], "embeddings");
        assert_eq!(activity["history"][0]["prompt_tokens"], 5);
        assert_eq!(activity["history"][0]["completion_tokens"], 0);

        let base64_response = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/embeddings")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model":"qwen/embed",
                            "input":[[1,2],[3]],
                            "dimensions":2,
                            "encoding_format":"base64"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let base64_body = json_body(base64_response).await;
        assert_eq!(base64_body["data"][0]["embedding"], "mpkZP83MTD8=");
        let invalid_dimensions = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/embeddings")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model":"qwen/embed",
                            "input":["first","second"],
                            "dimensions":4
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_dimensions.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(invalid_dimensions).await["error"]["param"],
            "dimensions"
        );
        state.shutdown().await;
    }

    #[tokio::test]
    async fn model_retrieval_uses_public_catalog_and_uniform_not_found() {
        let state = test_state_with_config(true, false, |config| {
            config.auth.api_key = Some("secret".to_string());
        });
        let app = router_with_ui_routes(state, Router::new());
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models/paddlepaddle%2Fpp-ocrv6-tiny")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let known = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models/paddlepaddle%2Fpp-ocrv6-tiny")
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(known.status(), StatusCode::OK);
        assert_eq!(json_body(known).await["object"], "model");
        let missing = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/Private%2FUnknown-Sentinel")
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        let missing = json_body(missing).await;
        assert_eq!(missing["error"]["code"], "model_not_found");
        assert!(!missing.to_string().contains("Private/Unknown-Sentinel"));
    }

    #[tokio::test]
    #[allow(
        clippy::too_many_lines,
        reason = "covers shared generation runtime and both ordinary and resumable completion transports"
    )]
    async fn legacy_completions_and_response_token_count_share_generation_runtime() {
        let root = tempfile::tempdir().unwrap();
        let gguf = root.path().join("main.gguf");
        std::fs::write(&gguf, "scripted generation fixture").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let id = ModelId::parse("qwen/test").unwrap();
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: None,
            model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
            mmproj_model: None,
            runtime: LlmRuntimeConfig::default(),
            chat_template: ChatTemplateConfig::default(),
            prompt_cache: crate::settings::PromptCacheConfig::default(),
            generation: LlmGenerationConfig::default(),
            kind: crate::settings::LlmDeploymentKind::Generation,
            vision: crate::settings::LlmVisionLimits::default(),
        }];
        let usage = LlmUsage {
            prompt_tokens: 2,
            completion_tokens: 1,
            reasoning_tokens: 0,
            total_tokens: 3,
            queue_time_ms: None,
            eval_time_ms: None,
            timings: LlmTimings::default(),
        };
        let (engine, control) = scripted_llm_engine(vec![
            GenerationEvent::ContentDelta("hello".to_string()),
            GenerationEvent::Finished {
                reason: GenerationFinishReason::Stop,
                usage,
            },
        ]);
        let state = AppState::load_with_test_llm_engine(config, engine)
            .await
            .unwrap();
        let app = router_with_ui_routes(state.clone(), Router::new());
        let completion = tokio::spawn(
            app.clone().oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"model":"qwen/test","prompt":"raw prompt","max_tokens":8})
                            .to_string(),
                    ))
                    .unwrap(),
            ),
        );
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !control.has_started() && !completion.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completion must start or return before the admission deadline");
        if completion.is_finished() {
            let response = completion.await.unwrap().unwrap();
            let status = response.status();
            let body = json_body(response).await;
            panic!("completion ended before native start: {status} {body}");
        }
        control.release_ready();
        let prepared = control.clone();
        tokio::task::spawn_blocking(move || prepared.wait_preparation_started())
            .await
            .unwrap();
        control.release_cleanup();
        let completion = completion.await.unwrap().unwrap();
        assert_eq!(completion.status(), StatusCode::OK);
        let completion = json_body(completion).await;
        assert_eq!(completion["object"], "text_completion");
        assert_eq!(completion["choices"][0]["text"], "hello");
        assert_eq!(completion["usage"]["total_tokens"], 3);

        let streamed = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            app.clone().oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"model":"qwen/test","prompt":"raw prompt","stream":true})
                            .to_string(),
                    ))
                    .unwrap(),
            ),
        )
        .await
        .expect("normal completion stream dispatch must finish")
        .unwrap();
        assert_eq!(streamed.headers()[CONTENT_TYPE], "text/event-stream");
        let stream_body = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            streamed.into_body().collect(),
        )
        .await
        .expect("normal completion stream body must reach terminal")
        .unwrap()
        .to_bytes();
        let stream_body = String::from_utf8(stream_body.to_vec()).unwrap();
        assert!(stream_body.contains("\"object\":\"text_completion\""));
        assert!(stream_body.contains("\"total_tokens\":3"));
        assert!(stream_body.contains("id: 1\n"));
        assert!(stream_body.contains("data: [DONE]\n\n"));

        let resumable = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            app.clone().oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .header("x-orchion-resumable", "true")
                    .body(Body::from(
                        json!({"model":"qwen/test","prompt":"raw prompt","stream":true})
                            .to_string(),
                    ))
                    .unwrap(),
            ),
        )
        .await
        .expect("resumable completion stream dispatch must finish")
        .unwrap();
        assert_eq!(resumable.status(), StatusCode::OK);
        let stream_id = resumable.headers()["x-orchion-stream-id"]
            .to_str()
            .unwrap()
            .to_string();
        assert!(stream_id.starts_with("strm_"));
        assert_eq!(resumable.headers()["x-orchion-stream-ttl-seconds"], "300");
        let initial = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            resumable.into_body().collect(),
        )
        .await
        .expect("resumable completion stream body must reach terminal")
        .unwrap()
        .to_bytes();
        assert!(initial.windows(5).any(|value| value == b"id: 1"));

        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/stream?stream_id={stream_id}&follow=false"))
                    .header("last-event-id", "1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay = replay.into_body().collect().await.unwrap().to_bytes();
        assert!(!replay.windows(5).any(|value| value == b"id: 1"));
        assert!(replay.windows(5).any(|value| value == b"id: 2"));

        let invalid = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .header("x-orchion-resumable", "false")
                    .body(Body::from(
                        json!({"model":"qwen/test","prompt":"raw prompt","stream":true})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            json_body(invalid).await["error"]["code"],
            "invalid_resumable_stream"
        );

        let inference = state.acquire_inference().await;
        let count = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            app.oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/responses/input_tokens")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model":"qwen/test",
                            "instructions":"be concise",
                            "input":"hello world"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            ),
        )
        .await
        .expect("response token counting must not wait for global generation capacity")
        .unwrap();
        assert_eq!(count.status(), StatusCode::OK);
        assert_eq!(
            json_body(count).await,
            json!({"object":"response.input_tokens","input_tokens":5})
        );
        drop(inference);
        tokio::time::timeout(std::time::Duration::from_secs(2), state.shutdown())
            .await
            .expect("completed normal and resumable streams must release the model runtime");
    }

    #[tokio::test]
    async fn chat_reasoning_control_normal_and_resumable_lifecycle() {
        for resumable in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let gguf = root.path().join("main.gguf");
            std::fs::write(&gguf, "test gguf fixture").unwrap();
            let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
            config.services.asr.enabled = false;
            config.services.tts.enabled = false;
            let id = ModelId::parse("qwen/test").unwrap();
            config.services.llm.enabled = true;
            config.services.llm.default_model = Some(id.clone());
            let mut chat_template = ChatTemplateConfig::default();
            chat_template.enable_thinking = true;
            config.services.llm.models = vec![LlmModelDeployment {
                id,
                name: None,
                model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
                mmproj_model: None,
                runtime: LlmRuntimeConfig::default(),
                chat_template,
                prompt_cache: crate::settings::PromptCacheConfig::default(),
                generation: LlmGenerationConfig::default(),
                kind: crate::settings::LlmDeploymentKind::Generation,
                vision: crate::settings::LlmVisionLimits::default(),
            }];
            let usage = LlmUsage {
                prompt_tokens: 2,
                completion_tokens: 1,
                reasoning_tokens: 1,
                total_tokens: 3,
                queue_time_ms: None,
                eval_time_ms: None,
                timings: LlmTimings::default(),
            };
            let (engine, control) = scripted_reasoning_llm_engine("thinking", usage);
            let state = AppState::load_with_test_llm_engine(config, engine)
                .await
                .unwrap();
            let shutdown = ServerShutdown::new();
            let app =
                router_with_ui_routes_and_shutdown(state.clone(), Router::new(), shutdown.clone());
            let mut request = Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json");
            if resumable {
                request = request.header("x-orchion-resumable", "true");
            }
            let start = tokio::spawn(
                app.clone().oneshot(
                    request
                        .body(Body::from(
                            json!({
                                "model":"qwen/test",
                                "messages":[{"role":"user","content":"hello"}],
                                "stream":true,
                                "reasoning_control":true
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                ),
            );
            let started = control.clone();
            tokio::task::spawn_blocking(move || started.wait_started())
                .await
                .unwrap();
            control.release_ready();
            let mut response = start.await.unwrap().unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let completion_id = response.headers()["x-orchion-completion-id"]
                .to_str()
                .unwrap()
                .to_string();
            let stream_id = response
                .headers()
                .get("x-orchion-stream-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            assert!(crate::api::http_llm::valid_chat_completion_id(
                &completion_id
            ));
            if resumable {
                let first = response
                    .body_mut()
                    .frame()
                    .await
                    .unwrap()
                    .unwrap()
                    .into_data()
                    .unwrap();
                assert!(
                    String::from_utf8_lossy(&first)
                        .contains(&format!("\"id\":\"{completion_id}\""))
                );
            }

            let invoke = |model: Option<&str>| {
                let mut body = json!({"id":completion_id.clone(),"action":"reasoning_end"});
                if let Some(model) = model {
                    body["model"] = json!(model);
                }
                app.clone().oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/chat/completions/control")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
            };
            assert_eq!(
                json_body(invoke(None).await.unwrap()).await["success"],
                true
            );
            let mismatch = json_body(invoke(Some("qwen/other")).await.unwrap()).await;
            assert_eq!(mismatch["success"], false);
            let duplicate = json_body(invoke(None).await.unwrap()).await;
            assert_eq!(duplicate["success"], false);
            assert_eq!(mismatch["message"], duplicate["message"]);
            if let Some(stream_id) = &stream_id {
                let deleted = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(Method::DELETE)
                            .uri(format!("/v1/stream?stream_id={stream_id}"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
                let after_delete = json_body(invoke(None).await.unwrap()).await;
                assert_eq!(after_delete["success"], false);
                assert_eq!(after_delete["message"], mismatch["message"]);
            } else {
                shutdown.trigger();
                let after_shutdown = json_body(invoke(None).await.unwrap()).await;
                assert_eq!(after_shutdown["success"], false);
                assert_eq!(after_shutdown["message"], mismatch["message"]);
            }

            control.release_cleanup();
            let body = String::from_utf8(
                response
                    .into_body()
                    .collect()
                    .await
                    .unwrap()
                    .to_bytes()
                    .to_vec(),
            )
            .unwrap();
            if !resumable {
                assert!(body.contains(&format!("\"id\":\"{completion_id}\"")));
            }
            let terminal = json_body(invoke(None).await.unwrap()).await;
            assert_eq!(terminal["success"], false);
            assert_eq!(terminal["message"], mismatch["message"]);
            let invalid = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/chat/completions/control")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(json!({"id":"bad","action":"stop"}).to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
            let invalid_action = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/chat/completions/control")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            json!({"id":completion_id,"action":"stop"}).to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(invalid_action.status(), StatusCode::BAD_REQUEST);
            assert_eq!(json_body(invalid_action).await["error"]["param"], "action");
            let metrics = state.metrics().encode().unwrap();
            for (outcome, count) in [
                ("success", 1),
                ("model_mismatch", 1),
                ("not_reasoning", 1),
                ("not_found", 2),
            ] {
                assert!(metrics.contains(&format!(
                    "orchion_llm_reasoning_controls_total{{outcome=\"{outcome}\"}} {count}"
                )));
            }
            state.shutdown().await;
        }
    }

    #[tokio::test]
    async fn llm_context_limit_is_json_4xx_before_sync_or_stream_headers() {
        for stream in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let gguf = root.path().join("main.gguf");
            std::fs::write(&gguf, "test gguf fixture").unwrap();
            let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
            config.services.asr.enabled = false;
            config.services.tts.enabled = false;
            let id = ModelId::parse("qwen/test").unwrap();
            config.services.llm.enabled = true;
            config.services.llm.default_model = Some(id.clone());
            config.services.llm.models = vec![LlmModelDeployment {
                id,
                name: None,
                model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
                mmproj_model: None,
                runtime: LlmRuntimeConfig::default(),
                chat_template: ChatTemplateConfig::default(),
                prompt_cache: crate::settings::PromptCacheConfig::default(),
                generation: LlmGenerationConfig::default(),
                kind: crate::settings::LlmDeploymentKind::Generation,
                vision: crate::settings::LlmVisionLimits::default(),
            }];
            let state = AppState::load_with_test_llm_engine(
                config,
                scripted_context_limit_llm_engine(100, 50, 128),
            )
            .await
            .unwrap();
            let response = router_with_ui_routes(state.clone(), Router::new())
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/chat/completions")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "model":"qwen/test",
                                "messages":[{"role":"user","content":"long prompt"}],
                                "stream":stream
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_ne!(response.headers()[CONTENT_TYPE], "text/event-stream");
            let body = json_body(response).await;
            assert_eq!(body["error"]["code"], "context_length_exceeded");
            assert_eq!(body["error"]["param"], "max_completion_tokens");

            let response = router_with_ui_routes(state.clone(), Router::new())
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/responses")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "model":"qwen/test",
                                "input":"long prompt",
                                "store":false,
                                "stream":stream
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_ne!(response.headers()[CONTENT_TYPE], "text/event-stream");
            let body = json_body(response).await;
            assert_eq!(body["error"]["code"], "context_length_exceeded");
            assert_eq!(body["error"]["param"], "max_output_tokens");
            state.shutdown().await;
        }
    }

    #[tokio::test]
    async fn disabled_activity_returns_an_empty_disabled_page() {
        let state = test_state_with_config(false, false, |config| {
            config.activity.enabled = false;
        });
        let app = router_with_ui_routes(state, Router::new());
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["enabled"], false);
        assert_eq!(activity["summary"]["retained"], 0);
        assert!(activity["history"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn activity_sse_begins_with_a_snapshot() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let mut body = response.into_body();
        let frame = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let data = frame.into_data().unwrap();
        let text = String::from_utf8(data.to_vec()).unwrap();

        assert!(text.contains("event: snapshot"));
        assert!(text.contains("\"active\":[]"));
    }

    #[tokio::test]
    async fn activity_sse_closes_on_server_shutdown() {
        let shutdown = ServerShutdown::new();
        let app = router_with_ui_routes_and_shutdown(
            test_state(false, false),
            Router::new(),
            shutdown.clone(),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body();
        let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap();
        assert!(first.is_some());

        shutdown.trigger();

        let end = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap();
        assert!(end.is_none());
    }

    #[tokio::test]
    async fn activity_records_oversized_multipart_rejections_as_client_errors() {
        let boundary = "activity-limit";
        let body = multipart_body(
            boundary,
            &[("model", "alibaba/qwen3-asr-0.6b")],
            Some(("file", "audio.wav", &[0_u8; 64])),
        );
        let content_length = body.len();
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_upload_size = 32;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .header(axum::http::header::CONTENT_LENGTH, content_length)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["history"][0]["http_status"], 400, "{activity}");
        assert_eq!(activity["history"][0]["outcome"], "client_error");
    }

    #[tokio::test]
    async fn ocr_route_is_absent_when_ocr_services_are_inactive() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn websocket_transcription_route_uses_stream_suffix() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });

        let old_response = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/audio/transcriptions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let stream_response = router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(old_response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_ne!(stream_response.status(), StatusCode::NOT_FOUND);
        assert_ne!(stream_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn failed_websocket_upgrade_remains_http_activity() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        response.into_body().collect().await.unwrap();

        let activity = wait_for_activity_history(app).await;
        assert_eq!(activity["history"][0]["transport"], "http");
    }

    #[tokio::test]
    async fn websocket_disconnect_before_start_is_recorded_and_releases_pending_capacity() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_pending_websocket_connections = 1;
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(Arc::clone(&state), Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();

        socket.send(Message::Close(None)).await.unwrap();
        drop(socket);
        let pending = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(permit) = state.try_acquire_pending_websocket() {
                    break permit;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(pending);

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "disconnected");
        assert_eq!(activity["history"][0]["transport"], "websocket");

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn websocket_activity_counts_input_rejected_before_ready() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_concurrent_inference = 1;
        });
        let inference = state.acquire_inference().await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(Arc::clone(&state), Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket
            .send(Message::Text(
                r#"{"type":"start","model":"alibaba/qwen3-asr-0.6b","input_audio_format":"pcm_s16le","sample_rate":16000}"#
                    .into(),
            ))
            .await
            .unwrap();
        socket
            .send(Message::Binary(vec![1_u8, 2, 3, 4].into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap();

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "client_error");
        assert_eq!(activity["history"][0]["error_code"], "invalid_stream_state");
        assert_eq!(activity["history"][0]["input_bytes"], 4);

        drop(socket);
        drop(inference);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn websocket_activity_counts_a_rejected_binary_first_frame() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(state, Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket
            .send(Message::Binary(vec![1_u8, 2, 3].into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap();

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "client_error");
        assert_eq!(
            activity["history"][0]["error_code"],
            "missing_start_message"
        );
        assert_eq!(activity["history"][0]["input_bytes"], 3);

        drop(socket);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn multipart_ocr_requires_file() {
        let boundary = "orchion-ocr-missing-file";
        let body = multipart_body(boundary, &[("model", "paddlepaddle/pp-ocrv6-tiny")], None);

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "missing_required_parameter");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_empty_file() {
        let boundary = "orchion-ocr-empty-file";
        let body = multipart_body(
            boundary,
            &[("model", "paddlepaddle/pp-ocrv6-tiny")],
            Some(("file", "empty.png", b"")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_invalid_response_format() {
        let boundary = "orchion-ocr-invalid-format";
        let body = multipart_body(
            boundary,
            &[("response_format", "verbose_json")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "unsupported_response_format");
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_unknown_model() {
        let boundary = "orchion-ocr-unknown-model";
        let body = multipart_body(
            boundary,
            &[("model", "Acme/Experimental-OCR")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "model_not_available");
        assert_eq!(body["error"]["param"], "model");
    }

    #[tokio::test]
    async fn pdf_images_route_requires_file() {
        let boundary = "orchion-pdf-images-missing-file";
        let body = multipart_body(boundary, &[("response_format", "png")], None);

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "missing_required_parameter");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_empty_file() {
        let boundary = "orchion-pdf-images-empty-file";
        let body = multipart_body(boundary, &[], Some(("file", "empty.pdf", b"")));

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_non_pdf_file() {
        let boundary = "orchion-pdf-images-non-pdf-file";
        let body = multipart_body(boundary, &[], Some(("file", "document.png", b"image")));

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_response_format() {
        let boundary = "orchion-pdf-images-invalid-format";
        let body = multipart_body(
            boundary,
            &[("response_format", "gif")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_pages() {
        let boundary = "orchion-pdf-images-invalid-pages";
        let body = multipart_body(
            boundary,
            &[("pages", "2-1")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "pages");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_scale() {
        let boundary = "orchion-pdf-images-invalid-scale";
        let body = multipart_body(
            boundary,
            &[("scale", "4.1")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "scale");
    }

    async fn post_ocr(state: Arc<AppState>, boundary: &str, body: Vec<u8>) -> Response {
        router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn post_pdf_images(state: Arc<AppState>, boundary: &str, body: Vec<u8>) -> Response {
        router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/pdf/images")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn wait_for_activity_history(app: Router) -> Value {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri("/api/activity")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let activity = json_body(response).await;
                if activity["summary"]["retained"].as_u64().unwrap_or_default() > 0 {
                    return activity;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap()
    }

    fn multipart_body(
        boundary: &str,
        fields: &[(&str, &str)],
        file: Option<(&str, &str, &[u8])>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        if let Some((field_name, file_name, file_bytes)) = file {
            let content_type = if file_name.to_ascii_lowercase().ends_with(".pdf") {
                "application/pdf"
            } else {
                "image/png"
            };
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_name}\"\r\nContent-Type: {content_type}\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(file_bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn find_deployment_manifest(root: &std::path::Path) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(root).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                if let Some(manifest) = find_deployment_manifest(&path) {
                    return Some(manifest);
                }
            } else if path
                .file_name()
                .is_some_and(|name| name == "orchion-deployment.json")
            {
                return Some(path);
            }
        }
        None
    }

    fn test_state(ocr_active: bool, ocr_vl_active: bool) -> Arc<AppState> {
        test_state_with_config(ocr_active, ocr_vl_active, |_| {})
    }

    fn test_state_with_config(
        ocr_active: bool,
        ocr_vl_active: bool,
        configure: impl FnOnce(&mut ServerConfig),
    ) -> Arc<AppState> {
        let root = tempfile::tempdir().unwrap().keep();
        let mut config = ServerConfig::default_for_exe(&root.join("orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config.services.ocr.enabled = ocr_active;
        config.services.ocr.default_model =
            Some(ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap());
        config.services.ocr.models = ["paddlepaddle/pp-ocrv6-tiny", "paddlepaddle/pp-ocrv6-small"]
            .into_iter()
            .map(|id| {
                OcrModelDeployment::from_runtime(OcrModel::new(
                    ModelId::parse(id).unwrap(),
                    OcrModelKind::TraditionalOcr,
                ))
            })
            .collect();
        config.services.ocr_vl.enabled = ocr_vl_active;
        config.services.ocr_vl.default_model =
            Some(ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap());
        config.services.ocr_vl.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap(),
            OcrModelKind::OcrVl,
        ))];
        config.services.asr.models = vec![ModelDeployment::from_asr_runtime(
            AsrModel::parse("alibaba/qwen3-asr-0.6b").unwrap(),
        )];
        config.services.tts.models = vec![ModelDeployment::from_tts_runtime(
            TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap(),
        )];
        config.services.asr.idle_timeout = Duration::from_mins(10);
        config.services.tts.idle_timeout = Duration::from_mins(10);
        configure(&mut config);
        if config.services.asr.enabled {
            for deployment in &config.services.asr.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "huggingface",
                );
            }
        }
        if config.services.tts.enabled {
            for deployment in &config.services.tts.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "huggingface",
                );
            }
        }
        if config.services.ocr.enabled {
            for deployment in &config.services.ocr.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "modelscope",
                );
            }
        }
        if config.services.ocr_vl.enabled {
            for deployment in &config.services.ocr_vl.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "modelscope",
                );
            }
        }

        Arc::new(AppState::from_prepared_config(config).unwrap())
    }

    fn write_ready_fixture<M: ModelSpec>(
        models_dir: &std::path::Path,
        model: &M,
        model_url: &ModelUrl,
        neutral_source: &str,
    ) {
        if model_url.source() == ModelUrlSource::File {
            return;
        }
        let locator_repo = format!(
            "{}/{}",
            model_url.owner().unwrap(),
            model_url.repository().unwrap()
        );
        let source = match model_url.source() {
            ModelUrlSource::HuggingFace => "huggingface",
            ModelUrlSource::ModelScope => "modelscope",
            ModelUrlSource::Neutral => neutral_source,
            ModelUrlSource::File => unreachable!(),
        };
        let target = if matches!(model.category(), ModelCategory::Ocr | ModelCategory::OcrVl) {
            ModelSpec::cache_path(model, models_dir)
        } else {
            models_dir.join(&locator_repo)
        };
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("fixture.json"), "{}").unwrap();
        let cache_repo = target
            .strip_prefix(models_dir)
            .unwrap()
            .to_string_lossy()
            .to_string();
        let repo_id = if matches!(model.category(), ModelCategory::Ocr | ModelCategory::OcrVl) {
            if source == "modelscope" {
                model.modelscope_repo()
            } else {
                model.huggingface_repo()
            }
        } else {
            locator_repo.as_str()
        };
        let requested_revision = if source == "modelscope" {
            "master"
        } else {
            "main"
        };
        let resolved_revision = "1111111111111111111111111111111111111111";
        let mut repository_identities = ModelDownloader::model_artifact_plan(model, model_url)
            .unwrap()
            .into_iter()
            .map(|artifact| artifact.repository)
            .collect::<Vec<_>>();
        repository_identities.sort();
        repository_identities.dedup();
        let repositories = repository_identities
            .iter()
            .map(|identity| {
                let actual_repo = if identity == model.huggingface_repo() && source == "modelscope"
                {
                    model.modelscope_repo()
                } else {
                    identity
                };
                serde_json::json!({
                    "identity": identity,
                    "repo_id": actual_repo,
                    "requested_revision": requested_revision,
                    "resolved_revision": resolved_revision,
                })
            })
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "schema_version": 3,
            "source": source,
            "repo_id": repo_id,
            "downloaded_repos": repositories.iter().map(|repository| repository["repo_id"].clone()).collect::<Vec<_>>(),
            "revision": requested_revision,
            "resolved_revision": resolved_revision,
            "repositories": repositories,
            "layout": "model-hub-native",
            "files": [{
                "repo": cache_repo,
                "path": "fixture.json",
                "file_type": "file",
                "size": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }],
        });
        std::fs::write(
            target.join(".orchion-ready.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
}
