use crate::application::model_cache::{ModelLoadFailurePhase, ModelResidencyStatus};
use crate::application::model_lifecycle::ModelService;
use orchion::{LlmUsage, ModelId};
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use std::sync::Arc;
use std::time::Duration;

pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

const SECONDS_BUCKETS: [f64; 15] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

fn seconds_histogram() -> Histogram {
    Histogram::new(SECONDS_BUCKETS)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Delete,
    Options,
    Other,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum HttpRoute {
    Root,
    Healthz,
    Readyz,
    Metrics,
    Models,
    Model,
    ModelStatus,
    ModelLoad,
    ModelUnload,
    Activity,
    ActivityEvents,
    Speech,
    Transcriptions,
    TranscriptionsStream,
    Ocr,
    PdfImages,
    ChatCompletions,
    ChatCompletionsControl,
    Completions,
    Responses,
    ResponsesInputTokens,
    Embeddings,
    Stream,
    StreamsLookup,
    Docs,
    Openapi,
    Ui,
    Unmatched,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Outcome {
    Success,
    ClientError,
    ServerError,
    Cancelled,
    Timeout,
    ResourceExhausted,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum InferenceOperation {
    Chat,
    Completion,
    Responses,
    InputTokens,
    Embeddings,
    Asr,
    AsrStream,
    Tts,
    Ocr,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TerminationReason {
    Stop,
    Length,
    Cancelled,
    Timeout,
    ClientDisconnect,
    ServerShutdown,
    ResourceExhausted,
    StreamBufferExceeded,
    Error,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadOutcome {
    Success,
    Failure,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadPhase {
    Provision,
    Load,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ResidencyState {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ServiceLabel {
    Asr,
    Tts,
    Ocr,
    OcrVl,
    Llm,
    Unknown,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ReasoningControlOutcome {
    Success,
    NotFound,
    NotReasoning,
    Disabled,
    ModelMismatch,
    Unavailable,
    Invalid,
}

macro_rules! encode_label_value {
    ($name:ty, {$($variant:path => $value:literal),+ $(,)?}) => {
        impl EncodeLabelValue for $name {
            fn encode(
                &self,
                encoder: &mut prometheus_client::encoding::LabelValueEncoder,
            ) -> Result<(), std::fmt::Error> {
                use std::fmt::Write as _;
                encoder.write_str(match self { $($variant => $value),+ })
            }
        }
    };
}

encode_label_value!(HttpMethod, { HttpMethod::Get => "GET", HttpMethod::Post => "POST", HttpMethod::Delete => "DELETE", HttpMethod::Options => "OPTIONS", HttpMethod::Other => "OTHER" });
encode_label_value!(HttpRoute, {
    HttpRoute::Root => "/", HttpRoute::Healthz => "/healthz", HttpRoute::Readyz => "/readyz", HttpRoute::Metrics => "/metrics", HttpRoute::Models => "/v1/models", HttpRoute::Model => "/v1/models/{model}", HttpRoute::ModelStatus => "/api/models/status", HttpRoute::ModelLoad => "/api/models/load", HttpRoute::ModelUnload => "/api/models/unload", HttpRoute::Activity => "/api/activity", HttpRoute::ActivityEvents => "/api/activity/events", HttpRoute::Speech => "/v1/audio/speech", HttpRoute::Transcriptions => "/v1/audio/transcriptions", HttpRoute::TranscriptionsStream => "/v1/audio/transcriptions/stream", HttpRoute::Ocr => "/v1/ocr", HttpRoute::PdfImages => "/v1/pdf/images", HttpRoute::ChatCompletions => "/v1/chat/completions", HttpRoute::ChatCompletionsControl => "/v1/chat/completions/control", HttpRoute::Completions => "/v1/completions", HttpRoute::Responses => "/v1/responses", HttpRoute::ResponsesInputTokens => "/v1/responses/input_tokens", HttpRoute::Embeddings => "/v1/embeddings", HttpRoute::Stream => "/v1/stream", HttpRoute::StreamsLookup => "/v1/streams/lookup", HttpRoute::Docs => "/docs/{*path}", HttpRoute::Openapi => "/openapi/v1.json", HttpRoute::Ui => "/ui/{*path}", HttpRoute::Unmatched => "unmatched"
});
encode_label_value!(Outcome, { Outcome::Success => "success", Outcome::ClientError => "client_error", Outcome::ServerError => "server_error", Outcome::Cancelled => "cancelled", Outcome::Timeout => "timeout", Outcome::ResourceExhausted => "resource_exhausted" });
encode_label_value!(InferenceOperation, { InferenceOperation::Chat => "chat", InferenceOperation::Completion => "completion", InferenceOperation::Responses => "responses", InferenceOperation::InputTokens => "input_tokens", InferenceOperation::Embeddings => "embeddings", InferenceOperation::Asr => "asr", InferenceOperation::AsrStream => "asr_stream", InferenceOperation::Tts => "tts", InferenceOperation::Ocr => "ocr" });
encode_label_value!(TerminationReason, { TerminationReason::Stop => "stop", TerminationReason::Length => "length", TerminationReason::Cancelled => "cancelled", TerminationReason::Timeout => "timeout", TerminationReason::ClientDisconnect => "client_disconnect", TerminationReason::ServerShutdown => "server_shutdown", TerminationReason::ResourceExhausted => "resource_exhausted", TerminationReason::StreamBufferExceeded => "stream_buffer_exceeded", TerminationReason::Error => "error" });
encode_label_value!(LoadOutcome, { LoadOutcome::Success => "success", LoadOutcome::Failure => "failure" });
encode_label_value!(LoadPhase, { LoadPhase::Provision => "provision", LoadPhase::Load => "load" });
encode_label_value!(ResidencyState, { ResidencyState::Unloaded => "unloaded", ResidencyState::Loading => "loading", ResidencyState::Loaded => "loaded", ResidencyState::Unloading => "unloading" });
encode_label_value!(ServiceLabel, { ServiceLabel::Asr => "asr", ServiceLabel::Tts => "tts", ServiceLabel::Ocr => "ocr", ServiceLabel::OcrVl => "ocr_vl", ServiceLabel::Llm => "llm", ServiceLabel::Unknown => "unknown" });
encode_label_value!(ReasoningControlOutcome, { ReasoningControlOutcome::Success => "success", ReasoningControlOutcome::NotFound => "not_found", ReasoningControlOutcome::NotReasoning => "not_reasoning", ReasoningControlOutcome::Disabled => "disabled", ReasoningControlOutcome::ModelMismatch => "model_mismatch", ReasoningControlOutcome::Unavailable => "unavailable", ReasoningControlOutcome::Invalid => "invalid" });

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct HttpLabels {
    method: HttpMethod,
    route: HttpRoute,
    outcome: Outcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct HttpDurationLabels {
    method: HttpMethod,
    route: HttpRoute,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct InferenceLabels {
    operation: InferenceOperation,
    model: String,
    outcome: Outcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OperationLabels {
    operation: InferenceOperation,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct TerminationLabels {
    operation: InferenceOperation,
    reason: TerminationReason,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ModelLabels {
    service: ServiceLabel,
    model: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ModelOutcomeLabels {
    service: ServiceLabel,
    model: String,
    outcome: LoadOutcome,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ModelFailureLabels {
    service: ServiceLabel,
    model: String,
    phase: LoadPhase,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ResidencyLabels {
    service: ServiceLabel,
    model: String,
    state: ResidencyState,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ReasoningControlLabels {
    outcome: ReasoningControlOutcome,
}

#[derive(Debug, Clone)]
pub struct ModelObservation {
    pub service: ModelService,
    pub model: ModelId,
    pub residency: ModelResidencyStatus,
    pub load_epoch: u64,
    pub worker_healthy: bool,
    pub active_leases: usize,
    pub last_load_failure: Option<ModelLoadFailurePhase>,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StreamObservation {
    pub active: usize,
    pub retained: usize,
    pub followers: usize,
    pub buffered_events: usize,
    pub buffered_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ObservabilitySnapshot {
    pub shutdown: bool,
    pub models: Vec<ModelObservation>,
}

#[derive(Clone)]
pub struct Metrics {
    inner: Arc<Inner>,
}

pub struct InferenceLifecycle {
    metrics: Metrics,
    operation: InferenceOperation,
    model: ModelId,
    started: std::time::Instant,
    finished: bool,
}

struct Inner {
    registry: Registry,
    scrape_lock: std::sync::Mutex<()>,
    http_requests: Family<HttpLabels, Counter>,
    http_active: Gauge,
    http_duration: Family<HttpDurationLabels, Histogram, fn() -> Histogram>,
    inference_requests: Family<InferenceLabels, Counter>,
    inference_terminations: Family<TerminationLabels, Counter>,
    inference_active: Family<OperationLabels, Gauge>,
    inference_duration: Family<OperationLabels, Histogram, fn() -> Histogram>,
    queue_duration: Family<OperationLabels, Histogram, fn() -> Histogram>,
    generation_duration: Family<OperationLabels, Histogram, fn() -> Histogram>,
    ttft_duration: Family<OperationLabels, Histogram, fn() -> Histogram>,
    model_loads: Family<ModelOutcomeLabels, Counter>,
    model_load_failures: Family<ModelFailureLabels, Counter>,
    model_residency: Family<ResidencyLabels, Gauge>,
    model_load_epoch: Family<ModelLabels, Gauge>,
    model_worker_health: Family<ModelLabels, Gauge>,
    model_active_leases: Family<ModelLabels, Gauge>,
    llm_logical_tokens: Counter,
    llm_processed_tokens: Counter,
    llm_cached_tokens: Counter,
    llm_completion_tokens: Counter,
    llm_reasoning_tokens: Counter,
    llm_prompt_seconds: Counter<f64>,
    llm_decode_seconds: Counter<f64>,
    resumable_created: Counter,
    resumable_terminal: Counter,
    resumable_attachments: Counter,
    resumable_evictions: Counter,
    resumable_truncations: Counter,
    resumable_active: Gauge,
    resumable_retained: Gauge,
    resumable_followers: Gauge,
    resumable_buffered_events: Gauge,
    resumable_buffered_bytes: Gauge,
    reasoning_controls: Family<ReasoningControlLabels, Counter>,
}

impl Metrics {
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "registers the complete observability contract in one registry"
    )]
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let http_requests = Family::default();
        let http_active = Gauge::default();
        let http_duration = Family::new_with_constructor(seconds_histogram as fn() -> Histogram);
        let inference_requests = Family::default();
        let inference_terminations = Family::default();
        let inference_active = Family::default();
        let inference_duration =
            Family::new_with_constructor(seconds_histogram as fn() -> Histogram);
        let queue_duration = Family::new_with_constructor(seconds_histogram as fn() -> Histogram);
        let generation_duration =
            Family::new_with_constructor(seconds_histogram as fn() -> Histogram);
        let ttft_duration = Family::new_with_constructor(seconds_histogram as fn() -> Histogram);
        let model_loads = Family::default();
        let model_load_failures = Family::default();
        let model_residency = Family::default();
        let model_load_epoch = Family::default();
        let model_worker_health = Family::default();
        let model_active_leases = Family::default();
        let llm_logical_tokens = Counter::default();
        let llm_processed_tokens = Counter::default();
        let llm_cached_tokens = Counter::default();
        let llm_completion_tokens = Counter::default();
        let llm_reasoning_tokens = Counter::default();
        let llm_prompt_seconds = Counter::<f64>::default();
        let llm_decode_seconds = Counter::<f64>::default();
        let resumable_created = Counter::default();
        let resumable_terminal = Counter::default();
        let resumable_attachments = Counter::default();
        let resumable_evictions = Counter::default();
        let resumable_truncations = Counter::default();
        let resumable_active = Gauge::default();
        let resumable_retained = Gauge::default();
        let resumable_followers = Gauge::default();
        let resumable_buffered_events = Gauge::default();
        let resumable_buffered_bytes = Gauge::default();
        let reasoning_controls = Family::default();

        macro_rules! register {
            ($name:literal, $help:literal, $metric:ident) => {
                registry.register($name, $help, $metric.clone());
            };
        }
        register!(
            "orchion_http_requests",
            "HTTP requests by bounded route and outcome",
            http_requests
        );
        register!(
            "orchion_http_active",
            "HTTP requests currently executing",
            http_active
        );
        register!(
            "orchion_http_duration_seconds",
            "HTTP request duration",
            http_duration
        );
        register!(
            "orchion_inference_requests",
            "Inference requests by operation, configured model, and outcome",
            inference_requests
        );
        register!(
            "orchion_inference_terminations",
            "Inference terminal reasons",
            inference_terminations
        );
        register!(
            "orchion_inference_active",
            "Inference requests currently executing",
            inference_active
        );
        register!(
            "orchion_inference_duration_seconds",
            "Inference request duration",
            inference_duration
        );
        register!(
            "orchion_inference_queue_seconds",
            "Inference queue duration",
            queue_duration
        );
        register!(
            "orchion_inference_generation_seconds",
            "Inference generation duration",
            generation_duration
        );
        register!(
            "orchion_inference_ttft_seconds",
            "Inference time to first token",
            ttft_duration
        );
        register!("orchion_model_loads", "Model load attempts", model_loads);
        register!(
            "orchion_model_load_failures",
            "Model load failures by phase",
            model_load_failures
        );
        register!(
            "orchion_model_residency",
            "Configured model residency one-hot state",
            model_residency
        );
        register!(
            "orchion_model_load_epoch",
            "Successful model load epoch",
            model_load_epoch
        );
        register!(
            "orchion_model_worker_health",
            "Resident model worker health",
            model_worker_health
        );
        register!(
            "orchion_model_active_leases",
            "Active model leases",
            model_active_leases
        );
        register!(
            "orchion_llm_logical_tokens",
            "Logical LLM input tokens",
            llm_logical_tokens
        );
        register!(
            "orchion_llm_processed_tokens",
            "Processed LLM prompt tokens",
            llm_processed_tokens
        );
        register!(
            "orchion_llm_cached_tokens",
            "Cached LLM prompt tokens",
            llm_cached_tokens
        );
        register!(
            "orchion_llm_completion_tokens",
            "Generated LLM completion tokens",
            llm_completion_tokens
        );
        register!(
            "orchion_llm_reasoning_tokens",
            "Generated LLM tokens classified as reasoning",
            llm_reasoning_tokens
        );
        register!(
            "orchion_llm_prompt_seconds",
            "LLM prompt processing seconds",
            llm_prompt_seconds
        );
        register!(
            "orchion_llm_decode_seconds",
            "LLM decode seconds",
            llm_decode_seconds
        );
        register!(
            "orchion_resumable_created",
            "Resumable sessions created",
            resumable_created
        );
        register!(
            "orchion_resumable_terminal",
            "Resumable sessions reaching terminal state",
            resumable_terminal
        );
        register!(
            "orchion_resumable_attachments",
            "Resumable stream follower attachments",
            resumable_attachments
        );
        register!(
            "orchion_resumable_evictions",
            "Resumable sessions evicted",
            resumable_evictions
        );
        register!(
            "orchion_resumable_truncations",
            "Resumable event buffers truncated",
            resumable_truncations
        );
        register!(
            "orchion_resumable_sessions_active",
            "Active resumable sessions",
            resumable_active
        );
        register!(
            "orchion_resumable_sessions_retained",
            "Retained terminal resumable sessions",
            resumable_retained
        );
        register!(
            "orchion_resumable_followers",
            "Attached resumable followers",
            resumable_followers
        );
        register!(
            "orchion_resumable_buffered_events",
            "Retained resumable events",
            resumable_buffered_events
        );
        register!(
            "orchion_resumable_buffered_bytes",
            "Retained resumable bytes",
            resumable_buffered_bytes
        );
        register!(
            "orchion_llm_reasoning_controls",
            "Chat reasoning control requests by bounded outcome",
            reasoning_controls
        );

        Self {
            inner: Arc::new(Inner {
                registry,
                scrape_lock: std::sync::Mutex::new(()),
                http_requests,
                http_active,
                http_duration,
                inference_requests,
                inference_terminations,
                inference_active,
                inference_duration,
                queue_duration,
                generation_duration,
                ttft_duration,
                model_loads,
                model_load_failures,
                model_residency,
                model_load_epoch,
                model_worker_health,
                model_active_leases,
                llm_logical_tokens,
                llm_processed_tokens,
                llm_cached_tokens,
                llm_completion_tokens,
                llm_reasoning_tokens,
                llm_prompt_seconds,
                llm_decode_seconds,
                resumable_created,
                resumable_terminal,
                resumable_attachments,
                resumable_evictions,
                resumable_truncations,
                resumable_active,
                resumable_retained,
                resumable_followers,
                resumable_buffered_events,
                resumable_buffered_bytes,
                reasoning_controls,
            }),
        }
    }

    pub fn begin_http(&self) {
        self.inner.http_active.inc();
    }
    pub fn finish_http(
        &self,
        method: HttpMethod,
        route: HttpRoute,
        outcome: Outcome,
        duration: Duration,
    ) {
        self.inner.http_active.dec();
        self.inner
            .http_requests
            .get_or_create(&HttpLabels {
                method: method.clone(),
                route: route.clone(),
                outcome,
            })
            .inc();
        self.inner
            .http_duration
            .get_or_create(&HttpDurationLabels { method, route })
            .observe(duration.as_secs_f64());
    }
    pub fn begin_inference(&self, operation: InferenceOperation) {
        self.inner
            .inference_active
            .get_or_create(&OperationLabels { operation })
            .inc();
    }
    #[must_use]
    pub fn start_inference(
        &self,
        operation: InferenceOperation,
        model: ModelId,
    ) -> InferenceLifecycle {
        self.begin_inference(operation.clone());
        InferenceLifecycle {
            metrics: self.clone(),
            operation,
            model,
            started: std::time::Instant::now(),
            finished: false,
        }
    }
    pub fn finish_inference(
        &self,
        operation: InferenceOperation,
        model: &ModelId,
        outcome: Outcome,
        duration: Duration,
    ) {
        self.inner
            .inference_active
            .get_or_create(&OperationLabels {
                operation: operation.clone(),
            })
            .dec();
        self.inner
            .inference_requests
            .get_or_create(&InferenceLabels {
                operation: operation.clone(),
                model: model.to_string(),
                outcome,
            })
            .inc();
        self.inner
            .inference_duration
            .get_or_create(&OperationLabels { operation })
            .observe(duration.as_secs_f64());
    }
    pub fn observe_termination(&self, operation: InferenceOperation, reason: TerminationReason) {
        self.inner
            .inference_terminations
            .get_or_create(&TerminationLabels { operation, reason })
            .inc();
    }
    pub fn observe_llm_usage(&self, operation: InferenceOperation, usage: LlmUsage) {
        let cached = usage.timings.cache_n.min(usage.prompt_tokens);
        self.inner
            .llm_logical_tokens
            .inc_by(usage.prompt_tokens as u64);
        self.inner
            .llm_processed_tokens
            .inc_by(usage.timings.prompt_n as u64);
        self.inner.llm_cached_tokens.inc_by(cached as u64);
        self.inner
            .llm_completion_tokens
            .inc_by(usage.completion_tokens as u64);
        self.inner
            .llm_reasoning_tokens
            .inc_by(usage.reasoning_tokens as u64);
        self.inner
            .llm_prompt_seconds
            .inc_by(usage.timings.prompt_ms / 1000.0);
        self.inner
            .llm_decode_seconds
            .inc_by(usage.timings.predicted_ms / 1000.0);
        if let Some(queue_ms) = usage.queue_time_ms {
            self.inner
                .queue_duration
                .get_or_create(&OperationLabels {
                    operation: operation.clone(),
                })
                .observe(Duration::from_millis(queue_ms).as_secs_f64());
        }
        if let Some(eval_ms) = usage.eval_time_ms {
            self.inner
                .generation_duration
                .get_or_create(&OperationLabels { operation })
                .observe(Duration::from_millis(eval_ms).as_secs_f64());
        }
    }
    pub fn observe_ttft(&self, operation: InferenceOperation, duration: Duration) {
        self.inner
            .ttft_duration
            .get_or_create(&OperationLabels { operation })
            .observe(duration.as_secs_f64());
    }
    pub fn observe_model_load(
        &self,
        service: ModelService,
        model: &ModelId,
        result: Result<(), ModelLoadFailurePhase>,
    ) {
        let labels = ModelOutcomeLabels {
            service: service.into(),
            model: model.to_string(),
            outcome: if result.is_ok() {
                LoadOutcome::Success
            } else {
                LoadOutcome::Failure
            },
        };
        self.inner.model_loads.get_or_create(&labels).inc();
        if let Err(phase) = result {
            self.inner
                .model_load_failures
                .get_or_create(&ModelFailureLabels {
                    service: service.into(),
                    model: model.to_string(),
                    phase: phase.into(),
                })
                .inc();
        }
    }
    pub fn update_models(&self, models: &[ModelObservation]) {
        for model in models {
            let service = ServiceLabel::from(model.service);
            for state in [
                ResidencyState::Unloaded,
                ResidencyState::Loading,
                ResidencyState::Loaded,
                ResidencyState::Unloading,
            ] {
                let selected = state == ResidencyState::from(model.residency);
                self.inner
                    .model_residency
                    .get_or_create(&ResidencyLabels {
                        service: service.clone(),
                        model: model.model.to_string(),
                        state,
                    })
                    .set(i64::from(selected));
            }
            let labels = ModelLabels {
                service,
                model: model.model.to_string(),
            };
            self.inner
                .model_load_epoch
                .get_or_create(&labels)
                .set(i64::try_from(model.load_epoch).unwrap_or(i64::MAX));
            self.inner
                .model_worker_health
                .get_or_create(&labels)
                .set(i64::from(model.worker_healthy));
            self.inner
                .model_active_leases
                .get_or_create(&labels)
                .set(i64::try_from(model.active_leases).unwrap_or(i64::MAX));
        }
    }
    pub fn update_streams(&self, streams: StreamObservation) {
        self.inner.resumable_active.set(to_i64(streams.active));
        self.inner.resumable_retained.set(to_i64(streams.retained));
        self.inner
            .resumable_followers
            .set(to_i64(streams.followers));
        self.inner
            .resumable_buffered_events
            .set(to_i64(streams.buffered_events));
        self.inner
            .resumable_buffered_bytes
            .set(to_i64(streams.buffered_bytes));
    }

    /// Updates dynamic gauges and encodes one coherent `OpenMetrics` scrape.
    ///
    /// # Errors
    ///
    /// Returns a formatting error if an individual metric cannot be encoded.
    pub fn scrape(
        &self,
        models: &[ModelObservation],
        streams: StreamObservation,
    ) -> Result<String, std::fmt::Error> {
        let _scrape = self
            .inner
            .scrape_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.update_models(models);
        self.update_streams(streams);
        self.encode()
    }
    pub fn resumable_created(&self) {
        self.inner.resumable_created.inc();
    }
    pub fn resumable_terminal(&self) {
        self.inner.resumable_terminal.inc();
    }
    pub fn resumable_attachment(&self) {
        self.inner.resumable_attachments.inc();
    }
    pub fn resumable_eviction(&self) {
        self.inner.resumable_evictions.inc();
    }
    pub fn resumable_truncation(&self) {
        self.inner.resumable_truncations.inc();
    }
    pub fn observe_reasoning_control(&self, outcome: ReasoningControlOutcome) {
        self.inner
            .reasoning_controls
            .get_or_create(&ReasoningControlLabels { outcome })
            .inc();
    }
    /// Encodes a complete `OpenMetrics` 1.0 exposition.
    ///
    /// # Errors
    ///
    /// Returns a formatting error if an individual metric cannot be encoded.
    pub fn encode(&self) -> Result<String, std::fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.inner.registry)?;
        Ok(output)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceLifecycle {
    pub fn finish(mut self, outcome: Outcome) {
        self.metrics.finish_inference(
            self.operation.clone(),
            &self.model,
            outcome,
            self.started.elapsed(),
        );
        self.finished = true;
    }
}

impl Drop for InferenceLifecycle {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.finish_inference(
                self.operation.clone(),
                &self.model,
                Outcome::Cancelled,
                self.started.elapsed(),
            );
        }
    }
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

impl From<ModelService> for ServiceLabel {
    fn from(value: ModelService) -> Self {
        match value {
            ModelService::Asr => Self::Asr,
            ModelService::Tts => Self::Tts,
            ModelService::Ocr => Self::Ocr,
            ModelService::OcrVl => Self::OcrVl,
            ModelService::Llm => Self::Llm,
            _ => Self::Unknown,
        }
    }
}
impl From<ModelLoadFailurePhase> for LoadPhase {
    fn from(value: ModelLoadFailurePhase) -> Self {
        match value {
            ModelLoadFailurePhase::Provision => Self::Provision,
            ModelLoadFailurePhase::Load => Self::Load,
        }
    }
}
impl From<ModelResidencyStatus> for ResidencyState {
    fn from(value: ModelResidencyStatus) -> Self {
        match value {
            ModelResidencyStatus::Unloaded => Self::Unloaded,
            ModelResidencyStatus::Loading => Self::Loading,
            ModelResidencyStatus::Loaded => Self::Loaded,
            ModelResidencyStatus::Unloading => Self::Unloading,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_are_monotonic_and_inference_guards_balance_active() {
        let metrics = Metrics::new();
        metrics.begin_http();
        metrics.finish_http(
            HttpMethod::Get,
            HttpRoute::Healthz,
            Outcome::Success,
            Duration::from_millis(2),
        );
        metrics.begin_http();
        metrics.finish_http(
            HttpMethod::Get,
            HttpRoute::Healthz,
            Outcome::Success,
            Duration::from_millis(3),
        );

        let model = ModelId::parse("qwen/test").unwrap();
        metrics
            .start_inference(InferenceOperation::Chat, model.clone())
            .finish(Outcome::Success);
        drop(metrics.start_inference(InferenceOperation::Chat, model));

        let output = metrics.encode().unwrap();
        assert!(output.contains(
            "orchion_http_requests_total{method=\"GET\",route=\"/healthz\",outcome=\"success\"} 2"
        ));
        assert!(output.contains("orchion_inference_active{operation=\"chat\"} 0"));
        assert!(output.contains("model=\"qwen/test\",outcome=\"success\"} 1"));
        assert!(output.contains("model=\"qwen/test\",outcome=\"cancelled\"} 1"));
        assert!(output.ends_with("# EOF\n"));
    }
}
