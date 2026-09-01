use orchion_llama_cpp as backend;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
#[cfg(feature = "server-support")]
use std::sync::Arc;

use crate::{LlmModel, OrchionError, Result};

pub fn validate_llm_json_schema(schema: &backend::JsonValue) -> Result<()> {
    backend::validate_strict_schema(schema).map_err(map_backend_error)
}

/// A text-only LLM deployment backed by one exact local GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmDeployment {
    model: LlmModel,
    path: PathBuf,
}

impl LlmDeployment {
    pub fn from_file(model: LlmModel, path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let metadata = std::fs::metadata(&path).map_err(|error| OrchionError::ModelLoad {
            message: format!("cannot access LLM GGUF `{}`: {error}", path.display()),
        })?;
        if !metadata.is_file() {
            return Err(OrchionError::ModelLoad {
                message: format!("LLM GGUF path `{}` is not a file", path.display()),
            });
        }
        if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
        {
            return Err(OrchionError::ModelLoad {
                message: format!("LLM model `{}` is not a GGUF file", path.display()),
            });
        }
        Ok(Self { model, path })
    }

    #[must_use]
    pub const fn model(&self) -> &LlmModel {
        &self.model
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(feature = "download-all")]
    pub async fn provision(
        model: LlmModel,
        source: crate::ModelUrl,
        cache_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::provision_with_downloader(
            model,
            source,
            cache_dir,
            &crate::ModelDownloader::default(),
        )
        .await
    }

    #[cfg(feature = "download-all")]
    pub async fn provision_with_downloader(
        model: LlmModel,
        source: crate::ModelUrl,
        cache_dir: impl AsRef<Path>,
        downloader: &crate::ModelDownloader,
    ) -> Result<Self> {
        let plan = llm_deployment_artifact_plan(&model, &source)?;
        let publication = downloader
            .provision_logical_deployment(model.id(), crate::ModelCategory::Llm, &plan, cache_dir)
            .await?;
        let path = publication
            .artifact_file(crate::ArtifactRole::LlmModel)
            .ok_or_else(|| OrchionError::ModelLoad {
                message: format!("published LLM deployment `{model}` has no model artifact"),
            })?
            .to_path_buf();
        Self::from_file(model, path)
    }
}

#[doc(hidden)]
#[cfg(feature = "server-support")]
#[derive(Debug, Clone)]
pub struct LlmBackendGuard {
    _inner: Arc<backend::BackendOwner>,
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
#[derive(Clone)]
pub struct LlmScriptedControl(backend::ScriptedControl);

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
impl LlmScriptedControl {
    pub fn wait_started(&self) {
        self.0.wait_started();
    }

    pub fn release_ready(&self) {
        self.0.release_ready();
    }

    pub fn release_cleanup(&self) {
        self.0.release_cleanup();
    }

    pub fn wait_preparation_started(&self) {
        self.0.wait_preparation_started();
    }

    pub fn release_preparation(&self) {
        self.0.release_preparation();
    }

    pub fn has_executed(&self) -> bool {
        self.0.has_executed()
    }

    pub fn has_started(&self) -> bool {
        self.0.has_started()
    }
}

#[doc(hidden)]
#[cfg(feature = "server-support")]
pub fn initialize_llm_backend() -> crate::Result<LlmBackendGuard> {
    backend::BackendOwner::acquire()
        .map(|inner| LlmBackendGuard { _inner: inner })
        .map_err(|error| crate::OrchionError::ModelLoad {
            message: error.to_string(),
        })
}

#[doc(hidden)]
#[cfg(feature = "server-support")]
#[must_use]
pub fn llm_build_metadata_json() -> String {
    backend::build_metadata_json()
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_llm_engine(script: Vec<GenerationEvent>) -> (LlmEngine, LlmScriptedControl) {
    let script = script
        .into_iter()
        .map(|event| match event {
            GenerationEvent::ContentDelta(content) => backend::Event::Content(content),
            GenerationEvent::Finished { reason, usage } => backend::Event::Finished {
                reason: match reason {
                    GenerationFinishReason::Stop => backend::FinishReason::Stop,
                    GenerationFinishReason::Length => backend::FinishReason::Length,
                    GenerationFinishReason::Cancelled => backend::FinishReason::Cancelled,
                },
                usage: backend::Usage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    reasoning_tokens: usage.reasoning_tokens,
                    timings: backend::Timings {
                        cache_n: usage.timings.cache_n,
                        prompt_n: usage.timings.prompt_n,
                        prompt_ms: usage.timings.prompt_ms,
                        prompt_per_token_ms: usage.timings.prompt_per_token_ms,
                        prompt_per_second: usage.timings.prompt_per_second,
                        predicted_n: usage.timings.predicted_n,
                        predicted_ms: usage.timings.predicted_ms,
                        predicted_per_token_ms: usage.timings.predicted_per_token_ms,
                        predicted_per_second: usage.timings.predicted_per_second,
                    },
                },
            },
        })
        .collect();
    let (inner, control) = backend::scripted_engine(script, 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_reasoning_llm_engine(
    reasoning: impl Into<String>,
    usage: LlmUsage,
) -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_engine(
        vec![
            backend::Event::Semantic(backend::SemanticDelta::Reasoning(reasoning.into())),
            backend::Event::Finished {
                reason: backend::FinishReason::Stop,
                usage: backend::Usage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    reasoning_tokens: usage.reasoning_tokens,
                    timings: backend::Timings {
                        cache_n: usage.timings.cache_n,
                        prompt_n: usage.timings.prompt_n,
                        prompt_ms: usage.timings.prompt_ms,
                        prompt_per_token_ms: usage.timings.prompt_per_token_ms,
                        prompt_per_second: usage.timings.prompt_per_second,
                        predicted_n: usage.timings.predicted_n,
                        predicted_ms: usage.timings.predicted_ms,
                        predicted_per_token_ms: usage.timings.predicted_per_token_ms,
                        predicted_per_second: usage.timings.predicted_per_second,
                    },
                },
            },
        ],
        1,
    );
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_context_limit_llm_engine(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: usize,
) -> LlmEngine {
    LlmEngine {
        inner: backend::scripted_context_limit_engine(prompt_tokens, max_tokens, context_size),
        event_queue_capacity: 1,
        model: None,
        embedding_config: None,
    }
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_engine(
        vec![backend::Event::Failed("__orchion_test_panic__".to_string())],
        1,
    );
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_failing_llm_engine(message: impl Into<String>) -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) =
        backend::scripted_engine(vec![backend::Event::Failed(message.into())], 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_preparation_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_preparation_panicking_engine(1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_slow_preparation_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_slow_preparation_engine(Vec::new(), 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_embedding_llm_engine(
    embeddings: Vec<Vec<f32>>,
    prompt_tokens: usize,
) -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_embedding_engine(
        backend::EmbeddingOutput {
            embeddings,
            prompt_tokens,
        },
        1,
    );
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
            embedding_config: Some(LlmEmbeddingConfig {
                pooling: LlmEmbeddingPooling::Last,
                min_dimensions: 1,
                max_input_tokens: 8192,
            }),
        },
        LlmScriptedControl(control),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmRole {
    System,
    Developer,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmSemanticRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmContentPart {
    Text { text: String },
    Image(LlmImageInput),
    Reasoning { text: String },
    ToolResult(LlmToolResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmImageFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmImageInput {
    pub bytes: Vec<u8>,
    pub format: LlmImageFormat,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmToolCall {
    pub id: String,
    pub name: String,
    pub arguments: backend::JsonValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRichMessage {
    pub role: LlmSemanticRole,
    pub content: Vec<LlmContentPart>,
    pub tool_calls: Vec<LlmToolCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: backend::JsonValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LlmToolChoice {
    #[default]
    None,
    Auto,
    Required,
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LlmReasoningOptions {
    /// `None` inherits the deployment default; `Some(false)` explicitly disables reasoning.
    pub enabled: Option<bool>,
    pub effort: Option<LlmReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum LlmOutputConstraint {
    #[default]
    Text,
    JsonObject,
    JsonSchema(backend::JsonValue),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmLogitBias {
    pub token_id: i32,
    pub bias: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LlmLogprobsOptions {
    pub top_logprobs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LlmSamplingExtensions {
    pub typical_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmAdvancedInput {
    Messages(Vec<LlmRichMessage>),
    Prompt(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmAdvancedRequest {
    pub input: LlmAdvancedInput,
    pub options: GenerationOptions,
    pub tools: Vec<LlmToolDefinition>,
    pub tool_choice: LlmToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: LlmReasoningOptions,
    pub output: LlmOutputConstraint,
    pub logprobs: Option<LlmLogprobsOptions>,
    pub logit_bias: Vec<LlmLogitBias>,
    pub sampling: LlmSamplingExtensions,
    pub choices: usize,
    #[doc(hidden)]
    pub reasoning_control_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmSemanticTokenCountRequest {
    pub messages: Vec<LlmRichMessage>,
    pub tools: Vec<LlmToolDefinition>,
    pub tool_choice: LlmToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: LlmReasoningOptions,
    pub output: LlmOutputConstraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmReasoningControlResult {
    Success,
    NotFound,
    NotReasoning,
    Disabled,
}

#[derive(Debug, Clone)]
pub struct LlmReasoningControl {
    inner: backend::ReasoningControlHandle,
}

#[doc(hidden)]
pub struct LlmReasoningControlAttempt {
    inner: backend::ReasoningControlAttempt,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct LlmReasoningControlCancellation {
    inner: backend::ReasoningControlCancellation,
}

impl LlmReasoningControl {
    pub async fn reasoning_end(&self) -> crate::Result<LlmReasoningControlResult> {
        self.begin_reasoning_end()?.result().await
    }

    #[doc(hidden)]
    pub fn begin_reasoning_end(&self) -> crate::Result<LlmReasoningControlAttempt> {
        self.inner
            .begin_reasoning_end()
            .map(|inner| LlmReasoningControlAttempt { inner })
            .map_err(map_backend_error)
    }
}

impl LlmReasoningControlAttempt {
    pub fn cancellation_handle(&self) -> LlmReasoningControlCancellation {
        LlmReasoningControlCancellation {
            inner: self.inner.cancellation_handle(),
        }
    }

    pub async fn result(self) -> crate::Result<LlmReasoningControlResult> {
        self.inner
            .result()
            .await
            .map(|result| match result {
                backend::ReasoningControlResult::Success => LlmReasoningControlResult::Success,
                backend::ReasoningControlResult::NotFound => LlmReasoningControlResult::NotFound,
                backend::ReasoningControlResult::NotReasoning => {
                    LlmReasoningControlResult::NotReasoning
                }
                backend::ReasoningControlResult::Disabled => LlmReasoningControlResult::Disabled,
            })
            .map_err(map_backend_error)
    }
}

impl LlmReasoningControlCancellation {
    pub fn cancel_pending(&self) -> bool {
        self.inner.cancel_pending()
    }
}

impl LlmRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            seed: u32::MAX,
            stop: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub messages: Vec<LlmMessage>,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmTimings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

impl Default for LlmTimings {
    fn default() -> Self {
        Self {
            cache_n: 0,
            prompt_n: 0,
            prompt_ms: 0.0,
            prompt_per_token_ms: 0.0,
            prompt_per_second: 0.0,
            predicted_n: 0,
            predicted_ms: 0.0,
            predicted_per_token_ms: 0.0,
            predicted_per_second: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub total_tokens: usize,
    pub queue_time_ms: Option<u64>,
    pub eval_time_ms: Option<u64>,
    pub timings: LlmTimings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationEvent {
    ContentDelta(String),
    Finished {
        reason: GenerationFinishReason,
        usage: LlmUsage,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmChoiceFinishReason {
    Stop,
    Length,
    Cancelled,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmTokenAlternative {
    pub token_id: i32,
    pub bytes: Vec<u8>,
    pub logprob: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmTokenLogprobs {
    pub chosen: LlmTokenAlternative,
    pub top: Vec<LlmTokenAlternative>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmChoiceEvent {
    Delta {
        index: usize,
        text: String,
        logprobs: Option<LlmTokenLogprobs>,
    },
    SemanticDelta {
        index: usize,
        delta: LlmSemanticDelta,
    },
    Finished {
        index: usize,
        reason: LlmChoiceFinishReason,
        usage: LlmUsage,
    },
    FinishedAll {
        usage: LlmUsage,
    },
    Failed {
        index: Option<usize>,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmSemanticDelta {
    Text(String),
    Reasoning(String),
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmComplete {
    pub text: String,
    pub finish_reason: GenerationFinishReason,
    pub usage: LlmUsage,
}

#[derive(Debug, Clone)]
pub struct LlmEngineConfig {
    pub context_size: Option<NonZeroU32>,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub threads: i32,
    pub gpu_layers: u32,
    pub parallel_sequences: u32,
    pub request_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub chat_template: Option<String>,
    pub template_engine: LlmTemplateEngine,
    pub enable_thinking: bool,
    pub prompt_cache: LlmPromptCacheConfig,
    pub deployment_kind: LlmDeploymentKind,
    pub vision: Option<LlmVisionConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmVisionConfig {
    pub mmproj: PathBuf,
    pub limits: LlmVisionLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmVisionLimits {
    pub max_images: usize,
    pub max_bytes_per_image: usize,
    pub max_total_bytes: usize,
    pub max_side: u32,
    pub max_pixels_per_image: u64,
    pub max_total_pixels: u64,
}

impl Default for LlmVisionLimits {
    fn default() -> Self {
        let limits = backend::VisionLimits::default();
        Self {
            max_images: limits.max_images,
            max_bytes_per_image: limits.max_bytes_per_image,
            max_total_bytes: limits.max_total_bytes,
            max_side: limits.max_side,
            max_pixels_per_image: limits.max_pixels_per_image,
            max_total_pixels: limits.max_total_pixels,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmPromptCacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub min_prefix_tokens: usize,
}

impl Default for LlmPromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: 4,
            max_bytes: 268_435_456,
            min_prefix_tokens: 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmDeploymentKind {
    Generation,
    Embeddings(LlmEmbeddingConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmEmbeddingConfig {
    pub pooling: LlmEmbeddingPooling,
    pub min_dimensions: usize,
    pub max_input_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmEmbeddingPooling {
    Last,
}

impl Default for LlmEngineConfig {
    fn default() -> Self {
        Self {
            context_size: None,
            batch_size: 512,
            micro_batch_size: 512,
            threads: 0,
            gpu_layers: u32::MAX,
            parallel_sequences: 1,
            request_queue_capacity: 8,
            event_queue_capacity: 16,
            chat_template: None,
            template_engine: LlmTemplateEngine::LlamaCpp,
            enable_thinking: false,
            prompt_cache: LlmPromptCacheConfig::default(),
            deployment_kind: LlmDeploymentKind::Generation,
            vision: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmTemplateEngine {
    LlamaCpp,
    Jinja,
}

#[derive(Debug, Clone)]
pub struct LlmEngine {
    inner: backend::Engine,
    event_queue_capacity: usize,
    model: Option<LlmModel>,
    embedding_config: Option<LlmEmbeddingConfig>,
}

pub struct LlmGeneration {
    inner: backend::Generation,
    terminal_received: bool,
}

pub struct LlmChoiceGeneration {
    inner: backend::ChoiceGeneration,
}

pub struct LlmChoiceReservation {
    inner: backend::ChoiceReservation,
}

#[doc(hidden)]
pub struct LlmReservation {
    inner: backend::Reservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEmbeddingInput {
    Text(String),
    Tokens(Vec<i32>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmEmbeddingRequest {
    pub inputs: Vec<LlmEmbeddingInput>,
    pub dimensions: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmEmbeddingResult {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[doc(hidden)]
pub struct LlmEmbeddingReservation {
    inner: backend::EmbeddingReservation,
    dimensions: Option<usize>,
    min_dimensions: usize,
    input_count: usize,
}

#[doc(hidden)]
pub struct LlmEmbeddingOperation {
    inner: backend::Embedding,
    dimensions: Option<usize>,
    min_dimensions: usize,
    input_count: usize,
}

#[allow(
    dead_code,
    reason = "server-support lifecycle methods are consumed by the downstream server crate"
)]
impl LlmEmbeddingReservation {
    pub async fn commit_reserved(&mut self) -> crate::Result<LlmEmbeddingOperation> {
        self.inner
            .commit()
            .await
            .map(|inner| LlmEmbeddingOperation {
                inner,
                dimensions: self.dimensions,
                min_dimensions: self.min_dimensions,
                input_count: self.input_count,
            })
            .map_err(map_backend_error)
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }

    pub fn abort(self) {
        self.inner.abort();
    }
}

#[allow(
    dead_code,
    reason = "server-support lifecycle methods are consumed by the downstream server crate"
)]
impl LlmEmbeddingOperation {
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub async fn result(&mut self) -> crate::Result<LlmEmbeddingResult> {
        let output = self.inner.result().await.map_err(map_backend_error)?;
        finalize_embeddings(
            output,
            self.dimensions,
            self.min_dimensions,
            self.input_count,
        )
    }

    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }
}

impl LlmReservation {
    pub async fn commit(mut self) -> crate::Result<LlmGeneration> {
        self.commit_inner().await
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn commit_reserved(&mut self) -> crate::Result<LlmGeneration> {
        self.commit_inner().await
    }

    async fn commit_inner(&mut self) -> crate::Result<LlmGeneration> {
        self.inner
            .commit()
            .await
            .map(|inner| LlmGeneration {
                inner,
                terminal_received: false,
            })
            .map_err(map_backend_error)
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }

    #[cfg(feature = "server-support")]
    pub fn abort(self) {
        self.inner.abort();
    }
}

impl LlmChoiceReservation {
    pub async fn commit(mut self) -> crate::Result<LlmChoiceGeneration> {
        self.inner
            .commit()
            .await
            .map(|inner| LlmChoiceGeneration { inner })
            .map_err(map_backend_error)
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn commit_reserved(&mut self) -> crate::Result<LlmChoiceGeneration> {
        self.inner
            .commit()
            .await
            .map(|inner| LlmChoiceGeneration { inner })
            .map_err(map_backend_error)
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn cancel_and_wait(&mut self) -> crate::Result<()> {
        self.inner
            .cancel_and_wait()
            .await
            .map_err(map_backend_error)
    }
}

impl LlmEngine {
    /// Loads a GGUF model synchronously.
    ///
    /// This method performs blocking model initialization. Async callers should use
    /// [`Self::load_deployment`], which offloads initialization to Tokio's blocking pool.
    /// Engines created through this compatibility API do not expose a typed model identity.
    pub fn load(model: PathBuf, config: LlmEngineConfig) -> crate::Result<Self> {
        let event_queue_capacity = config.event_queue_capacity;
        let embedding_config = match config.deployment_kind {
            LlmDeploymentKind::Generation => None,
            LlmDeploymentKind::Embeddings(config) => Some(config),
        };
        let inner = backend::Engine::load(
            model,
            backend::RuntimeConfig {
                context_size: config.context_size,
                batch_size: config.batch_size,
                micro_batch_size: config.micro_batch_size,
                threads: config.threads,
                gpu_layers: config.gpu_layers,
                parallel_sequences: config.parallel_sequences,
                request_queue_capacity: config.request_queue_capacity,
                event_queue_capacity,
                chat_template: config.chat_template,
                template_engine: match config.template_engine {
                    LlmTemplateEngine::LlamaCpp => backend::TemplateEngine::LlamaCpp,
                    LlmTemplateEngine::Jinja => backend::TemplateEngine::Jinja,
                },
                enable_thinking: config.enable_thinking,
                prompt_cache: backend::PromptCacheConfig {
                    enabled: config.prompt_cache.enabled,
                    max_entries: config.prompt_cache.max_entries,
                    max_bytes: config.prompt_cache.max_bytes,
                    min_prefix_tokens: config.prompt_cache.min_prefix_tokens,
                },
                mode: match config.deployment_kind {
                    LlmDeploymentKind::Generation => backend::RuntimeMode::Generation,
                    LlmDeploymentKind::Embeddings(embedding) => backend::RuntimeMode::Embeddings {
                        pooling: match embedding.pooling {
                            LlmEmbeddingPooling::Last => backend::EmbeddingPooling::Last,
                        },
                        max_input_tokens: embedding.max_input_tokens,
                    },
                },
                vision: config.vision.map(|vision| backend::LlmVisionConfig {
                    mmproj: vision.mmproj,
                    limits: backend::VisionLimits {
                        max_images: vision.limits.max_images,
                        max_bytes_per_image: vision.limits.max_bytes_per_image,
                        max_total_bytes: vision.limits.max_total_bytes,
                        max_side: vision.limits.max_side,
                        max_pixels_per_image: vision.limits.max_pixels_per_image,
                        max_total_pixels: vision.limits.max_total_pixels,
                    },
                }),
            },
        )
        .map_err(|error| crate::OrchionError::ModelLoad {
            message: error.to_string(),
        })?;
        Ok(Self {
            inner,
            event_queue_capacity,
            model: None,
            embedding_config,
        })
    }

    /// Loads a typed deployment synchronously on the current thread.
    pub fn load_deployment_blocking(
        deployment: LlmDeployment,
        config: LlmEngineConfig,
    ) -> crate::Result<Self> {
        let LlmDeployment { model, path } = deployment;
        let mut engine = Self::load(path, config)?;
        engine.model = Some(model);
        Ok(engine)
    }

    /// Loads a typed deployment on Tokio's blocking thread pool.
    pub async fn load_deployment(
        deployment: LlmDeployment,
        config: LlmEngineConfig,
    ) -> crate::Result<Self> {
        tokio::task::spawn_blocking(move || Self::load_deployment_blocking(deployment, config))
            .await
            .map_err(|error| crate::OrchionError::BlockingTask {
                message: error.to_string(),
            })?
    }

    /// Returns the typed identity when the engine was loaded from an [`LlmDeployment`].
    #[must_use]
    pub const fn model(&self) -> Option<&LlmModel> {
        self.model.as_ref()
    }

    pub async fn stream(&self, request: GenerationRequest) -> crate::Result<LlmGeneration> {
        self.reserve_generation(request).await?.commit().await
    }

    pub async fn stream_advanced(
        &self,
        request: LlmAdvancedRequest,
    ) -> crate::Result<LlmChoiceGeneration> {
        self.reserve_advanced(request).await?.commit().await
    }

    pub async fn reserve_advanced(
        &self,
        request: LlmAdvancedRequest,
    ) -> crate::Result<LlmChoiceReservation> {
        let backend_request = advanced_backend_request(request)?;
        let inner = match backend_request {
            AdvancedBackendRequest::Raw(request) => {
                self.inner
                    .reserve_choices(request, self.event_queue_capacity)
                    .await
            }
            AdvancedBackendRequest::Semantic(request) => {
                self.inner
                    .reserve_choice_semantic(request, self.event_queue_capacity)
                    .await
            }
        }
        .map_err(map_backend_error)?;
        Ok(LlmChoiceReservation { inner })
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn reserve(&self, request: GenerationRequest) -> crate::Result<LlmReservation> {
        self.reserve_generation(request).await
    }

    async fn reserve_generation(
        &self,
        request: GenerationRequest,
    ) -> crate::Result<LlmReservation> {
        let request = backend::Request {
            input: backend::Input::Messages(
                request
                    .messages
                    .into_iter()
                    .map(|message| backend::Message {
                        role: message.role.as_str().to_string(),
                        content: message.content,
                    })
                    .collect(),
            ),
            options: backend_options(request.options),
        };
        let inner = self
            .inner
            .reserve(request, self.event_queue_capacity)
            .await
            .map_err(map_backend_error)?;
        Ok(LlmReservation { inner })
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn reserve_prompt(
        &self,
        prompt: String,
        options: GenerationOptions,
    ) -> crate::Result<LlmReservation> {
        let inner = self
            .inner
            .reserve(
                backend::Request {
                    input: backend::Input::Prompt(prompt),
                    options: backend_options(options),
                },
                self.event_queue_capacity,
            )
            .await
            .map_err(map_backend_error)?;
        Ok(LlmReservation { inner })
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn count_input_tokens(&self, messages: Vec<LlmMessage>) -> crate::Result<usize> {
        self.inner
            .count_tokens(backend::TokenCountRequest {
                messages: messages
                    .into_iter()
                    .map(|message| backend::Message {
                        role: message.role.as_str().to_string(),
                        content: message.content,
                    })
                    .collect(),
            })
            .await
            .map_err(map_backend_error)
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn count_semantic_input_tokens(
        &self,
        request: LlmSemanticTokenCountRequest,
    ) -> crate::Result<usize> {
        self.inner
            .count_semantic_tokens(backend::SemanticTokenCountRequest {
                messages: request
                    .messages
                    .into_iter()
                    .map(backend_rich_message)
                    .collect(),
                tools: request
                    .tools
                    .into_iter()
                    .map(|tool| backend::ToolDefinition {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.parameters,
                    })
                    .collect(),
                tool_choice: match request.tool_choice {
                    LlmToolChoice::None => backend::ToolChoice::None,
                    LlmToolChoice::Auto => backend::ToolChoice::Auto,
                    LlmToolChoice::Required => backend::ToolChoice::Required,
                    LlmToolChoice::Named(name) => backend::ToolChoice::Named(name),
                },
                parallel_tool_calls: request.parallel_tool_calls,
                reasoning: backend::ReasoningOptions {
                    enabled: request.reasoning.enabled,
                    effort: request.reasoning.effort.map(|effort| match effort {
                        LlmReasoningEffort::Low => backend::ReasoningEffort::Low,
                        LlmReasoningEffort::Medium => backend::ReasoningEffort::Medium,
                        LlmReasoningEffort::High => backend::ReasoningEffort::High,
                    }),
                },
                output: match request.output {
                    LlmOutputConstraint::Text => backend::OutputConstraint::Text,
                    LlmOutputConstraint::JsonObject => backend::OutputConstraint::JsonObject,
                    LlmOutputConstraint::JsonSchema(schema) => {
                        backend::OutputConstraint::JsonSchema(schema)
                    }
                },
            })
            .await
            .map_err(map_backend_error)
    }

    pub async fn complete(&self, request: GenerationRequest) -> crate::Result<LlmComplete> {
        collect_generation(self.stream(request).await?).await
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn reserve_embedding(
        &self,
        request: LlmEmbeddingRequest,
    ) -> crate::Result<LlmEmbeddingReservation> {
        let min_dimensions = self
            .embedding_config
            .map_or(1, |config| config.min_dimensions);
        let dimensions = request.dimensions;
        let input_count = request.inputs.len();
        if dimensions.is_some_and(|value| value < min_dimensions) {
            return Err(crate::OrchionError::Inference {
                message: format!("embedding dimensions must be at least {min_dimensions}"),
            });
        }
        let inner = self
            .inner
            .reserve_embedding(backend::EmbeddingRequest {
                inputs: request
                    .inputs
                    .into_iter()
                    .map(|input| match input {
                        LlmEmbeddingInput::Text(text) => backend::EmbeddingInput::Text(text),
                        LlmEmbeddingInput::Tokens(tokens) => {
                            backend::EmbeddingInput::Tokens(tokens)
                        }
                    })
                    .collect(),
            })
            .await
            .map_err(map_backend_error)?;
        Ok(LlmEmbeddingReservation {
            inner,
            dimensions,
            min_dimensions,
            input_count,
        })
    }

    pub async fn embed(&self, request: LlmEmbeddingRequest) -> crate::Result<LlmEmbeddingResult> {
        let mut reservation = self.reserve_embedding_internal(request).await?;
        let mut operation = reservation.commit_reserved().await?;
        operation.result().await
    }

    async fn reserve_embedding_internal(
        &self,
        request: LlmEmbeddingRequest,
    ) -> crate::Result<LlmEmbeddingReservation> {
        let dimensions = request.dimensions;
        let input_count = request.inputs.len();
        let min_dimensions = self
            .embedding_config
            .map_or(1, |config| config.min_dimensions);
        let inner = self
            .inner
            .reserve_embedding(backend::EmbeddingRequest {
                inputs: request
                    .inputs
                    .into_iter()
                    .map(|input| match input {
                        LlmEmbeddingInput::Text(text) => backend::EmbeddingInput::Text(text),
                        LlmEmbeddingInput::Tokens(tokens) => {
                            backend::EmbeddingInput::Tokens(tokens)
                        }
                    })
                    .collect(),
            })
            .await
            .map_err(map_backend_error)?;
        Ok(LlmEmbeddingReservation {
            inner,
            dimensions,
            min_dimensions,
            input_count,
        })
    }

    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    #[doc(hidden)]
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
}

fn backend_options(options: GenerationOptions) -> backend::GenerationOptions {
    backend::GenerationOptions {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        min_p: options.min_p,
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        repeat_penalty: options.repeat_penalty,
        seed: options.seed,
        stop: options.stop,
    }
}

enum AdvancedBackendRequest {
    Raw(backend::AdvancedRequest),
    Semantic(backend::AdvancedSemanticRequest),
}

fn advanced_backend_request(request: LlmAdvancedRequest) -> crate::Result<AdvancedBackendRequest> {
    let output = match request.output {
        LlmOutputConstraint::Text => backend::OutputConstraint::Text,
        LlmOutputConstraint::JsonObject => backend::OutputConstraint::JsonObject,
        LlmOutputConstraint::JsonSchema(schema) => backend::OutputConstraint::JsonSchema(schema),
    };
    let logprobs = request.logprobs.map(|options| backend::LogprobsOptions {
        top_logprobs: options.top_logprobs,
    });
    let logit_bias = request
        .logit_bias
        .into_iter()
        .map(|bias| backend::LogitBias {
            token_id: bias.token_id,
            bias: bias.bias,
        })
        .collect();
    match request.input {
        LlmAdvancedInput::Prompt(prompt) => {
            if !request.tools.is_empty() || request.tool_choice != LlmToolChoice::None {
                return Err(crate::OrchionError::LlmUnsupported {
                    field: "tools",
                    detail: "raw prompts do not have a truthful tool template/parser contract"
                        .to_string(),
                });
            }
            if request.reasoning != LlmReasoningOptions::default() {
                return Err(crate::OrchionError::LlmUnsupported {
                    field: "reasoning",
                    detail: "raw prompts do not have a reasoning parser contract".to_string(),
                });
            }
            Ok(AdvancedBackendRequest::Raw(backend::AdvancedRequest {
                input: backend::Input::Prompt(prompt),
                options: backend_options(request.options),
                output,
                logprobs,
                logit_bias,
                sampling: backend::SamplingExtensions {
                    typical_p: request.sampling.typical_p,
                    top_n_sigma: request.sampling.top_n_sigma,
                },
                choices: request.choices,
                reasoning_control_id: request.reasoning_control_id,
            }))
        }
        LlmAdvancedInput::Messages(messages) => Ok(AdvancedBackendRequest::Semantic(
            backend::AdvancedSemanticRequest {
                messages: messages.into_iter().map(backend_rich_message).collect(),
                options: backend_options(request.options),
                tools: request
                    .tools
                    .into_iter()
                    .map(|tool| backend::ToolDefinition {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.parameters,
                    })
                    .collect(),
                tool_choice: match request.tool_choice {
                    LlmToolChoice::None => backend::ToolChoice::None,
                    LlmToolChoice::Auto => backend::ToolChoice::Auto,
                    LlmToolChoice::Required => backend::ToolChoice::Required,
                    LlmToolChoice::Named(name) => backend::ToolChoice::Named(name),
                },
                parallel_tool_calls: request.parallel_tool_calls,
                reasoning: backend::ReasoningOptions {
                    enabled: request.reasoning.enabled,
                    effort: request.reasoning.effort.map(|effort| match effort {
                        LlmReasoningEffort::Low => backend::ReasoningEffort::Low,
                        LlmReasoningEffort::Medium => backend::ReasoningEffort::Medium,
                        LlmReasoningEffort::High => backend::ReasoningEffort::High,
                    }),
                },
                output,
                logprobs,
                logit_bias,
                sampling: backend::SamplingExtensions {
                    typical_p: request.sampling.typical_p,
                    top_n_sigma: request.sampling.top_n_sigma,
                },
                choices: request.choices,
                reasoning_control_id: request.reasoning_control_id,
            },
        )),
    }
}

fn backend_rich_message(message: LlmRichMessage) -> backend::RichMessage {
    backend::RichMessage {
        role: match message.role {
            LlmSemanticRole::System => backend::Role::System,
            LlmSemanticRole::Developer => backend::Role::Developer,
            LlmSemanticRole::User => backend::Role::User,
            LlmSemanticRole::Assistant => backend::Role::Assistant,
            LlmSemanticRole::Tool => backend::Role::Tool,
            LlmSemanticRole::Other(role) => backend::Role::Other(role),
        },
        content: message
            .content
            .into_iter()
            .map(|part| match part {
                LlmContentPart::Text { text } => backend::ContentPart::Text { text },
                LlmContentPart::Image(image) => backend::ContentPart::Image(backend::ImageInput {
                    bytes: image.bytes,
                    format: match image.format {
                        LlmImageFormat::Png => backend::ImageFormat::Png,
                        LlmImageFormat::Jpeg => backend::ImageFormat::Jpeg,
                    },
                    width: image.width,
                    height: image.height,
                }),
                LlmContentPart::Reasoning { text } => backend::ContentPart::Reasoning { text },
                LlmContentPart::ToolResult(result) => {
                    backend::ContentPart::ToolResult(backend::ToolResult {
                        tool_call_id: result.tool_call_id,
                        content: result.content,
                        is_error: result.is_error,
                    })
                }
            })
            .collect(),
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|call| backend::ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
    }
}

fn map_token_logprobs(logprobs: backend::TokenLogprobs) -> LlmTokenLogprobs {
    LlmTokenLogprobs {
        chosen: map_token_alternative(logprobs.chosen),
        top: logprobs
            .top
            .into_iter()
            .map(map_token_alternative)
            .collect(),
    }
}

fn map_token_alternative(alternative: backend::TokenAlternative) -> LlmTokenAlternative {
    LlmTokenAlternative {
        token_id: alternative.token_id,
        bytes: alternative.bytes,
        logprob: alternative.logprob,
    }
}

const fn map_choice_finish_reason(reason: backend::FinishReason) -> LlmChoiceFinishReason {
    match reason {
        backend::FinishReason::Stop => LlmChoiceFinishReason::Stop,
        backend::FinishReason::Length => LlmChoiceFinishReason::Length,
        backend::FinishReason::Cancelled => LlmChoiceFinishReason::Cancelled,
        backend::FinishReason::ToolCalls => LlmChoiceFinishReason::ToolCalls,
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the API output is f32 while the norm intentionally accumulates in f64"
)]
fn finalize_embeddings(
    output: backend::EmbeddingOutput,
    dimensions: Option<usize>,
    min_dimensions: usize,
    input_count: usize,
) -> crate::Result<LlmEmbeddingResult> {
    let mut embeddings = output.embeddings;
    if embeddings.len() != input_count {
        return Err(crate::OrchionError::Inference {
            message: format!(
                "model returned {} embeddings for {input_count} inputs",
                embeddings.len()
            ),
        });
    }
    for embedding in &mut embeddings {
        let dimensions = dimensions.unwrap_or(embedding.len());
        if dimensions < min_dimensions || dimensions > embedding.len() {
            return Err(crate::OrchionError::Inference {
                message: format!(
                    "embedding dimensions must be in {min_dimensions}..={}",
                    embedding.len()
                ),
            });
        }
        embedding.truncate(dimensions);
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(crate::OrchionError::Inference {
                message: "model returned a non-finite embedding".to_string(),
            });
        }
        let norm = embedding
            .iter()
            .fold(0.0_f64, |sum, value| {
                f64::from(*value).mul_add(f64::from(*value), sum)
            })
            .sqrt();
        if norm > 0.0 {
            for value in embedding {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }
    Ok(LlmEmbeddingResult {
        embeddings,
        prompt_tokens: output.prompt_tokens,
        total_tokens: output.prompt_tokens,
    })
}

async fn collect_generation(mut generation: LlmGeneration) -> crate::Result<LlmComplete> {
    let mut text = String::new();
    while let Some(event) = generation.next().await? {
        match event {
            GenerationEvent::ContentDelta(delta) => text.push_str(&delta),
            GenerationEvent::Finished { reason, usage } => {
                return Ok(LlmComplete {
                    text,
                    finish_reason: reason,
                    usage,
                });
            }
        }
    }
    Err(crate::OrchionError::Inference {
        message: "LLM generation ended without a terminal event".to_string(),
    })
}

impl LlmGeneration {
    pub async fn next(&mut self) -> crate::Result<Option<GenerationEvent>> {
        if self.terminal_received {
            return Ok(None);
        }
        match self.inner.events.recv().await {
            Some(backend::Event::Content(content)) => {
                Ok(Some(GenerationEvent::ContentDelta(content)))
            }
            Some(
                backend::Event::Token { text, .. }
                | backend::Event::Semantic(backend::SemanticDelta::Text(text)),
            ) => Ok(Some(GenerationEvent::ContentDelta(text))),
            Some(backend::Event::Semantic(_)) => Err(crate::OrchionError::LlmUnsupported {
                field: "semantic_stream",
                detail: "use advanced choice streaming for reasoning and tool deltas".to_string(),
            }),
            Some(backend::Event::Finished { reason, usage }) => {
                self.terminal_received = true;
                Ok(Some(map_terminal(reason, usage)))
            }
            Some(backend::Event::Failed(message)) => {
                self.terminal_received = true;
                Err(crate::OrchionError::Inference { message })
            }
            None => {
                self.terminal_received = true;
                match self
                    .inner
                    .recv_terminal()
                    .await
                    .map_err(map_backend_error)?
                {
                    backend::Event::Finished { reason, usage } => {
                        Ok(Some(map_terminal(reason, usage)))
                    }
                    backend::Event::Failed(message) => {
                        Err(crate::OrchionError::Inference { message })
                    }
                    backend::Event::Content(content) => {
                        Ok(Some(GenerationEvent::ContentDelta(content)))
                    }
                    backend::Event::Token { text, .. }
                    | backend::Event::Semantic(backend::SemanticDelta::Text(text)) => {
                        Ok(Some(GenerationEvent::ContentDelta(text)))
                    }
                    backend::Event::Semantic(_) => Err(crate::OrchionError::LlmUnsupported {
                        field: "semantic_stream",
                        detail: "use advanced choice streaming for reasoning and tool deltas"
                            .to_string(),
                    }),
                }
            }
        }
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }
}

impl LlmChoiceGeneration {
    #[must_use]
    pub fn reasoning_control(&self) -> Option<LlmReasoningControl> {
        self.inner
            .reasoning_control()
            .map(|inner| LlmReasoningControl { inner })
    }

    pub async fn next(&mut self) -> crate::Result<Option<LlmChoiceEvent>> {
        match self.inner.events.recv().await {
            Some(backend::ChoiceEvent::Delta {
                index,
                text,
                logprobs,
            }) => Ok(Some(LlmChoiceEvent::Delta {
                index,
                text,
                logprobs: logprobs.map(map_token_logprobs),
            })),
            Some(backend::ChoiceEvent::Finished {
                index,
                reason,
                usage,
            }) => Ok(Some(LlmChoiceEvent::Finished {
                index,
                reason: map_choice_finish_reason(reason),
                usage: map_usage(usage),
            })),
            Some(backend::ChoiceEvent::SemanticDelta { index, delta }) => {
                Ok(Some(LlmChoiceEvent::SemanticDelta {
                    index,
                    delta: map_semantic_delta(delta),
                }))
            }
            Some(backend::ChoiceEvent::FinishedAll { usage }) => {
                Ok(Some(LlmChoiceEvent::FinishedAll {
                    usage: map_usage(usage),
                }))
            }
            Some(backend::ChoiceEvent::Failed { index, message }) => {
                Ok(Some(LlmChoiceEvent::Failed { index, message }))
            }
            None => Ok(None),
        }
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }
}

fn map_semantic_delta(delta: backend::SemanticDelta) -> LlmSemanticDelta {
    match delta {
        backend::SemanticDelta::Text(text) => LlmSemanticDelta::Text(text),
        backend::SemanticDelta::Reasoning(text) => LlmSemanticDelta::Reasoning(text),
        backend::SemanticDelta::ToolCall {
            index,
            id,
            name,
            arguments,
        } => LlmSemanticDelta::ToolCall {
            index,
            id,
            name,
            arguments,
        },
    }
}

fn map_terminal(reason: backend::FinishReason, usage: backend::Usage) -> GenerationEvent {
    GenerationEvent::Finished {
        reason: match reason {
            backend::FinishReason::Stop | backend::FinishReason::ToolCalls => {
                GenerationFinishReason::Stop
            }
            backend::FinishReason::Length => GenerationFinishReason::Length,
            backend::FinishReason::Cancelled => GenerationFinishReason::Cancelled,
        },
        usage: map_usage(usage),
    }
}

fn map_usage(usage: backend::Usage) -> LlmUsage {
    LlmUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        total_tokens: usage.prompt_tokens + usage.completion_tokens,
        queue_time_ms: None,
        eval_time_ms: None,
        timings: LlmTimings {
            cache_n: usage.timings.cache_n,
            prompt_n: usage.timings.prompt_n,
            prompt_ms: usage.timings.prompt_ms,
            prompt_per_token_ms: usage.timings.prompt_per_token_ms,
            prompt_per_second: usage.timings.prompt_per_second,
            predicted_n: usage.timings.predicted_n,
            predicted_ms: usage.timings.predicted_ms,
            predicted_per_token_ms: usage.timings.predicted_per_token_ms,
            predicted_per_second: usage.timings.predicted_per_second,
        },
    }
}

fn map_backend_error(error: backend::Error) -> crate::OrchionError {
    match error {
        backend::Error::ContextLimit {
            prompt_tokens,
            max_tokens,
            context_size,
        } => crate::OrchionError::LlmContextLimit {
            prompt_tokens,
            max_tokens,
            context_size,
        },
        backend::Error::WorkerPanic(message) => crate::OrchionError::LlmWorkerFailed { message },
        backend::Error::WorkerUnavailable => crate::OrchionError::LlmWorkerFailed {
            message: "worker is unavailable".to_string(),
        },
        backend::Error::Unsupported { field, detail } => {
            crate::OrchionError::LlmUnsupported { field, detail }
        }
        backend::Error::InvalidRequest { field, detail } => {
            crate::OrchionError::LlmInvalidRequest { field, detail }
        }
        other => crate::OrchionError::Inference {
            message: other.to_string(),
        },
    }
}

#[cfg(feature = "download-all")]
fn llm_deployment_artifact_plan(
    model: &LlmModel,
    source: &crate::ModelUrl,
) -> Result<crate::DeploymentArtifactPlan> {
    use crate::{
        ArtifactRole, DeploymentArtifactRequest, DeploymentArtifactSource, DownloadSource,
        ModelUrlSource,
    };

    let artifact = match source.source() {
        ModelUrlSource::File => {
            let path = PathBuf::from(source.path().expect("validated file URL has a path"));
            if path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
            {
                return Err(OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` is not a GGUF file"),
                });
            }
            let file_name = path
                .file_name()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` does not identify a file"),
                })?
                .to_string_lossy()
                .to_string();
            DeploymentArtifactRequest {
                role: ArtifactRole::LlmModel,
                source: DeploymentArtifactSource::File(path),
                repository: None,
                files: vec![file_name],
                required_source: None,
            }
        }
        ModelUrlSource::Neutral | ModelUrlSource::HuggingFace | ModelUrlSource::ModelScope => {
            let path = source.path().ok_or_else(|| OrchionError::ModelLoad {
                message: format!("LLM source `{source}` must identify an exact GGUF file"),
            })?;
            if !path.to_ascii_lowercase().ends_with(".gguf") {
                return Err(OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` is not a GGUF file"),
                });
            }
            DeploymentArtifactRequest {
                role: ArtifactRole::LlmModel,
                source: match source.source() {
                    ModelUrlSource::Neutral => DeploymentArtifactSource::Neutral,
                    ModelUrlSource::HuggingFace => DeploymentArtifactSource::HuggingFace,
                    ModelUrlSource::ModelScope => DeploymentArtifactSource::ModelScope,
                    ModelUrlSource::File => unreachable!("file source handled above"),
                },
                repository: Some(format!(
                    "{}/{}",
                    source.owner().expect("validated hub URL has an owner"),
                    source
                        .repository()
                        .expect("validated hub URL has a repository")
                )),
                files: vec![path.to_string()],
                required_source: None,
            }
        }
    };
    let neutral_candidates = vec![DownloadSource::HuggingFace, DownloadSource::ModelScope];
    let neutral_suffix = if source.source() == ModelUrlSource::Neutral {
        "|neutral-policy=huggingface,modelscope"
    } else {
        ""
    };
    Ok(crate::DeploymentArtifactPlan {
        deployment_id: model.id().clone(),
        category: crate::ModelCategory::Llm,
        source_intent: format!("model={source}|mmproj=none{neutral_suffix}"),
        artifacts: vec![artifact],
        neutral_candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> LlmModel {
        LlmModel::new(crate::ModelId::parse("acme/test-llm").unwrap())
    }

    fn temporary_gguf(contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "orchion-llm-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[tokio::test]
    async fn complete_collects_the_same_deltas_and_terminal_as_stream() {
        let usage = backend::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            reasoning_tokens: 0,
            timings: backend::Timings::default(),
        };
        let script = vec![
            backend::Event::Content("hel".to_string()),
            backend::Event::Content("lo".to_string()),
            backend::Event::Finished {
                reason: backend::FinishReason::Length,
                usage,
            },
        ];
        let mut streamed = LlmGeneration {
            inner: backend::deterministic_generation(script.clone()),
            terminal_received: false,
        };
        let mut stream_text = String::new();
        let mut terminal = None;
        while let Some(event) = streamed.next().await.unwrap() {
            match event {
                GenerationEvent::ContentDelta(delta) => stream_text.push_str(&delta),
                GenerationEvent::Finished { reason, usage } => terminal = Some((reason, usage)),
            }
        }
        let complete = collect_generation(LlmGeneration {
            inner: backend::deterministic_generation(script),
            terminal_received: false,
        })
        .await
        .unwrap();
        assert_eq!(complete.text, stream_text);
        assert_eq!(Some((complete.finish_reason, complete.usage)), terminal);
    }

    #[tokio::test]
    async fn deterministic_generation_cancel_is_idempotent() {
        let generation = LlmGeneration {
            inner: backend::deterministic_generation([]),
            terminal_received: false,
        };
        generation.cancel();
        generation.cancel();
    }

    #[tokio::test]
    async fn indexed_choice_script_maps_logprobs_terminals_and_parent_usage() {
        let usage = backend::Usage {
            prompt_tokens: 2,
            completion_tokens: 1,
            reasoning_tokens: 0,
            timings: backend::Timings::default(),
        };
        let mut generation = LlmChoiceGeneration {
            inner: backend::deterministic_choice_generation([
                backend::ChoiceEvent::Delta {
                    index: 1,
                    text: "x".to_string(),
                    logprobs: Some(backend::TokenLogprobs {
                        chosen: backend::TokenAlternative {
                            token_id: 7,
                            bytes: vec![b'x'],
                            logprob: -0.25,
                        },
                        top: vec![backend::TokenAlternative {
                            token_id: 7,
                            bytes: vec![b'x'],
                            logprob: -0.25,
                        }],
                    }),
                },
                backend::ChoiceEvent::Finished {
                    index: 1,
                    reason: backend::FinishReason::Length,
                    usage,
                },
                backend::ChoiceEvent::FinishedAll { usage },
            ]),
        };
        let LlmChoiceEvent::Delta {
            index,
            text,
            logprobs: Some(logprobs),
        } = generation.next().await.unwrap().unwrap()
        else {
            panic!("expected indexed delta");
        };
        assert_eq!((index, text), (1, "x".to_string()));
        assert_eq!(logprobs.chosen.bytes, [b'x']);
        assert_eq!(logprobs.top.len(), 1);
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::Finished {
                index: 1,
                reason: LlmChoiceFinishReason::Length,
                ..
            })
        ));
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::FinishedAll { usage }) if usage.total_tokens == 3
        ));
        assert!(generation.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn indexed_choice_script_preserves_child_and_parent_failures() {
        let mut generation = LlmChoiceGeneration {
            inner: backend::deterministic_choice_generation([
                backend::ChoiceEvent::Failed {
                    index: Some(0),
                    message: "choice failed".to_string(),
                },
                backend::ChoiceEvent::Failed {
                    index: None,
                    message: "choice failed".to_string(),
                },
            ]),
        };
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::Failed { index: Some(0), .. })
        ));
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::Failed { index: None, .. })
        ));
    }

    #[tokio::test]
    async fn indexed_semantic_script_maps_reasoning_and_tool_deltas() {
        let mut generation = LlmChoiceGeneration {
            inner: backend::deterministic_choice_generation([
                backend::ChoiceEvent::SemanticDelta {
                    index: 1,
                    delta: backend::SemanticDelta::Reasoning("thinking".into()),
                },
                backend::ChoiceEvent::SemanticDelta {
                    index: 1,
                    delta: backend::SemanticDelta::ToolCall {
                        index: 0,
                        id: Some("call_1".into()),
                        name: Some("weather".into()),
                        arguments: "{\"city\":".into(),
                    },
                },
            ]),
        };
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::SemanticDelta {
                index: 1,
                delta: LlmSemanticDelta::Reasoning(ref text),
            }) if text == "thinking"
        ));
        assert!(matches!(
            generation.next().await.unwrap(),
            Some(LlmChoiceEvent::SemanticDelta {
                index: 1,
                delta: LlmSemanticDelta::ToolCall {
                    index: 0,
                    id: Some(ref id),
                    ..
                },
            }) if id == "call_1"
        ));
    }

    #[test]
    fn advanced_facade_preserves_constraints_sampling_bias_and_choice_count() {
        let request = LlmAdvancedRequest {
            input: LlmAdvancedInput::Prompt("json".to_string()),
            options: GenerationOptions::default(),
            tools: Vec::new(),
            tool_choice: LlmToolChoice::None,
            parallel_tool_calls: false,
            reasoning: LlmReasoningOptions::default(),
            output: LlmOutputConstraint::JsonObject,
            logprobs: Some(LlmLogprobsOptions { top_logprobs: 4 }),
            logit_bias: vec![LlmLogitBias {
                token_id: 5,
                bias: 2.0,
            }],
            sampling: LlmSamplingExtensions {
                typical_p: Some(0.8),
                top_n_sigma: Some(2.0),
            },
            choices: 2,
            reasoning_control_id: None,
        };
        let AdvancedBackendRequest::Raw(mapped) = advanced_backend_request(request).unwrap() else {
            panic!("expected raw request");
        };
        assert_eq!(mapped.choices, 2);
        assert!(matches!(
            mapped.output,
            backend::OutputConstraint::JsonObject
        ));
        assert_eq!(mapped.logprobs.unwrap().top_logprobs, 4);
        assert_eq!(mapped.logit_bias[0].token_id, 5);
        assert_eq!(mapped.sampling.typical_p, Some(0.8));
    }

    #[test]
    fn raw_tool_and_reasoning_requests_are_truthfully_typed_unsupported() {
        let request = LlmAdvancedRequest {
            input: LlmAdvancedInput::Prompt("call a tool".to_string()),
            options: GenerationOptions::default(),
            tools: vec![LlmToolDefinition {
                name: "lookup".to_string(),
                description: None,
                parameters: backend::json_value!({"type":"object"}),
            }],
            tool_choice: LlmToolChoice::Auto,
            parallel_tool_calls: false,
            reasoning: LlmReasoningOptions::default(),
            output: LlmOutputConstraint::Text,
            logprobs: None,
            logit_bias: Vec::new(),
            sampling: LlmSamplingExtensions::default(),
            choices: 1,
            reasoning_control_id: None,
        };
        assert!(matches!(
            advanced_backend_request(request),
            Err(OrchionError::LlmUnsupported { field: "tools", .. })
        ));
    }

    #[test]
    fn engine_config_default_matches_server_runtime_defaults() {
        let config = LlmEngineConfig::default();
        assert_eq!(config.context_size, None);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.micro_batch_size, 512);
        assert_eq!(config.threads, 0);
        assert_eq!(config.gpu_layers, u32::MAX);
        assert_eq!(config.parallel_sequences, 1);
        assert_eq!(config.request_queue_capacity, 8);
        assert_eq!(config.event_queue_capacity, 16);
        assert_eq!(config.chat_template, None);
        assert_eq!(config.template_engine, LlmTemplateEngine::LlamaCpp);
        assert!(!config.enable_thinking);
        assert_eq!(config.deployment_kind, LlmDeploymentKind::Generation);
    }

    #[test]
    fn embedding_postprocessing_truncates_then_normalizes_and_preserves_zero_vectors() {
        let result = finalize_embeddings(
            backend::EmbeddingOutput {
                embeddings: vec![vec![3.0, 4.0, 100.0], vec![0.0, 0.0, 1.0]],
                prompt_tokens: 7,
            },
            Some(2),
            1,
            2,
        )
        .unwrap();
        assert_eq!(result.embeddings, vec![vec![0.6, 0.8], vec![0.0, 0.0]]);
        assert_eq!(result.prompt_tokens, 7);
        assert_eq!(result.total_tokens, 7);
    }

    #[test]
    fn embedding_postprocessing_rejects_dimensions_and_nonfinite_values() {
        for (embedding, dimensions) in [(vec![1.0, 2.0], Some(3)), (vec![1.0, f32::NAN], None)] {
            assert!(
                finalize_embeddings(
                    backend::EmbeddingOutput {
                        embeddings: vec![embedding],
                        prompt_tokens: 1,
                    },
                    dimensions,
                    1,
                    1,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn deployment_checks_file_and_preserves_identity_and_path() {
        let path = temporary_gguf(b"not a real model");
        let model = test_model();
        let deployment = LlmDeployment::from_file(model.clone(), path.clone()).unwrap();
        assert_eq!(deployment.model(), &model);
        assert_eq!(deployment.path(), path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(
            LlmDeployment::from_file(test_model(), "/definitely/missing/model.gguf"),
            Err(OrchionError::ModelLoad { .. })
        ));
    }

    #[tokio::test]
    async fn typed_load_maps_invalid_gguf_to_model_load_without_nested_runtime() {
        let path = temporary_gguf(b"not a real model");
        let deployment = LlmDeployment::from_file(test_model(), path.clone()).unwrap();
        let result = LlmEngine::load_deployment(deployment, LlmEngineConfig::default()).await;
        assert!(matches!(result, Err(OrchionError::ModelLoad { .. })));
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "download-all")]
    #[test]
    fn llm_plan_preserves_exact_sources_roles_and_identity() {
        use crate::{
            ArtifactRole, DeploymentArtifactSource, DownloadSource, ModelCategory, ModelUrl,
        };

        let cases = [
            (
                "hf://owner/repo/models/main.gguf",
                DeploymentArtifactSource::HuggingFace,
                Some("owner/repo"),
                "models/main.gguf",
            ),
            (
                "ms://owner/repo/main.gguf",
                DeploymentArtifactSource::ModelScope,
                Some("owner/repo"),
                "main.gguf",
            ),
            (
                "//owner/repo/main.gguf",
                DeploymentArtifactSource::Neutral,
                Some("owner/repo"),
                "main.gguf",
            ),
            (
                "file:///tmp/main.gguf",
                DeploymentArtifactSource::File(PathBuf::from("/tmp/main.gguf")),
                None,
                "main.gguf",
            ),
        ];
        for (url, expected_source, repository, file) in cases {
            let model = test_model();
            let source = ModelUrl::parse(url).unwrap();
            let plan = llm_deployment_artifact_plan(&model, &source).unwrap();
            assert_eq!(plan.deployment_id, *model.id());
            assert_eq!(plan.category, ModelCategory::Llm);
            assert!(plan.source_intent.contains("mmproj=none"));
            assert_eq!(
                plan.neutral_candidates,
                vec![DownloadSource::HuggingFace, DownloadSource::ModelScope]
            );
            assert_eq!(plan.artifacts.len(), 1);
            let artifact = &plan.artifacts[0];
            assert_eq!(artifact.role, ArtifactRole::LlmModel);
            assert_eq!(artifact.source, expected_source);
            assert_eq!(artifact.repository.as_deref(), repository);
            assert_eq!(artifact.files, [file]);
        }
    }

    #[cfg(feature = "download-all")]
    #[test]
    fn llm_plan_rejects_repository_only_and_non_gguf_hub_sources() {
        for source in [
            "//owner/repo",
            "hf://owner/repo/model.bin",
            "file:///tmp/model.bin",
        ] {
            let source = crate::ModelUrl::parse(source).unwrap();
            assert!(matches!(
                llm_deployment_artifact_plan(&test_model(), &source),
                Err(OrchionError::ModelLoad { .. })
            ));
        }
    }

    #[tokio::test]
    #[ignore = "requires ORCHION_TEST_EMBEDDING_GGUF pointing to a real embedding GGUF"]
    async fn real_embedding_model_produces_finite_ordered_unit_vectors() {
        let path = PathBuf::from(
            std::env::var("ORCHION_TEST_EMBEDDING_GGUF")
                .expect("ORCHION_TEST_EMBEDDING_GGUF must be set"),
        );
        let engine = LlmEngine::load(
            path,
            LlmEngineConfig {
                context_size: NonZeroU32::new(8192),
                batch_size: 8192,
                micro_batch_size: 8192,
                threads: 2,
                gpu_layers: 0,
                deployment_kind: LlmDeploymentKind::Embeddings(LlmEmbeddingConfig {
                    pooling: LlmEmbeddingPooling::Last,
                    min_dimensions: 32,
                    max_input_tokens: 8192,
                }),
                ..LlmEngineConfig::default()
            },
        )
        .unwrap();
        let result = engine
            .embed(LlmEmbeddingRequest {
                inputs: vec![
                    LlmEmbeddingInput::Text("A cat sits on a mat.".to_string()),
                    LlmEmbeddingInput::Text("A cat sits on a mat.".to_string()),
                    LlmEmbeddingInput::Text("A kitten rests on a rug.".to_string()),
                    LlmEmbeddingInput::Text("Database indexes speed up queries.".to_string()),
                ],
                dimensions: Some(64),
            })
            .await
            .unwrap();
        assert!(result.prompt_tokens > 0);
        assert_eq!(result.total_tokens, result.prompt_tokens);
        for embedding in &result.embeddings {
            assert_eq!(embedding.len(), 64);
            assert!(embedding.iter().all(|value| value.is_finite()));
            let norm = embedding
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
        assert_eq!(result.embeddings[0], result.embeddings[1]);
        let cosine = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>()
        };
        assert!(
            cosine(&result.embeddings[0], &result.embeddings[2])
                > cosine(&result.embeddings[0], &result.embeddings[3])
        );
        engine.shutdown();
    }
}
