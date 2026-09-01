use std::num::NonZeroU32;
use std::path::PathBuf;

/// Maximum bounded event channel size accepted by the native scheduler.
pub const MAX_EVENT_CAPACITY: usize = 4096;

pub const BINDING_REVISION: &str = "0ad1788017e39de25c431d9323f90a4b4b538d9e";
pub const LLAMA_CPP_REVISION: &str = "e79e4bf660e19f2ad851e06c6913f7a8c5852621";

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BuildMetadata {
    pub binding_revision: &'static str,
    pub llama_cpp_revision: &'static str,
    pub binding_features: &'static str,
    pub cargo_features: &'static str,
    pub rustc_version: &'static str,
    pub rustc_verbose_version: &'static str,
    pub toolchain: &'static str,
    pub target: &'static str,
    pub profile: &'static str,
    pub common_chat_bridge: bool,
    pub cmake_input: CmakeInputMetadata,
    pub cmake_resolved: ResolvedCmakeBuildMetadata,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CmakeInputMetadata {
    pub ggml_metal: &'static str,
    pub ggml_cuda: &'static str,
    pub ggml_openmp: &'static str,
    pub build_type: &'static str,
    pub generator: &'static str,
    pub osx_deployment_target: &'static str,
    pub macosx_deployment_target: &'static str,
    pub toolchain_file: &'static str,
    pub cuda_compute_cap: &'static str,
    pub llama_build_shared_libs: &'static str,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ResolvedCmakeBuildMetadata {
    pub cache_path_relative: &'static str,
    pub cache_sha256: &'static str,
    pub build_type: &'static str,
    pub generator: &'static str,
    pub osx_deployment_target: &'static str,
    pub build_shared_libs: &'static str,
    pub ggml_metal: &'static str,
    pub ggml_openmp: &'static str,
    pub ggml_cuda: &'static str,
    pub ggml_vulkan: &'static str,
    pub ggml_native: &'static str,
    pub c_compiler: CompilerBuildMetadata,
    pub cxx_compiler: CompilerBuildMetadata,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CompilerBuildMetadata {
    pub basename: &'static str,
    pub id: &'static str,
    pub version: &'static str,
}

#[must_use]
pub const fn build_metadata() -> BuildMetadata {
    BuildMetadata {
        binding_revision: BINDING_REVISION,
        llama_cpp_revision: LLAMA_CPP_REVISION,
        binding_features: "common,mtmd",
        cargo_features: env!("ORCHION_LLAMA_CARGO_FEATURES"),
        rustc_version: env!("ORCHION_RUSTC_VERSION"),
        rustc_verbose_version: env!("ORCHION_RUSTC_VERBOSE_VERSION"),
        toolchain: env!("ORCHION_RUST_TOOLCHAIN"),
        target: env!("TARGET"),
        profile: env!("PROFILE"),
        common_chat_bridge: true,
        cmake_input: CmakeInputMetadata {
            ggml_metal: env!("ORCHION_BUILD_INPUT_GGML_METAL"),
            ggml_cuda: env!("ORCHION_BUILD_INPUT_GGML_CUDA"),
            ggml_openmp: env!("ORCHION_BUILD_INPUT_GGML_OPENMP"),
            build_type: env!("ORCHION_BUILD_INPUT_CMAKE_BUILD_TYPE"),
            generator: env!("ORCHION_BUILD_INPUT_CMAKE_GENERATOR"),
            osx_deployment_target: env!("ORCHION_BUILD_INPUT_CMAKE_OSX_DEPLOYMENT_TARGET"),
            macosx_deployment_target: env!("ORCHION_BUILD_INPUT_MACOSX_DEPLOYMENT_TARGET"),
            toolchain_file: env!("ORCHION_BUILD_INPUT_CMAKE_TOOLCHAIN_FILE"),
            cuda_compute_cap: env!("ORCHION_BUILD_INPUT_CUDA_COMPUTE_CAP"),
            llama_build_shared_libs: env!("ORCHION_BUILD_INPUT_LLAMA_BUILD_SHARED_LIBS"),
        },
        cmake_resolved: ResolvedCmakeBuildMetadata {
            cache_path_relative: env!("ORCHION_BUILD_CMAKE_CACHE_RELATIVE_PATH"),
            cache_sha256: env!("ORCHION_BUILD_CMAKE_CACHE_SHA256"),
            build_type: env!("ORCHION_BUILD_RESOLVED_CMAKE_BUILD_TYPE"),
            generator: env!("ORCHION_BUILD_RESOLVED_CMAKE_GENERATOR"),
            osx_deployment_target: env!("ORCHION_BUILD_RESOLVED_CMAKE_OSX_DEPLOYMENT_TARGET"),
            build_shared_libs: env!("ORCHION_BUILD_RESOLVED_BUILD_SHARED_LIBS"),
            ggml_metal: env!("ORCHION_BUILD_RESOLVED_GGML_METAL"),
            ggml_openmp: env!("ORCHION_BUILD_RESOLVED_GGML_OPENMP"),
            ggml_cuda: env!("ORCHION_BUILD_RESOLVED_GGML_CUDA"),
            ggml_vulkan: env!("ORCHION_BUILD_RESOLVED_GGML_VULKAN"),
            ggml_native: env!("ORCHION_BUILD_RESOLVED_GGML_NATIVE"),
            c_compiler: CompilerBuildMetadata {
                basename: env!("ORCHION_BUILD_C_COMPILER"),
                id: env!("ORCHION_BUILD_C_COMPILER_ID"),
                version: env!("ORCHION_BUILD_C_COMPILER_VERSION"),
            },
            cxx_compiler: CompilerBuildMetadata {
                basename: env!("ORCHION_BUILD_CXX_COMPILER"),
                id: env!("ORCHION_BUILD_CXX_COMPILER_ID"),
                version: env!("ORCHION_BUILD_CXX_COMPILER_VERSION"),
            },
        },
    }
}

#[must_use]
pub fn build_metadata_json() -> String {
    serde_json::to_string_pretty(&build_metadata()).unwrap_or_else(|error| {
        serde_json::json!({"metadata_error": error.to_string()}).to_string()
    })
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl Role {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role,
        }
    }
}

impl From<String> for Role {
    fn from(role: String) -> Self {
        match role.as_str() {
            "system" => Self::System,
            "developer" => Self::Developer,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            _ => Self::Other(role),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaType {
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct MediaPlaceholder {
    pub media_type: MediaType,
    pub id: String,
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image(ImageInput),
    Reasoning { text: String },
    Media(MediaPlaceholder),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RichMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    None,
    Auto,
    Required,
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub struct ReasoningOptions {
    /// `None` inherits the deployment default; `Some(false)` explicitly disables reasoning.
    pub enabled: Option<bool>,
    pub effort: Option<ReasoningEffort>,
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

#[derive(Debug, Clone, PartialEq, Default)]
pub enum OutputConstraint {
    #[default]
    Text,
    JsonObject,
    JsonSchema(serde_json::Value),
    #[doc(hidden)]
    Grammar(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogprobsOptions {
    pub top_logprobs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SamplingExtensions {
    pub typical_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Messages(Vec<Message>),
    Prompt(String),
    #[doc(hidden)]
    Semantic(Box<SemanticInput>),
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticInput {
    pub messages: Vec<RichMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub input: Input,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedRequest {
    pub input: Input,
    pub options: GenerationOptions,
    pub output: OutputConstraint,
    pub logprobs: Option<LogprobsOptions>,
    pub logit_bias: Vec<LogitBias>,
    pub sampling: SamplingExtensions,
    pub choices: usize,
    pub reasoning_control_id: Option<String>,
}

impl From<Request> for AdvancedRequest {
    fn from(request: Request) -> Self {
        Self {
            input: request.input,
            options: request.options,
            output: OutputConstraint::Text,
            logprobs: None,
            logit_bias: Vec::new(),
            sampling: SamplingExtensions::default(),
            choices: 1,
            reasoning_control_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticRequest {
    pub messages: Vec<RichMessage>,
    pub options: GenerationOptions,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedSemanticRequest {
    pub messages: Vec<RichMessage>,
    pub options: GenerationOptions,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningOptions,
    pub output: OutputConstraint,
    pub logprobs: Option<LogprobsOptions>,
    pub logit_bias: Vec<LogitBias>,
    pub sampling: SamplingExtensions,
    pub choices: usize,
    pub reasoning_control_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenCountRequest {
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticTokenCountRequest {
    pub messages: Vec<RichMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning: ReasoningOptions,
    pub output: OutputConstraint,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<i32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRequest {
    pub inputs: Vec<EmbeddingInput>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenAlternative {
    pub token_id: i32,
    pub bytes: Vec<u8>,
    pub logprob: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    pub chosen: TokenAlternative,
    pub top: Vec<TokenAlternative>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Timings {
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

impl Default for Timings {
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

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub timings: Timings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Content(String),
    Token {
        text: String,
        logprobs: TokenLogprobs,
    },
    Semantic(SemanticDelta),
    Finished {
        reason: FinishReason,
        usage: Usage,
    },
    Failed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChoiceEvent {
    Delta {
        index: usize,
        text: String,
        logprobs: Option<TokenLogprobs>,
    },
    SemanticDelta {
        index: usize,
        delta: SemanticDelta,
    },
    Finished {
        index: usize,
        reason: FinishReason,
        usage: Usage,
    },
    FinishedAll {
        usage: Usage,
    },
    Failed {
        index: Option<usize>,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticDelta {
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
pub enum SemanticEvent {
    Delta(SemanticDelta),
    Finished { reason: FinishReason, usage: Usage },
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_bytes: usize,
    pub min_prefix_tokens: usize,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: 4,
            max_bytes: 268_435_456,
            min_prefix_tokens: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub context_size: Option<NonZeroU32>,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub threads: i32,
    pub gpu_layers: u32,
    pub parallel_sequences: u32,
    pub request_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub chat_template: Option<String>,
    pub template_engine: TemplateEngine,
    pub enable_thinking: bool,
    pub prompt_cache: PromptCacheConfig,
    pub mode: RuntimeMode,
    pub vision: Option<LlmVisionConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmVisionConfig {
    pub mmproj: PathBuf,
    pub limits: VisionLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionLimits {
    pub max_images: usize,
    pub max_bytes_per_image: usize,
    pub max_total_bytes: usize,
    pub max_side: u32,
    pub max_pixels_per_image: u64,
    pub max_total_pixels: u64,
}

impl Default for VisionLimits {
    fn default() -> Self {
        Self {
            max_images: 4,
            max_bytes_per_image: 10 * 1024 * 1024,
            max_total_bytes: 20 * 1024 * 1024,
            max_side: 8192,
            max_pixels_per_image: 16_777_216,
            max_total_pixels: 33_554_432,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Generation,
    Embeddings {
        pooling: EmbeddingPooling,
        max_input_tokens: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPooling {
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateEngine {
    LlamaCpp,
    Jinja,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("failed to initialize llama.cpp backend: {0}")]
    Backend(String),
    #[error("failed to start llama.cpp model worker: {0}")]
    WorkerStart(String),
    #[error("llama.cpp model worker is unavailable")]
    WorkerUnavailable,
    #[error("invalid llama.cpp runtime configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid generation field `{field}`: {detail}")]
    InvalidRequest { field: &'static str, detail: String },
    #[error("unsupported semantic field `{field}`: {detail}")]
    Unsupported { field: &'static str, detail: String },
    #[error(
        "prompt ({prompt_tokens} tokens) plus completion ({max_tokens} tokens) exceeds context size {context_size}"
    )]
    ContextLimit {
        prompt_tokens: usize,
        max_tokens: usize,
        context_size: usize,
    },
    #[error("generation was cancelled before worker commit")]
    Cancelled,
    #[error("llama.cpp model worker panicked: {0}")]
    WorkerPanic(String),
    #[error("llama.cpp generation failed: {0}")]
    Generation(String),
    #[error("llama.cpp embedding failed: {0}")]
    Embedding(String),
}

impl From<Event> for SemanticEvent {
    fn from(event: Event) -> Self {
        match event {
            Event::Content(text) | Event::Token { text, .. } => {
                Self::Delta(SemanticDelta::Text(text))
            }
            Event::Semantic(delta) => Self::Delta(delta),
            Event::Finished { reason, usage } => Self::Finished { reason, usage },
            Event::Failed(message) => Self::Failed(message),
        }
    }
}
