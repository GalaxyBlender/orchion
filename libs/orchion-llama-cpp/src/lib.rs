#![allow(clippy::missing_errors_doc)]

//! Safe llama.cpp model execution behind a bounded cooperative scheduler.

mod common_chat;
mod constraints;
mod contract;
mod multimodal;
mod prefix_cache;
mod scheduler;
mod slot;
mod template;
mod test_support;
mod worker;

pub use constraints::validate_strict_schema;
pub use contract::{
    AdvancedRequest, AdvancedSemanticRequest, BINDING_REVISION, BuildMetadata, ChoiceEvent,
    CmakeInputMetadata, CompilerBuildMetadata, ContentPart, EmbeddingInput, EmbeddingOutput,
    EmbeddingPooling, EmbeddingRequest, Error, Event, FinishReason, GenerationOptions, ImageFormat,
    ImageInput, Input, LLAMA_CPP_REVISION, LlmVisionConfig, LogitBias, LogprobsOptions,
    MAX_EVENT_CAPACITY, MediaPlaceholder, MediaType, Message, OutputConstraint, PromptCacheConfig,
    ReasoningEffort, ReasoningOptions, Request, RichMessage, Role, RuntimeConfig, RuntimeMode,
    SamplingExtensions, SemanticDelta, SemanticEvent, SemanticInput, SemanticRequest,
    SemanticTokenCountRequest, TemplateEngine, Timings, TokenAlternative, TokenCountRequest,
    TokenLogprobs, ToolCall, ToolChoice, ToolDefinition, ToolResult, Usage, VisionLimits,
    build_metadata, build_metadata_json,
};
pub use serde_json::{Value as JsonValue, json as json_value};
pub use worker::{
    BackendOwner, ChoiceGeneration, ChoiceReservation, Embedding, EmbeddingReservation, Engine,
    Generation, ReasoningControlAttempt, ReasoningControlCancellation, ReasoningControlHandle,
    ReasoningControlResult, Reservation,
};

#[doc(hidden)]
pub use test_support::{
    SchedulerInstrumentation, ScriptedControl, deterministic_choice_generation,
    deterministic_generation, reset_scheduler_instrumentation, scheduler_instrumentation,
    scripted_context_limit_engine, scripted_embedding_engine, scripted_engine,
    scripted_preparation_panicking_engine, scripted_slow_preparation_engine,
};
