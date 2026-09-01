#![allow(
    clippy::needless_pass_by_value,
    clippy::struct_field_names,
    clippy::large_stack_arrays,
    reason = "wire DTO values are consumed into owned JSON/SSE frames and retain protocol field names"
)]

use crate::api::activity::{ActivityContext, ActivityOutcome};
use crate::api::chat_controls::{ApplyResult, ChatControls, Registration};
use crate::api::http_shared::authorize;
use crate::api::llm_streams::{LlmStreams, StartError, StreamProtocol, StreamTerminalSignal};
use crate::api::openai::{ApiError, ErrorBody};
use crate::api::sse;
use crate::application::LlmVisionPolicy;
use crate::application::llm::ManagedChoiceCancellation;
use crate::application::llm::{LlmGenerationOverrides, ManagedChoiceGeneration};
use crate::application::{RuntimeError, ServerApplication, UseCaseError};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{Extension, State};
use axum::http::HeaderMap;
#[cfg(test)]
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use orchion::{
    GenerationOptions, LlmAdvancedInput, LlmAdvancedRequest, LlmChoiceEvent, LlmChoiceFinishReason,
    LlmContentPart, LlmImageFormat, LlmImageInput, LlmLogitBias, LlmLogprobsOptions,
    LlmOutputConstraint, LlmReasoningEffort, LlmReasoningOptions, LlmRichMessage,
    LlmSamplingExtensions, LlmSemanticDelta, LlmSemanticRole, LlmTimings, LlmTokenLogprobs,
    LlmToolCall, LlmToolChoice, LlmToolDefinition, LlmToolResult, LlmUsage,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use utoipa::ToSchema;

const MIN_RESPONSES_MAX_OUTPUT_TOKENS: usize = 16;
const MAX_EMBEDDING_INPUTS: usize = 2048;
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_IMAGES: usize = 32;
const MAX_TOTAL_IMAGE_BYTES: usize = 128 * 1024 * 1024;
const MAX_IMAGE_SIDE: u32 = 16_384;
const MAX_IMAGE_PIXELS: u64 = 67_108_864;
const MAX_TOTAL_IMAGE_PIXELS: u64 = 134_217_728;

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub reasoning_control: bool,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream_options: Option<ChatStreamOptions>,
    #[serde(default)]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoiceValue>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub functions: UnsupportedField,
    #[serde(default)]
    pub function_call: UnsupportedField,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    pub modalities: UnsupportedField,
    #[serde(default)]
    pub audio: UnsupportedField,
    #[serde(default)]
    pub store: UnsupportedField,
    #[serde(default)]
    pub metadata: UnsupportedField,
    #[serde(default)]
    pub service_tier: UnsupportedField,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub prediction: UnsupportedField,
    #[serde(default)]
    pub user: UnsupportedField,
    #[serde(default)]
    pub web_search_options: UnsupportedField,
    #[serde(default)]
    pub prompt_cache_key: UnsupportedField,
    #[serde(default)]
    pub prompt_cache_retention: UnsupportedField,
    #[serde(default)]
    pub safety_identifier: UnsupportedField,
    #[serde(default)]
    pub verbosity: UnsupportedField,
    #[serde(default)]
    pub moderation: UnsupportedField,
    #[serde(default)]
    pub prompt_cache_options: UnsupportedField,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatReasoningControlRequest {
    pub id: String,
    pub action: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatReasoningControlResponse {
    pub id: String,
    pub action: &'static str,
    pub success: bool,
    #[schema(required = true, nullable)]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatContent>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct FunctionTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

impl<'de> Deserialize<'de> for FunctionTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireTool {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            function: Option<FunctionDefinition>,
        }
        let value = WireTool::deserialize(deserializer)?;
        Ok(Self {
            kind: value.kind,
            function: value.function.unwrap_or(FunctionDefinition {
                name: String::new(),
                description: None,
                parameters: empty_object(),
                strict: None,
            }),
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "empty_object")]
    pub parameters: Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatFunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolChoiceValue {
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub function: Option<NamedFunctionChoice>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct NamedFunctionChoice {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaFormat>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub description: Option<String>,
    pub schema: Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatTextPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatTextPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ChatImageUrl>,
    #[serde(default)]
    pub input_audio: Option<Value>,
    #[serde(default)]
    pub file: Option<Value>,
    #[serde(default)]
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageObject,
    pub timings: TimingObject,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
    #[schema(required = true, nullable, value_type = Object)]
    pub logprobs: Option<Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AssistantMessage {
    pub role: &'static str,
    #[schema(required = true, nullable)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    #[schema(required = true, nullable)]
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, ToSchema)]
pub struct UsageObject {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    #[serde(default)]
    pub encoding_format: EmbeddingEncodingFormat,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<i32>),
    TokenBatches(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Copy, Default, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: EmbeddingValue,
    pub index: usize,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub suffix: UnsupportedField,
    #[serde(default)]
    pub best_of: UnsupportedField,
    #[serde(default)]
    pub logprobs: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageObject,
    pub timings: TimingObject,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    #[schema(required = true, nullable, value_type = Object)]
    pub logprobs: Option<Value>,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize, ToSchema)]
pub struct TimingObject {
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

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    pub store: Option<bool>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    #[schema(minimum = 16)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<ResponsesStreamOptions>,
    #[serde(default)]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoiceValue>,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    pub background: UnsupportedField,
    #[serde(default)]
    pub conversation: UnsupportedField,
    #[serde(default)]
    pub previous_response_id: UnsupportedField,
    #[serde(default)]
    pub context_management: UnsupportedField,
    #[serde(default)]
    pub include: UnsupportedField,
    #[serde(default)]
    pub metadata: UnsupportedField,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub text: Option<ResponsesText>,
    #[serde(default)]
    pub truncation: UnsupportedField,
    #[serde(default)]
    pub service_tier: UnsupportedField,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub prompt_cache_key: UnsupportedField,
    #[serde(default)]
    pub prompt_cache_retention: UnsupportedField,
    #[serde(default)]
    pub safety_identifier: UnsupportedField,
    #[serde(default)]
    pub user: UnsupportedField,
    #[serde(default)]
    pub max_tool_calls: UnsupportedField,
    #[serde(default)]
    pub moderation: UnsupportedField,
    #[serde(default)]
    pub prompt_cache_options: UnsupportedField,
    #[serde(default)]
    pub prompt: UnsupportedField,
}

#[derive(Debug, Clone, Default, ToSchema)]
pub struct UnsupportedField {
    present: bool,
}

impl UnsupportedField {
    const fn is_present(&self) -> bool {
        self.present
    }
}

impl<'de> Deserialize<'de> for UnsupportedField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<serde::de::IgnoredAny>::deserialize(deserializer)?;
        Ok(Self {
            present: value.is_some(),
        })
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseInputItem {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<ResponseItemContent>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<Value>,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub summary: Option<Vec<ResponseSummaryPart>>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseSummaryPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: UnsupportedField,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ResponsesText {
    #[serde(default)]
    pub format: Option<ResponseTextFormat>,
    #[serde(default)]
    pub verbosity: UnsupportedField,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct ResponseTextFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub schema: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseItemContent {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponseInputPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(default)]
    pub file_id: Option<String>,
    #[serde(default)]
    #[schema(value_type = ResponseImageDetail)]
    pub detail: Option<RuntimeResponseImageDetail>,
}

#[derive(Debug, Clone, Copy, ToSchema)]
#[serde(rename_all = "lowercase")]
#[allow(dead_code, reason = "published OpenAPI-only enum")]
pub enum ResponseImageDetail {
    Auto,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeResponseImageDetail {
    Auto,
    Low,
    High,
    Original,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResponsesStreamOptions {
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: &'static str,
    pub background: bool,
    #[schema(required = true, nullable, value_type = Object)]
    pub error: Option<Value>,
    #[schema(required = true, nullable, value_type = Object)]
    pub incomplete_details: Option<Value>,
    #[schema(required = true, nullable)]
    pub instructions: Option<String>,
    #[schema(required = true, nullable)]
    pub max_output_tokens: Option<usize>,
    pub model: String,
    pub output: Vec<Value>,
    pub output_text: String,
    pub parallel_tool_calls: bool,
    #[schema(required = true, nullable)]
    pub previous_response_id: Option<String>,
    pub reasoning: Value,
    pub store: bool,
    #[schema(required = true, nullable)]
    pub temperature: Option<f32>,
    pub text: Value,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub top_logprobs: usize,
    #[schema(required = true, nullable)]
    pub top_p: Option<f32>,
    pub truncation: &'static str,
    pub metadata: Value,
    #[schema(required = true, nullable)]
    pub usage: Option<ResponsesUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<TimingObject>,
}

#[derive(ToSchema)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum ChatCompletionSseEvent {
    Chunk(ChatCompletionStreamChunk),
    Error(ErrorBody),
}

#[derive(ToSchema)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum CompletionSseEvent {
    Chunk(CompletionStreamChunk),
    Error(ErrorBody),
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct CompletionStreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionStreamChoice>,
    #[schema(required = true, nullable)]
    usage: Option<UsageObject>,
    #[schema(required = true, nullable)]
    timings: Option<TimingObject>,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct CompletionStreamChoice {
    text: String,
    index: usize,
    #[schema(required = true, nullable, value_type = Object)]
    logprobs: Option<Value>,
    #[schema(required = true, nullable)]
    finish_reason: Option<String>,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ChatCompletionStreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionStreamChoice>,
    #[schema(required = true, nullable)]
    usage: Option<UsageObject>,
    timings: Option<TimingObject>,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ChatCompletionStreamChoice {
    index: usize,
    delta: Value,
    #[schema(required = true, nullable, value_type = Object)]
    logprobs: Option<Value>,
    #[schema(required = true, nullable)]
    finish_reason: Option<String>,
}

#[derive(ToSchema)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum ResponsesSseEvent {
    Created(ResponseCreatedSseEvent),
    InProgress(ResponseInProgressSseEvent),
    OutputItemAdded(ResponseOutputItemAddedSseEvent),
    ContentPartAdded(ResponseContentPartAddedSseEvent),
    OutputTextDelta(ResponseOutputTextDeltaSseEvent),
    OutputTextDone(ResponseOutputTextDoneSseEvent),
    ContentPartDone(ResponseContentPartDoneSseEvent),
    FunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaSseEvent),
    FunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneSseEvent),
    ReasoningSummaryPartAdded(ResponseReasoningSummaryPartAddedSseEvent),
    ReasoningSummaryTextDelta(ResponseReasoningSummaryTextDeltaSseEvent),
    ReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneSseEvent),
    ReasoningSummaryPartDone(ResponseReasoningSummaryPartDoneSseEvent),
    OutputItemDone(ResponseOutputItemDoneSseEvent),
    Completed(ResponseCompletedSseEvent),
    Incomplete(ResponseIncompleteSseEvent),
    Error(LlmStreamErrorEvent),
}

macro_rules! response_sse_event_type {
    ($name:ident, $variant:ident, $value:literal) => {
        #[derive(ToSchema)]
        #[allow(dead_code)]
        pub enum $name {
            #[schema(rename = $value)]
            $variant,
        }
    };
}

response_sse_event_type!(ResponseCreatedSseEventType, Created, "response.created");
response_sse_event_type!(
    ResponseInProgressSseEventType,
    InProgress,
    "response.in_progress"
);
response_sse_event_type!(
    ResponseOutputItemAddedSseEventType,
    OutputItemAdded,
    "response.output_item.added"
);
response_sse_event_type!(
    ResponseContentPartAddedSseEventType,
    ContentPartAdded,
    "response.content_part.added"
);
response_sse_event_type!(
    ResponseOutputTextDeltaSseEventType,
    OutputTextDelta,
    "response.output_text.delta"
);
response_sse_event_type!(
    ResponseOutputTextDoneSseEventType,
    OutputTextDone,
    "response.output_text.done"
);
response_sse_event_type!(
    ResponseContentPartDoneSseEventType,
    ContentPartDone,
    "response.content_part.done"
);
response_sse_event_type!(
    ResponseOutputItemDoneSseEventType,
    OutputItemDone,
    "response.output_item.done"
);
response_sse_event_type!(
    ResponseCompletedSseEventType,
    Completed,
    "response.completed"
);
response_sse_event_type!(
    ResponseIncompleteSseEventType,
    Incomplete,
    "response.incomplete"
);
response_sse_event_type!(LlmStreamErrorEventType, Error, "error");
response_sse_event_type!(
    ResponseFunctionCallArgumentsDeltaSseEventType,
    Delta,
    "response.function_call_arguments.delta"
);
response_sse_event_type!(
    ResponseFunctionCallArgumentsDoneSseEventType,
    Done,
    "response.function_call_arguments.done"
);
response_sse_event_type!(
    ResponseReasoningSummaryPartAddedSseEventType,
    Added,
    "response.reasoning_summary_part.added"
);
response_sse_event_type!(
    ResponseReasoningSummaryTextDeltaSseEventType,
    Delta,
    "response.reasoning_summary_text.delta"
);
response_sse_event_type!(
    ResponseReasoningSummaryTextDoneSseEventType,
    Done,
    "response.reasoning_summary_text.done"
);
response_sse_event_type!(
    ResponseReasoningSummaryPartDoneSseEventType,
    Done,
    "response.reasoning_summary_part.done"
);

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseCreatedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseCreatedSseEventType,
    response: ResponsesResponse,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseInProgressSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseInProgressSseEventType,
    response: ResponsesResponse,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseOutputItemAddedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseOutputItemAddedSseEventType,
    output_index: usize,
    item: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseContentPartAddedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseContentPartAddedSseEventType,
    item_id: String,
    output_index: usize,
    content_index: usize,
    part: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseOutputTextDeltaSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseOutputTextDeltaSseEventType,
    item_id: String,
    output_index: usize,
    content_index: usize,
    delta: String,
    logprobs: Vec<Value>,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseOutputTextDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseOutputTextDoneSseEventType,
    item_id: String,
    output_index: usize,
    content_index: usize,
    text: String,
    logprobs: Vec<Value>,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseContentPartDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseContentPartDoneSseEventType,
    item_id: String,
    output_index: usize,
    content_index: usize,
    part: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseOutputItemDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseOutputItemDoneSseEventType,
    output_index: usize,
    item: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseFunctionCallArgumentsDeltaSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseFunctionCallArgumentsDeltaSseEventType,
    item_id: String,
    output_index: usize,
    delta: String,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseFunctionCallArgumentsDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseFunctionCallArgumentsDoneSseEventType,
    item_id: String,
    output_index: usize,
    arguments: String,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseReasoningSummaryPartAddedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseReasoningSummaryPartAddedSseEventType,
    item_id: String,
    output_index: usize,
    summary_index: usize,
    part: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseReasoningSummaryTextDeltaSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseReasoningSummaryTextDeltaSseEventType,
    item_id: String,
    output_index: usize,
    summary_index: usize,
    delta: String,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseReasoningSummaryTextDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseReasoningSummaryTextDoneSseEventType,
    item_id: String,
    output_index: usize,
    summary_index: usize,
    text: String,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseReasoningSummaryPartDoneSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseReasoningSummaryPartDoneSseEventType,
    item_id: String,
    output_index: usize,
    summary_index: usize,
    part: Value,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseCompletedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseCompletedSseEventType,
    response: ResponsesResponse,
    timings: TimingObject,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseIncompleteSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseIncompleteSseEventType,
    response: ResponsesResponse,
    timings: TimingObject,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct LlmStreamErrorEvent {
    #[schema(rename = "type", inline)]
    kind: LlmStreamErrorEventType,
    sequence_number: u64,
    #[schema(required = true, nullable)]
    code: Option<String>,
    message: String,
    #[schema(required = true, nullable)]
    param: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub input_tokens_details: Value,
    pub output_tokens: usize,
    pub output_tokens_details: Value,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResponsesInputTokensResponse {
    pub object: &'static str,
    pub input_tokens: usize,
}

#[derive(Debug, Clone)]
struct ResponseConfig {
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    tools: Vec<FunctionTool>,
    tool_choice: Value,
    parallel_tool_calls: bool,
    text: Value,
    reasoning: Value,
    top_logprobs: usize,
}

#[allow(
    clippy::too_many_lines,
    reason = "owns chat validation, admission, and JSON/SSE transport selection"
)]
pub(super) async fn create_chat_completion<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let controls = streams.controls();
    let principal = authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    let resumable = resumable_requested(&headers, request.stream)?;
    validate_chat(&request)?;
    if resumable {
        ensure_resumable_capacity(&streams)?;
    }
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|value| value.include_usage);
    let model = request.model.clone();
    let id = next_id("chatcmpl");
    let vision = model_vision_policy(state.as_ref(), &request.model);
    let (mut advanced, overrides) = chat_advanced_request_with_limits(&request, vision)?;
    if request.reasoning_control {
        advanced.reasoning_control_id = Some(id.clone());
    }
    let max_tokens_param = if request.max_tokens.is_some() {
        "max_tokens"
    } else {
        "max_completion_tokens"
    };
    let generation = state
        .start_choice_generation(
            crate::application::metrics::InferenceOperation::Chat,
            model.clone(),
            advanced,
            overrides,
            max_tokens_param,
            None,
            None,
        )
        .await
        .map_err(|error| map_llm_start_error(error, "messages", max_tokens_param))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let activity = activity.map(|Extension(activity)| activity);
    if let Some(activity) = &activity {
        activity.set_model(model.clone());
    }
    let created = now_seconds();
    if request.stream {
        let completion_header = id.clone();
        let cancellation = generation.cancellation_handle();
        let registration = if request.reasoning_control {
            let control = generation.reasoning_control().ok_or_else(|| {
                ApiError::internal("reasoning control was armed without a worker handle")
            })?;
            Some(controls.register(id.clone(), principal, model.clone(), control))
        } else {
            None
        };
        let completion_id = registration
            .as_ref()
            .map(Registration::id)
            .map(str::to_owned);
        let response = chat_stream(
            generation,
            id,
            created,
            model,
            include_usage,
            activity.clone(),
            registration,
        );
        let mut response = stream_response(
            state.as_ref(),
            &streams,
            principal,
            StreamProtocol::Chat,
            resumable,
            response,
            cancellation,
            activity.clone(),
            completion_id.as_deref(),
        )?;
        response.headers_mut().insert(
            "x-orchion-completion-id",
            completion_header
                .parse()
                .expect("generated completion ID is a valid header value"),
        );
        Ok(response)
    } else {
        let result = collect_choices(generation, request.n.unwrap_or(1)).await?;
        let usage = result.usage;
        if let Some(activity) = &activity {
            activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
            set_activity_timing(activity, usage);
        }
        Ok(Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: result.choices.into_iter().map(chat_choice).collect(),
            usage: usage.into(),
            timings: usage.timings.into(),
        })
        .into_response())
    }
}

pub(super) async fn control_chat_completion<S>(
    State(state): State<Arc<S>>,
    Extension(controls): Extension<ChatControls>,
    headers: HeaderMap,
    request: Result<Json<ChatReasoningControlRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    let request = parse_json(request).inspect_err(|_| controls.observe_invalid())?;
    if !valid_chat_completion_id(&request.id) {
        controls.observe_invalid();
        return Err(invalid("id must be a valid chat completion id", "id"));
    }
    if request.action != "reasoning_end" {
        controls.observe_invalid();
        return Err(invalid("action must be reasoning_end", "action"));
    }
    if request
        .model
        .as_deref()
        .is_some_and(|model| orchion::ModelId::parse(model).is_err())
    {
        controls.observe_invalid();
        return Err(invalid("model must be a configured model id", "model"));
    }
    let (result, message) = controls
        .reasoning_end(&request.id, principal, request.model.as_deref())
        .await;
    if matches!(result, ApplyResult::Unavailable) {
        return Err(ApiError::control_unavailable());
    }
    Ok(Json(ChatReasoningControlResponse {
        id: request.id,
        action: "reasoning_end",
        success: matches!(result, ApplyResult::Applied),
        message: (!message.is_empty()).then(|| message.to_string()),
    })
    .into_response())
}

pub(super) async fn create_completion<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    let resumable = resumable_requested(&headers, request.stream)?;
    validate_completion(&request)?;
    if resumable {
        ensure_resumable_capacity(&streams)?;
    }
    let model = request.model.clone();
    let (advanced, overrides) = completion_advanced_request(&request)?;
    let generation = state
        .start_choice_generation(
            crate::application::metrics::InferenceOperation::Completion,
            model.clone(),
            advanced,
            overrides,
            "max_tokens",
            None,
            None,
        )
        .await
        .map_err(|error| map_llm_start_error(error, "prompt", "max_tokens"))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let activity = activity.map(|Extension(activity)| activity);
    if let Some(activity) = &activity {
        activity.set_model(model.clone());
    }
    let id = next_id("cmpl");
    let created = now_seconds();
    if request.stream {
        let cancellation = generation.cancellation_handle();
        let response = completion_stream(generation, id, created, model, activity.clone());
        stream_response(
            state.as_ref(),
            &streams,
            principal,
            StreamProtocol::Completions,
            resumable,
            response,
            cancellation,
            activity.clone(),
            None,
        )
    } else {
        let result = collect_choices(generation, request.n.unwrap_or(1)).await?;
        let usage = result.usage;
        if let Some(activity) = &activity {
            activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
            set_activity_timing(activity, usage);
        }
        Ok(Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: result
                .choices
                .into_iter()
                .map(|choice| CompletionChoice {
                    text: choice.text,
                    index: choice.index,
                    logprobs: completion_logprobs(&choice.logprobs),
                    finish_reason: choice_finish_reason(
                        choice
                            .finish_reason
                            .expect("collected choice has finish reason"),
                    ),
                })
                .collect(),
            usage: usage.into(),
            timings: usage.timings.into(),
        })
        .into_response())
    }
}

pub(super) async fn create_embeddings<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    let _ = &request.user;
    let inputs = embedding_inputs(&request.input)?;
    let model = request.model.clone();
    let result = state
        .create_embeddings(crate::application::llm::LlmEmbeddingCommand {
            model: model.clone(),
            inputs,
            dimensions: request.dimensions,
            queue_timeout: None,
            embedding_timeout: None,
        })
        .await
        .map_err(map_embedding_error)?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    if let Some(Extension(activity)) = activity {
        activity.set_model(model.clone());
        activity.set_llm_usage(result.prompt_tokens, 0);
    }
    let data = result
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            object: "embedding",
            embedding: match request.encoding_format {
                EmbeddingEncodingFormat::Float => EmbeddingValue::Float(embedding),
                EmbeddingEncodingFormat::Base64 => {
                    let bytes = embedding
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>();
                    EmbeddingValue::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
            },
            index,
        })
        .collect();
    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: EmbeddingUsage {
            prompt_tokens: result.prompt_tokens,
            total_tokens: result.total_tokens,
        },
    })
    .into_response())
}

fn embedding_inputs(input: &EmbeddingsInput) -> Result<Vec<orchion::LlmEmbeddingInput>, ApiError> {
    let inputs = match input {
        EmbeddingsInput::Text(text) => vec![orchion::LlmEmbeddingInput::Text(text.clone())],
        EmbeddingsInput::Texts(texts) => texts
            .iter()
            .cloned()
            .map(orchion::LlmEmbeddingInput::Text)
            .collect(),
        EmbeddingsInput::Tokens(tokens) => {
            vec![orchion::LlmEmbeddingInput::Tokens(tokens.clone())]
        }
        EmbeddingsInput::TokenBatches(batches) => batches
            .iter()
            .cloned()
            .map(orchion::LlmEmbeddingInput::Tokens)
            .collect(),
    };
    if inputs.is_empty() || inputs.len() > MAX_EMBEDDING_INPUTS {
        return Err(invalid("input must contain 1 to 2048 items", "input"));
    }
    if inputs.iter().any(|input| match input {
        orchion::LlmEmbeddingInput::Text(text) => text.is_empty(),
        orchion::LlmEmbeddingInput::Tokens(tokens) => tokens.is_empty(),
    }) {
        return Err(invalid("embedding inputs must not be empty", "input"));
    }
    Ok(inputs)
}

fn map_embedding_error(error: RuntimeError) -> ApiError {
    match error {
        RuntimeError::Core(orchion::OrchionError::LlmContextLimit { .. }) => {
            ApiError::invalid_request(
                error.to_string(),
                Some("input"),
                Some("context_length_exceeded"),
            )
        }
        RuntimeError::Core(orchion::OrchionError::Inference { ref message })
            if message.starts_with("embedding dimensions") =>
        {
            ApiError::invalid_request(
                message.clone(),
                Some("dimensions"),
                Some("invalid_parameter"),
            )
        }
        RuntimeError::Core(orchion::OrchionError::Inference { ref message })
            if message.contains("token id ")
                || message.contains("embedding request exceeds ")
                || message.contains("each embedding input ") =>
        {
            ApiError::invalid_request(message.clone(), Some("input"), Some("invalid_parameter"))
        }
        other => ApiError::from(UseCaseError::from(other)),
    }
}

pub(super) async fn create_response<S>(
    State(state): State<Arc<S>>,
    Extension(streams): Extension<LlmStreams>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let principal = authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    let resumable = resumable_requested(&headers, request.stream)?;
    validate_responses(&request)?;
    if resumable {
        ensure_resumable_capacity(&streams)?;
    }
    let model = request.model.clone();
    let vision = model_vision_policy(state.as_ref(), &request.model);
    let (advanced, overrides) = responses_advanced_request_with_limits(&request, vision)?;
    let generation = state
        .start_choice_generation(
            crate::application::metrics::InferenceOperation::Responses,
            model.clone(),
            advanced,
            overrides,
            "max_output_tokens",
            None,
            None,
        )
        .await
        .map_err(|error| map_llm_start_error(error, "input", "max_output_tokens"))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let activity = activity.map(|Extension(activity)| activity);
    if let Some(activity) = &activity {
        activity.set_model(model.clone());
    }
    let id = next_id("resp");
    let message_id = next_id("msg");
    let created = now_seconds();
    let response_config = ResponseConfig {
        instructions: request.instructions.clone(),
        max_output_tokens: request.max_output_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: request.tools.clone().unwrap_or_default(),
        tool_choice: tool_choice_wire(request.tool_choice.as_ref(), request.tools.as_deref()),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        text: responses_text_wire(request.text.as_ref()),
        reasoning: responses_reasoning_wire(request.reasoning.as_ref()),
        top_logprobs: request.top_logprobs.unwrap_or(0),
    };
    if request.stream {
        let cancellation = generation.cancellation_handle();
        let response = responses_stream(
            generation,
            id,
            message_id,
            created,
            model,
            response_config,
            activity.clone(),
        );
        stream_response(
            state.as_ref(),
            &streams,
            principal,
            StreamProtocol::Responses,
            resumable,
            response,
            cancellation,
            activity.clone(),
            None,
        )
    } else {
        let result = collect_choices(generation, 1).await?;
        let usage = result.usage;
        let choice = result
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::internal("generation returned no response choice"))?;
        let reason = choice
            .finish_reason
            .expect("collected choice has a finish reason");
        let text = choice.text.clone();
        if let Some(activity) = &activity {
            activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
            set_activity_timing(activity, usage);
        }
        let output = response_output_items(&message_id, &choice);
        Ok(Json(response_object(
            id,
            created,
            model,
            output,
            text,
            Some(reason),
            Some(usage),
            true,
            &response_config,
        ))
        .into_response())
    }
}

pub(super) async fn count_response_input_tokens<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    validate_responses(&request)?;
    if request.stream {
        return Err(invalid(
            "stream must be false for input token counting",
            "stream",
        ));
    }
    let vision = model_vision_policy(state.as_ref(), &request.model);
    let (advanced, _) = responses_advanced_request_with_limits(&request, vision)?;
    let LlmAdvancedInput::Messages(messages) = advanced.input else {
        return Err(ApiError::internal(
            "Responses preparation did not produce semantic input",
        ));
    };
    let semantic = orchion::LlmSemanticTokenCountRequest {
        messages,
        tools: advanced.tools,
        tool_choice: advanced.tool_choice,
        parallel_tool_calls: advanced.parallel_tool_calls,
        reasoning: advanced.reasoning,
        output: advanced.output,
    };
    let model = request.model;
    let input_tokens = state
        .count_semantic_input_tokens(model.clone(), semantic)
        .await
        .map_err(|error| map_llm_start_error(error, "input", "input"))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    if let Some(Extension(activity)) = activity {
        activity.set_model(model);
        activity.set_llm_usage(input_tokens, 0);
    }
    Ok(Json(ResponsesInputTokensResponse {
        object: "response.input_tokens",
        input_tokens,
    })
    .into_response())
}

fn parse_json<T>(request: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    request
        .map(|Json(value)| value)
        .map_err(|error| ApiError::invalid_request(error.body_text(), None, Some("invalid_json")))
}

fn resumable_requested(headers: &HeaderMap, stream: bool) -> Result<bool, ApiError> {
    let Some(value) = headers.get("x-orchion-resumable") else {
        return Ok(false);
    };
    if !stream || value.as_bytes() != b"true" {
        return Err(ApiError::invalid_request(
            "X-Orchion-Resumable is only valid as `true` for streaming requests",
            None,
            Some("invalid_resumable_stream"),
        ));
    }
    Ok(true)
}

fn ensure_resumable_capacity(streams: &LlmStreams) -> Result<(), ApiError> {
    streams
        .ensure_start_capacity()
        .map_err(|error| match error {
            StartError::Capacity => ApiError::stream_capacity("resumable stream"),
            StartError::ShuttingDown => ApiError::shutting_down(),
            StartError::Entropy => ApiError::internal("unexpected stream entropy preflight error"),
        })
}

#[allow(
    clippy::too_many_arguments,
    reason = "keeps protocol-neutral stream ownership explicit"
)]
fn stream_response<S: ServerApplication>(
    state: &S,
    streams: &LlmStreams,
    principal: crate::api::llm_streams::PrincipalId,
    protocol: StreamProtocol,
    resumable: bool,
    mut response: Response,
    cancellation: ManagedChoiceCancellation,
    activity: Option<ActivityContext>,
    completion_id: Option<&str>,
) -> Result<Response, ApiError> {
    let keepalive = state.api_policy().streaming.keepalive_interval;
    if !resumable {
        let (parts, body) = response.into_parts();
        return Ok(Response::from_parts(
            parts,
            sse::numbered_with_keepalive(body, keepalive),
        ));
    }
    let terminal = response
        .extensions_mut()
        .remove::<StreamTerminalSignal>()
        .ok_or_else(|| ApiError::internal("stream producer omitted terminal tracking"))?;
    let (stream_id, body) = streams
        .start_with_completion(
            principal,
            protocol,
            response.into_body(),
            cancellation,
            activity.clone(),
            completion_id.map(str::to_owned),
            terminal,
        )
        .map_err(|error| match error {
            StartError::Capacity => ApiError::stream_capacity("resumable stream"),
            StartError::ShuttingDown => ApiError::shutting_down(),
            StartError::Entropy => ApiError::internal("OS entropy unavailable for stream ID"),
        })?;
    if let Some(activity) = &activity {
        activity.handoff_to_owner();
    }
    let mut response = sse::response(sse::keepalive(body, keepalive));
    response.headers_mut().insert(
        "x-orchion-stream-id",
        stream_id
            .parse()
            .expect("generated stream ID is a valid header value"),
    );
    response.headers_mut().insert(
        "x-orchion-stream-ttl-seconds",
        streams
            .ttl()
            .as_secs()
            .to_string()
            .parse()
            .expect("stream TTL is a valid header value"),
    );
    Ok(response)
}

fn map_llm_start_error(
    error: RuntimeError,
    input_param: &'static str,
    max_tokens_param: &'static str,
) -> ApiError {
    match error {
        RuntimeError::Core(orchion::OrchionError::LlmContextLimit {
            prompt_tokens,
            max_tokens,
            context_size,
        }) => {
            let param = if prompt_tokens >= context_size {
                input_param
            } else {
                max_tokens_param
            };
            ApiError::invalid_request(
                format!(
                    "prompt ({prompt_tokens} tokens) plus completion ({max_tokens} tokens) exceeds context size {context_size}"
                ),
                Some(param),
                Some("context_length_exceeded"),
            )
        }
        other => ApiError::from(UseCaseError::from(other)),
    }
}

fn validate_chat(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    for (name, value) in [
        ("functions", &request.functions),
        ("function_call", &request.function_call),
        ("modalities", &request.modalities),
        ("audio", &request.audio),
        ("store", &request.store),
        ("metadata", &request.metadata),
        ("service_tier", &request.service_tier),
        ("prediction", &request.prediction),
        ("user", &request.user),
        ("web_search_options", &request.web_search_options),
        ("prompt_cache_key", &request.prompt_cache_key),
        ("prompt_cache_retention", &request.prompt_cache_retention),
        ("safety_identifier", &request.safety_identifier),
        ("verbosity", &request.verbosity),
        ("moderation", &request.moderation),
        ("prompt_cache_options", &request.prompt_cache_options),
    ] {
        if value.is_present() {
            return Err(unsupported(name));
        }
    }
    if request.messages.is_empty() {
        return Err(invalid("messages must not be empty", "messages"));
    }
    if request.max_completion_tokens.is_some() && request.max_tokens.is_some() {
        return Err(invalid(
            "max_completion_tokens and max_tokens are mutually exclusive",
            "max_completion_tokens",
        ));
    }
    if request.n == Some(0) {
        return Err(invalid("n must be greater than zero", "n"));
    }
    if request.reasoning_control && !request.stream {
        return Err(invalid(
            "reasoning_control requires stream=true",
            "reasoning_control",
        ));
    }
    if request.reasoning_control && request.n.unwrap_or(1) != 1 {
        return Err(invalid(
            "reasoning_control requires n=1",
            "reasoning_control",
        ));
    }
    if request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_obfuscation)
        .is_some_and(|value| value)
    {
        return Err(unsupported("stream_options.include_obfuscation"));
    }
    validate_sampling(
        request.temperature,
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
    )?;
    validate_tools(request.tools.as_deref(), request.tool_choice.as_ref())?;
    validate_response_format(request.response_format.as_ref(), "response_format")?;
    validate_logprobs(request.logprobs, request.top_logprobs, 20)?;
    validate_logit_bias(request.logit_bias.as_ref())?;
    reasoning_options(request.reasoning_effort.as_deref(), "reasoning_effort")?;
    stop_values(request.stop.as_ref())?;
    Ok(())
}

fn validate_completion(request: &CompletionRequest) -> Result<(), ApiError> {
    for (name, value) in [("suffix", &request.suffix), ("best_of", &request.best_of)] {
        if value.is_present() {
            return Err(unsupported(name));
        }
    }
    if request.n == Some(0) {
        return Err(invalid("n must be greater than zero", "n"));
    }
    if request.echo.is_some_and(|value| value) {
        return Err(unsupported("echo"));
    }
    validate_sampling(request.temperature, request.top_p, None, None)?;
    if request.logprobs.is_some_and(|value| value > 5) {
        return Err(invalid("logprobs must be in [0, 5]", "logprobs"));
    }
    validate_logit_bias(request.logit_bias.as_ref())?;
    stop_values(request.stop.as_ref())?;
    Ok(())
}

fn validate_responses(request: &ResponsesRequest) -> Result<(), ApiError> {
    if request.store == Some(true) {
        return Err(unsupported("store"));
    }
    if request.stream_options.is_some() && !request.stream {
        return Err(invalid(
            "stream_options requires stream=true",
            "stream_options",
        ));
    }
    if request
        .max_output_tokens
        .is_some_and(|value| value < MIN_RESPONSES_MAX_OUTPUT_TOKENS)
    {
        return Err(invalid(
            "max_output_tokens must be at least 16",
            "max_output_tokens",
        ));
    }
    if request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_obfuscation)
        .is_some_and(|value| value)
    {
        return Err(unsupported("stream_options.include_obfuscation"));
    }
    for (name, value) in [
        ("background", &request.background),
        ("conversation", &request.conversation),
        ("previous_response_id", &request.previous_response_id),
        ("context_management", &request.context_management),
        ("include", &request.include),
        ("metadata", &request.metadata),
        ("truncation", &request.truncation),
        ("service_tier", &request.service_tier),
        ("prompt_cache_key", &request.prompt_cache_key),
        ("prompt_cache_retention", &request.prompt_cache_retention),
        ("safety_identifier", &request.safety_identifier),
        ("user", &request.user),
        ("max_tool_calls", &request.max_tool_calls),
        ("moderation", &request.moderation),
        ("prompt_cache_options", &request.prompt_cache_options),
        ("prompt", &request.prompt),
    ] {
        if value.is_present() {
            return Err(unsupported(name));
        }
    }
    validate_sampling(request.temperature, request.top_p, None, None)?;
    validate_tools(request.tools.as_deref(), request.tool_choice.as_ref())?;
    if request.top_logprobs.is_some_and(|value| value > 20) {
        return Err(invalid("top_logprobs must be in [0, 20]", "top_logprobs"));
    }
    if let Some(reasoning) = &request.reasoning {
        if reasoning.summary.is_present() {
            return Err(unsupported("reasoning.summary"));
        }
        reasoning_options(reasoning.effort.as_deref(), "reasoning.effort")?;
    }
    if let Some(text) = &request.text {
        if text.verbosity.is_present() {
            return Err(unsupported("text.verbosity"));
        }
        validate_response_text_format(text.format.as_ref())?;
    }
    Ok(())
}

fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence: Option<f32>,
    frequency: Option<f32>,
) -> Result<(), ApiError> {
    if temperature.is_some_and(|v| !v.is_finite() || !(0.0..=2.0).contains(&v)) {
        return Err(invalid("temperature must be in [0, 2]", "temperature"));
    }
    if top_p.is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0) {
        return Err(invalid("top_p must be in (0, 1]", "top_p"));
    }
    if presence.is_some_and(|v| !v.is_finite() || !(-2.0..=2.0).contains(&v)) {
        return Err(invalid(
            "presence_penalty must be in [-2, 2]",
            "presence_penalty",
        ));
    }
    if frequency.is_some_and(|v| !v.is_finite() || !(-2.0..=2.0).contains(&v)) {
        return Err(invalid(
            "frequency_penalty must be in [-2, 2]",
            "frequency_penalty",
        ));
    }
    Ok(())
}

fn validate_tools(
    tools: Option<&[FunctionTool]>,
    choice: Option<&ToolChoiceValue>,
) -> Result<(), ApiError> {
    let tools = tools.unwrap_or_default();
    let mut tool_names = std::collections::BTreeSet::new();
    for tool in tools {
        if tool.kind != "function" {
            return Err(unsupported("tools.type"));
        }
        if tool.function.name.is_empty() || !tool_names.insert(tool.function.name.as_str()) {
            return Err(invalid(
                "function tool names must be nonempty and unique",
                "tools",
            ));
        }
        if !tool.function.parameters.is_object() {
            return Err(invalid("tool parameters must be an object", "tools"));
        }
    }
    match choice {
        None => Ok(()),
        Some(ToolChoiceValue::Mode(mode)) if mode == "none" => Ok(()),
        Some(ToolChoiceValue::Mode(mode)) if matches!(mode.as_str(), "auto" | "required") => {
            if tools.is_empty() {
                Err(invalid(
                    "tool_choice requires at least one tool",
                    "tool_choice",
                ))
            } else {
                Ok(())
            }
        }
        Some(ToolChoiceValue::Named(named)) => {
            if named.kind != "function" {
                return Err(unsupported("tool_choice.type"));
            }
            let name = named_tool_name(named)?;
            if tools.iter().any(|tool| tool.function.name == name) {
                Ok(())
            } else {
                Err(invalid(
                    "named tool_choice must reference a supplied tool",
                    "tool_choice",
                ))
            }
        }
        Some(ToolChoiceValue::Mode(_)) => Err(invalid("invalid tool_choice", "tool_choice")),
    }
}

fn validate_response_format(
    format: Option<&ResponseFormat>,
    param: &'static str,
) -> Result<(), ApiError> {
    let Some(format) = format else { return Ok(()) };
    match format.kind.as_str() {
        "text" | "json_object" if format.json_schema.is_none() => Ok(()),
        "json_schema" => {
            let schema = format
                .json_schema
                .as_ref()
                .ok_or_else(|| invalid("json_schema configuration is required", param))?;
            if schema.name.is_empty() || schema.strict != Some(true) {
                return Err(invalid(
                    "json_schema requires a nonempty name and strict=true",
                    param,
                ));
            }
            validate_schema(&schema.schema, param)
        }
        _ => Err(invalid("invalid response format", param)),
    }
}

fn validate_response_text_format(format: Option<&ResponseTextFormat>) -> Result<(), ApiError> {
    let Some(format) = format else { return Ok(()) };
    match format.kind.as_str() {
        "text" | "json_object" if format.schema.is_none() => Ok(()),
        "json_schema" => {
            if format.name.as_deref().is_none_or(str::is_empty) || format.strict != Some(true) {
                return Err(invalid(
                    "text.format json_schema requires a nonempty name and strict=true",
                    "text.format",
                ));
            }
            let schema = format
                .schema
                .as_ref()
                .ok_or_else(|| invalid("text.format schema is required", "text.format"))?;
            validate_schema(schema, "text.format")
        }
        _ => Err(invalid("invalid text.format", "text.format")),
    }
}

fn validate_schema(schema: &Value, param: &'static str) -> Result<(), ApiError> {
    orchion::validate_llm_json_schema(schema).map_err(|error| {
        ApiError::invalid_request(error.to_string(), Some(param), Some("invalid_json_schema"))
    })
}

fn validate_logprobs(
    enabled: Option<bool>,
    top: Option<usize>,
    maximum: usize,
) -> Result<(), ApiError> {
    if top.is_some_and(|value| value > maximum) {
        return Err(invalid("top_logprobs must be in [0, 20]", "top_logprobs"));
    }
    if top.is_some() && enabled != Some(true) {
        return Err(invalid(
            "top_logprobs requires logprobs=true",
            "top_logprobs",
        ));
    }
    Ok(())
}

fn validate_logit_bias(bias: Option<&BTreeMap<String, f32>>) -> Result<(), ApiError> {
    if bias.is_some_and(|bias| bias.len() > 256) {
        return Err(invalid(
            "logit_bias must contain at most 256 entries",
            "logit_bias",
        ));
    }
    for (token, bias) in bias.into_iter().flatten() {
        if token.parse::<i32>().is_err() || !bias.is_finite() || !(-100.0..=100.0).contains(bias) {
            return Err(invalid(
                "logit_bias keys must be token IDs and values must be in [-100, 100]",
                "logit_bias",
            ));
        }
    }
    Ok(())
}

fn reasoning_options(
    effort: Option<&str>,
    param: &'static str,
) -> Result<LlmReasoningOptions, ApiError> {
    let effort = match effort {
        None => return Ok(LlmReasoningOptions::default()),
        Some("none") => {
            return Ok(LlmReasoningOptions {
                enabled: Some(false),
                effort: None,
            });
        }
        Some("low") => LlmReasoningEffort::Low,
        Some("medium") => LlmReasoningEffort::Medium,
        Some("high") => LlmReasoningEffort::High,
        Some(_) => return Err(unsupported(param)),
    };
    Ok(LlmReasoningOptions {
        enabled: Some(true),
        effort: Some(effort),
    })
}

fn empty_object() -> Value {
    json!({})
}

#[cfg(test)]
fn chat_advanced_request(
    request: &ChatCompletionRequest,
) -> Result<(LlmAdvancedRequest, LlmGenerationOverrides), ApiError> {
    chat_advanced_request_with_limits(request, hard_vision_policy())
}

fn chat_advanced_request_with_limits(
    request: &ChatCompletionRequest,
    limits: LlmVisionPolicy,
) -> Result<(LlmAdvancedRequest, LlmGenerationOverrides), ApiError> {
    let mut budget = ImageBudget::new(limits);
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        messages.push(normalize_rich_chat_message(message, &mut budget)?);
    }
    validate_media_parts(&messages, limits)?;
    if request.n.unwrap_or(1) != 1 && messages_have_images(&messages) {
        return Err(invalid("multimodal requests require n == 1", "n"));
    }
    Ok((
        LlmAdvancedRequest {
            input: LlmAdvancedInput::Messages(messages),
            options: GenerationOptions::default(),
            tools: tool_definitions(request.tools.as_deref()),
            tool_choice: effective_tool_choice(
                request.tool_choice.as_ref(),
                request.tools.as_deref(),
            )?,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            reasoning: reasoning_options(request.reasoning_effort.as_deref(), "reasoning_effort")?,
            output: chat_output_constraint(request.response_format.as_ref())?,
            logprobs: request
                .logprobs
                .unwrap_or(false)
                .then_some(LlmLogprobsOptions {
                    top_logprobs: request.top_logprobs.unwrap_or(0),
                }),
            logit_bias: logit_bias(request.logit_bias.as_ref())?,
            sampling: LlmSamplingExtensions::default(),
            choices: request.n.unwrap_or(1),
            reasoning_control_id: None,
        },
        LlmGenerationOverrides {
            max_tokens: request.max_completion_tokens.or(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            seed: request.seed,
            stop: stop_values(request.stop.as_ref())?,
        },
    ))
}

fn completion_advanced_request(
    request: &CompletionRequest,
) -> Result<(LlmAdvancedRequest, LlmGenerationOverrides), ApiError> {
    Ok((
        LlmAdvancedRequest {
            input: LlmAdvancedInput::Prompt(request.prompt.clone()),
            options: GenerationOptions::default(),
            tools: Vec::new(),
            tool_choice: LlmToolChoice::None,
            parallel_tool_calls: false,
            reasoning: LlmReasoningOptions::default(),
            output: LlmOutputConstraint::Text,
            logprobs: request
                .logprobs
                .map(|top_logprobs| LlmLogprobsOptions { top_logprobs }),
            logit_bias: logit_bias(request.logit_bias.as_ref())?,
            sampling: LlmSamplingExtensions::default(),
            choices: request.n.unwrap_or(1),
            reasoning_control_id: None,
        },
        LlmGenerationOverrides {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stop: stop_values(request.stop.as_ref())?,
            ..LlmGenerationOverrides::default()
        },
    ))
}

#[allow(
    clippy::too_many_lines,
    reason = "normalizes the complete role-sensitive Chat message contract"
)]
fn normalize_rich_chat_message(
    message: &ChatMessage,
    budget: &mut ImageBudget,
) -> Result<LlmRichMessage, ApiError> {
    let role = match message.role.as_str() {
        "system" => LlmSemanticRole::System,
        "developer" => LlmSemanticRole::Developer,
        "user" => LlmSemanticRole::User,
        "assistant" => LlmSemanticRole::Assistant,
        "tool" => LlmSemanticRole::Tool,
        "function" => return Err(unsupported("messages.role")),
        _ => return Err(invalid("unsupported message role", "messages.role")),
    };
    let mut content = match &message.content {
        None => Vec::new(),
        Some(ChatContent::Text(text)) => vec![LlmContentPart::Text { text: text.clone() }],
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part.kind.as_str() {
                "text"
                    if part.image_url.is_none()
                        && part.input_audio.is_none()
                        && part.file.is_none()
                        && part.refusal.is_none() =>
                {
                    part.text
                        .clone()
                        .map(|text| LlmContentPart::Text { text })
                        .ok_or_else(|| {
                            invalid("text content part requires text", "messages.content")
                        })
                }
                "image_url"
                    if part.text.is_none()
                        && part.input_audio.is_none()
                        && part.file.is_none()
                        && part.refusal.is_none() =>
                {
                    if message.role != "user" {
                        return Err(invalid(
                            "images are only accepted in user messages",
                            "messages.content",
                        ));
                    }
                    let image = part.image_url.as_ref().ok_or_else(|| {
                        invalid(
                            "image_url content part requires image_url",
                            "messages.content",
                        )
                    })?;
                    validate_image_detail(
                        image.detail.as_deref(),
                        "messages.content.image_url.detail",
                    )?;
                    parse_data_image_with_budget(
                        &image.url,
                        "messages.content.image_url.url",
                        budget,
                    )
                    .map(LlmContentPart::Image)
                }
                _ => Err(unsupported("messages.content")),
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let text = content
        .iter()
        .filter_map(|part| match part {
            LlmContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    if message.role != "assistant"
        && (!message.tool_calls.is_empty() || message.reasoning_content.is_some())
    {
        return Err(invalid(
            "tool_calls and reasoning_content require assistant role",
            "messages",
        ));
    }
    if let Some(reasoning) = &message.reasoning_content {
        content.push(LlmContentPart::Reasoning {
            text: reasoning.clone(),
        });
    }
    if message.role == "tool" {
        let call_id = message.tool_call_id.clone().ok_or_else(|| {
            invalid(
                "tool messages require tool_call_id",
                "messages.tool_call_id",
            )
        })?;
        content.clear();
        content.push(LlmContentPart::ToolResult(LlmToolResult {
            tool_call_id: call_id,
            content: text,
            is_error: false,
        }));
    } else if message.tool_call_id.is_some() {
        return Err(invalid(
            "tool_call_id requires tool role",
            "messages.tool_call_id",
        ));
    }
    let tool_calls: Vec<LlmToolCall> = message
        .tool_calls
        .iter()
        .map(|call| {
            if call.kind != "function" {
                return Err(unsupported("messages.tool_calls.type"));
            }
            Ok(LlmToolCall {
                id: call.id.clone(),
                name: call.function.name.clone(),
                arguments: function_arguments(&call.function.arguments, "messages.tool_calls")?,
            })
        })
        .collect::<Result<_, _>>()?;
    if content.is_empty() && tool_calls.is_empty() {
        return Err(invalid(
            "message content may be null only for assistant tool calls",
            "messages.content",
        ));
    }
    Ok(LlmRichMessage {
        role,
        content,
        tool_calls,
    })
}

#[cfg(test)]
fn responses_advanced_request(
    request: &ResponsesRequest,
) -> Result<(LlmAdvancedRequest, LlmGenerationOverrides), ApiError> {
    responses_advanced_request_with_limits(request, hard_vision_policy())
}

fn responses_advanced_request_with_limits(
    request: &ResponsesRequest,
    limits: LlmVisionPolicy,
) -> Result<(LlmAdvancedRequest, LlmGenerationOverrides), ApiError> {
    let messages =
        responses_rich_messages_with_limits(&request.input, request.instructions.as_ref(), limits)?;
    Ok((
        LlmAdvancedRequest {
            input: LlmAdvancedInput::Messages(messages),
            options: GenerationOptions::default(),
            tools: tool_definitions(request.tools.as_deref()),
            tool_choice: effective_tool_choice(
                request.tool_choice.as_ref(),
                request.tools.as_deref(),
            )?,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            reasoning: reasoning_options(
                request
                    .reasoning
                    .as_ref()
                    .and_then(|value| value.effort.as_deref()),
                "reasoning.effort",
            )?,
            output: responses_output_constraint(request.text.as_ref())?,
            logprobs: request
                .top_logprobs
                .map(|top_logprobs| LlmLogprobsOptions { top_logprobs }),
            logit_bias: Vec::new(),
            sampling: LlmSamplingExtensions::default(),
            choices: 1,
            reasoning_control_id: None,
        },
        LlmGenerationOverrides {
            max_tokens: request.max_output_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            ..LlmGenerationOverrides::default()
        },
    ))
}

fn tool_definitions(tools: Option<&[FunctionTool]>) -> Vec<LlmToolDefinition> {
    tools
        .unwrap_or_default()
        .iter()
        .map(|tool| LlmToolDefinition {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
        })
        .collect()
}

fn tool_choice(choice: Option<&ToolChoiceValue>) -> Result<LlmToolChoice, ApiError> {
    match choice {
        None => Ok(LlmToolChoice::None),
        Some(ToolChoiceValue::Mode(mode)) if mode == "none" => Ok(LlmToolChoice::None),
        Some(ToolChoiceValue::Mode(mode)) if mode == "auto" => Ok(LlmToolChoice::Auto),
        Some(ToolChoiceValue::Mode(mode)) if mode == "required" => Ok(LlmToolChoice::Required),
        Some(ToolChoiceValue::Named(named)) => {
            Ok(LlmToolChoice::Named(named_tool_name(named)?.to_string()))
        }
        _ => Err(invalid("invalid tool_choice", "tool_choice")),
    }
}

fn effective_tool_choice(
    choice: Option<&ToolChoiceValue>,
    tools: Option<&[FunctionTool]>,
) -> Result<LlmToolChoice, ApiError> {
    match choice {
        Some(choice) => tool_choice(Some(choice)),
        None if tools.is_some_and(|tools| !tools.is_empty()) => Ok(LlmToolChoice::Auto),
        None => Ok(LlmToolChoice::None),
    }
}

fn named_tool_name(named: &NamedToolChoice) -> Result<&str, ApiError> {
    named
        .function
        .as_ref()
        .map(|function| function.name.as_str())
        .or(named.name.as_deref())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("named tool_choice requires name", "tool_choice"))
}

fn logit_bias(values: Option<&BTreeMap<String, f32>>) -> Result<Vec<LlmLogitBias>, ApiError> {
    values
        .into_iter()
        .flatten()
        .map(|(token, bias)| {
            Ok(LlmLogitBias {
                token_id: token
                    .parse()
                    .map_err(|_| invalid("invalid token id", "logit_bias"))?,
                bias: *bias,
            })
        })
        .collect()
}

fn chat_output_constraint(
    format: Option<&ResponseFormat>,
) -> Result<LlmOutputConstraint, ApiError> {
    match format.map(|format| format.kind.as_str()) {
        None | Some("text") => Ok(LlmOutputConstraint::Text),
        Some("json_object") => Ok(LlmOutputConstraint::JsonObject),
        Some("json_schema") => Ok(LlmOutputConstraint::JsonSchema(
            format
                .and_then(|format| format.json_schema.as_ref())
                .ok_or_else(|| invalid("json_schema is required", "response_format"))?
                .schema
                .clone(),
        )),
        Some(_) => Err(invalid("invalid response format", "response_format")),
    }
}

fn responses_output_constraint(
    text: Option<&ResponsesText>,
) -> Result<LlmOutputConstraint, ApiError> {
    match text
        .and_then(|text| text.format.as_ref())
        .map(|format| format.kind.as_str())
    {
        None | Some("text") => Ok(LlmOutputConstraint::Text),
        Some("json_object") => Ok(LlmOutputConstraint::JsonObject),
        Some("json_schema") => Ok(LlmOutputConstraint::JsonSchema(
            text.and_then(|text| text.format.as_ref())
                .and_then(|format| format.schema.clone())
                .ok_or_else(|| invalid("schema is required", "text.format"))?,
        )),
        Some(_) => Err(invalid("invalid text.format", "text.format")),
    }
}

fn function_arguments(value: &Value, param: &'static str) -> Result<Value, ApiError> {
    match value {
        Value::String(arguments) => serde_json::from_str(arguments)
            .map_err(|_| invalid("function arguments must contain valid JSON", param)),
        value if value.is_object() => Ok(value.clone()),
        _ => Err(invalid("function arguments must be JSON", param)),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "normalizes all supported stateless Responses input item variants"
)]
#[cfg(test)]
fn responses_rich_messages(
    input: &ResponsesInput,
    instructions: Option<&String>,
) -> Result<Vec<LlmRichMessage>, ApiError> {
    responses_rich_messages_with_limits(input, instructions, hard_vision_policy())
}

#[allow(
    clippy::too_many_lines,
    reason = "normalizes all supported Responses items under one shared media budget"
)]
fn responses_rich_messages_with_limits(
    input: &ResponsesInput,
    instructions: Option<&String>,
    limits: LlmVisionPolicy,
) -> Result<Vec<LlmRichMessage>, ApiError> {
    let mut messages = Vec::new();
    let mut budget = ImageBudget::new(limits);
    if let Some(instructions) = instructions {
        messages.push(rich_text_message(
            LlmSemanticRole::Developer,
            instructions.clone(),
        ));
    }
    match input {
        ResponsesInput::Text(text) => {
            messages.push(rich_text_message(LlmSemanticRole::User, text.clone()));
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match item.kind.as_deref().unwrap_or("message") {
                    "message" => {
                        let role = match item.role.as_deref() {
                            Some("system") => LlmSemanticRole::System,
                            Some("developer") => LlmSemanticRole::Developer,
                            Some("user") => LlmSemanticRole::User,
                            Some("assistant") => LlmSemanticRole::Assistant,
                            _ => {
                                return Err(invalid(
                                    "message input requires a valid role",
                                    "input.role",
                                ));
                            }
                        };
                        messages.push(LlmRichMessage {
                            role: role.clone(),
                            content: response_item_parts(
                                item.content.as_ref(),
                                &role,
                                &mut budget,
                            )?,
                            tool_calls: Vec::new(),
                        });
                    }
                    "function_call" => {
                        let call_id = item
                            .call_id
                            .clone()
                            .or_else(|| item.id.clone())
                            .ok_or_else(|| {
                                invalid("function_call requires call_id", "input.call_id")
                            })?;
                        let name = item
                            .name
                            .clone()
                            .ok_or_else(|| invalid("function_call requires name", "input.name"))?;
                        let arguments = item.arguments.as_ref().ok_or_else(|| {
                            invalid("function_call requires arguments", "input.arguments")
                        })?;
                        messages.push(LlmRichMessage {
                            role: LlmSemanticRole::Assistant,
                            content: Vec::new(),
                            tool_calls: vec![LlmToolCall {
                                id: call_id,
                                name,
                                arguments: function_arguments(arguments, "input.arguments")?,
                            }],
                        });
                    }
                    "function_call_output" => {
                        let call_id = item.call_id.clone().ok_or_else(|| {
                            invalid("function_call_output requires call_id", "input.call_id")
                        })?;
                        let output = item
                            .output
                            .clone()
                            .or_else(|| {
                                item.content.as_ref().and_then(|content| match content {
                                    ResponseItemContent::Text(text) => Some(text.clone()),
                                    ResponseItemContent::Parts(_) => None,
                                })
                            })
                            .ok_or_else(|| {
                                invalid("function_call_output requires output", "input.output")
                            })?;
                        messages.push(LlmRichMessage {
                            role: LlmSemanticRole::Tool,
                            content: vec![LlmContentPart::ToolResult(LlmToolResult {
                                tool_call_id: call_id,
                                content: output,
                                is_error: false,
                            })],
                            tool_calls: Vec::new(),
                        });
                    }
                    "reasoning" => {
                        let text = item
                            .summary
                            .as_ref()
                            .map(|parts| {
                                parts
                                    .iter()
                                    .map(|part| {
                                        if part.kind != "summary_text" {
                                            return Err(unsupported("input.summary.type"));
                                        }
                                        Ok(part.text.as_str())
                                    })
                                    .collect::<Result<Vec<_>, _>>()
                                    .map(|parts| parts.join(""))
                            })
                            .transpose()?
                            .unwrap_or_default();
                        if text.is_empty() {
                            return Err(invalid(
                                "reasoning input requires summary text",
                                "input.summary",
                            ));
                        }
                        messages.push(LlmRichMessage {
                            role: LlmSemanticRole::Assistant,
                            content: vec![LlmContentPart::Reasoning { text }],
                            tool_calls: Vec::new(),
                        });
                    }
                    _ => return Err(unsupported("input.type")),
                }
            }
        }
    }
    if messages.is_empty() {
        return Err(invalid("input must not be empty", "input"));
    }
    validate_media_parts(&messages, limits)?;
    Ok(messages)
}

fn rich_text_message(role: LlmSemanticRole, text: String) -> LlmRichMessage {
    LlmRichMessage {
        role,
        content: vec![LlmContentPart::Text { text }],
        tool_calls: Vec::new(),
    }
}

fn response_item_parts(
    content: Option<&ResponseItemContent>,
    role: &LlmSemanticRole,
    budget: &mut ImageBudget,
) -> Result<Vec<LlmContentPart>, ApiError> {
    match content {
        Some(ResponseItemContent::Text(text)) => {
            Ok(vec![LlmContentPart::Text { text: text.clone() }])
        }
        Some(ResponseItemContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part.kind.as_str() {
                "input_text" | "output_text"
                    if part.image_url.is_none()
                        && part.file_id.is_none()
                        && part.detail.is_none() =>
                {
                    part.text
                        .clone()
                        .map(|text| LlmContentPart::Text { text })
                        .ok_or_else(|| invalid("text part requires text", "input.content"))
                }
                "input_image" if part.text.is_none() => {
                    if part.file_id.is_some() {
                        return Err(unsupported("input.content.file_id"));
                    }
                    if !matches!(role, LlmSemanticRole::User) {
                        return Err(invalid(
                            "images are only accepted in user messages",
                            "input.content",
                        ));
                    }
                    if !matches!(part.detail, None | Some(RuntimeResponseImageDetail::Auto)) {
                        return Err(unsupported("input.content.detail"));
                    }
                    let url = part.image_url.as_deref().ok_or_else(|| {
                        invalid("input_image requires image_url", "input.content.image_url")
                    })?;
                    parse_data_image_with_budget(url, "input.content.image_url", budget)
                        .map(LlmContentPart::Image)
                }
                _ => Err(unsupported("input.content")),
            })
            .collect(),
        None => Err(invalid("message input requires content", "input.content")),
    }
}

fn validate_image_detail(detail: Option<&str>, param: &'static str) -> Result<(), ApiError> {
    match detail {
        None | Some("auto") => Ok(()),
        Some(_) => Err(unsupported(param)),
    }
}

#[cfg(test)]
fn parse_data_image(url: &str, param: &'static str) -> Result<LlmImageInput, ApiError> {
    parse_data_image_with_budget(url, param, &mut ImageBudget::default())
}

#[derive(Debug)]
struct ImageBudget {
    limits: LlmVisionPolicy,
    count: usize,
    decoded_bytes: usize,
    pixels: u64,
}

impl ImageBudget {
    const fn new(limits: LlmVisionPolicy) -> Self {
        Self {
            limits,
            count: 0,
            decoded_bytes: 0,
            pixels: 0,
        }
    }

    fn reserve_decoded(&mut self, payload: &str, param: &'static str) -> Result<usize, ApiError> {
        if self.count >= self.limits.max_images {
            return Err(invalid(
                "request exceeds the model image-count limit",
                param,
            ));
        }
        let estimated = exact_base64_decoded_len(payload, param)?;
        if estimated > self.limits.max_bytes_per_image {
            return Err(invalid("image exceeds the model decoded-size limit", param));
        }
        let remaining = self
            .limits
            .max_total_bytes
            .saturating_sub(self.decoded_bytes);
        if estimated > remaining {
            return Err(invalid(
                "aggregate image decoded-size limit exceeded",
                param,
            ));
        }
        self.count += 1;
        self.decoded_bytes += estimated;
        Ok(estimated)
    }

    fn record_dimensions(
        &mut self,
        width: u32,
        height: u32,
        param: &'static str,
    ) -> Result<(), ApiError> {
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| invalid("image dimensions overflow", param))?;
        if width == 0
            || height == 0
            || width > self.limits.max_side
            || height > self.limits.max_side
            || pixels > self.limits.max_pixels_per_image
        {
            return Err(invalid("image dimensions exceed safety limits", param));
        }
        let total = self
            .pixels
            .checked_add(pixels)
            .ok_or_else(|| invalid("aggregate image dimensions overflow", param))?;
        if total > self.limits.max_total_pixels {
            return Err(invalid("aggregate image pixel limit exceeded", param));
        }
        self.pixels = total;
        Ok(())
    }
}

fn exact_base64_decoded_len(payload: &str, param: &'static str) -> Result<usize, ApiError> {
    if payload.is_empty() || !payload.len().is_multiple_of(4) {
        return Err(invalid("image contains invalid base64", param));
    }
    let padding = usize::from(payload.ends_with('=')) + usize::from(payload.ends_with("=="));
    let data_length = payload.len() - padding;
    if !payload[..data_length]
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        || !payload[data_length..].bytes().all(|byte| byte == b'=')
    {
        return Err(invalid("image contains invalid base64", param));
    }
    payload
        .len()
        .checked_div(4)
        .and_then(|groups| groups.checked_mul(3))
        .and_then(|length| length.checked_sub(padding))
        .ok_or_else(|| invalid("image base64 length overflow", param))
}

fn parse_data_image_with_budget(
    url: &str,
    param: &'static str,
    budget: &mut ImageBudget,
) -> Result<LlmImageInput, ApiError> {
    let (format, payload) = if let Some(payload) = url.strip_prefix("data:image/png;base64,") {
        (LlmImageFormat::Png, payload)
    } else if let Some(payload) = url.strip_prefix("data:image/jpeg;base64,") {
        (LlmImageFormat::Jpeg, payload)
    } else {
        return Err(invalid(
            "image_url must be a strict PNG or JPEG base64 data URL",
            param,
        ));
    };
    if payload.is_empty() || payload.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(invalid(
            "image base64 must be nonempty and contain no whitespace",
            param,
        ));
    }
    let estimated = budget.reserve_decoded(payload, param)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| invalid("image contains invalid base64", param))?;
    if bytes.len() != estimated {
        return Err(invalid(
            "image decoded size did not match its encoding",
            param,
        ));
    }
    let image_format = match format {
        LlmImageFormat::Png => image::ImageFormat::Png,
        LlmImageFormat::Jpeg => image::ImageFormat::Jpeg,
    };
    let (width, height) =
        image::ImageReader::with_format(std::io::Cursor::new(&bytes), image_format)
            .into_dimensions()
            .map_err(|_| {
                invalid(
                    "image bytes do not match the declared PNG/JPEG format",
                    param,
                )
            })?;
    budget.record_dimensions(width, height, param)?;
    Ok(LlmImageInput {
        bytes,
        format,
        width,
        height,
    })
}

impl Default for ImageBudget {
    fn default() -> Self {
        Self::new(hard_vision_policy())
    }
}

fn messages_have_images(messages: &[LlmRichMessage]) -> bool {
    messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, LlmContentPart::Image(_)))
    })
}

fn validate_media_parts(
    messages: &[LlmRichMessage],
    limits: LlmVisionPolicy,
) -> Result<(), ApiError> {
    let images = messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|part| match part {
            LlmContentPart::Image(image) => Some(image),
            _ => None,
        })
        .collect::<Vec<_>>();
    if images.len() > limits.max_images {
        return Err(invalid(
            "request exceeds the hard image-count limit",
            "input",
        ));
    }
    let mut total_bytes = 0usize;
    let mut total_pixels = 0u64;
    for image in images {
        let pixels = u64::from(image.width) * u64::from(image.height);
        if image.width == 0
            || image.height == 0
            || image.width > limits.max_side
            || image.height > limits.max_side
            || pixels > limits.max_pixels_per_image
        {
            return Err(invalid("image dimensions exceed safety limits", "input"));
        }
        total_bytes = total_bytes
            .checked_add(image.bytes.len())
            .ok_or_else(|| invalid("aggregate image size overflow", "input"))?;
        total_pixels = total_pixels
            .checked_add(pixels)
            .ok_or_else(|| invalid("aggregate image dimensions overflow", "input"))?;
    }
    if total_bytes > limits.max_total_bytes || total_pixels > limits.max_total_pixels {
        return Err(invalid("aggregate image limits exceeded", "input"));
    }
    Ok(())
}

fn model_vision_policy<S: ServerApplication>(state: &S, model: &str) -> LlmVisionPolicy {
    orchion::ModelId::parse(model)
        .ok()
        .and_then(|id| state.api_policy().llm_vision_limits.get(&id).copied())
        .unwrap_or_else(hard_vision_policy)
}

const fn hard_vision_policy() -> LlmVisionPolicy {
    LlmVisionPolicy {
        max_images: MAX_IMAGES,
        max_bytes_per_image: MAX_IMAGE_BYTES,
        max_total_bytes: MAX_TOTAL_IMAGE_BYTES,
        max_side: MAX_IMAGE_SIDE,
        max_pixels_per_image: MAX_IMAGE_PIXELS,
        max_total_pixels: MAX_TOTAL_IMAGE_PIXELS,
    }
}

fn stop_values(stop: Option<&StopSequences>) -> Result<Vec<String>, ApiError> {
    let values = match stop {
        None => Vec::new(),
        Some(StopSequences::One(value)) => vec![value.clone()],
        Some(StopSequences::Many(values)) => values.clone(),
    };
    if values.is_empty() && stop.is_some()
        || values.len() > 4
        || values.iter().any(String::is_empty)
    {
        return Err(invalid("stop must contain 1 to 4 nonempty strings", "stop"));
    }
    Ok(values)
}

#[derive(Default)]
struct ChoiceAccumulator {
    index: usize,
    text: String,
    reasoning: String,
    reasoning_id: String,
    reasoning_output_index: Option<usize>,
    message_output_index: Option<usize>,
    tool_calls: Vec<AccumulatedToolCall>,
    logprobs: Vec<LlmTokenLogprobs>,
    finish_reason: Option<LlmChoiceFinishReason>,
    next_output_index: usize,
}

#[derive(Default)]
struct AccumulatedToolCall {
    item_id: String,
    id: String,
    name: String,
    arguments: String,
    output_index: Option<usize>,
    started: bool,
}

struct CollectedChoices {
    choices: Vec<ChoiceAccumulator>,
    usage: LlmUsage,
}

impl ChoiceAccumulator {
    fn message_output_index(&mut self) -> usize {
        *self.message_output_index.get_or_insert_with(|| {
            let index = self.next_output_index;
            self.next_output_index += 1;
            index
        })
    }

    fn reasoning_output_index(&mut self) -> usize {
        *self.reasoning_output_index.get_or_insert_with(|| {
            let index = self.next_output_index;
            self.next_output_index += 1;
            self.reasoning_id = next_id("rs");
            index
        })
    }

    fn tool_call_mut(&mut self, index: usize) -> Result<&mut AccumulatedToolCall, ApiError> {
        if index > self.tool_calls.len() {
            return Err(ApiError::internal("tool call indexes were not contiguous"));
        }
        if index == self.tool_calls.len() {
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            self.tool_calls.push(AccumulatedToolCall {
                item_id: next_id("fc"),
                output_index: Some(output_index),
                ..AccumulatedToolCall::default()
            });
        }
        Ok(&mut self.tool_calls[index])
    }
}

async fn collect_choices(
    mut generation: ManagedChoiceGeneration,
    count: usize,
) -> Result<CollectedChoices, ApiError> {
    let mut choices = (0..count)
        .map(|index| ChoiceAccumulator {
            index,
            ..ChoiceAccumulator::default()
        })
        .collect::<Vec<_>>();
    while let Some(event) = generation.next().await {
        match event.map_err(|error| ApiError::from(UseCaseError::from(error)))? {
            LlmChoiceEvent::Delta {
                index,
                text,
                logprobs,
            } => {
                let choice = choice_mut(&mut choices, index)?;
                choice.message_output_index();
                choice.text.push_str(&text);
                if let Some(logprobs) = logprobs {
                    choice.logprobs.push(logprobs);
                }
            }
            LlmChoiceEvent::SemanticDelta { index, delta } => {
                apply_semantic_delta(choice_mut(&mut choices, index)?, delta)?;
            }
            LlmChoiceEvent::Finished { index, reason, .. } => {
                choice_mut(&mut choices, index)?.finish_reason = Some(reason);
            }
            LlmChoiceEvent::FinishedAll { usage } => {
                if choices.iter().any(|choice| choice.finish_reason.is_none()) {
                    return Err(ApiError::internal(
                        "aggregate terminal arrived before every choice finished",
                    ));
                }
                return Ok(CollectedChoices { choices, usage });
            }
            LlmChoiceEvent::Failed { message, .. } => return Err(ApiError::internal(message)),
        }
    }
    Err(ApiError::internal(
        "generation ended without aggregate terminal acknowledgement",
    ))
}

fn choice_mut(
    choices: &mut [ChoiceAccumulator],
    index: usize,
) -> Result<&mut ChoiceAccumulator, ApiError> {
    choices
        .get_mut(index)
        .ok_or_else(|| ApiError::internal("generation returned an out-of-range choice index"))
}

fn apply_semantic_delta(
    choice: &mut ChoiceAccumulator,
    delta: LlmSemanticDelta,
) -> Result<(), ApiError> {
    match delta {
        LlmSemanticDelta::Text(text) => {
            choice.message_output_index();
            choice.text.push_str(&text);
        }
        LlmSemanticDelta::Reasoning(reasoning) => {
            choice.reasoning_output_index();
            choice.reasoning.push_str(&reasoning);
        }
        LlmSemanticDelta::ToolCall {
            index,
            id,
            name,
            arguments,
        } => {
            let call = choice.tool_call_mut(index)?;
            if let Some(id) = id {
                call.id = id;
            }
            if let Some(name) = name {
                call.name = name;
            }
            call.arguments.push_str(&arguments);
        }
    }
    Ok(())
}

fn chat_choice(choice: ChoiceAccumulator) -> ChatChoice {
    ChatChoice {
        index: choice.index,
        message: AssistantMessage {
            role: "assistant",
            content: (!choice.text.is_empty()).then_some(choice.text),
            reasoning_content: (!choice.reasoning.is_empty()).then_some(choice.reasoning),
            tool_calls: wire_tool_calls(choice.tool_calls),
            refusal: None,
        },
        finish_reason: choice_finish_reason(
            choice
                .finish_reason
                .expect("collected choice has a finish reason"),
        ),
        logprobs: chat_logprobs(&choice.logprobs),
    }
}

fn wire_tool_calls(calls: Vec<AccumulatedToolCall>) -> Vec<ChatToolCall> {
    calls
        .into_iter()
        .map(|call| ChatToolCall {
            id: call.id,
            kind: "function".to_string(),
            function: ChatFunctionCall {
                name: call.name,
                arguments: Value::String(call.arguments),
            },
        })
        .collect()
}

fn token_logprob(value: &orchion::LlmTokenAlternative) -> Value {
    json!({
        "token": String::from_utf8_lossy(&value.bytes),
        "logprob": value.logprob,
        "bytes": value.bytes,
    })
}

fn chat_logprobs(values: &[LlmTokenLogprobs]) -> Option<Value> {
    (!values.is_empty()).then(|| {
        json!({
            "content": values.iter().map(|value| {
                let mut chosen = token_logprob(&value.chosen);
                chosen["top_logprobs"] = Value::Array(value.top.iter().map(token_logprob).collect());
                chosen
            }).collect::<Vec<_>>(),
            "refusal": null,
        })
    })
}

fn completion_logprobs(values: &[LlmTokenLogprobs]) -> Option<Value> {
    (!values.is_empty()).then(|| {
        let mut offset = 0_usize;
        let mut offsets = Vec::with_capacity(values.len());
        let mut tokens = Vec::with_capacity(values.len());
        let mut chosen = Vec::with_capacity(values.len());
        let mut top = Vec::with_capacity(values.len());
        for value in values {
            offsets.push(offset);
            offset += value.chosen.bytes.len();
            tokens.push(String::from_utf8_lossy(&value.chosen.bytes).into_owned());
            chosen.push(value.chosen.logprob);
            top.push(
                value
                    .top
                    .iter()
                    .map(|alternative| {
                        (
                            String::from_utf8_lossy(&alternative.bytes).into_owned(),
                            alternative.logprob,
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            );
        }
        json!({"tokens":tokens,"token_logprobs":chosen,"top_logprobs":top,"text_offset":offsets})
    })
}

fn chat_stream(
    mut generation: ManagedChoiceGeneration,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    activity: Option<ActivityContext>,
    registration: Option<Registration>,
) -> Response {
    let terminal = StreamTerminalSignal::default();
    let producer_terminal = terminal.clone();
    let stream = async_stream::stream! {
        let _registration = registration;
        let mut started = std::collections::BTreeSet::new();
        while let Some(event) = generation.next().await {
            match event {
                Ok(LlmChoiceEvent::Delta { index, text, logprobs }) => {
                    if started.insert(index) {
                        yield Ok::<Bytes, Infallible>(Bytes::from(chat_choice_chunk(&id, created, &model, index, json!({"role":"assistant"}), None, None)));
                    }
                    yield Ok(Bytes::from(chat_choice_chunk(&id, created, &model, index, json!({"content":text}), logprobs.as_ref().map(chat_delta_logprobs), None)));
                }
                Ok(LlmChoiceEvent::SemanticDelta { index, delta }) => {
                    if started.insert(index) {
                        yield Ok(Bytes::from(chat_choice_chunk(&id, created, &model, index, json!({"role":"assistant"}), None, None)));
                    }
                    let delta = match delta {
                        LlmSemanticDelta::Text(text) => json!({"content":text}),
                        LlmSemanticDelta::Reasoning(text) => json!({"reasoning_content":text}),
                        LlmSemanticDelta::ToolCall { index, id, name, arguments } => json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]}),
                    };
                    yield Ok(Bytes::from(chat_choice_chunk(&id, created, &model, index, delta, None, None)));
                }
                Ok(LlmChoiceEvent::Finished { index, reason, .. }) => {
                    if started.insert(index) {
                        yield Ok(Bytes::from(chat_choice_chunk(&id, created, &model, index, json!({"role":"assistant"}), None, None)));
                    }
                    yield Ok(Bytes::from(chat_choice_chunk(&id, created, &model, index, json!({}), None, Some(choice_finish_reason(reason)))));
                }
                Ok(LlmChoiceEvent::FinishedAll { usage }) => {
                    if let Some(activity) = &activity {
                        activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
                        set_activity_timing(activity, usage);
                    }
                    yield Ok(Bytes::from(chat_aggregate_chunk(&id, created, &model, include_usage.then_some(usage), usage.timings)));
                    producer_terminal.complete();
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    break;
                }
                Ok(LlmChoiceEvent::Failed { index: Some(_), .. }) => {}
                Ok(LlmChoiceEvent::Failed { index: None, message }) => {
                    let api = ApiError::internal(message);
                    if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                    producer_terminal.fail();
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&ErrorBody { error: api.error }).unwrap())));
                    break;
                }
                Err(error) => {
                    let api = ApiError::from(UseCaseError::from(error));
                    if let Some(activity) = &activity {
                        activity.complete_stream_failure(activity_outcome(&api), api.activity_error());
                    }
                    let body = ErrorBody { error: api.error };
                    producer_terminal.fail();
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&body).unwrap())));
                    break;
                }
            }
        }
    };
    let mut response = sse_response(Body::from_stream(stream));
    response.extensions_mut().insert(terminal);
    response
}

fn completion_stream(
    mut generation: ManagedChoiceGeneration,
    id: String,
    created: u64,
    model: String,
    activity: Option<ActivityContext>,
) -> Response {
    let terminal = StreamTerminalSignal::default();
    let producer_terminal = terminal.clone();
    let stream = async_stream::stream! {
        let mut text_offsets = BTreeMap::<usize, usize>::new();
        while let Some(event) = generation.next().await {
            match event {
                Ok(LlmChoiceEvent::Delta { index, text, logprobs }) => {
                    let offset = *text_offsets.entry(index).or_default();
                    yield Ok::<Bytes, Infallible>(Bytes::from(completion_choice_chunk(&id, created, &model, index, &text, logprobs.as_ref(), Some(offset), None, None)));
                    let raw_bytes = logprobs
                        .as_ref()
                        .map_or_else(|| text.len(), |value| value.chosen.bytes.len());
                    *text_offsets.entry(index).or_default() = offset.saturating_add(raw_bytes);
                }
                Ok(LlmChoiceEvent::SemanticDelta { .. }) => {
                    let api = ApiError::internal("legacy completion received a semantic event");
                    if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                    producer_terminal.fail();
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&ErrorBody { error: api.error }).unwrap())));
                    break;
                }
                Ok(LlmChoiceEvent::Finished { index, reason, .. }) => {
                    yield Ok(Bytes::from(completion_choice_chunk(&id, created, &model, index, "", None, None, Some(choice_finish_reason(reason)), None)));
                }
                Ok(LlmChoiceEvent::FinishedAll { usage }) => {
                    if let Some(activity) = &activity {
                        activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
                        set_activity_timing(activity, usage);
                    }
                    yield Ok(Bytes::from(completion_choice_chunk(&id, created, &model, 0, "", None, None, None, Some(usage))));
                    producer_terminal.complete();
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    break;
                }
                Ok(LlmChoiceEvent::Failed { index: Some(_), .. }) => {}
                Ok(LlmChoiceEvent::Failed { index: None, message }) => {
                    let api = ApiError::internal(message);
                    if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                    producer_terminal.fail();
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&ErrorBody { error: api.error }).unwrap())));
                    break;
                }
                Err(error) => {
                    let api = ApiError::from(UseCaseError::from(error));
                    if let Some(activity) = &activity {
                        activity.complete_stream_failure(activity_outcome(&api), api.activity_error());
                    }
                    let body = ErrorBody { error: api.error };
                    producer_terminal.fail();
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&body).unwrap())));
                    break;
                }
            }
        }
    };
    let mut response = sse_response(Body::from_stream(stream));
    response.extensions_mut().insert(terminal);
    response
}

#[allow(
    clippy::too_many_arguments,
    reason = "the legacy completion chunk has independent wire fields"
)]
fn completion_choice_chunk(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    text: &str,
    logprobs: Option<&LlmTokenLogprobs>,
    text_offset: Option<usize>,
    finish_reason: Option<&str>,
    usage: Option<LlmUsage>,
) -> String {
    let timings = usage.map(|usage| TimingObject::from(usage.timings));
    let usage = usage.map(UsageObject::from);
    let choices = if usage.is_some() {
        json!([])
    } else {
        let logprobs = logprobs.map(|value| {
            let mut value =
                completion_logprobs(std::slice::from_ref(value)).expect("one logprob is nonempty");
            value["text_offset"] = json!([text_offset.unwrap_or(0)]);
            value
        });
        json!([{"text": text, "index": index, "logprobs": logprobs, "finish_reason": finish_reason}])
    };
    format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": choices,
            "usage": usage,
            "timings": timings,
        })
    )
}

fn chat_choice_chunk(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    delta: Value,
    logprobs: Option<Value>,
    finish: Option<&str>,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id":id,"object":"chat.completion.chunk","created":created,"model":model,
            "choices":[{"index":index,"delta":delta,"logprobs":logprobs,"finish_reason":finish}],
            "usage":null
        })
    )
}

fn chat_aggregate_chunk(
    id: &str,
    created: u64,
    model: &str,
    usage: Option<LlmUsage>,
    timings: LlmTimings,
) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id":id,"object":"chat.completion.chunk","created":created,"model":model,
            "choices":[],"usage":usage.map(UsageObject::from),"timings":TimingObject::from(timings)
        })
    )
}

fn chat_delta_logprobs(value: &LlmTokenLogprobs) -> Value {
    chat_logprobs(std::slice::from_ref(value)).expect("one logprob is nonempty")
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps the ordered Responses SSE protocol state machine in one producer"
)]
fn responses_stream(
    mut generation: ManagedChoiceGeneration,
    id: String,
    message_id: String,
    created: u64,
    model: String,
    config: ResponseConfig,
    activity: Option<ActivityContext>,
) -> Response {
    let terminal = StreamTerminalSignal::default();
    let producer_terminal = terminal.clone();
    let stream = async_stream::stream! {
        let mut sequence = 0_u64;
        let initial = response_snapshot(&id, created, &model, Vec::new(), None, None, &config);
        for (event, data) in [("response.created", json!({"type":"response.created","response":initial})),
            ("response.in_progress", json!({"type":"response.in_progress","response":response_snapshot(&id, created, &model, Vec::new(), None, None, &config)}))] {
            yield Ok::<Bytes, Infallible>(Bytes::from(event_frame(event, with_sequence(data, sequence))));
            sequence += 1;
        }
        let mut choice = ChoiceAccumulator::default();
        let mut choice_finished = None;
        while let Some(event) = generation.next().await {
            match event {
                Ok(LlmChoiceEvent::Delta { index: 0, text: delta, logprobs }) => {
                    choice.text.push_str(&delta);
                    if let Some(value) = logprobs { choice.logprobs.push(value); }
                    let output_index = choice.message_output_index();
                    if choice.text == delta {
                        for (name, data) in [
                            ("response.output_item.added", json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":message_id,"type":"message","status":"in_progress","role":"assistant","content":[]}})),
                            ("response.content_part.added", json!({"type":"response.content_part.added","item_id":message_id,"output_index":output_index,"content_index":0,"part":output_part("", &[])})),
                        ] { yield Ok(Bytes::from(event_frame(name, with_sequence(data, sequence)))); sequence += 1; }
                    }
                    let logprobs = choice.logprobs.last().map(|value| vec![response_token_logprob(value)]).unwrap_or_default();
                    let data = json!({"type":"response.output_text.delta","item_id":message_id,"output_index":output_index,"content_index":0,"delta":delta,"logprobs":logprobs});
                    yield Ok(Bytes::from(event_frame("response.output_text.delta", with_sequence(data, sequence)))); sequence += 1;
                }
                Ok(LlmChoiceEvent::SemanticDelta { index: 0, delta: LlmSemanticDelta::Text(delta) }) => {
                    let first = choice.text.is_empty();
                    choice.text.push_str(&delta);
                    let output_index = choice.message_output_index();
                    if first {
                        for (name, data) in [
                            ("response.output_item.added", json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":message_id,"type":"message","status":"in_progress","role":"assistant","content":[]}})),
                            ("response.content_part.added", json!({"type":"response.content_part.added","item_id":message_id,"output_index":output_index,"content_index":0,"part":output_part("", &[])})),
                        ] { yield Ok(Bytes::from(event_frame(name, with_sequence(data, sequence)))); sequence += 1; }
                    }
                    let data = json!({"type":"response.output_text.delta","item_id":message_id,"output_index":output_index,"content_index":0,"delta":delta,"logprobs":[]});
                    yield Ok(Bytes::from(event_frame("response.output_text.delta", with_sequence(data, sequence)))); sequence += 1;
                }
                Ok(LlmChoiceEvent::SemanticDelta { index: 0, delta: LlmSemanticDelta::Reasoning(delta) }) => {
                    let first = choice.reasoning.is_empty();
                    choice.reasoning.push_str(&delta);
                    let output_index = choice.reasoning_output_index();
                    if first {
                        for (name, data) in [
                            ("response.output_item.added", json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":choice.reasoning_id,"type":"reasoning","status":"in_progress","summary":[]}})),
                            ("response.reasoning_summary_part.added", json!({"type":"response.reasoning_summary_part.added","item_id":choice.reasoning_id,"output_index":output_index,"summary_index":0,"part":{"type":"summary_text","text":""}})),
                        ] { yield Ok(Bytes::from(event_frame(name, with_sequence(data, sequence)))); sequence += 1; }
                    }
                    let data = json!({"type":"response.reasoning_summary_text.delta","item_id":choice.reasoning_id,"output_index":output_index,"summary_index":0,"delta":delta});
                    yield Ok(Bytes::from(event_frame("response.reasoning_summary_text.delta", with_sequence(data, sequence)))); sequence += 1;
                }
                Ok(LlmChoiceEvent::SemanticDelta { index: 0, delta: LlmSemanticDelta::ToolCall { index, id: call_id, name, arguments } }) => {
                    let call = match choice.tool_call_mut(index) {
                        Ok(call) => call,
                        Err(api) => {
                            if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                            let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
                            producer_terminal.fail();
                            yield Ok(Bytes::from(event_frame("error", data)));
                            break;
                        }
                    };
                    let first = !call.started;
                    call.started = true;
                    if let Some(value) = call_id { call.id = value; }
                    if let Some(value) = name { call.name = value; }
                    let output_index = call.output_index.expect("tool call receives an output index");
                    if first {
                        let data = json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":call.item_id,"type":"function_call","status":"in_progress","call_id":call.id,"name":call.name,"arguments":""}});
                        yield Ok(Bytes::from(event_frame("response.output_item.added", with_sequence(data, sequence)))); sequence += 1;
                    }
                    call.arguments.push_str(&arguments);
                    let data = json!({"type":"response.function_call_arguments.delta","item_id":call.item_id,"output_index":output_index,"delta":arguments});
                    yield Ok(Bytes::from(event_frame("response.function_call_arguments.delta", with_sequence(data, sequence)))); sequence += 1;
                }
                Ok(LlmChoiceEvent::Finished { index: 0, reason, .. }) => {
                    choice.finish_reason = Some(reason);
                    choice_finished = Some(reason);
                    let status = if reason == LlmChoiceFinishReason::Length { "incomplete" } else { "completed" };
                    let mut done_groups = Vec::<(usize, Vec<(&'static str, Value)>)>::new();
                    if let Some(output_index) = choice.reasoning_output_index {
                        done_groups.push((output_index, vec![
                            ("response.reasoning_summary_text.done", json!({"type":"response.reasoning_summary_text.done","item_id":choice.reasoning_id,"output_index":output_index,"summary_index":0,"text":choice.reasoning})),
                            ("response.reasoning_summary_part.done", json!({"type":"response.reasoning_summary_part.done","item_id":choice.reasoning_id,"output_index":output_index,"summary_index":0,"part":{"type":"summary_text","text":choice.reasoning}})),
                            ("response.output_item.done", json!({"type":"response.output_item.done","output_index":output_index,"item":response_reasoning(&choice.reasoning_id,status,&choice.reasoning)})),
                        ]));
                    }
                    if let Some(output_index) = choice.message_output_index {
                        let logprobs = response_logprobs(&choice.logprobs);
                        done_groups.push((output_index, vec![
                            ("response.output_text.done", json!({"type":"response.output_text.done","item_id":message_id,"output_index":output_index,"content_index":0,"text":choice.text,"logprobs":logprobs})),
                            ("response.content_part.done", json!({"type":"response.content_part.done","item_id":message_id,"output_index":output_index,"content_index":0,"part":output_part(&choice.text,&choice.logprobs)})),
                            ("response.output_item.done", json!({"type":"response.output_item.done","output_index":output_index,"item":response_message(&message_id,status,&choice.text,&choice.logprobs)})),
                        ]));
                    }
                    for call in &choice.tool_calls {
                        let Some(output_index) = call.output_index else { continue };
                        done_groups.push((output_index, vec![
                            ("response.function_call_arguments.done", json!({"type":"response.function_call_arguments.done","item_id":call.item_id,"output_index":output_index,"arguments":call.arguments})),
                            ("response.output_item.done", json!({"type":"response.output_item.done","output_index":output_index,"item":response_function_call(call,status)})),
                        ]));
                    }
                    done_groups.sort_by_key(|(output_index, _)| *output_index);
                    for (_, events) in done_groups {
                        for (name, data) in events {
                            yield Ok(Bytes::from(event_frame(name, with_sequence(data, sequence))));
                            sequence += 1;
                        }
                    }
                }
                Ok(LlmChoiceEvent::FinishedAll { usage }) => {
                    let Some(reason) = choice_finished else {
                        let api = ApiError::internal("aggregate terminal arrived before the response choice finished");
                        if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                        let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
                        producer_terminal.fail();
                        yield Ok(Bytes::from(event_frame("error", data)));
                        break;
                    };
                    if let Some(activity) = &activity { activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens); set_activity_timing(activity, usage); }
                    let final_event = if reason == LlmChoiceFinishReason::Length { "response.incomplete" } else { "response.completed" };
                    let terminal_response = response_snapshot(&id,created,&model,response_output_items(&message_id,&choice),Some(reason),Some(usage),&config);
                    let data = json!({"type":final_event,"response":terminal_response,"timings":TimingObject::from(usage.timings)});
                    producer_terminal.complete();
                    yield Ok(Bytes::from(event_frame(final_event, with_sequence(data, sequence))));
                    break;
                }
                Ok(LlmChoiceEvent::Failed { index: Some(_), .. }) => {}
                Ok(LlmChoiceEvent::Failed { index: None, message }) => {
                    let api = ApiError::internal(message);
                    if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
                    let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
                    producer_terminal.fail();
                    yield Ok(Bytes::from(event_frame("error", data)));
                    break;
                }
                Ok(_) => { let api = ApiError::internal("Responses generation returned an invalid choice index"); if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); } let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param}); producer_terminal.fail(); yield Ok(Bytes::from(event_frame("error",data))); break; }
                Err(error) => {
                    let api = ApiError::from(UseCaseError::from(error));
                    if let Some(activity) = &activity {
                        activity.complete_stream_failure(activity_outcome(&api), api.activity_error());
                    }
                    let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
                    producer_terminal.fail();
                    yield Ok(Bytes::from(event_frame("error", data)));
                    break;
                }
            }
        }
        if producer_terminal.is_pending() {
            let api = ApiError::internal("Responses generation ended without a terminal event");
            if let Some(activity) = &activity { activity.complete_stream_failure(activity_outcome(&api), api.activity_error()); }
            let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
            producer_terminal.fail();
            yield Ok(Bytes::from(event_frame("error", data)));
        }
    };
    let mut response = sse_response(Body::from_stream(stream));
    response.extensions_mut().insert(terminal);
    response
}

fn response_snapshot(
    id: &str,
    created: u64,
    model: &str,
    output: Vec<Value>,
    reason: Option<LlmChoiceFinishReason>,
    usage: Option<LlmUsage>,
    config: &ResponseConfig,
) -> Value {
    let output_text = output
        .iter()
        .filter(|item| item["type"] == "message")
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect::<String>();
    serde_json::to_value(response_object(
        id.to_string(),
        created,
        model.to_string(),
        output,
        output_text,
        reason,
        usage,
        false,
        config,
    ))
    .expect("Responses response object is serializable")
}

#[allow(clippy::too_many_arguments)]
fn response_object(
    id: String,
    created_at: u64,
    model: String,
    output: Vec<Value>,
    output_text: String,
    reason: Option<LlmChoiceFinishReason>,
    usage: Option<LlmUsage>,
    include_timings: bool,
    config: &ResponseConfig,
) -> ResponsesResponse {
    let incomplete = reason == Some(LlmChoiceFinishReason::Length);
    let timings = if include_timings {
        usage.map(|usage| usage.timings.into())
    } else {
        None
    };
    ResponsesResponse {
        id,
        object: "response",
        created_at,
        status: if incomplete {
            "incomplete"
        } else if reason.is_some() {
            "completed"
        } else {
            "in_progress"
        },
        background: false,
        error: None,
        incomplete_details: incomplete.then(|| json!({"reason":"max_output_tokens"})),
        instructions: config.instructions.clone(),
        max_output_tokens: config.max_output_tokens,
        model,
        output,
        output_text,
        parallel_tool_calls: config.parallel_tool_calls,
        previous_response_id: None,
        reasoning: config.reasoning.clone(),
        store: false,
        temperature: config.temperature,
        text: config.text.clone(),
        tool_choice: config.tool_choice.clone(),
        tools: config
            .tools
            .iter()
            .map(|tool| serde_json::to_value(tool).expect("tool serializes"))
            .collect(),
        top_logprobs: config.top_logprobs,
        top_p: config.top_p,
        truncation: "disabled",
        metadata: json!({}),
        usage: usage.map(ResponsesUsage::from),
        timings,
    }
}

fn response_message(id: &str, status: &str, text: &str, logprobs: &[LlmTokenLogprobs]) -> Value {
    json!({"id":id,"type":"message","status":status,"role":"assistant","content":[output_part(text,logprobs)]})
}

fn output_part(text: &str, logprobs: &[LlmTokenLogprobs]) -> Value {
    json!({"type":"output_text","text":text,"annotations":[],"logprobs":response_logprobs(logprobs)})
}

fn response_output_items(message_id: &str, choice: &ChoiceAccumulator) -> Vec<Value> {
    let status = if choice.finish_reason == Some(LlmChoiceFinishReason::Length) {
        "incomplete"
    } else {
        "completed"
    };
    let mut output = Vec::new();
    let mut fallback_index = 0_usize;
    if choice.reasoning_output_index.is_some() {
        let id = if choice.reasoning_id.is_empty() {
            next_id("rs")
        } else {
            choice.reasoning_id.clone()
        };
        output.push((
            choice.reasoning_output_index.unwrap_or_else(|| {
                let index = fallback_index;
                fallback_index += 1;
                index
            }),
            response_reasoning(&id, status, &choice.reasoning),
        ));
    }
    if choice.message_output_index.is_some() {
        output.push((
            choice.message_output_index.unwrap_or_else(|| {
                let index = fallback_index;
                fallback_index += 1;
                index
            }),
            response_message(message_id, status, &choice.text, &choice.logprobs),
        ));
    }
    for call in &choice.tool_calls {
        let index = call.output_index.unwrap_or_else(|| {
            let index = fallback_index;
            fallback_index += 1;
            index
        });
        output.push((index, response_function_call(call, status)));
    }
    if output.is_empty() {
        output.push((
            0,
            response_message(message_id, status, "", &choice.logprobs),
        ));
    }
    output.sort_by_key(|(index, _)| *index);
    output.into_iter().map(|(_, item)| item).collect()
}

fn response_reasoning(id: &str, status: &str, text: &str) -> Value {
    json!({"id":id,"type":"reasoning","status":status,"summary":[{"type":"summary_text","text":text}]})
}

fn response_function_call(call: &AccumulatedToolCall, status: &str) -> Value {
    let item_id = if call.item_id.is_empty() {
        next_id("fc")
    } else {
        call.item_id.clone()
    };
    json!({
        "id":item_id,"type":"function_call","status":status,
        "call_id":call.id,"name":call.name,"arguments":call.arguments
    })
}

fn response_token_logprob(value: &LlmTokenLogprobs) -> Value {
    let mut chosen = token_logprob(&value.chosen);
    chosen["top_logprobs"] = Value::Array(value.top.iter().map(token_logprob).collect());
    chosen
}

fn response_logprobs(values: &[LlmTokenLogprobs]) -> Vec<Value> {
    values.iter().map(response_token_logprob).collect()
}

fn tool_choice_wire(choice: Option<&ToolChoiceValue>, tools: Option<&[FunctionTool]>) -> Value {
    match choice {
        None if tools.is_some_and(|tools| !tools.is_empty()) => json!("auto"),
        None => json!("none"),
        Some(ToolChoiceValue::Mode(mode)) => json!(mode),
        Some(ToolChoiceValue::Named(named)) => json!({
            "type":"function","name":named_tool_name(named).ok()
        }),
    }
}

fn responses_text_wire(text: Option<&ResponsesText>) -> Value {
    let format = text.and_then(|text| text.format.as_ref()).map_or_else(
        || json!({"type":"text"}),
        |format| match format.kind.as_str() {
            "json_schema" => json!({
                "type":"json_schema","name":format.name,"description":format.description,
                "schema":format.schema,"strict":format.strict
            }),
            kind => json!({"type":kind}),
        },
    );
    json!({"format":format,"verbosity":"medium"})
}

fn responses_reasoning_wire(reasoning: Option<&ResponsesReasoning>) -> Value {
    json!({
        "effort":reasoning.and_then(|reasoning| reasoning.effort.as_deref()),
        "summary":null
    })
}
fn with_sequence(mut data: Value, sequence: u64) -> Value {
    data["sequence_number"] = json!(sequence);
    data
}
fn event_frame(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}
fn sse_response(body: Body) -> Response {
    sse::response(body)
}
fn next_id(prefix: &str) -> String {
    let mut entropy = [0_u8; 24];
    getrandom::fill(&mut entropy).expect("OS entropy is required for completion identifiers");
    format!(
        "{prefix}_{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(entropy)
    )
}

pub(super) fn valid_chat_completion_id(id: &str) -> bool {
    id.strip_prefix("chatcmpl_").is_some_and(|value| {
        value.len() == 32
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    })
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn choice_finish_reason(reason: LlmChoiceFinishReason) -> &'static str {
    match reason {
        LlmChoiceFinishReason::Stop | LlmChoiceFinishReason::Cancelled => "stop",
        LlmChoiceFinishReason::Length => "length",
        LlmChoiceFinishReason::ToolCalls => "tool_calls",
    }
}
fn unsupported(param: &'static str) -> ApiError {
    ApiError::invalid_request(
        format!("`{param}` is not supported"),
        Some(param),
        Some("unsupported_parameter"),
    )
}
fn invalid(message: &'static str, param: &'static str) -> ApiError {
    ApiError::invalid_request(message, Some(param), Some("invalid_parameter"))
}
fn activity_outcome(error: &ApiError) -> ActivityOutcome {
    match error.error.code.as_deref() {
        Some("request_timeout") => ActivityOutcome::Timeout,
        Some("resource_exhausted") => ActivityOutcome::ResourceExhausted,
        Some("server_shutdown") => ActivityOutcome::Cancelled,
        _ => ActivityOutcome::ServerError,
    }
}
impl From<LlmUsage> for UsageObject {
    fn from(value: LlmUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}
impl From<LlmTimings> for TimingObject {
    fn from(value: LlmTimings) -> Self {
        Self {
            cache_n: value.cache_n,
            prompt_n: value.prompt_n,
            prompt_ms: value.prompt_ms,
            prompt_per_token_ms: value.prompt_per_token_ms,
            prompt_per_second: value.prompt_per_second,
            predicted_n: value.predicted_n,
            predicted_ms: value.predicted_ms,
            predicted_per_token_ms: value.predicted_per_token_ms,
            predicted_per_second: value.predicted_per_second,
        }
    }
}

fn set_activity_timing(activity: &ActivityContext, usage: LlmUsage) {
    activity.set_llm_reasoning_tokens(usage.reasoning_tokens);
    activity.set_llm_timing(
        usage.queue_time_ms,
        usage.eval_time_ms,
        usage.timings.prompt_per_second,
        usage.timings.predicted_per_second,
    );
}
impl From<LlmUsage> for ResponsesUsage {
    fn from(value: LlmUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            input_tokens_details: json!({
                "cached_tokens": value.timings.cache_n,
                "cache_write_tokens": 0
            }),
            output_tokens: value.completion_tokens,
            output_tokens_details: json!({"reasoning_tokens":value.reasoning_tokens}),
            total_tokens: value.total_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::activity::{ActivityFilter, ActivityHub, ActivityOperation, ActivityTransport};
    use crate::application::ActivityPolicy;
    use crate::application::RuntimeError;
    use http_body_util::BodyExt;
    use orchion::{GenerationEvent, GenerationFinishReason};
    use std::sync::atomic::AtomicBool;
    use tokio::sync::{Notify, mpsc};

    fn png_data_url() -> String {
        let mut bytes = Vec::new();
        image::ImageEncoder::write_image(
            image::codecs::png::PngEncoder::new(&mut bytes),
            &[255, 0, 0, 255],
            1,
            1,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    fn jpeg_data_url() -> String {
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode(&[255, 0, 0], 1, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    #[test]
    fn strict_data_image_validation_accepts_png_and_jpeg_only() {
        assert_eq!(
            parse_data_image(&png_data_url(), "image").unwrap().format,
            LlmImageFormat::Png
        );
        assert_eq!(
            parse_data_image(&jpeg_data_url(), "image").unwrap().format,
            LlmImageFormat::Jpeg
        );
        for invalid_url in [
            "https://example.com/a.png",
            "file:///tmp/a.png",
            "/tmp/a.png",
            "data:image/png;charset=utf-8;base64,AAAA",
            "data:image/jpg;base64,AAAA",
            "data:image/png;base64,",
            "data:image/png;base64,AA A=",
            "data:image/png;base64,!!!!",
        ] {
            assert!(
                parse_data_image(invalid_url, "image").is_err(),
                "{invalid_url}"
            );
        }
        let jpeg_payload = jpeg_data_url()
            .strip_prefix("data:image/jpeg;base64,")
            .unwrap()
            .to_string();
        assert!(
            parse_data_image(&format!("data:image/png;base64,{jpeg_payload}"), "image").is_err()
        );
    }

    #[test]
    fn chat_images_preserve_order_and_reject_role_detail_and_multiple_choices() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":[
                {"type":"text","text":"before"},
                {"type":"image_url","image_url":{"url":png_data_url(),"detail":"auto"}},
                {"type":"text","text":"after"}
            ]}]
        }))
        .unwrap();
        let (advanced, _) = chat_advanced_request(&request).unwrap();
        let LlmAdvancedInput::Messages(messages) = advanced.input else {
            panic!()
        };
        assert!(matches!(messages[0].content.as_slice(), [
            LlmContentPart::Text { text: before },
            LlmContentPart::Image(_),
            LlmContentPart::Text { text: after },
        ] if before == "before" && after == "after"));

        for value in [
            json!({"model":"a/b","messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":png_data_url()}}]}]}),
            json!({"model":"a/b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":png_data_url(),"detail":"high"}}]}]}),
            json!({"model":"a/b","n":2,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":png_data_url()}}]}]}),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(value).unwrap();
            assert!(chat_advanced_request(&request).is_err());
        }
    }

    #[test]
    fn responses_images_reject_file_remote_and_non_user_inputs() {
        let accepted: ResponsesInput = serde_json::from_value(json!([{
            "type":"message","role":"user","content":[
                {"type":"input_text","text":"before"},
                {"type":"input_image","image_url":png_data_url(),"detail":"auto"},
                {"type":"input_text","text":"after"}
            ]
        }]))
        .unwrap();
        let messages = responses_rich_messages(&accepted, None).unwrap();
        assert!(matches!(messages[0].content[1], LlmContentPart::Image(_)));
        for value in [
            json!([{"type":"message","role":"user","content":[{"type":"input_image","file_id":"file-1"}]}]),
            json!([{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png"}]}]),
            json!([{"type":"message","role":"assistant","content":[{"type":"input_image","image_url":png_data_url()}]}]),
            json!([{"type":"message","role":"user","content":[{"type":"input_image","image_url":png_data_url(),"detail":"low"}]}]),
            json!([{"type":"message","role":"user","content":[{"type":"input_image","image_url":png_data_url(),"detail":"high"}]}]),
            json!([{"type":"message","role":"user","content":[{"type":"input_image","image_url":png_data_url(),"detail":"original"}]}]),
        ] {
            let input: ResponsesInput = serde_json::from_value(value).unwrap();
            assert!(responses_rich_messages(&input, None).is_err());
        }
    }

    #[test]
    fn chat_normalization_enforces_the_incremental_image_count_limit() {
        let parts = (0..=MAX_IMAGES)
            .map(|_| json!({"type":"image_url","image_url":{"url":png_data_url()}}))
            .collect::<Vec<_>>();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":parts}]
        }))
        .unwrap();
        assert!(chat_advanced_request(&request).is_err());
    }

    #[test]
    fn normalization_budget_rejects_count_and_estimated_bytes_before_decode() {
        let mut count = ImageBudget {
            count: MAX_IMAGES,
            ..ImageBudget::default()
        };
        assert!(count.reserve_decoded("AAAA", "image").is_err());

        let mut bytes = ImageBudget {
            decoded_bytes: MAX_TOTAL_IMAGE_BYTES - 2,
            ..ImageBudget::default()
        };
        assert!(bytes.reserve_decoded("AAAA", "image").is_err());
        assert_eq!(bytes.decoded_bytes, MAX_TOTAL_IMAGE_BYTES - 2);
    }

    #[test]
    fn normalization_budget_rejects_aggregate_pixels_immediately() {
        let mut budget = ImageBudget {
            pixels: MAX_TOTAL_IMAGE_PIXELS,
            ..ImageBudget::default()
        };
        assert!(budget.record_dimensions(1, 1, "image").is_err());
        assert_eq!(budget.pixels, MAX_TOTAL_IMAGE_PIXELS);
    }

    #[test]
    fn error_nullable_fields_are_explicit() {
        let value = serde_json::to_value(ErrorBody {
            error: ApiError::invalid_request("bad", None, None).error,
        })
        .unwrap();
        assert_eq!(value["error"]["param"], Value::Null);
        assert_eq!(value["error"]["code"], Value::Null);
    }

    #[test]
    fn shutdown_error_is_sanitized_and_classified_as_cancelled() {
        let error = ApiError::from(UseCaseError::from(RuntimeError::ShuttingDown));
        assert_eq!(error.error.code.as_deref(), Some("server_shutdown"));
        assert_eq!(activity_outcome(&error), ActivityOutcome::Cancelled);
        assert_eq!(error.error.message, "server is shutting down");
    }

    #[test]
    fn timeout_error_stays_408_and_is_classified_as_timeout() {
        let error = ApiError::from(UseCaseError::from(RuntimeError::Timeout(
            "LLM generation timed out".to_string(),
        )));
        assert_eq!(error.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error.error.code.as_deref(), Some("request_timeout"));
        assert_eq!(activity_outcome(&error), ActivityOutcome::Timeout);
    }

    #[test]
    fn embedding_input_and_dimension_errors_are_client_errors() {
        let token = map_embedding_error(RuntimeError::Core(orchion::OrchionError::Inference {
            message: "llama.cpp embedding failed: token id 999 is outside model vocabulary"
                .to_string(),
        }));
        assert_eq!(token.status(), StatusCode::BAD_REQUEST);
        assert_eq!(token.error.param.as_deref(), Some("input"));

        let dimensions =
            map_embedding_error(RuntimeError::Core(orchion::OrchionError::Inference {
                message: "embedding dimensions must be in 32..=1024".to_string(),
            }));
        assert_eq!(dimensions.status(), StatusCode::BAD_REQUEST);
        assert_eq!(dimensions.error.param.as_deref(), Some("dimensions"));
    }

    #[test]
    fn chat_rejects_known_unsupported_fields() {
        let request: ChatCompletionRequest = serde_json::from_value(
            json!({"model":"a/b","messages":[{"role":"user","content":"hi"}],"functions":[]}),
        )
        .unwrap();
        assert_eq!(
            validate_chat(&request).unwrap_err().error.code.as_deref(),
            Some("unsupported_parameter")
        );
        let null_tools: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b","messages":[{"role":"user","content":"hi"}],"tools":null
        }))
        .unwrap();
        assert!(validate_chat(&null_tools).is_ok());
        for field in [
            "user",
            "web_search_options",
            "moderation",
            "prompt_cache_options",
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model":"a/b","messages":[{"role":"user","content":"hi"}], (field):null
            }))
            .unwrap();
            assert!(validate_chat(&request).is_ok(), "{field}");
        }
    }

    #[test]
    fn chat_reasoning_control_requires_stream_and_one_choice() {
        for (value, param) in [
            (
                json!({"model":"a/b","messages":[{"role":"user","content":"hi"}],"reasoning_control":true}),
                "reasoning_control",
            ),
            (
                json!({"model":"a/b","messages":[{"role":"user","content":"hi"}],"stream":true,"n":2,"reasoning_control":true}),
                "reasoning_control",
            ),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(value).unwrap();
            assert_eq!(
                validate_chat(&request).unwrap_err().error.param.as_deref(),
                Some(param)
            );
        }
        assert!(valid_chat_completion_id(
            "chatcmpl_0123456789abcdefghijklmnopqrstuv"
        ));
        assert!(!valid_chat_completion_id("chatcmpl_1"));
    }

    #[test]
    fn openai_requests_ignore_unknown_and_null_sdk_fields_but_keep_known_types_strict() {
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":"hi","sdk_extra":"ignored"}],
            "tools":null,
            "response_format":null,
            "sdk_request_id":"ignored"
        }))
        .unwrap();
        assert!(validate_chat(&chat).is_ok());

        let responses: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b",
            "input":[{"type":"message","role":"user","content":"hi","id":"sdk-item"}],
            "store":null,
            "metadata":null,
            "sdk_extension":{"harmless":true}
        }))
        .unwrap();
        assert!(validate_responses(&responses).is_ok());

        assert!(
            serde_json::from_value::<ChatCompletionRequest>(json!({
                "model":"a/b","messages":[],"temperature":"hot"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ResponsesRequest>(json!({
                "model":"a/b","input":"hi","stream":"yes"
            }))
            .is_err()
        );
    }

    #[test]
    fn chat_http_omitted_tool_choice_defaults_to_auto_only_when_tools_are_supplied() {
        let with_tools: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]
        }))
        .unwrap();
        let without_tools: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b","messages":[{"role":"user","content":"hi"}]
        }))
        .unwrap();

        assert_eq!(
            chat_advanced_request(&with_tools).unwrap().0.tool_choice,
            LlmToolChoice::Auto
        );
        assert_eq!(
            chat_advanced_request(&without_tools).unwrap().0.tool_choice,
            LlmToolChoice::None
        );
    }

    #[test]
    fn chat_logit_bias_is_bounded_and_reasoning_none_is_explicit() {
        let bias = (0..=256)
            .map(|token| (token.to_string(), 0.0))
            .collect::<BTreeMap<_, _>>();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":"hi"}],
            "logit_bias": bias
        }))
        .unwrap();
        let error = validate_chat(&request).unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.error.param.as_deref(), Some("logit_bias"));

        assert_eq!(
            reasoning_options(None, "reasoning_effort").unwrap().enabled,
            None
        );
        assert_eq!(
            reasoning_options(Some("none"), "reasoning_effort")
                .unwrap()
                .enabled,
            Some(false)
        );
    }

    #[test]
    fn responses_http_omitted_tool_choice_defaults_to_auto_only_when_tools_are_supplied() {
        let with_tools: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b",
            "input":"hi",
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]
        }))
        .unwrap();
        let without_tools: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi"})).unwrap();

        assert_eq!(
            responses_advanced_request(&with_tools)
                .unwrap()
                .0
                .tool_choice,
            LlmToolChoice::Auto
        );
        assert_eq!(
            responses_advanced_request(&without_tools)
                .unwrap()
                .0
                .tool_choice,
            LlmToolChoice::None
        );
    }

    #[test]
    fn responses_input_token_preparation_reuses_full_stateless_semantics() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b",
            "input":"hi",
            "instructions":"be concise",
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],
            "tool_choice":"required",
            "parallel_tool_calls":false,
            "reasoning":{"effort":"low"},
            "text":{"format":{"type":"json_object"}},
            "metadata":null,
            "unknown_future_field":null
        }))
        .unwrap();
        validate_responses(&request).unwrap();
        let (prepared, _) = responses_advanced_request(&request).unwrap();
        assert_eq!(prepared.tools.len(), 1);
        assert_eq!(prepared.tool_choice, LlmToolChoice::Required);
        assert!(!prepared.parallel_tool_calls);
        assert_eq!(prepared.reasoning.enabled, Some(true));
        assert_eq!(prepared.output, LlmOutputConstraint::JsonObject);
        let LlmAdvancedInput::Messages(messages) = prepared.input else {
            panic!("expected semantic input")
        };
        assert_eq!(messages[0].role, LlmSemanticRole::Developer);
        assert_eq!(messages[1].role, LlmSemanticRole::User);
    }

    #[test]
    fn embeddings_accept_all_input_shapes_and_reject_empty_or_oversized_batches() {
        for (value, expected) in [
            (json!("hello"), 1),
            (json!(["hello", "world"]), 2),
            (json!([1, 2, 3]), 1),
            (json!([[1, 2], [3]]), 2),
        ] {
            let input: EmbeddingsInput = serde_json::from_value(value).unwrap();
            assert_eq!(embedding_inputs(&input).unwrap().len(), expected);
        }
        for value in [json!(""), json!([]), json!([[]])] {
            let input: EmbeddingsInput = serde_json::from_value(value).unwrap();
            assert_eq!(
                embedding_inputs(&input).unwrap_err().error.param.as_deref(),
                Some("input")
            );
        }
        let input = EmbeddingsInput::Texts(vec!["x".to_string(); MAX_EMBEDDING_INPUTS + 1]);
        assert!(embedding_inputs(&input).is_err());
    }

    #[test]
    fn responses_defaults_to_stateless_and_rejects_store_true() {
        let omitted: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi"})).unwrap();
        let enabled: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi","store":true})).unwrap();
        let disabled: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi","store":false})).unwrap();
        assert!(validate_responses(&omitted).is_ok());
        assert!(validate_responses(&enabled).is_err());
        assert!(validate_responses(&disabled).is_ok());
        let effectful: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b","input":"hi","background":true
        }))
        .unwrap();
        assert_eq!(
            validate_responses(&effectful)
                .unwrap_err()
                .error
                .param
                .as_deref(),
            Some("background")
        );
        for field in [
            "parallel_tool_calls",
            "text",
            "truncation",
            "service_tier",
            "moderation",
            "prompt_cache_options",
            "prompt",
        ] {
            let request: ResponsesRequest = serde_json::from_value(json!({
                "model":"a/b","input":"hi","store":false,(field):null
            }))
            .unwrap();
            assert!(validate_responses(&request).is_ok(), "{field}");
        }

        let too_small: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b","input":"hi","store":false,"max_output_tokens":15
        }))
        .unwrap();
        let error = validate_responses(&too_small).unwrap_err();
        assert_eq!(error.error.param.as_deref(), Some("max_output_tokens"));
        assert_eq!(error.error.code.as_deref(), Some("invalid_parameter"));
    }

    #[test]
    fn completion_validation_rejects_effectful_options_and_uses_raw_prompt_input() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model":"a/b","prompt":"raw prompt","suffix":null,"n":1,"echo":false,
            "sdk_field":"ignored"
        }))
        .unwrap();
        assert!(validate_completion(&request).is_ok());
        let (command, _) = completion_advanced_request(&request).unwrap();
        assert_eq!(
            command.input,
            LlmAdvancedInput::Prompt("raw prompt".to_string())
        );

        for value in [
            json!({"model":"a/b","prompt":"x","suffix":"effect"}),
            json!({"model":"a/b","prompt":"x","best_of":2}),
            json!({"model":"a/b","prompt":"x","echo":true}),
        ] {
            let request: CompletionRequest = serde_json::from_value(value).unwrap();
            assert_eq!(
                validate_completion(&request)
                    .unwrap_err()
                    .error
                    .code
                    .as_deref(),
                Some("unsupported_parameter")
            );
        }
        for value in [
            json!({"model":"a/b","prompt":"x","logprobs":6}),
            json!({"model":"a/b","prompt":"x","n":0}),
        ] {
            let request: CompletionRequest = serde_json::from_value(value).unwrap();
            assert_eq!(
                validate_completion(&request)
                    .unwrap_err()
                    .error
                    .code
                    .as_deref(),
                Some("invalid_parameter")
            );
        }
    }

    #[test]
    fn stream_options_accept_usage_and_explicitly_disabled_obfuscation_only() {
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"a/b",
            "messages":[{"role":"user","content":"hi"}],
            "stream":true,
            "stream_options":{"include_usage":true,"include_obfuscation":false}
        }))
        .unwrap();
        assert!(validate_chat(&chat).is_ok());

        let responses: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b",
            "input":"hi",
            "store":false,
            "stream":true,
            "stream_options":{"include_obfuscation":false}
        }))
        .unwrap();
        assert!(validate_responses(&responses).is_ok());

        let obfuscated: ResponsesRequest = serde_json::from_value(json!({
            "model":"a/b",
            "input":"hi",
            "store":false,
            "stream":true,
            "stream_options":{"include_obfuscation":true}
        }))
        .unwrap();
        assert_eq!(
            validate_responses(&obfuscated)
                .unwrap_err()
                .error
                .param
                .as_deref(),
            Some("stream_options.include_obfuscation")
        );
    }

    fn generation(
        events: impl IntoIterator<Item = Result<GenerationEvent, RuntimeError>>,
    ) -> ManagedChoiceGeneration {
        let (sender, receiver) = mpsc::channel(16);
        for event in events {
            match event {
                Ok(GenerationEvent::ContentDelta(text)) => sender
                    .try_send(Ok(LlmChoiceEvent::Delta {
                        index: 0,
                        text,
                        logprobs: None,
                    }))
                    .unwrap(),
                Ok(GenerationEvent::Finished { reason, usage }) => {
                    let reason = match reason {
                        GenerationFinishReason::Stop => LlmChoiceFinishReason::Stop,
                        GenerationFinishReason::Length => LlmChoiceFinishReason::Length,
                        GenerationFinishReason::Cancelled => LlmChoiceFinishReason::Cancelled,
                    };
                    sender
                        .try_send(Ok(LlmChoiceEvent::Finished {
                            index: 0,
                            reason,
                            usage,
                        }))
                        .unwrap();
                    sender
                        .try_send(Ok(LlmChoiceEvent::FinishedAll { usage }))
                        .unwrap();
                }
                Err(error) => sender.try_send(Err(error)).unwrap(),
            }
        }
        drop(sender);
        ManagedChoiceGeneration::new(
            receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        )
    }

    fn choice_generation(
        events: impl IntoIterator<Item = Result<LlmChoiceEvent, RuntimeError>>,
    ) -> ManagedChoiceGeneration {
        let (sender, receiver) = mpsc::channel(32);
        for event in events {
            sender.try_send(event).unwrap();
        }
        drop(sender);
        ManagedChoiceGeneration::new(
            receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        )
    }

    fn usage() -> LlmUsage {
        LlmUsage {
            prompt_tokens: 2,
            completion_tokens: 1,
            reasoning_tokens: 0,
            total_tokens: 3,
            queue_time_ms: Some(4),
            eval_time_ms: Some(5),
            timings: LlmTimings {
                cache_n: 0,
                prompt_n: 2,
                prompt_ms: 8.0,
                prompt_per_token_ms: 4.0,
                prompt_per_second: 250.0,
                predicted_n: 1,
                predicted_ms: 5.0,
                predicted_per_token_ms: 5.0,
                predicted_per_second: 200.0,
            },
        }
    }

    #[test]
    fn responses_usage_reports_restored_prompt_tokens_as_cached() {
        let mut value = usage();
        value.timings.cache_n = 1;
        value.timings.prompt_n = 1;
        let response = serde_json::to_value(ResponsesUsage::from(value)).unwrap();
        assert_eq!(response["input_tokens"], 2);
        assert_eq!(response["input_tokens_details"]["cached_tokens"], 1);
        assert_eq!(response["input_tokens_details"]["cache_write_tokens"], 0);
    }

    #[tokio::test]
    async fn indexed_semantic_choices_collect_interleaved_tools_reasoning_and_aggregate_usage() {
        let result = collect_choices(
            choice_generation([
                Ok(LlmChoiceEvent::Delta {
                    index: 1,
                    text: "plain".to_string(),
                    logprobs: None,
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::Reasoning("think".to_string()),
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: Some("call_1".to_string()),
                        name: Some("weather".to_string()),
                        arguments: "{\"city\":".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: None,
                        name: None,
                        arguments: "\"Paris\"}".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 1,
                    reason: LlmChoiceFinishReason::Length,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::ToolCalls,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]),
            2,
        )
        .await
        .unwrap();
        let choices = result
            .choices
            .into_iter()
            .map(chat_choice)
            .collect::<Vec<_>>();

        assert_eq!(choices[0].message.content, None);
        assert_eq!(
            choices[0].message.reasoning_content.as_deref(),
            Some("think")
        );
        assert_eq!(choices[0].message.tool_calls[0].id, "call_1");
        assert_eq!(
            choices[0].message.tool_calls[0].function.arguments,
            json!(r#"{"city":"Paris"}"#)
        );
        assert_eq!(choices[0].finish_reason, "tool_calls");
        assert_eq!(choices[1].message.content.as_deref(), Some("plain"));
        assert_eq!(choices[1].finish_reason, "length");
        assert_eq!(result.usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn chat_and_responses_mapping_emit_one_complete_tool_name_with_argument_suffixes() {
        fn tool_events() -> Vec<Result<LlmChoiceEvent, RuntimeError>> {
            vec![
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: Some("call_1".to_string()),
                        name: Some("weather".to_string()),
                        arguments: "{".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: None,
                        name: None,
                        arguments: r#""city":"Paris"}"#.to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::ToolCalls,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]
        }

        let chat = chat_stream(
            choice_generation(tool_events()),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            None,
            None,
        );
        let chat = String::from_utf8(
            chat.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert_eq!(chat.matches(r#""name":"weather""#).count(), 1);
        assert!(!chat.contains(r#""name":"w""#));

        let responses = responses_stream(
            choice_generation(tool_events()),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
        let responses = String::from_utf8(
            responses
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(responses.contains(r#""name":"weather""#));
        assert!(responses.contains(r#""arguments":"{\"city\":\"Paris\"}""#));
        assert!(!responses.contains(r#""name":"w""#));
    }

    #[tokio::test]
    async fn chat_sse_has_role_content_terminal_usage_and_one_done() {
        let response = chat_stream(
            generation([
                Ok(GenerationEvent::ContentDelta("hello".to_string())),
                Ok(GenerationEvent::Finished {
                    reason: GenerationFinishReason::Stop,
                    usage: usage(),
                }),
            ]),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            true,
            None,
            None,
        );
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
        assert!(body.contains("\"role\":\"assistant\""));
        assert!(body.contains("\"content\":\"hello\""));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("\"choices\":[]"));
        assert!(body.contains("\"prompt_tokens\":2"));
        let usage_chunk = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|chunk| chunk["choices"] == json!([]))
            .unwrap();
        assert_eq!(usage_chunk["timings"]["prompt_per_second"], 250.0);
        assert_eq!(usage_chunk["timings"]["predicted_per_second"], 200.0);
        assert_eq!(body.matches("data: [DONE]").count(), 1);
    }

    #[tokio::test]
    async fn chat_sse_without_usage_puts_timings_on_the_terminal_choice_chunk() {
        let response = chat_stream(
            generation([Ok(GenerationEvent::Finished {
                reason: GenerationFinishReason::Stop,
                usage: usage(),
            })]),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            None,
            None,
        );
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
        let terminal = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|chunk| chunk["choices"] == json!([]))
            .unwrap();
        assert_eq!(terminal["usage"], Value::Null);
        assert_eq!(terminal["timings"]["prompt_n"], 2);
        assert_eq!(terminal["timings"]["predicted_n"], 1);
    }

    #[tokio::test]
    async fn completion_sse_logprob_offsets_are_utf8_bytes_per_choice() {
        let logprobs = |bytes: &[u8]| LlmTokenLogprobs {
            chosen: orchion::LlmTokenAlternative {
                token_id: 1,
                bytes: bytes.to_vec(),
                logprob: -0.1,
            },
            top: Vec::new(),
        };
        let response = completion_stream(
            choice_generation([
                Ok(LlmChoiceEvent::Delta {
                    index: 0,
                    text: String::from_utf8_lossy(&[0xc3]).into_owned(),
                    logprobs: Some(logprobs(&[0xc3])),
                }),
                Ok(LlmChoiceEvent::Delta {
                    index: 1,
                    text: "x".to_string(),
                    logprobs: Some(logprobs(b"x")),
                }),
                Ok(LlmChoiceEvent::Delta {
                    index: 0,
                    text: String::from_utf8_lossy(&[0xa9]).into_owned(),
                    logprobs: Some(logprobs(&[0xa9])),
                }),
                Ok(LlmChoiceEvent::Delta {
                    index: 1,
                    text: "yz".to_string(),
                    logprobs: Some(logprobs(b"yz")),
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::Stop,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 1,
                    reason: LlmChoiceFinishReason::Stop,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]),
            "cmpl-1".to_string(),
            1,
            "qwen/test".to_string(),
            None,
        );
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
        let offsets = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|chunk| !chunk["choices"].as_array().is_some_and(Vec::is_empty))
            .filter_map(|chunk| {
                let choice = &chunk["choices"][0];
                choice["logprobs"]["text_offset"][0]
                    .as_u64()
                    .map(|offset| (choice["index"].as_u64().unwrap(), offset))
            })
            .collect::<Vec<_>>();
        assert_eq!(offsets, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn configured_vision_budget_rejects_over_one_mib_before_decode() {
        let limits = LlmVisionPolicy {
            max_images: 2,
            max_bytes_per_image: 1024 * 1024,
            max_total_bytes: 2 * 1024 * 1024,
            max_side: 4096,
            max_pixels_per_image: 4_000_000,
            max_total_pixels: 8_000_000,
        };
        let payload = "A".repeat(1_398_104);
        let error = ImageBudget::new(limits)
            .reserve_decoded(&payload, "messages.content.image_url.url")
            .unwrap_err();
        assert_eq!(
            error.error.param.as_deref(),
            Some("messages.content.image_url.url")
        );
    }

    #[tokio::test]
    async fn collected_responses_preserve_first_seen_order_and_empty_message() {
        let collected = collect_choices(
            choice_generation([
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::Text("answer".to_string()),
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::Reasoning("why".to_string()),
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: Some("call-1".to_string()),
                        name: Some("lookup".to_string()),
                        arguments: "{}".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::Stop,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]),
            1,
        )
        .await
        .unwrap();
        let choice = &collected.choices[0];
        let output = response_output_items("msg-1", choice);
        assert_eq!(
            output
                .iter()
                .map(|item| item["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["message", "reasoning", "function_call"]
        );

        let empty = ChoiceAccumulator {
            finish_reason: Some(LlmChoiceFinishReason::Stop),
            ..ChoiceAccumulator::default()
        };
        let output = response_output_items("msg-empty", &empty);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "");
    }

    #[tokio::test]
    async fn synchronous_chat_and_responses_shapes_include_required_nulls_and_usage() {
        let result = collect_choices(
            generation([
                Ok(GenerationEvent::ContentDelta("hello".to_string())),
                Ok(GenerationEvent::Finished {
                    reason: GenerationFinishReason::Stop,
                    usage: usage(),
                }),
            ]),
            1,
        )
        .await
        .unwrap();
        let usage = result.usage;
        let choice = result.choices.into_iter().next().unwrap();
        let text = choice.text.clone();
        let reason = choice.finish_reason.unwrap();
        let chat = serde_json::to_value(ChatCompletionResponse {
            id: "chat-1".to_string(),
            object: "chat.completion",
            created: 1,
            model: "qwen/test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some(text.clone()),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    refusal: None,
                },
                finish_reason: choice_finish_reason(reason),
                logprobs: None,
            }],
            usage: usage.into(),
            timings: usage.timings.into(),
        })
        .unwrap();
        assert_eq!(chat["choices"][0]["message"]["refusal"], Value::Null);
        assert_eq!(chat["choices"][0]["logprobs"], Value::Null);
        assert_eq!(chat["usage"]["total_tokens"], 3);
        assert_eq!(chat["timings"]["cache_n"], 0);
        assert_eq!(chat["timings"]["prompt_n"], 2);
        assert_eq!(chat["timings"]["predicted_n"], 1);
        assert_eq!(chat["timings"]["prompt_per_second"], 250.0);

        let responses = serde_json::to_value(response_object(
            "resp-1".to_string(),
            1,
            "qwen/test".to_string(),
            vec![response_message("msg-1", "completed", &text, &[])],
            text,
            Some(reason),
            Some(usage),
            true,
            &response_config(),
        ))
        .unwrap();
        assert_eq!(responses["output"][0]["role"], "assistant");
        assert_eq!(responses["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(responses["usage"]["total_tokens"], 3);
        assert_eq!(responses["timings"]["prompt_ms"], 8.0);
        assert_eq!(responses["timings"]["predicted_ms"], 5.0);
        assert_eq!(
            responses["usage"]["input_tokens_details"]["cached_tokens"],
            0
        );
        assert_eq!(
            responses["usage"]["output_tokens_details"]["reasoning_tokens"],
            0
        );
        assert_eq!(responses["error"], Value::Null);
        assert_eq!(responses["incomplete_details"], Value::Null);
        assert_eq!(responses["store"], false);
        assert_eq!(responses["instructions"], "be concise");
        assert_eq!(
            responses["usage"]["input_tokens_details"]["cache_write_tokens"],
            0
        );
        assert!(responses["reasoning"].is_object());
        assert!(responses["text"].is_object());
    }

    #[tokio::test]
    async fn responses_sse_has_fixed_lifecycle_and_no_done_sentinel() {
        let response = responses_stream(
            generation([
                Ok(GenerationEvent::ContentDelta("hello".to_string())),
                Ok(GenerationEvent::Finished {
                    reason: GenerationFinishReason::Stop,
                    usage: usage(),
                }),
            ]),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
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
        for event in [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ] {
            assert!(
                body.contains(&format!("event: {event}\n")),
                "missing {event}: {body}"
            );
        }
        assert!(body.contains("\"logprobs\":[]"));
        assert!(body.contains("\"content\":[]"));
        let completed = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(completed["timings"]["prompt_n"], 2);
        assert_eq!(completed["timings"]["predicted_n"], 1);
        assert!(completed["response"].get("timings").is_none());
        assert!(!body.contains("[DONE]"));
        assert_monotonic_sequence_numbers(&body);
    }

    #[tokio::test]
    async fn responses_sse_reuses_item_identity_and_real_output_order() {
        let response = responses_stream(
            choice_generation([
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::Text("answer".to_string()),
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::Reasoning("because".to_string()),
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::Stop,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
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
        let events = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let added = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.added")
            .collect::<Vec<_>>();
        let done = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .collect::<Vec<_>>();
        assert_eq!(added.len(), 2);
        assert_eq!(
            done.iter()
                .map(|event| event["output_index"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        for item in &added {
            let index = item["output_index"].as_u64().unwrap();
            let finished = done
                .iter()
                .find(|done| done["output_index"].as_u64() == Some(index))
                .unwrap();
            assert_eq!(item["item"]["id"], finished["item"]["id"]);
        }
        let terminal = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(terminal["response"]["output"][0]["id"], "msg-1");
        assert_eq!(
            terminal["response"]["output"][0]["id"],
            added[0]["item"]["id"]
        );
        assert_eq!(
            terminal["response"]["output"][1]["id"],
            added[1]["item"]["id"]
        );
    }

    #[tokio::test]
    async fn responses_tool_call_first_delta_emits_complete_ordered_item_lifecycle() {
        let response = responses_stream(
            choice_generation([
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: Some("call-1".to_string()),
                        name: Some("lookup".to_string()),
                        arguments: "{\"city\":".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::SemanticDelta {
                    index: 0,
                    delta: LlmSemanticDelta::ToolCall {
                        index: 0,
                        id: None,
                        name: None,
                        arguments: "\"Paris\"}".to_string(),
                    },
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 0,
                    reason: LlmChoiceFinishReason::ToolCalls,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::FinishedAll { usage: usage() }),
            ]),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
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
        let events = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let tool_events = events
            .iter()
            .filter(|event| {
                matches!(
                    event["type"].as_str(),
                    Some(
                        "response.output_item.added"
                            | "response.function_call_arguments.delta"
                            | "response.function_call_arguments.done"
                            | "response.output_item.done"
                    )
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tool_events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
            ]
        );
        let added = tool_events[0];
        assert_eq!(added["item"]["call_id"], "call-1");
        assert_eq!(added["item"]["name"], "lookup");
        assert_eq!(added["output_index"], 0);
        let item_id = &added["item"]["id"];
        for event in &tool_events[1..4] {
            assert_eq!(&event["item_id"], item_id);
            assert_eq!(event["output_index"], 0);
        }
        assert_eq!(tool_events[3]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(tool_events[4]["item"]["id"], *item_id);
        assert_eq!(tool_events[4]["output_index"], 0);
        assert_monotonic_sequence_numbers(&body);
    }

    #[tokio::test]
    async fn malformed_responses_choice_sequences_emit_protocol_errors() {
        let cases = [
            vec![Ok(LlmChoiceEvent::SemanticDelta {
                index: 0,
                delta: LlmSemanticDelta::ToolCall {
                    index: 1,
                    id: Some("call-gap".to_string()),
                    name: Some("tool".to_string()),
                    arguments: String::new(),
                },
            })],
            vec![Ok(LlmChoiceEvent::FinishedAll { usage: usage() })],
            Vec::new(),
        ];
        for events in cases {
            let response = responses_stream(
                choice_generation(events),
                "resp-1".to_string(),
                "msg-1".to_string(),
                1,
                "qwen/test".to_string(),
                response_config(),
                None,
            );
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
            assert!(body.contains("event: error\n"), "{body}");
            assert!(!body.contains("event: response.completed\n"), "{body}");
        }
    }

    #[tokio::test]
    async fn chat_sse_error_uses_nested_error_body_contract() {
        let response = chat_stream(
            generation([Err(RuntimeError::Timeout("deadline".to_string()))]),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            None,
            None,
        );
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
        let error_line = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .find(|line| line.contains("request_timeout"))
            .unwrap();
        let event: Value = serde_json::from_str(error_line).unwrap();
        assert_eq!(
            event,
            json!({"error":{"message":"deadline","type":"invalid_request_error","param":null,"code":"request_timeout"}})
        );
    }

    #[tokio::test]
    async fn responses_length_uses_incomplete_sync_and_sse_contract() {
        let object = serde_json::to_value(response_object(
            "resp-1".to_string(),
            1,
            "qwen/test".to_string(),
            vec![response_message("msg-1", "incomplete", "partial", &[])],
            "partial".to_string(),
            Some(LlmChoiceFinishReason::Length),
            Some(usage()),
            true,
            &response_config(),
        ))
        .unwrap();
        assert_eq!(object["status"], "incomplete");
        assert!(object.get("finish_reason").is_none());
        assert_eq!(object["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(object["output"][0]["status"], "incomplete");

        let response = responses_stream(
            generation([Ok(GenerationEvent::Finished {
                reason: GenerationFinishReason::Length,
                usage: usage(),
            })]),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
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
        assert!(body.contains("event: response.incomplete\n"));
        assert!(!body.contains("event: response.completed\n"));
        assert!(body.contains("\"status\":\"incomplete\""));
        assert!(body.contains("\"reason\":\"max_output_tokens\""));
        assert_monotonic_sequence_numbers(&body);
    }

    #[tokio::test]
    async fn post_header_errors_use_protocol_specific_event_and_no_terminal() {
        let chat = chat_stream(
            generation([Err(RuntimeError::Internal("failed".to_string()))]),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            None,
            None,
        );
        let chat = String::from_utf8(
            chat.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(chat.contains("\"error\":{"));
        assert!(!chat.contains("[DONE]"));

        let responses = responses_stream(
            generation([Err(RuntimeError::Internal("failed".to_string()))]),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        );
        let responses = String::from_utf8(
            responses
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(responses.contains("event: error\n"));
        assert!(responses.contains("\"code\":\"internal_error\""));
        assert!(!responses.contains("\"error\":{"));
        assert!(!responses.contains("response.completed"));
        assert!(!responses.contains("[DONE]"));
    }

    #[tokio::test]
    async fn per_choice_failure_waits_for_one_aggregate_wire_error() {
        let failure_events = || {
            [
                Ok(LlmChoiceEvent::Failed {
                    index: Some(0),
                    message: "choice zero failed".to_string(),
                }),
                Ok(LlmChoiceEvent::Finished {
                    index: 1,
                    reason: LlmChoiceFinishReason::Stop,
                    usage: usage(),
                }),
                Ok(LlmChoiceEvent::Failed {
                    index: None,
                    message: "aggregate failed".to_string(),
                }),
            ]
        };
        let chat = chat_stream(
            choice_generation(failure_events()),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            None,
            None,
        )
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
        let chat = String::from_utf8(chat.to_vec()).unwrap();
        assert_eq!(chat.matches("\"error\":{").count(), 1);
        assert!(!chat.contains("choice zero failed"));
        assert!(!chat.contains("[DONE]"));

        let completion = completion_stream(
            choice_generation(failure_events()),
            "cmpl-1".to_string(),
            1,
            "qwen/test".to_string(),
            None,
        )
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
        let completion = String::from_utf8(completion.to_vec()).unwrap();
        assert_eq!(completion.matches("\"error\":{").count(), 1);
        assert!(!completion.contains("choice zero failed"));
        assert!(!completion.contains("[DONE]"));

        let responses = responses_stream(
            choice_generation(failure_events()),
            "resp-1".to_string(),
            "msg-1".to_string(),
            1,
            "qwen/test".to_string(),
            response_config(),
            None,
        )
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
        let responses = String::from_utf8(responses.to_vec()).unwrap();
        assert_eq!(responses.matches("event: error\n").count(), 1);
        assert_eq!(responses.matches("\"type\":\"error\"").count(), 1);
        assert!(!responses.contains("choice zero failed"));
        assert!(!responses.contains("response.completed"));
    }

    #[tokio::test]
    async fn sse_failure_records_activity_failure_instead_of_http_success() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 2,
        });
        let activity = hub
            .start(
                ActivityOperation::Chat,
                ActivityTransport::Http,
                "POST",
                "/v1/chat/completions",
                None,
            )
            .unwrap();
        let response = chat_stream(
            generation([Err(RuntimeError::Internal("failed".to_string()))]),
            "chat-1".to_string(),
            1,
            "qwen/test".to_string(),
            false,
            Some(activity),
            None,
        );
        response.into_body().collect().await.unwrap();
        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert_eq!(page.history[0].http_status, Some(200));
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::ServerError));
        assert_eq!(
            page.history[0].error_code.as_deref(),
            Some("internal_error")
        );
    }

    fn response_config() -> ResponseConfig {
        ResponseConfig {
            instructions: Some("be concise".to_string()),
            max_output_tokens: Some(16),
            temperature: Some(1.0),
            top_p: Some(0.9),
            tools: Vec::new(),
            tool_choice: json!("none"),
            parallel_tool_calls: true,
            text: json!({"format":{"type":"text"},"verbosity":"medium"}),
            reasoning: json!({"effort":null,"summary":null}),
            top_logprobs: 0,
        }
    }

    fn assert_monotonic_sequence_numbers(body: &str) {
        let numbers = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .filter_map(|data| data["sequence_number"].as_u64())
            .collect::<Vec<_>>();
        assert_eq!(numbers, (0..numbers.len() as u64).collect::<Vec<_>>());
    }
}
