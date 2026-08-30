#![allow(
    clippy::needless_pass_by_value,
    clippy::struct_field_names,
    clippy::large_stack_arrays,
    reason = "wire DTO values are consumed into owned JSON/SSE frames and retain protocol field names"
)]

use crate::api::activity::{ActivityContext, ActivityOutcome};
use crate::api::http_shared::authorize;
use crate::api::openai::{ApiError, ErrorBody};
use crate::application::llm::{LlmCommand, LlmGenerationOverrides, ManagedGeneration};
use crate::application::{RuntimeError, ServerApplication, UseCaseError};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use orchion::{GenerationEvent, GenerationFinishReason, LlmMessage, LlmRole, LlmUsage};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use utoipa::ToSchema;

static NEXT_LLM_ID: AtomicU64 = AtomicU64::new(1);
const MIN_RESPONSES_MAX_OUTPUT_TOKENS: usize = 16;

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
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
    pub tools: UnsupportedField,
    #[serde(default)]
    pub tool_choice: UnsupportedField,
    #[serde(default)]
    pub parallel_tool_calls: UnsupportedField,
    #[serde(default)]
    pub functions: UnsupportedField,
    #[serde(default)]
    pub function_call: UnsupportedField,
    #[serde(default)]
    pub response_format: UnsupportedField,
    #[serde(default)]
    pub logprobs: UnsupportedField,
    #[serde(default)]
    pub top_logprobs: UnsupportedField,
    #[serde(default)]
    pub logit_bias: UnsupportedField,
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
    pub reasoning_effort: UnsupportedField,
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
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatTextPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatTextPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<Value>,
    #[serde(default)]
    pub input_audio: Option<Value>,
    #[serde(default)]
    pub file: Option<Value>,
    #[serde(default)]
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
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
    pub content: String,
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
#[serde(deny_unknown_fields)]
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
    pub tools: UnsupportedField,
    #[serde(default)]
    pub tool_choice: UnsupportedField,
    #[serde(default)]
    pub reasoning: UnsupportedField,
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
    pub parallel_tool_calls: UnsupportedField,
    #[serde(default)]
    pub text: UnsupportedField,
    #[serde(default)]
    pub truncation: UnsupportedField,
    #[serde(default)]
    pub service_tier: UnsupportedField,
    #[serde(default)]
    pub top_logprobs: UnsupportedField,
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

#[derive(Debug, Default, ToSchema)]
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
        let _ = serde::de::IgnoredAny::deserialize(deserializer)?;
        Ok(Self { present: true })
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseInputItem {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    pub role: String,
    pub content: ResponseItemContent,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseItemContent {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponseInputPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<Value>,
    #[serde(default)]
    pub file_id: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
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
}

#[derive(ToSchema)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum ChatCompletionSseEvent {
    Chunk(ChatCompletionStreamChunk),
    Error(ErrorBody),
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
pub struct ResponseCompletedSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseCompletedSseEventType,
    response: ResponsesResponse,
    sequence_number: u64,
}

#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ResponseIncompleteSseEvent {
    #[schema(rename = "type", inline)]
    kind: ResponseIncompleteSseEventType,
    response: ResponsesResponse,
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

#[derive(Debug, Clone)]
struct ResponseConfig {
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

pub(super) async fn create_chat_completion<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    request: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let request = parse_json(request)?;
    validate_chat(&request)?;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|value| value.include_usage);
    let model = request.model.clone();
    let command = chat_command(&request)?;
    let max_tokens_param = if request.max_tokens.is_some() {
        "max_tokens"
    } else {
        "max_completion_tokens"
    };
    let generation = state
        .start_generation(command)
        .await
        .map_err(|error| map_llm_start_error(error, "messages", max_tokens_param))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let activity = activity.map(|Extension(activity)| activity);
    if let Some(activity) = &activity {
        activity.set_model(model.clone());
    }
    let id = next_id("chatcmpl");
    let created = now_seconds();
    if request.stream {
        Ok(chat_stream(
            generation,
            id,
            created,
            model,
            include_usage,
            activity,
        ))
    } else {
        let (text, reason, usage) = collect_generation(generation).await?;
        if let Some(activity) = &activity {
            activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
            activity.set_llm_timing(usage.queue_time_ms, usage.eval_time_ms);
        }
        Ok(Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: text,
                    refusal: None,
                },
                finish_reason: finish_reason(reason),
                logprobs: None,
            }],
            usage: usage.into(),
        })
        .into_response())
    }
}

pub(super) async fn create_response<S>(
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
    let model = request.model.clone();
    let command = responses_command(&request)?;
    let generation = state
        .start_generation(command)
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
    };
    if request.stream {
        Ok(responses_stream(
            generation,
            id,
            message_id,
            created,
            model,
            response_config,
            activity,
        ))
    } else {
        let (text, reason, usage) = collect_generation(generation).await?;
        if let Some(activity) = &activity {
            activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
            activity.set_llm_timing(usage.queue_time_ms, usage.eval_time_ms);
        }
        let output = vec![response_message(
            &message_id,
            if reason == GenerationFinishReason::Length {
                "incomplete"
            } else {
                "completed"
            },
            &text,
        )];
        Ok(Json(response_object(
            id,
            created,
            model,
            output,
            text,
            Some(reason),
            Some(usage),
            &response_config,
        ))
        .into_response())
    }
}

fn parse_json<T>(request: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    request
        .map(|Json(value)| value)
        .map_err(|error| ApiError::invalid_request(error.body_text(), None, Some("invalid_json")))
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
        ("tools", &request.tools),
        ("tool_choice", &request.tool_choice),
        ("parallel_tool_calls", &request.parallel_tool_calls),
        ("functions", &request.functions),
        ("function_call", &request.function_call),
        ("response_format", &request.response_format),
        ("logprobs", &request.logprobs),
        ("top_logprobs", &request.top_logprobs),
        ("logit_bias", &request.logit_bias),
        ("modalities", &request.modalities),
        ("audio", &request.audio),
        ("store", &request.store),
        ("metadata", &request.metadata),
        ("service_tier", &request.service_tier),
        ("reasoning_effort", &request.reasoning_effort),
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
    if request.n.is_some_and(|value| value != 1) {
        return Err(unsupported("n"));
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
    stop_values(request.stop.as_ref())?;
    Ok(())
}

fn validate_responses(request: &ResponsesRequest) -> Result<(), ApiError> {
    if request.store != Some(false) {
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
        ("tools", &request.tools),
        ("tool_choice", &request.tool_choice),
        ("reasoning", &request.reasoning),
        ("background", &request.background),
        ("conversation", &request.conversation),
        ("previous_response_id", &request.previous_response_id),
        ("context_management", &request.context_management),
        ("include", &request.include),
        ("metadata", &request.metadata),
        ("parallel_tool_calls", &request.parallel_tool_calls),
        ("text", &request.text),
        ("truncation", &request.truncation),
        ("service_tier", &request.service_tier),
        ("top_logprobs", &request.top_logprobs),
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
    validate_sampling(request.temperature, request.top_p, None, None)
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

fn chat_command(request: &ChatCompletionRequest) -> Result<LlmCommand, ApiError> {
    let messages = request
        .messages
        .iter()
        .map(normalize_chat_message)
        .collect::<Result<_, _>>()?;
    Ok(LlmCommand {
        model: request.model.clone(),
        messages,
        max_tokens_param: if request.max_tokens.is_some() {
            "max_tokens"
        } else {
            "max_completion_tokens"
        },
        queue_timeout: None,
        generation_timeout: None,
        options: LlmGenerationOverrides {
            max_tokens: request.max_completion_tokens.or(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            seed: request.seed,
            stop: stop_values(request.stop.as_ref())?,
        },
    })
}

fn normalize_chat_message(message: &ChatMessage) -> Result<LlmMessage, ApiError> {
    let role = match message.role.as_str() {
        "system" => LlmRole::System,
        "developer" => LlmRole::Developer,
        "user" => LlmRole::User,
        "assistant" => LlmRole::Assistant,
        "tool" | "function" => return Err(unsupported("messages.role")),
        _ => return Err(invalid("unsupported message role", "messages.role")),
    };
    let content = match &message.content {
        ChatContent::Text(text) => text.clone(),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|part| {
                if part.kind != "text"
                    || part.image_url.is_some()
                    || part.input_audio.is_some()
                    || part.file.is_some()
                    || part.refusal.is_some()
                {
                    return Err(unsupported("messages.content"));
                }
                part.text
                    .clone()
                    .ok_or_else(|| invalid("text content part requires text", "messages.content"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(""),
    };
    Ok(LlmMessage { role, content })
}

fn responses_command(request: &ResponsesRequest) -> Result<LlmCommand, ApiError> {
    let mut messages = Vec::new();
    if let Some(instructions) = &request.instructions {
        messages.push(LlmMessage {
            role: LlmRole::Developer,
            content: instructions.clone(),
        });
    }
    match &request.input {
        ResponsesInput::Text(text) => messages.push(LlmMessage {
            role: LlmRole::User,
            content: text.clone(),
        }),
        ResponsesInput::Items(items) => {
            for item in items {
                if item.kind.as_deref().is_some_and(|kind| kind != "message") {
                    return Err(unsupported("input.type"));
                }
                let role = match item.role.as_str() {
                    "system" => LlmRole::System,
                    "developer" => LlmRole::Developer,
                    "user" => LlmRole::User,
                    "assistant" => LlmRole::Assistant,
                    _ => return Err(unsupported("input.role")),
                };
                let content = match &item.content {
                    ResponseItemContent::Text(text) => text.clone(),
                    ResponseItemContent::Parts(parts) => parts
                        .iter()
                        .map(|part| {
                            if part.kind != "input_text"
                                || part.image_url.is_some()
                                || part.file_id.is_some()
                            {
                                return Err(unsupported("input.content"));
                            }
                            part.text.clone().ok_or_else(|| {
                                invalid("input_text part requires text", "input.content")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .join(""),
                };
                messages.push(LlmMessage { role, content });
            }
        }
    }
    if messages.is_empty() {
        return Err(invalid("input must not be empty", "input"));
    }
    Ok(LlmCommand {
        model: request.model.clone(),
        messages,
        max_tokens_param: "max_output_tokens",
        queue_timeout: None,
        generation_timeout: None,
        options: LlmGenerationOverrides {
            max_tokens: request.max_output_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            ..LlmGenerationOverrides::default()
        },
    })
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

async fn collect_generation(
    mut generation: ManagedGeneration,
) -> Result<(String, GenerationFinishReason, LlmUsage), ApiError> {
    let mut text = String::new();
    while let Some(event) = generation.next().await {
        match event.map_err(|error| ApiError::from(UseCaseError::from(error)))? {
            GenerationEvent::ContentDelta(delta) => text.push_str(&delta),
            GenerationEvent::Finished {
                reason: GenerationFinishReason::Cancelled,
                ..
            } => return Err(ApiError::internal("generation was cancelled")),
            GenerationEvent::Finished { reason, usage } => return Ok((text, reason, usage)),
        }
    }
    Err(ApiError::internal(
        "generation ended without terminal acknowledgement",
    ))
}

fn chat_stream(
    mut generation: ManagedGeneration,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    activity: Option<ActivityContext>,
) -> Response {
    let stream = async_stream::stream! {
        yield Ok::<Bytes, Infallible>(Bytes::from(chat_chunk(&id, created, &model, json!({"role":"assistant"}), None, None)));
        while let Some(event) = generation.next().await {
            match event {
                Ok(GenerationEvent::ContentDelta(delta)) => yield Ok(Bytes::from(chat_chunk(&id, created, &model, json!({"content":delta}), None, None))),
                Ok(GenerationEvent::Finished { reason: GenerationFinishReason::Cancelled, .. }) => break,
                Ok(GenerationEvent::Finished { reason, usage }) => {
                    if let Some(activity) = &activity {
                        activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
                        activity.set_llm_timing(usage.queue_time_ms, usage.eval_time_ms);
                    }
                    yield Ok(Bytes::from(chat_chunk(&id, created, &model, json!({}), Some(finish_reason(reason)), None)));
                    if include_usage { yield Ok(Bytes::from(chat_chunk(&id, created, &model, json!({}), None, Some(usage)))); }
                    yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    break;
                }
                Err(error) => {
                    let api = ApiError::from(UseCaseError::from(error));
                    if let Some(activity) = &activity {
                        activity.complete_stream_failure(activity_outcome(&api), api.activity_error());
                    }
                    let body = ErrorBody { error: api.error };
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&body).unwrap())));
                    break;
                }
            }
        }
    };
    sse_response(Body::from_stream(stream))
}

fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Option<&str>,
    usage: Option<LlmUsage>,
) -> String {
    let (choices, usage) = if let Some(usage) = usage {
        (
            json!([]),
            serde_json::to_value(UsageObject::from(usage)).unwrap(),
        )
    } else {
        (
            json!([{"index":0,"delta":delta,"logprobs":null,"finish_reason":finish}]),
            Value::Null,
        )
    };
    format!(
        "data: {}\n\n",
        json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":choices,"usage":usage})
    )
}

fn responses_stream(
    mut generation: ManagedGeneration,
    id: String,
    message_id: String,
    created: u64,
    model: String,
    config: ResponseConfig,
    activity: Option<ActivityContext>,
) -> Response {
    let stream = async_stream::stream! {
        let mut sequence = 0_u64;
        let initial = response_snapshot(&id, created, &model, Vec::new(), None, None, &config);
        for (event, data) in [("response.created", json!({"type":"response.created","response":initial})),
            ("response.in_progress", json!({"type":"response.in_progress","response":response_snapshot(&id, created, &model, Vec::new(), None, None, &config)})),
            ("response.output_item.added", json!({"type":"response.output_item.added","output_index":0,"item":{"id":message_id,"type":"message","status":"in_progress","role":"assistant","content":[]}})),
            ("response.content_part.added", json!({"type":"response.content_part.added","item_id":message_id,"output_index":0,"content_index":0,"part":output_part("")}))] {
            yield Ok::<Bytes, Infallible>(Bytes::from(event_frame(event, with_sequence(data, sequence))));
            sequence += 1;
        }
        let mut text = String::new();
        while let Some(event) = generation.next().await {
            match event {
                Ok(GenerationEvent::ContentDelta(delta)) => {
                    text.push_str(&delta);
                    let data = json!({"type":"response.output_text.delta","item_id":message_id,"output_index":0,"content_index":0,"delta":delta,"logprobs":[]});
                    yield Ok(Bytes::from(event_frame("response.output_text.delta", with_sequence(data, sequence)))); sequence += 1;
                }
                Ok(GenerationEvent::Finished { reason: GenerationFinishReason::Cancelled, .. }) => break,
                Ok(GenerationEvent::Finished { reason, usage }) => {
                    if let Some(activity) = &activity {
                        activity.set_llm_usage(usage.prompt_tokens, usage.completion_tokens);
                        activity.set_llm_timing(usage.queue_time_ms, usage.eval_time_ms);
                    }
                    let final_event = if reason == GenerationFinishReason::Length { "response.incomplete" } else { "response.completed" };
                    let output_status = if reason == GenerationFinishReason::Length { "incomplete" } else { "completed" };
                    let events = [
                        ("response.output_text.done", json!({"type":"response.output_text.done","item_id":message_id,"output_index":0,"content_index":0,"text":text,"logprobs":[]})),
                        ("response.content_part.done", json!({"type":"response.content_part.done","item_id":message_id,"output_index":0,"content_index":0,"part":output_part(&text)})),
                        ("response.output_item.done", json!({"type":"response.output_item.done","output_index":0,"item":response_message(&message_id,output_status,&text)})),
                        (final_event, json!({"type":final_event,"response":response_snapshot(&id,created,&model,vec![response_message(&message_id,output_status,&text)],Some(reason),Some(usage),&config)})),
                    ];
                    for (name, data) in events { yield Ok(Bytes::from(event_frame(name, with_sequence(data, sequence)))); sequence += 1; }
                    break;
                }
                Err(error) => {
                    let api = ApiError::from(UseCaseError::from(error));
                    if let Some(activity) = &activity {
                        activity.complete_stream_failure(activity_outcome(&api), api.activity_error());
                    }
                    let data = json!({"type":"error","sequence_number":sequence,"code":api.error.code,"message":api.error.message,"param":api.error.param});
                    yield Ok(Bytes::from(event_frame("error", data)));
                    break;
                }
            }
        }
    };
    sse_response(Body::from_stream(stream))
}

fn response_snapshot(
    id: &str,
    created: u64,
    model: &str,
    output: Vec<Value>,
    reason: Option<GenerationFinishReason>,
    usage: Option<LlmUsage>,
    config: &ResponseConfig,
) -> Value {
    let output_text = output
        .first()
        .and_then(|item| item["content"][0]["text"].as_str())
        .unwrap_or("")
        .to_string();
    serde_json::to_value(response_object(
        id.to_string(),
        created,
        model.to_string(),
        output,
        output_text,
        reason,
        usage,
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
    reason: Option<GenerationFinishReason>,
    usage: Option<LlmUsage>,
    config: &ResponseConfig,
) -> ResponsesResponse {
    let incomplete = reason == Some(GenerationFinishReason::Length);
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
        parallel_tool_calls: false,
        previous_response_id: None,
        reasoning: json!({"effort":null,"summary":null}),
        store: false,
        temperature: config.temperature,
        text: json!({"format":{"type":"text"},"verbosity":"medium"}),
        tool_choice: json!("none"),
        tools: Vec::new(),
        top_logprobs: 0,
        top_p: config.top_p,
        truncation: "disabled",
        metadata: json!({}),
        usage: usage.map(ResponsesUsage::from),
    }
}

fn response_message(id: &str, status: &str, text: &str) -> Value {
    json!({"id":id,"type":"message","status":status,"role":"assistant","content":[output_part(text)]})
}

fn output_part(text: &str) -> Value {
    json!({"type":"output_text","text":text,"annotations":[],"logprobs":[]})
}
fn with_sequence(mut data: Value, sequence: u64) -> Value {
    data["sequence_number"] = json!(sequence);
    data
}
fn event_frame(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}
fn sse_response(body: Body) -> Response {
    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
            ("x-accel-buffering", "no"),
        ],
        body,
    )
        .into_response()
}
fn next_id(prefix: &str) -> String {
    format!("{prefix}-{}", NEXT_LLM_ID.fetch_add(1, Ordering::Relaxed))
}
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn finish_reason(reason: GenerationFinishReason) -> &'static str {
    match reason {
        GenerationFinishReason::Stop | GenerationFinishReason::Cancelled => "stop",
        GenerationFinishReason::Length => "length",
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
impl From<LlmUsage> for ResponsesUsage {
    fn from(value: LlmUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            input_tokens_details: json!({"cached_tokens":0,"cache_write_tokens":0}),
            output_tokens: value.completion_tokens,
            output_tokens_details: json!({"reasoning_tokens":0}),
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
    use std::sync::atomic::AtomicBool;
    use tokio::sync::{Notify, mpsc};

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
    fn chat_rejects_known_unsupported_fields() {
        let request: ChatCompletionRequest = serde_json::from_value(
            json!({"model":"a/b","messages":[{"role":"user","content":"hi"}],"tools":[]}),
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
        assert_eq!(
            validate_chat(&null_tools)
                .unwrap_err()
                .error
                .param
                .as_deref(),
            Some("tools")
        );
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
            assert_eq!(
                validate_chat(&request).unwrap_err().error.param.as_deref(),
                Some(field)
            );
        }
    }

    #[test]
    fn responses_requires_explicit_store_false() {
        let omitted: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi"})).unwrap();
        let enabled: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi","store":true})).unwrap();
        let disabled: ResponsesRequest =
            serde_json::from_value(json!({"model":"a/b","input":"hi","store":false})).unwrap();
        assert!(validate_responses(&omitted).is_err());
        assert!(validate_responses(&enabled).is_err());
        assert!(validate_responses(&disabled).is_ok());
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
            assert_eq!(
                validate_responses(&request)
                    .unwrap_err()
                    .error
                    .param
                    .as_deref(),
                Some(field)
            );
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
    ) -> ManagedGeneration {
        let (sender, receiver) = mpsc::channel(16);
        let (terminal_sender, terminal_receiver) = tokio::sync::oneshot::channel();
        let mut terminal = None;
        for event in events {
            if matches!(event, Ok(GenerationEvent::ContentDelta(_))) {
                sender.try_send(event).unwrap();
            } else {
                terminal = Some(event);
            }
        }
        drop(sender);
        if let Some(terminal) = terminal {
            let _ = terminal_sender.send(terminal);
        }
        ManagedGeneration::new(
            receiver,
            terminal_receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        )
    }

    fn usage() -> LlmUsage {
        LlmUsage {
            prompt_tokens: 2,
            completion_tokens: 1,
            total_tokens: 3,
            queue_time_ms: Some(4),
            eval_time_ms: Some(5),
        }
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
        assert_eq!(body.matches("data: [DONE]").count(), 1);
    }

    #[tokio::test]
    async fn synchronous_chat_and_responses_shapes_include_required_nulls_and_usage() {
        let (text, reason, usage) = collect_generation(generation([
            Ok(GenerationEvent::ContentDelta("hello".to_string())),
            Ok(GenerationEvent::Finished {
                reason: GenerationFinishReason::Stop,
                usage: usage(),
            }),
        ]))
        .await
        .unwrap();
        let chat = serde_json::to_value(ChatCompletionResponse {
            id: "chat-1".to_string(),
            object: "chat.completion",
            created: 1,
            model: "qwen/test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: text.clone(),
                    refusal: None,
                },
                finish_reason: finish_reason(reason),
                logprobs: None,
            }],
            usage: usage.into(),
        })
        .unwrap();
        assert_eq!(chat["choices"][0]["message"]["refusal"], Value::Null);
        assert_eq!(chat["choices"][0]["logprobs"], Value::Null);
        assert_eq!(chat["usage"]["total_tokens"], 3);

        let responses = serde_json::to_value(response_object(
            "resp-1".to_string(),
            1,
            "qwen/test".to_string(),
            vec![response_message("msg-1", "completed", &text)],
            text,
            Some(reason),
            Some(usage),
            &response_config(),
        ))
        .unwrap();
        assert_eq!(responses["output"][0]["role"], "assistant");
        assert_eq!(responses["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(responses["usage"]["total_tokens"], 3);
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
        assert!(!body.contains("[DONE]"));
        assert_monotonic_sequence_numbers(&body);
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
            vec![response_message("msg-1", "incomplete", "partial")],
            "partial".to_string(),
            Some(GenerationFinishReason::Length),
            Some(usage()),
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
