use crate::client::decode_json;
use crate::sse::SseStream;
use crate::{Client, ClientError, ServerErrorBody, StreamErrorObject};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for OpenAI-compatible LLM endpoints.
pub struct LlmClient<'a> {
    client: &'a Client,
}

impl<'a> LlmClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Creates a non-streaming chat completion.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or cannot be decoded.
    pub async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        request.validate()?;
        let response = self
            .client
            .post("/v1/chat/completions")?
            .json(&wire::ChatRequest {
                request: &request,
                stream: false,
                stream_options: None,
            })
            .send()
            .await?;
        decode_json(response).await
    }

    /// Starts a chat completion stream without automatic retries.
    ///
    /// The wire request always uses `stream: true` and requests the terminal usage chunk.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, fails, or is not an SSE response.
    pub async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/chat/completions")?
            .json(&wire::ChatRequest {
                request: &request,
                stream: true,
                stream_options: Some(wire::ChatStreamOptions {
                    include_usage: true,
                }),
            })
            .send()
            .await?;
        Ok(ChatCompletionStream {
            stream: SseStream::from_response(response).await?,
            terminal: false,
        })
    }

    /// Creates a non-streaming Responses API response.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or cannot be decoded.
    pub async fn create_response(
        &self,
        request: ResponsesRequest,
    ) -> Result<ResponsesResponse, ClientError> {
        request.validate()?;
        let response = self
            .client
            .post("/v1/responses")?
            .json(&wire::ResponsesRequest {
                request: &request,
                store: false,
                stream: false,
            })
            .send()
            .await?;
        decode_json(response).await
    }

    /// Starts a Responses API stream without automatic retries.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, fails, or is not an SSE response.
    pub async fn stream_response(
        &self,
        request: ResponsesRequest,
    ) -> Result<ResponsesStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/responses")?
            .json(&wire::ResponsesRequest {
                request: &request,
                store: false,
                stream: true,
            })
            .send()
            .await?;
        Ok(ResponsesStream {
            stream: SseStream::from_response(response).await?,
            terminal: false,
            next_sequence: 0,
            phase: ResponsesPhase::Start,
        })
    }
}

/// Role of a chat or Responses API message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
}

/// A typed text message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    #[must_use]
    pub fn developer(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Developer, content)
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }
}

/// One or more chat stop sequences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

/// Chat completion parameters controlled by the caller.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

impl ChatCompletionRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            seed: None,
        }
    }

    #[must_use]
    pub const fn with_max_completion_tokens(mut self, value: usize) -> Self {
        self.max_completion_tokens = Some(value);
        self
    }

    #[must_use]
    pub const fn with_temperature(mut self, value: f32) -> Self {
        self.temperature = Some(value);
        self
    }

    #[must_use]
    pub const fn with_top_p(mut self, value: f32) -> Self {
        self.top_p = Some(value);
        self
    }

    #[must_use]
    pub const fn with_presence_penalty(mut self, value: f32) -> Self {
        self.presence_penalty = Some(value);
        self
    }

    #[must_use]
    pub const fn with_frequency_penalty(mut self, value: f32) -> Self {
        self.frequency_penalty = Some(value);
        self
    }

    #[must_use]
    pub fn with_stop(mut self, value: impl Into<String>) -> Self {
        self.stop = Some(StopSequences::One(value.into()));
        self
    }

    #[must_use]
    pub fn with_stop_sequences(mut self, values: Vec<String>) -> Self {
        self.stop = Some(StopSequences::Many(values));
        self
    }

    #[must_use]
    pub const fn with_seed(mut self, value: u32) -> Self {
        self.seed = Some(value);
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if self.messages.is_empty() {
            return Err(ClientError::build_request("messages must not be empty"));
        }
        validate_sampling(
            self.temperature,
            self.top_p,
            self.presence_penalty,
            self.frequency_penalty,
        )?;
        if let Some(stop) = &self.stop {
            let values = match stop {
                StopSequences::One(value) => std::slice::from_ref(value),
                StopSequences::Many(values) => values.as_slice(),
            };
            if values.is_empty() || values.len() > 4 || values.iter().any(String::is_empty) {
                return Err(ClientError::build_request(
                    "stop must contain 1 to 4 nonempty strings",
                ));
            }
        }
        Ok(())
    }
}

/// A chat completion response.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
    pub timings: LlmTimings,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub logprobs: Option<Value>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: String,
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[non_exhaustive]
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

/// One chat completion stream chunk.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
    pub usage: Option<ChatUsage>,
    #[serde(default)]
    pub timings: Option<LlmTimings>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct ChatDelta {
    #[serde(default)]
    pub role: Option<MessageRole>,
    #[serde(default)]
    pub content: Option<String>,
}

/// A chat stream event. The `[DONE]` sentinel is consumed as the successful end of the stream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChatCompletionEvent {
    Chunk(ChatCompletionChunk),
}

pub struct ChatCompletionStream {
    stream: SseStream,
    terminal: bool,
}

impl ChatCompletionStream {
    /// Returns a chunk, or `None` after the unique `[DONE]` terminal sentinel.
    ///
    /// # Errors
    ///
    /// Any stream, server, or decode error terminates this stream.
    pub async fn next_event(&mut self) -> Result<Option<ChatCompletionEvent>, ClientError> {
        if self.terminal {
            return Ok(None);
        }
        let event = match self.stream.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                self.terminal = true;
                return Err(ClientError::UnexpectedEof {
                    stream: "chat completion",
                });
            }
            Err(error) => {
                self.terminal = true;
                return Err(error);
            }
        };
        if event.event != "message" {
            return self
                .decode_failure(format!("unexpected chat SSE event name `{}`", event.event));
        }
        if event.data == "[DONE]" {
            self.terminal = true;
            return Ok(None);
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => return self.decode_failure(format!("invalid chat event JSON: {error}")),
        };
        if value.get("error").is_some() {
            self.terminal = true;
            let body: ServerErrorBody = serde_json::from_value(value).map_err(|error| {
                ClientError::decode(format!("invalid chat streaming error: {error}"))
            })?;
            return Err(ClientError::StreamingServer { error: body.error });
        }
        let chunk = match serde_json::from_value(value) {
            Ok(chunk) => chunk,
            Err(error) => {
                return self.decode_failure(format!("invalid chat completion chunk: {error}"));
            }
        };
        Ok(Some(ChatCompletionEvent::Chunk(chunk)))
    }

    fn decode_failure<T>(&mut self, message: String) -> Result<T, ClientError> {
        self.terminal = true;
        Err(ClientError::decode(message))
    }
}

/// Input accepted by the Responses API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponseInputMessage>),
}

impl ResponsesInput {
    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    #[must_use]
    pub fn messages(value: Vec<ResponseInputMessage>) -> Self {
        Self::Messages(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ResponseInputMessage {
    #[serde(rename = "type")]
    kind: &'static str,
    pub role: MessageRole,
    pub content: String,
}

impl ResponseInputMessage {
    #[must_use]
    pub const fn new(role: MessageRole, content: String) -> Self {
        Self {
            kind: "message",
            role,
            content,
        }
    }

    #[must_use]
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self::new(role, content.into())
    }
}

/// Responses API parameters controlled by the caller.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

impl ResponsesRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, input: ResponsesInput) -> Self {
        Self {
            model: model.into(),
            input,
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    #[must_use]
    pub fn with_instructions(mut self, value: impl Into<String>) -> Self {
        self.instructions = Some(value.into());
        self
    }

    #[must_use]
    pub const fn with_max_output_tokens(mut self, value: usize) -> Self {
        self.max_output_tokens = Some(value);
        self
    }

    #[must_use]
    pub const fn with_temperature(mut self, value: f32) -> Self {
        self.temperature = Some(value);
        self
    }

    #[must_use]
    pub const fn with_top_p(mut self, value: f32) -> Self {
        self.top_p = Some(value);
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if matches!(&self.input, ResponsesInput::Messages(messages) if messages.is_empty()) {
            return Err(ClientError::build_request(
                "Responses message input must not be empty",
            ));
        }
        if self.max_output_tokens.is_some_and(|value| value < 16) {
            return Err(ClientError::build_request(
                "max_output_tokens must be at least 16",
            ));
        }
        validate_sampling(self.temperature, self.top_p, None, None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub input_tokens_details: InputTokenDetails,
    pub output_tokens: usize,
    pub output_tokens_details: OutputTokenDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct InputTokenDetails {
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct OutputTokenDetails {
    pub reasoning_tokens: usize,
}

/// A Responses API object. Extensible `OpenAI` item fields remain JSON values.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: ResponseStatus,
    pub background: bool,
    pub error: Option<Value>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub model: String,
    pub output: Vec<Value>,
    pub output_text: String,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: Value,
    pub store: bool,
    pub temperature: Option<f32>,
    pub text: Value,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub top_logprobs: usize,
    pub top_p: Option<f32>,
    pub truncation: String,
    pub metadata: Value,
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub timings: Option<LlmTimings>,
}

/// A semantically validated Responses API lifecycle event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResponsesEvent {
    Created {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    InProgress {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    OutputItemAdded {
        output_index: usize,
        item: Value,
        sequence_number: u64,
    },
    ContentPartAdded {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: Value,
        sequence_number: u64,
    },
    OutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
        logprobs: Vec<Value>,
        sequence_number: u64,
    },
    OutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
        logprobs: Vec<Value>,
        sequence_number: u64,
    },
    ContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: Value,
        sequence_number: u64,
    },
    OutputItemDone {
        output_index: usize,
        item: Value,
        sequence_number: u64,
    },
    Completed {
        response: ResponsesResponse,
        timings: LlmTimings,
        sequence_number: u64,
    },
    Incomplete {
        response: ResponsesResponse,
        timings: LlmTimings,
        sequence_number: u64,
    },
}

pub struct ResponsesStream {
    stream: SseStream,
    terminal: bool,
    next_sequence: u64,
    phase: ResponsesPhase,
}

impl ResponsesStream {
    /// Returns the next lifecycle event, or `None` after `completed` or `incomplete`.
    ///
    /// # Errors
    ///
    /// Any stream, server, sequence, lifecycle, or decode error terminates this stream.
    pub async fn next_event(&mut self) -> Result<Option<ResponsesEvent>, ClientError> {
        if self.terminal {
            return Ok(None);
        }
        let event = match self.stream.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                self.terminal = true;
                return Err(ClientError::UnexpectedEof {
                    stream: "Responses API",
                });
            }
            Err(error) => {
                self.terminal = true;
                return Err(error);
            }
        };
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => {
                return self.decode_failure(format!("invalid Responses event JSON: {error}"));
            }
        };
        let kind = match value.get("type").and_then(Value::as_str) {
            Some(kind) => kind.to_owned(),
            None => return self.decode_failure("Responses event is missing string `type`"),
        };
        if event.event != kind {
            return self.decode_failure(format!(
                "Responses SSE event `{}` does not match data type `{kind}`",
                event.event
            ));
        }
        let Some(sequence) = value.get("sequence_number").and_then(Value::as_u64) else {
            return self.decode_failure("Responses event is missing `sequence_number`");
        };
        if sequence != self.next_sequence {
            return self.decode_failure(format!(
                "expected Responses sequence {}, received {sequence}",
                self.next_sequence
            ));
        }

        if kind == "error" {
            self.terminal = true;
            let error =
                serde_json::from_value::<wire::FlatStreamError>(value).map_err(|error| {
                    ClientError::decode(format!("invalid Responses error: {error}"))
                })?;
            return Err(ClientError::StreamingServerEvent {
                error: StreamErrorObject {
                    code: error.code,
                    message: error.message,
                    param: error.param,
                },
            });
        }

        let Some(next_phase) = self.phase.transition(&kind) else {
            return self.decode_failure(format!(
                "Responses event `{kind}` is duplicated or out of lifecycle order"
            ));
        };
        let decoded = match wire::decode_response_event(&kind, value) {
            Ok(event) => event,
            Err(error) => return self.decode_failure(error),
        };
        self.next_sequence += 1;
        self.phase = next_phase;
        if matches!(
            decoded,
            ResponsesEvent::Completed { .. } | ResponsesEvent::Incomplete { .. }
        ) {
            self.terminal = true;
        }
        Ok(Some(decoded))
    }

    fn decode_failure<T>(&mut self, message: impl Into<String>) -> Result<T, ClientError> {
        self.terminal = true;
        Err(ClientError::decode(message))
    }
}

#[derive(Clone, Copy)]
enum ResponsesPhase {
    Start,
    Created,
    InProgress,
    ItemAdded,
    PartAdded,
    Delta,
    TextDone,
    PartDone,
    ItemDone,
    Terminal,
}

impl ResponsesPhase {
    fn transition(self, kind: &str) -> Option<Self> {
        match (self, kind) {
            (Self::Start, "response.created") => Some(Self::Created),
            (Self::Created, "response.in_progress") => Some(Self::InProgress),
            (Self::InProgress, "response.output_item.added") => Some(Self::ItemAdded),
            (Self::ItemAdded, "response.content_part.added") => Some(Self::PartAdded),
            (Self::PartAdded | Self::Delta, "response.output_text.delta") => Some(Self::Delta),
            (Self::PartAdded | Self::Delta, "response.output_text.done") => Some(Self::TextDone),
            (Self::TextDone, "response.content_part.done") => Some(Self::PartDone),
            (Self::PartDone, "response.output_item.done") => Some(Self::ItemDone),
            (Self::ItemDone, "response.completed" | "response.incomplete") => Some(Self::Terminal),
            _ => None,
        }
    }
}

fn validate_model(model: &str) -> Result<(), ClientError> {
    if model.is_empty() {
        Err(ClientError::build_request("model must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence: Option<f32>,
    frequency: Option<f32>,
) -> Result<(), ClientError> {
    if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
        return Err(ClientError::build_request("temperature must be in [0, 2]"));
    }
    if top_p.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0) {
        return Err(ClientError::build_request("top_p must be in (0, 1]"));
    }
    if presence.is_some_and(|value| !value.is_finite() || !(-2.0..=2.0).contains(&value)) {
        return Err(ClientError::build_request(
            "presence_penalty must be in [-2, 2]",
        ));
    }
    if frequency.is_some_and(|value| !value.is_finite() || !(-2.0..=2.0).contains(&value)) {
        return Err(ClientError::build_request(
            "frequency_penalty must be in [-2, 2]",
        ));
    }
    Ok(())
}

mod wire {
    use super::{ChatCompletionRequest, LlmTimings, ResponsesEvent, ResponsesResponse};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Serialize)]
    pub(super) struct ChatRequest<'a> {
        #[serde(flatten)]
        pub request: &'a ChatCompletionRequest,
        pub stream: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream_options: Option<ChatStreamOptions>,
    }

    #[derive(Serialize)]
    pub(super) struct ChatStreamOptions {
        pub include_usage: bool,
    }

    #[derive(Serialize)]
    pub(super) struct ResponsesRequest<'a> {
        #[serde(flatten)]
        pub request: &'a super::ResponsesRequest,
        pub store: bool,
        pub stream: bool,
    }

    #[derive(Deserialize)]
    pub(super) struct FlatStreamError {
        pub code: Option<String>,
        pub message: String,
        pub param: Option<String>,
    }

    #[derive(Deserialize)]
    struct SnapshotEvent {
        response: ResponsesResponse,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct ItemEvent {
        output_index: usize,
        item: Value,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct PartEvent {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: Value,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct TextEvent {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
        logprobs: Vec<Value>,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct DeltaEvent {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
        logprobs: Vec<Value>,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct TerminalEvent {
        response: ResponsesResponse,
        timings: LlmTimings,
        sequence_number: u64,
    }

    pub(super) fn decode_response_event(
        kind: &str,
        value: Value,
    ) -> Result<ResponsesEvent, String> {
        macro_rules! decode {
            ($type:ty) => {
                serde_json::from_value::<$type>(value)
                    .map_err(|error| format!("invalid `{kind}` event: {error}"))?
            };
        }
        Ok(match kind {
            "response.created" => {
                let event = decode!(SnapshotEvent);
                ResponsesEvent::Created {
                    response: event.response,
                    sequence_number: event.sequence_number,
                }
            }
            "response.in_progress" => {
                let event = decode!(SnapshotEvent);
                ResponsesEvent::InProgress {
                    response: event.response,
                    sequence_number: event.sequence_number,
                }
            }
            "response.output_item.added" => {
                let event = decode!(ItemEvent);
                ResponsesEvent::OutputItemAdded {
                    output_index: event.output_index,
                    item: event.item,
                    sequence_number: event.sequence_number,
                }
            }
            "response.content_part.added" => {
                let event = decode!(PartEvent);
                ResponsesEvent::ContentPartAdded {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    content_index: event.content_index,
                    part: event.part,
                    sequence_number: event.sequence_number,
                }
            }
            "response.output_text.delta" => {
                let event = decode!(DeltaEvent);
                ResponsesEvent::OutputTextDelta {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    content_index: event.content_index,
                    delta: event.delta,
                    logprobs: event.logprobs,
                    sequence_number: event.sequence_number,
                }
            }
            "response.output_text.done" => {
                let event = decode!(TextEvent);
                ResponsesEvent::OutputTextDone {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    content_index: event.content_index,
                    text: event.text,
                    logprobs: event.logprobs,
                    sequence_number: event.sequence_number,
                }
            }
            "response.content_part.done" => {
                let event = decode!(PartEvent);
                ResponsesEvent::ContentPartDone {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    content_index: event.content_index,
                    part: event.part,
                    sequence_number: event.sequence_number,
                }
            }
            "response.output_item.done" => {
                let event = decode!(ItemEvent);
                ResponsesEvent::OutputItemDone {
                    output_index: event.output_index,
                    item: event.item,
                    sequence_number: event.sequence_number,
                }
            }
            "response.completed" => {
                let event = decode!(TerminalEvent);
                ResponsesEvent::Completed {
                    response: event.response,
                    timings: event.timings,
                    sequence_number: event.sequence_number,
                }
            }
            "response.incomplete" => {
                let event = decode!(TerminalEvent);
                ResponsesEvent::Incomplete {
                    response: event.response,
                    timings: event.timings,
                    sequence_number: event.sequence_number,
                }
            }
            _ => return Err(format!("unknown Responses event type `{kind}`")),
        })
    }
}
