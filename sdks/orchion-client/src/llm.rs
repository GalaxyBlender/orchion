use crate::client::decode_json;
use crate::sse::SseStream;
use crate::{Client, ClientError, ServerErrorBody, StreamErrorObject};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

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
        if request.reasoning_control {
            return Err(ClientError::build_request(
                "reasoning_control requires a streaming chat request",
            ));
        }
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

    /// Creates a non-streaming legacy text completion.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, or response decoding fails.
    pub async fn create_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ClientError> {
        request.validate()?;
        let response = self
            .client
            .post("/v1/completions")?
            .json(&wire::CompletionRequest {
                request: &request,
                stream: false,
            })
            .send()
            .await?;
        decode_json(response).await
    }

    /// Starts a legacy text completion stream without automatic retries.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, or SSE setup fails.
    pub async fn stream_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/completions")?
            .json(&wire::CompletionRequest {
                request: &request,
                stream: true,
            })
            .send()
            .await?;
        Ok(CompletionStream {
            stream: SseStream::from_response(response).await?,
            terminal: false,
            stream_id: None,
        })
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
        let completion_id = completion_id_header(&response)?;
        Ok(ChatCompletionStream {
            stream: SseStream::from_response(response).await?,
            terminal: false,
            stream_id: None,
            completion_id,
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

    /// Counts tokens after applying the Responses prompt template.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, or response decoding fails.
    pub async fn count_response_input_tokens(
        &self,
        request: ResponsesInputTokensRequest,
    ) -> Result<ResponsesInputTokensResponse, ClientError> {
        request.validate()?;
        let response = self
            .client
            .post("/v1/responses/input_tokens")?
            .json(&request)
            .send()
            .await?;
        decode_json(response).await
    }

    /// Creates embeddings for text or token inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, or response decoding fails.
    pub async fn create_embeddings(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, ClientError> {
        request.validate()?;
        let response = self
            .client
            .post("/v1/embeddings")?
            .json(&request)
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
            next_sequence: Some(0),
            phase: ResponsesPhase::Start,
            stream_id: None,
            resumed: false,
        })
    }

    /// Starts an opt-in resumable chat completion stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, headers, or SSE setup fails.
    pub async fn start_resumable_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/chat/completions")?
            .header("x-orchion-resumable", "true")
            .json(&wire::ChatRequest {
                request: &request,
                stream: true,
                stream_options: Some(wire::ChatStreamOptions {
                    include_usage: true,
                }),
            })
            .send()
            .await?;
        let stream_id = resumable_stream_id(&response)?;
        let completion_id = completion_id_header(&response)?;
        Ok(ChatCompletionStream {
            stream: SseStream::from_resumable_response(response, None).await?,
            terminal: false,
            stream_id: Some(stream_id),
            completion_id,
        })
    }

    /// Starts an opt-in resumable legacy completion stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, headers, or SSE setup fails.
    pub async fn start_resumable_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/completions")?
            .header("x-orchion-resumable", "true")
            .json(&wire::CompletionRequest {
                request: &request,
                stream: true,
            })
            .send()
            .await?;
        let stream_id = resumable_stream_id(&response)?;
        Ok(CompletionStream {
            stream: SseStream::from_resumable_response(response, None).await?,
            terminal: false,
            stream_id: Some(stream_id),
        })
    }

    /// Starts an opt-in resumable Responses API stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when validation, transport, headers, or SSE setup fails.
    pub async fn start_resumable_response(
        &self,
        request: ResponsesRequest,
    ) -> Result<ResponsesStream, ClientError> {
        request.validate()?;
        let response = self
            .client
            .stream_post("/v1/responses")?
            .header("x-orchion-resumable", "true")
            .json(&wire::ResponsesRequest {
                request: &request,
                store: false,
                stream: true,
            })
            .send()
            .await?;
        let stream_id = resumable_stream_id(&response)?;
        Ok(ResponsesStream {
            stream: SseStream::from_resumable_response(response, None).await?,
            terminal: false,
            next_sequence: Some(0),
            phase: ResponsesPhase::Start,
            stream_id: Some(stream_id),
            resumed: false,
        })
    }

    /// Resumes a chat stream strictly after `last_event_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the stream cannot be resumed or decoded.
    pub async fn resume_chat_completion(
        &self,
        stream_id: impl Into<String>,
        last_event_id: Option<u64>,
    ) -> Result<ChatCompletionStream, ClientError> {
        let stream_id = stream_id.into();
        let response = self.resume_request(&stream_id, last_event_id).await?;
        Ok(ChatCompletionStream {
            stream: SseStream::from_resumable_response(response, last_event_id).await?,
            terminal: false,
            stream_id: Some(stream_id),
            completion_id: None,
        })
    }

    /// Resumes a legacy completion stream strictly after `last_event_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the stream cannot be resumed or decoded.
    pub async fn resume_completion(
        &self,
        stream_id: impl Into<String>,
        last_event_id: Option<u64>,
    ) -> Result<CompletionStream, ClientError> {
        let stream_id = stream_id.into();
        let response = self.resume_request(&stream_id, last_event_id).await?;
        Ok(CompletionStream {
            stream: SseStream::from_resumable_response(response, last_event_id).await?,
            terminal: false,
            stream_id: Some(stream_id),
        })
    }

    /// Resumes a Responses API stream strictly after `last_event_id`.
    ///
    /// `None` and `Some(0)` request a full replay and validate the complete sequence-zero
    /// `created` -> `in_progress` prefix. A positive cursor cannot validate events before the
    /// cursor, but the first replayed event establishes the lifecycle phase used to validate all
    /// subsequent events.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the stream cannot be resumed or decoded.
    pub async fn resume_response(
        &self,
        stream_id: impl Into<String>,
        last_event_id: Option<u64>,
    ) -> Result<ResponsesStream, ClientError> {
        let stream_id = stream_id.into();
        let response = self.resume_request(&stream_id, last_event_id).await?;
        Ok(ResponsesStream {
            stream: SseStream::from_resumable_response(response, last_event_id).await?,
            terminal: false,
            next_sequence: (last_event_id.unwrap_or(0) == 0).then_some(0),
            phase: ResponsesPhase::Start,
            stream_id: Some(stream_id),
            resumed: last_event_id.is_some_and(|cursor| cursor > 0),
        })
    }

    /// Looks up only the requested resumable stream IDs owned by this client principal.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the lookup request or response fails.
    pub async fn lookup_streams(
        &self,
        stream_ids: Vec<String>,
    ) -> Result<StreamLookupResponse, ClientError> {
        let response = self
            .client
            .post("/v1/streams/lookup")?
            .json(&orchion_protocol::LlmStreamLookupRequest { stream_ids })
            .send()
            .await?;
        decode_json(response).await
    }

    /// Idempotently deletes and cancels a resumable stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the delete request fails.
    pub async fn delete_stream(&self, stream_id: &str) -> Result<(), ClientError> {
        let response = self
            .client
            .delete("/v1/stream")?
            .query(&[("stream_id", stream_id)])
            .send()
            .await?;
        crate::client::ensure_success(response).await?;
        Ok(())
    }

    /// Requests an early end to the active reasoning block of an armed chat stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] for an invalid completion ID, transport failure, server rejection,
    /// or malformed response.
    pub async fn control_chat_reasoning(
        &self,
        request: ChatReasoningControlRequest,
    ) -> Result<ChatReasoningControlResult, ClientError> {
        validate_completion_id(&request.id)?;
        if let Some(model) = request.model.as_deref() {
            validate_model(model)?;
        }
        let response = self
            .client
            .post("/v1/chat/completions/control")?
            .json(&wire::ChatReasoningControlRequest {
                id: &request.id,
                action: "reasoning_end",
                model: request.model.as_deref(),
            })
            .send()
            .await?;
        decode_json(response).await
    }

    async fn resume_request(
        &self,
        stream_id: &str,
        last_event_id: Option<u64>,
    ) -> Result<reqwest::Response, ClientError> {
        let mut request = self
            .client
            .stream_get("/v1/stream")?
            .query(&[("stream_id", stream_id), ("follow", "true")]);
        if let Some(last_event_id) = last_event_id {
            request = request.header("last-event-id", last_event_id);
        }
        Ok(request.send().await?)
    }
}

fn resumable_stream_id(response: &reqwest::Response) -> Result<String, ClientError> {
    response
        .headers()
        .get("x-orchion-stream-id")
        .ok_or_else(|| ClientError::decode("missing X-Orchion-Stream-ID response header"))?
        .to_str()
        .map(ToOwned::to_owned)
        .map_err(|error| {
            ClientError::decode(format!("invalid X-Orchion-Stream-ID header: {error}"))
        })
}

fn completion_id_header(response: &reqwest::Response) -> Result<Option<String>, ClientError> {
    let Some(value) = response.headers().get("x-orchion-completion-id") else {
        return Ok(None);
    };
    value
        .to_str()
        .map(ToOwned::to_owned)
        .map(Some)
        .map_err(|error| ClientError::decode(format!("invalid completion ID header: {error}")))
}

fn validate_completion_id(id: &str) -> Result<(), ClientError> {
    let valid = id.strip_prefix("chatcmpl_").is_some_and(|value| {
        value.len() == 32
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    });
    if valid {
        Ok(())
    } else {
        Err(ClientError::build_request("invalid chat completion id"))
    }
}

pub type StreamLookupResponse = orchion_protocol::LlmStreamLookupResponse;
pub type ResumableStreamMetadata = orchion_protocol::LlmStreamMetadata;

/// Legacy prompt completion parameters controlled by the caller.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, f32>>,
}

impl CompletionRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            seed: None,
            n: None,
            logprobs: None,
            logit_bias: None,
        }
    }

    #[must_use]
    pub const fn with_max_tokens(mut self, value: usize) -> Self {
        self.max_tokens = Some(value);
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
    pub fn with_stop(mut self, value: impl Into<String>) -> Self {
        self.stop = Some(StopSequences::One(value.into()));
        self
    }

    #[must_use]
    pub const fn with_seed(mut self, value: u32) -> Self {
        self.seed = Some(value);
        self
    }

    #[must_use]
    pub const fn with_choices(mut self, value: usize) -> Self {
        self.n = Some(value);
        self
    }

    #[must_use]
    pub const fn with_logprobs(mut self, value: usize) -> Self {
        self.logprobs = Some(value);
        self
    }

    #[must_use]
    pub fn with_logit_bias(mut self, value: BTreeMap<String, f32>) -> Self {
        self.logit_bias = Some(value);
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if self.max_tokens == Some(0) {
            return Err(ClientError::build_request(
                "max_tokens must be greater than zero",
            ));
        }
        if self.n == Some(0) {
            return Err(ClientError::build_request("n must be greater than zero"));
        }
        if self.logprobs.is_some_and(|value| value > 5) {
            return Err(ClientError::build_request("logprobs must be in [0, 5]"));
        }
        validate_logit_bias(self.logit_bias.as_ref())?;
        validate_sampling(self.temperature, self.top_p, None, None)?;
        validate_stop(self.stop.as_ref())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: ChatUsage,
    #[serde(default)]
    pub timings: Option<LlmTimings>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<Value>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
    pub usage: Option<ChatUsage>,
    #[serde(default)]
    pub timings: Option<LlmTimings>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

pub struct CompletionStream {
    stream: SseStream,
    terminal: bool,
    stream_id: Option<String>,
}

impl CompletionStream {
    #[must_use]
    pub fn stream_id(&self) -> Option<&str> {
        self.stream_id.as_deref()
    }

    #[must_use]
    pub const fn last_event_id(&self) -> Option<u64> {
        self.stream.last_event_id()
    }
    /// Returns the next completion chunk, or `None` after `[DONE]`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] for malformed events, server errors, or premature EOF.
    pub async fn next_event(&mut self) -> Result<Option<CompletionChunk>, ClientError> {
        if self.terminal {
            return Ok(None);
        }
        let event = match self.stream.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                self.terminal = true;
                return Err(ClientError::UnexpectedEof {
                    stream: "text completion",
                });
            }
            Err(error) => {
                self.terminal = true;
                return Err(error);
            }
        };
        if event.event != "message" {
            self.terminal = true;
            return Err(ClientError::decode(format!(
                "unexpected completion SSE event name `{}`",
                event.event
            )));
        }
        if event.data == "[DONE]" {
            self.terminal = true;
            return Ok(None);
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => {
                self.terminal = true;
                return Err(ClientError::decode(format!(
                    "invalid completion event JSON: {error}"
                )));
            }
        };
        if value.get("error").is_some() {
            self.terminal = true;
            let body: ServerErrorBody = serde_json::from_value(value).map_err(|error| {
                ClientError::decode(format!("invalid completion streaming error: {error}"))
            })?;
            return Err(ClientError::StreamingServer { error: body.error });
        }
        match serde_json::from_value(value) {
            Ok(chunk) => Ok(Some(chunk)),
            Err(error) => {
                self.terminal = true;
                Err(ClientError::decode(format!(
                    "invalid completion chunk: {error}"
                )))
            }
        }
    }
}

/// Input accepted by the embeddings API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum EmbeddingsInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<i32>),
    TokenBatches(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    pub encoding_format: EmbeddingEncodingFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingsRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, input: EmbeddingsInput) -> Self {
        Self {
            model: model.into(),
            input,
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
            user: None,
        }
    }

    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    #[must_use]
    pub const fn with_encoding_format(mut self, format: EmbeddingEncodingFormat) -> Self {
        self.encoding_format = format;
        self
    }

    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        let valid = match &self.input {
            EmbeddingsInput::Text(text) => !text.is_empty(),
            EmbeddingsInput::Texts(texts) => {
                !texts.is_empty()
                    && texts.len() <= 2048
                    && texts.iter().all(|text| !text.is_empty())
            }
            EmbeddingsInput::Tokens(tokens) => !tokens.is_empty(),
            EmbeddingsInput::TokenBatches(batches) => {
                !batches.is_empty()
                    && batches.len() <= 2048
                    && batches.iter().all(|tokens| !tokens.is_empty())
            }
        };
        if !valid {
            return Err(ClientError::build_request(
                "embedding input must contain 1 to 2048 nonempty items",
            ));
        }
        if self.dimensions == Some(0) {
            return Err(ClientError::build_request(
                "embedding dimensions must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: EmbeddingValue,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
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
    Tool,
}

/// A typed text message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: ChatMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text { text: String },
    ImageUrl { image_url: ChatImageUrl },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: ChatMessageContent::Text(content.into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn with_content_parts(mut self, parts: Vec<ChatContentPart>) -> Self {
        self.content = ChatMessageContent::Parts(parts);
        self
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

    #[must_use]
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        let mut message = Self::assistant("");
        message.tool_calls = tool_calls;
        message
    }

    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let mut message = Self::new(MessageRole::Tool, content);
        message.tool_call_id = Some(tool_call_id.into());
        message
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    #[must_use]
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct FunctionTool {
    #[serde(rename = "type")]
    kind: &'static str,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

impl FunctionTool {
    #[must_use]
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function",
            function: FunctionDefinition {
                name: name.into(),
                description: None,
                parameters,
                strict: true,
            },
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.function.description = Some(description.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    kind: &'static str,
    function: NamedFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NamedFunction {
    name: String,
}

impl ToolChoice {
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(NamedToolChoice {
            kind: "function",
            function: NamedFunction { name: name.into() },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: Value,
    pub strict: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub reasoning_control: bool,
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
            n: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
            logprobs: None,
            top_logprobs: None,
            logit_bias: None,
            reasoning_effort: None,
            reasoning_control: false,
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

    #[must_use]
    pub const fn with_choices(mut self, value: usize) -> Self {
        self.n = Some(value);
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<FunctionTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    #[must_use]
    pub fn with_response_format(mut self, format: OutputFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    #[must_use]
    pub const fn with_logprobs(mut self, top_logprobs: usize) -> Self {
        self.logprobs = Some(true);
        self.top_logprobs = Some(top_logprobs);
        self
    }

    #[must_use]
    pub fn with_logit_bias(mut self, bias: BTreeMap<String, f32>) -> Self {
        self.logit_bias = Some(bias);
        self
    }

    #[must_use]
    pub const fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    #[must_use]
    pub const fn with_reasoning_control(mut self) -> Self {
        self.reasoning_control = true;
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if self.messages.is_empty() {
            return Err(ClientError::build_request("messages must not be empty"));
        }
        if self.n == Some(0) {
            return Err(ClientError::build_request("n must be greater than zero"));
        }
        if self.reasoning_control && self.n.unwrap_or(1) != 1 {
            return Err(ClientError::build_request("reasoning_control requires n=1"));
        }
        if self.top_logprobs.is_some_and(|value| value > 20) {
            return Err(ClientError::build_request(
                "top_logprobs must be in [0, 20]",
            ));
        }
        if self.top_logprobs.is_some() && self.logprobs != Some(true) {
            return Err(ClientError::build_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        validate_logit_bias(self.logit_bias.as_ref())?;
        validate_sampling(
            self.temperature,
            self.top_p,
            self.presence_penalty,
            self.frequency_penalty,
        )?;
        validate_stop(self.stop.as_ref())
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

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: String,
    pub content_is_null: bool,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub refusal: Option<String>,
}

impl<'de> Deserialize<'de> for AssistantMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireAssistantMessage {
            role: MessageRole,
            content: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            tool_calls: Vec<ToolCall>,
            #[serde(default)]
            refusal: Option<String>,
        }
        let value = WireAssistantMessage::deserialize(deserializer)?;
        let content_is_null = value.content.is_none();
        Ok(Self {
            role: value.role,
            content: value.content.unwrap_or_default(),
            content_is_null,
            reasoning_content: value.reasoning_content,
            tool_calls: value.tool_calls,
            refusal: value.refusal,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Unknown(String),
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            _ => Self::Unknown(value),
        })
    }
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
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: FunctionCallDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
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
    stream_id: Option<String>,
    completion_id: Option<String>,
}

impl ChatCompletionStream {
    #[must_use]
    pub fn stream_id(&self) -> Option<&str> {
        self.stream_id.as_deref()
    }

    #[must_use]
    pub fn completion_id(&self) -> Option<&str> {
        self.completion_id.as_deref()
    }

    #[must_use]
    pub const fn last_event_id(&self) -> Option<u64> {
        self.stream.last_event_id()
    }
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
        let chunk: ChatCompletionChunk = match serde_json::from_value(value) {
            Ok(chunk) => chunk,
            Err(error) => {
                return self.decode_failure(format!("invalid chat completion chunk: {error}"));
            }
        };
        if self
            .completion_id
            .as_deref()
            .is_some_and(|id| id != chunk.id)
        {
            return self.decode_failure("chat stream completion ID changed".to_string());
        }
        if self.completion_id.is_none() {
            self.completion_id = Some(chunk.id.clone());
        }
        Ok(Some(ChatCompletionEvent::Chunk(chunk)))
    }

    fn decode_failure<T>(&mut self, message: String) -> Result<T, ClientError> {
        self.terminal = true;
        Err(ClientError::decode(message))
    }
}

#[derive(Debug, Clone)]
pub struct ChatReasoningControlRequest {
    pub id: String,
    pub model: Option<String>,
}

impl ChatReasoningControlRequest {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: None,
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatReasoningControlResult {
    pub id: String,
    pub action: String,
    pub success: bool,
    pub message: Option<String>,
}

/// Input accepted by the Responses API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponseInputMessage>),
    Items(Vec<ResponseInputItem>),
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

    #[must_use]
    pub fn items(value: Vec<ResponseInputItem>) -> Self {
        Self::Items(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseInputItem {
    Message {
        role: MessageRole,
        content: String,
    },
    #[serde(rename = "message")]
    MessageParts {
        role: MessageRole,
        content: Vec<ResponseInputContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    Reasoning {
        summary: Vec<ResponseSummaryPart>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputContentPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResponseSummaryPart {
    #[serde(rename = "type")]
    kind: &'static str,
    pub text: String,
}

impl ResponseSummaryPart {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "summary_text",
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ResponseInputMessage {
    #[serde(rename = "type")]
    kind: &'static str,
    pub role: MessageRole,
    pub content: ResponseInputMessageContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ResponseInputMessageContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

impl ResponseInputMessage {
    #[must_use]
    pub const fn new(role: MessageRole, content: ResponseInputMessageContent) -> Self {
        Self {
            kind: "message",
            role,
            content,
        }
    }

    #[must_use]
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self::new(role, ResponseInputMessageContent::Text(content.into()))
    }

    #[must_use]
    pub fn parts(role: MessageRole, content: Vec<ResponseInputContentPart>) -> Self {
        Self::new(role, ResponseInputMessageContent::Parts(content))
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
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
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            text: None,
            top_logprobs: None,
            reasoning: None,
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

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<FunctionTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    #[must_use]
    pub fn with_text_format(mut self, format: ResponsesTextFormat) -> Self {
        self.text = Some(ResponsesText { format });
        self
    }

    #[must_use]
    pub const fn with_top_logprobs(mut self, value: usize) -> Self {
        self.top_logprobs = Some(value);
        self
    }

    #[must_use]
    pub const fn with_reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(ResponsesReasoning { effort });
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if matches!(&self.input, ResponsesInput::Messages(messages) if messages.is_empty())
            || matches!(&self.input, ResponsesInput::Items(items) if items.is_empty())
        {
            return Err(ClientError::build_request(
                "Responses message input must not be empty",
            ));
        }
        if self.max_output_tokens.is_some_and(|value| value < 16) {
            return Err(ClientError::build_request(
                "max_output_tokens must be at least 16",
            ));
        }
        if self.top_logprobs.is_some_and(|value| value > 20) {
            return Err(ClientError::build_request(
                "top_logprobs must be in [0, 20]",
            ));
        }
        validate_sampling(self.temperature, self.top_p, None, None)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponsesText {
    pub format: ResponsesTextFormat,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponsesTextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResponsesReasoning {
    pub effort: ReasoningEffort,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct ResponsesInputTokensRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesText>,
}

impl ResponsesInputTokensRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, input: ResponsesInput) -> Self {
        Self {
            model: model.into(),
            input,
            instructions: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            text: None,
        }
    }

    #[must_use]
    pub fn with_instructions(mut self, value: impl Into<String>) -> Self {
        self.instructions = Some(value.into());
        self
    }

    #[must_use]
    pub fn with_tools(mut self, tools: Vec<FunctionTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    #[must_use]
    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    #[must_use]
    pub const fn with_reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(ResponsesReasoning { effort });
        self
    }

    #[must_use]
    pub fn with_text_format(mut self, format: ResponsesTextFormat) -> Self {
        self.text = Some(ResponsesText { format });
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        validate_model(&self.model)?;
        if matches!(&self.input, ResponsesInput::Messages(messages) if messages.is_empty())
            || matches!(&self.input, ResponsesInput::Items(items) if items.is_empty())
        {
            return Err(ClientError::build_request(
                "Responses message input must not be empty",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[non_exhaustive]
pub struct ResponsesInputTokensResponse {
    pub object: String,
    pub input_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
    Unknown(String),
}

impl<'de> Deserialize<'de> for ResponseStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => Self::Unknown(value),
        })
    }
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
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: usize,
        delta: String,
        sequence_number: u64,
    },
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: usize,
        arguments: String,
        sequence_number: u64,
    },
    ReasoningSummaryPartAdded {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        part: Value,
        sequence_number: u64,
    },
    ReasoningSummaryTextDelta {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        delta: String,
        sequence_number: u64,
    },
    ReasoningSummaryTextDone {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        text: String,
        sequence_number: u64,
    },
    ReasoningSummaryPartDone {
        item_id: String,
        output_index: usize,
        summary_index: usize,
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
    Failed {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    Cancelled {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    Unknown {
        kind: String,
        data: Value,
        sequence_number: u64,
    },
}

pub struct ResponsesStream {
    stream: SseStream,
    terminal: bool,
    next_sequence: Option<u64>,
    phase: ResponsesPhase,
    stream_id: Option<String>,
    resumed: bool,
}

impl ResponsesStream {
    #[must_use]
    pub fn stream_id(&self) -> Option<&str> {
        self.stream_id.as_deref()
    }

    #[must_use]
    pub const fn last_event_id(&self) -> Option<u64> {
        self.stream.last_event_id()
    }
    /// Returns the next lifecycle event, or `None` after any terminal response event.
    ///
    /// Orchion servers currently emit `completed`, `incomplete`, or `error`. Typed `failed` and
    /// `cancelled` events are accepted for forward compatibility and compatible remote servers and
    /// are terminal with the same EOF behavior.
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
        if self
            .next_sequence
            .is_some_and(|expected| sequence != expected)
        {
            return self.decode_failure(format!(
                "expected Responses sequence {}, received {sequence}",
                self.next_sequence.unwrap_or(sequence)
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

        let next_phase = if self.resumed && self.phase == ResponsesPhase::Start {
            Some(ResponsesPhase::infer(&kind))
        } else {
            self.phase.transition(&kind)
        };
        let Some(next_phase) = next_phase else {
            return self.decode_failure(format!(
                "Responses event `{kind}` is duplicated or out of lifecycle order"
            ));
        };
        let trusted_terminal_status = value
            .get("response")
            .and_then(|response| response.get("status"))
            .or_else(|| value.get("status"))
            .and_then(Value::as_str)
            .is_some_and(|status| {
                matches!(status, "completed" | "incomplete" | "failed" | "cancelled")
            });
        let decoded = match wire::decode_response_event(&kind, value) {
            Ok(event) => event,
            Err(error) => return self.decode_failure(error),
        };
        self.next_sequence = Some(sequence.saturating_add(1));
        self.phase = next_phase;
        if matches!(
            decoded,
            ResponsesEvent::Completed { .. }
                | ResponsesEvent::Incomplete { .. }
                | ResponsesEvent::Failed { .. }
                | ResponsesEvent::Cancelled { .. }
        ) || matches!(decoded, ResponsesEvent::Unknown { .. }) && trusted_terminal_status
        {
            self.terminal = true;
        }
        Ok(Some(decoded))
    }

    fn decode_failure<T>(&mut self, message: impl Into<String>) -> Result<T, ClientError> {
        self.terminal = true;
        Err(ClientError::decode(message))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResponsesPhase {
    Start,
    Created,
    InProgress,
    Dynamic,
    Terminal,
}

impl ResponsesPhase {
    fn infer(kind: &str) -> Self {
        match kind {
            "response.created" => Self::Created,
            "response.in_progress" => Self::InProgress,
            "response.completed"
            | "response.incomplete"
            | "response.failed"
            | "response.cancelled" => Self::Terminal,
            _ => Self::Dynamic,
        }
    }

    fn transition(self, kind: &str) -> Option<Self> {
        match (self, kind) {
            (Self::Start, "response.created") => Some(Self::Created),
            (Self::Created, "response.in_progress") => Some(Self::InProgress),
            (
                Self::InProgress | Self::Dynamic,
                "response.completed"
                | "response.incomplete"
                | "response.failed"
                | "response.cancelled",
            ) => Some(Self::Terminal),
            (Self::InProgress | Self::Dynamic, kind)
                if !matches!(kind, "response.created" | "response.in_progress") =>
            {
                Some(Self::Dynamic)
            }
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

fn validate_stop(stop: Option<&StopSequences>) -> Result<(), ClientError> {
    if let Some(stop) = stop {
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

fn validate_logit_bias(values: Option<&BTreeMap<String, f32>>) -> Result<(), ClientError> {
    if values.is_some_and(|values| values.len() > 256) {
        return Err(ClientError::build_request(
            "logit_bias must contain at most 256 entries",
        ));
    }
    if values.into_iter().flatten().any(|(token, bias)| {
        token.parse::<i32>().is_err() || !bias.is_finite() || !(-100.0..=100.0).contains(bias)
    }) {
        Err(ClientError::build_request(
            "logit_bias keys must be token IDs and values must be in [-100, 100]",
        ))
    } else {
        Ok(())
    }
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if callbacks receive fields by reference"
)]
const fn is_false(value: &bool) -> bool {
    !*value
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
    pub(super) struct ChatReasoningControlRequest<'a> {
        pub id: &'a str,
        pub action: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<&'a str>,
    }

    #[derive(Serialize)]
    pub(super) struct CompletionRequest<'a> {
        #[serde(flatten)]
        pub request: &'a super::CompletionRequest,
        pub stream: bool,
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
    struct FunctionDeltaEvent {
        item_id: String,
        output_index: usize,
        delta: String,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct FunctionDoneEvent {
        item_id: String,
        output_index: usize,
        arguments: String,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct ReasoningPartEvent {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        part: Value,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct ReasoningTextDeltaEvent {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        delta: String,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct ReasoningTextDoneEvent {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        text: String,
        sequence_number: u64,
    }

    #[derive(Deserialize)]
    struct TerminalEvent {
        response: ResponsesResponse,
        timings: LlmTimings,
        sequence_number: u64,
    }

    #[allow(
        clippy::too_many_lines,
        reason = "keeps the forward-compatible Responses event discriminator mapping together"
    )]
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
            "response.function_call_arguments.delta" => {
                let event = decode!(FunctionDeltaEvent);
                ResponsesEvent::FunctionCallArgumentsDelta {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    delta: event.delta,
                    sequence_number: event.sequence_number,
                }
            }
            "response.function_call_arguments.done" => {
                let event = decode!(FunctionDoneEvent);
                ResponsesEvent::FunctionCallArgumentsDone {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    arguments: event.arguments,
                    sequence_number: event.sequence_number,
                }
            }
            "response.reasoning_summary_part.added" => {
                let event = decode!(ReasoningPartEvent);
                ResponsesEvent::ReasoningSummaryPartAdded {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    summary_index: event.summary_index,
                    part: event.part,
                    sequence_number: event.sequence_number,
                }
            }
            "response.reasoning_summary_text.delta" => {
                let event = decode!(ReasoningTextDeltaEvent);
                ResponsesEvent::ReasoningSummaryTextDelta {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    summary_index: event.summary_index,
                    delta: event.delta,
                    sequence_number: event.sequence_number,
                }
            }
            "response.reasoning_summary_text.done" => {
                let event = decode!(ReasoningTextDoneEvent);
                ResponsesEvent::ReasoningSummaryTextDone {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    summary_index: event.summary_index,
                    text: event.text,
                    sequence_number: event.sequence_number,
                }
            }
            "response.reasoning_summary_part.done" => {
                let event = decode!(ReasoningPartEvent);
                ResponsesEvent::ReasoningSummaryPartDone {
                    item_id: event.item_id,
                    output_index: event.output_index,
                    summary_index: event.summary_index,
                    part: event.part,
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
            "response.failed" => {
                let event = decode!(SnapshotEvent);
                ResponsesEvent::Failed {
                    response: event.response,
                    sequence_number: event.sequence_number,
                }
            }
            "response.cancelled" => {
                let event = decode!(SnapshotEvent);
                ResponsesEvent::Cancelled {
                    response: event.response,
                    sequence_number: event.sequence_number,
                }
            }
            _ => {
                let sequence_number = value
                    .get("sequence_number")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| format!("unknown Responses event `{kind}` has no sequence"))?;
                ResponsesEvent::Unknown {
                    kind: kind.to_string(),
                    data: value,
                    sequence_number,
                }
            }
        })
    }
}
