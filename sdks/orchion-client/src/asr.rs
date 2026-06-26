use crate::client::{decode_json, decode_text};
use crate::{Client, ClientError, ServerErrorObject};
use futures_util::{SinkExt, StreamExt};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message;

/// Client for the ASR API.
pub struct AsrClient<'a> {
    client: &'a Client,
}

impl<'a> AsrClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Transcribes audio using the multipart transcription endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or the response cannot
    /// be decoded.
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, ClientError> {
        let response_format = request.response_format;
        let response = self
            .client
            .post("/v1/audio/transcriptions")?
            .multipart(request.into_form()?)
            .send()
            .await?;

        match response_format {
            TranscriptionFormat::Json => {
                let response: TranscriptionJson = decode_json(response).await?;
                Ok(TranscriptionResponse::Json {
                    text: response.text,
                })
            }
            TranscriptionFormat::VerboseJson => {
                let response = decode_json(response).await?;
                Ok(TranscriptionResponse::VerboseJson(response))
            }
            TranscriptionFormat::Text | TranscriptionFormat::Srt => {
                let response = decode_text(response).await?;
                Ok(TranscriptionResponse::Text(response))
            }
        }
    }

    /// Starts a streaming ASR WebSocket session.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, the WebSocket cannot be opened, or the
    /// initial start message cannot be sent.
    pub async fn start_streaming(
        &self,
        request: StreamingStartRequest,
    ) -> Result<StreamingSession, ClientError> {
        if request.model.is_empty() {
            return Err(ClientError::build_request("model must not be empty"));
        }

        if request.input_audio_format == StreamingInputAudioFormat::PcmS16Le
            && request.sample_rate.is_none()
        {
            return Err(ClientError::build_request(
                "sample_rate is required for pcm_s16le input audio",
            ));
        }

        let url = self
            .client
            .websocket_url("/v1/audio/transcriptions/stream")?;
        let headers = self.client.websocket_headers()?;
        let mut websocket_request = url
            .as_str()
            .into_client_request()
            .map_err(websocket_error)?;
        websocket_request.headers_mut().extend(headers);

        let (mut stream, _) = tokio_tungstenite::connect_async(websocket_request)
            .await
            .map_err(websocket_error)?;
        let start_message = serde_json::to_string(&request).map_err(|error| {
            ClientError::decode(format!("invalid streaming start request: {error}"))
        })?;
        stream
            .send(Message::Text(start_message.into()))
            .await
            .map_err(websocket_error)?;

        Ok(StreamingSession { stream })
    }
}

/// Multipart transcription request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionRequest {
    pub model: String,
    pub filename: String,
    pub file_bytes: Vec<u8>,
    pub language: Option<String>,
    pub response_format: TranscriptionFormat,
    pub timestamp_granularities: Vec<TimestampGranularity>,
}

impl TranscriptionRequest {
    /// Creates a transcription request.
    #[must_use]
    pub fn new(model: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: filename.into(),
            file_bytes: Vec::new(),
            language: None,
            response_format: TranscriptionFormat::Json,
            timestamp_granularities: Vec::new(),
        }
    }

    /// Sets audio bytes for the multipart file field.
    #[must_use]
    pub fn with_file_bytes(mut self, file_bytes: Vec<u8>) -> Self {
        self.file_bytes = file_bytes;
        self
    }

    /// Reads audio bytes from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the file cannot be read.
    pub async fn with_file_path(mut self, path: impl AsRef<Path>) -> Result<Self, ClientError> {
        self.file_bytes = tokio::fs::read(path.as_ref()).await.map_err(|error| {
            ClientError::build_request(format!("failed to read audio file: {error}"))
        })?;
        Ok(self)
    }

    /// Sets the optional transcription language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the response format.
    #[must_use]
    pub fn with_response_format(mut self, response_format: TranscriptionFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// Adds a timestamp granularity.
    #[must_use]
    pub fn with_timestamp_granularity(
        mut self,
        timestamp_granularity: TimestampGranularity,
    ) -> Self {
        self.timestamp_granularities.push(timestamp_granularity);
        self
    }

    fn into_form(self) -> Result<Form, ClientError> {
        if self.model.is_empty() {
            return Err(ClientError::build_request("model must not be empty"));
        }

        if self.filename.is_empty() {
            return Err(ClientError::build_request("filename must not be empty"));
        }

        if self.file_bytes.is_empty() {
            return Err(ClientError::build_request("file bytes must not be empty"));
        }

        let file = Part::bytes(self.file_bytes).file_name(self.filename);
        let mut form = Form::new()
            .text("model", self.model)
            .text("response_format", self.response_format.as_str())
            .part("file", file);

        if let Some(language) = self.language {
            form = form.text("language", language);
        }

        for timestamp_granularity in self.timestamp_granularities {
            form = form.text("timestamp_granularities[]", timestamp_granularity.as_str());
        }

        Ok(form)
    }
}

/// ASR transcription response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
}

impl TranscriptionFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::VerboseJson => "verbose_json",
            Self::Srt => "srt",
        }
    }
}

/// Timestamp granularities supported by the transcription endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampGranularity {
    Segment,
}

impl TimestampGranularity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Segment => "segment",
        }
    }
}

/// Transcription response.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionResponse {
    Json { text: String },
    VerboseJson(VerboseTranscriptionResponse),
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct TranscriptionJson {
    text: String,
}

/// Verbose JSON transcription response.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct VerboseTranscriptionResponse {
    pub text: String,
    pub language: String,
    pub raw_output: String,
    pub segments: Option<Vec<AsrSegment>>,
}

/// Segment returned in a verbose transcription response.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AsrSegment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Start message sent to the ASR streaming WebSocket endpoint.
#[derive(Clone, Serialize)]
pub struct StreamingStartRequest {
    #[serde(rename = "type")]
    message_type: &'static str,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub response_format: &'static str,
    pub input_audio_format: StreamingInputAudioFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size_sec: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unfixed_chunk_num: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unfixed_token_num: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens_streaming: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens_final: Option<usize>,
}

impl fmt::Debug for StreamingStartRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingStartRequest")
            .field("message_type", &self.message_type)
            .field("model", &self.model)
            .field("language", &self.language)
            .field("prompt", &self.prompt)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("response_format", &self.response_format)
            .field("input_audio_format", &self.input_audio_format)
            .field("sample_rate", &self.sample_rate)
            .field("chunk_size_sec", &self.chunk_size_sec)
            .field("unfixed_chunk_num", &self.unfixed_chunk_num)
            .field("unfixed_token_num", &self.unfixed_token_num)
            .field("max_new_tokens_streaming", &self.max_new_tokens_streaming)
            .field("max_new_tokens_final", &self.max_new_tokens_final)
            .finish()
    }
}

impl StreamingStartRequest {
    /// Creates a streaming start request.
    #[must_use]
    pub fn new(model: impl Into<String>, input_audio_format: StreamingInputAudioFormat) -> Self {
        Self {
            message_type: "start",
            model: model.into(),
            language: None,
            prompt: None,
            api_key: None,
            response_format: "json",
            input_audio_format,
            sample_rate: None,
            chunk_size_sec: None,
            unfixed_chunk_num: None,
            unfixed_token_num: None,
            max_new_tokens_streaming: None,
            max_new_tokens_final: None,
        }
    }

    /// Sets the optional transcription language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the optional transcription prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the optional API key field in the start message.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the input audio sample rate.
    #[must_use]
    pub const fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Sets the requested streaming chunk size in seconds.
    #[must_use]
    pub const fn with_chunk_size_sec(mut self, chunk_size_sec: f32) -> Self {
        self.chunk_size_sec = Some(chunk_size_sec);
        self
    }
}

/// Input audio format used by the ASR streaming endpoint.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StreamingInputAudioFormat {
    Auto,
    #[serde(rename = "pcm_s16le")]
    PcmS16Le,
    WebmOpus,
    Mp3,
    Wav,
    M4a,
    Aac,
    Flac,
    Ogg,
}

/// Event received from the ASR streaming WebSocket endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingEvent {
    Ready,
    Partial { text: String },
    Final { text: String },
    Error { error: ServerErrorObject },
}

impl StreamingEvent {
    /// Decodes a streaming event from a WebSocket text message.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the event JSON is invalid or contains an unsupported event.
    pub fn from_text(text: &str) -> Result<Self, ClientError> {
        let event: StreamingEventBody = serde_json::from_str(text)
            .map_err(|error| ClientError::decode(format!("invalid streaming event: {error}")))?;

        match event.event_type.as_str() {
            "ready" => Ok(Self::Ready),
            "partial" => event
                .text
                .map(|text| Self::Partial { text })
                .ok_or_else(|| ClientError::decode("partial streaming event missing text")),
            "final" => event
                .text
                .map(|text| Self::Final { text })
                .ok_or_else(|| ClientError::decode("final streaming event missing text")),
            "error" => event
                .error
                .map(|error| Self::Error { error })
                .ok_or_else(|| ClientError::decode("streaming error event missing error object")),
            event_type => Err(ClientError::decode(format!(
                "unsupported streaming event type `{event_type}`"
            ))),
        }
    }
}

#[derive(Deserialize)]
struct StreamingEventBody {
    #[serde(rename = "type")]
    event_type: String,
    text: Option<String>,
    error: Option<ServerErrorObject>,
}

/// Active ASR streaming WebSocket session.
pub struct StreamingSession {
    stream: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
}

impl StreamingSession {
    /// Sends audio bytes to the streaming session.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket send fails.
    pub async fn send_audio(&mut self, audio: impl Into<Vec<u8>>) -> Result<(), ClientError> {
        self.stream
            .send(Message::Binary(audio.into().into()))
            .await
            .map_err(websocket_error)
    }

    /// Signals that no more audio will be sent.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket send fails.
    pub async fn finish(&mut self) -> Result<(), ClientError> {
        self.stream
            .send(Message::Text(r#"{"type":"end"}"#.to_string().into()))
            .await
            .map_err(websocket_error)
    }

    /// Receives the next streaming event.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket receives an unsupported message or the event
    /// cannot be decoded.
    pub async fn next_event(&mut self) -> Result<Option<StreamingEvent>, ClientError> {
        while let Some(message) = self.stream.next().await {
            match message.map_err(websocket_error)? {
                Message::Text(text) => return StreamingEvent::from_text(&text).map(Some),
                Message::Close(_) => return Ok(None),
                Message::Ping(_) | Message::Pong(_) => {}
                Message::Binary(_) | Message::Frame(_) => {
                    return Err(ClientError::decode(
                        "unsupported binary streaming WebSocket message",
                    ));
                }
            }
        }

        Ok(None)
    }

    /// Closes the streaming WebSocket session.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket close fails.
    pub async fn close(mut self) -> Result<(), ClientError> {
        self.stream.close(None).await.map_err(websocket_error)
    }
}

fn websocket_error(error: impl std::fmt::Display) -> ClientError {
    ClientError::WebSocket {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamingEvent, StreamingInputAudioFormat, StreamingStartRequest};
    use crate::ServerErrorObject;

    #[test]
    fn stream_start_serializes_server_protocol_fields() {
        let request =
            StreamingStartRequest::new("Qwen/Qwen3-ASR-Flash", StreamingInputAudioFormat::PcmS16Le)
                .with_sample_rate(16_000)
                .with_language("zh")
                .with_prompt("context")
                .with_chunk_size_sec(2.0);

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["type"], "start");
        assert_eq!(value["model"], "Qwen/Qwen3-ASR-Flash");
        assert_eq!(value["input_audio_format"], "pcm_s16le");
        assert_eq!(value["sample_rate"], 16000);
        assert_eq!(value["language"], "zh");
        assert_eq!(value["prompt"], "context");
        assert_eq!(value["chunk_size_sec"], 2.0);
    }

    #[test]
    fn stream_event_decodes_ready_partial_final_and_error() {
        let ready = StreamingEvent::from_text(r#"{"type":"ready"}"#).unwrap();
        let partial = StreamingEvent::from_text(r#"{"type":"partial","text":"hel"}"#).unwrap();
        let final_event = StreamingEvent::from_text(r#"{"type":"final","text":"hello"}"#).unwrap();
        let error = StreamingEvent::from_text(
            r#"{
                "type":"error",
                "error":{
                    "message":"bad",
                    "type":"invalid_request_error",
                    "param":"model",
                    "code":"model_not_available"
                }
            }"#,
        )
        .unwrap();

        assert_eq!(ready, StreamingEvent::Ready);
        assert_eq!(
            partial,
            StreamingEvent::Partial {
                text: "hel".to_string()
            }
        );
        assert_eq!(
            final_event,
            StreamingEvent::Final {
                text: "hello".to_string()
            }
        );
        assert_eq!(
            error,
            StreamingEvent::Error {
                error: ServerErrorObject {
                    message: "bad".to_string(),
                    error_type: "invalid_request_error".to_string(),
                    param: Some("model".to_string()),
                    code: Some("model_not_available".to_string()),
                }
            }
        );
    }
}
