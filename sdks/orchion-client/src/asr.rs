use crate::client::{decode_json, decode_text};
use crate::{Client, ClientError, ServerErrorObject};
use futures_util::{SinkExt, StreamExt};
use orchion_protocol::{
    AsrStreamControlMessage, AsrStreamEvent, AsrStreamStartMessage,
    CaptionEndpointingValidationError,
};
pub use orchion_protocol::{
    AsrStreamInputAudioFormat as StreamingInputAudioFormat, AsrStreamMode as StreamingMode,
    CaptionEndpointing,
};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;
use std::future::Future;
use std::path::Path;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Error as TungsteniteError;
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
    /// Returns [`ClientError`] when the request is invalid, the WebSocket cannot be opened, the
    /// initial start message cannot be sent, or a configured WebSocket operation times out.
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

        if request.endpointing.is_some() && request.mode != Some(StreamingMode::Caption) {
            return Err(ClientError::build_request(
                "endpointing is only supported when streaming mode is caption",
            ));
        }

        if request.mode == Some(StreamingMode::Caption)
            && request.input_audio_format == StreamingInputAudioFormat::PcmS16Le
            && request.sample_rate != Some(16_000)
        {
            return Err(ClientError::build_request(
                "caption mode requires pcm_s16le input audio at 16000 Hz",
            ));
        }

        if let Some(endpointing) = request.endpointing {
            validate_caption_endpointing(endpointing)?;
        }

        let url = self
            .client
            .websocket_url("/v1/audio/transcriptions/stream")?;
        let headers = self.client.websocket_headers()?;
        let timeout = self.client.config().timeout;
        let mut websocket_request = url
            .as_str()
            .into_client_request()
            .map_err(websocket_error)?;
        websocket_request.headers_mut().extend(headers);

        let (mut stream, _) = websocket_with_timeout(
            timeout,
            "websocket connect/handshake",
            Box::pin(async move {
                tokio_tungstenite::connect_async(websocket_request)
                    .await
                    .map_err(websocket_connect_error)
            }),
        )
        .await?;
        let start_message = request.to_protocol_message().to_text().map_err(|error| {
            ClientError::decode(format!("invalid streaming start request: {error}"))
        })?;
        websocket_with_timeout(timeout, "websocket start message", async {
            stream
                .send(Message::Text(start_message.into()))
                .await
                .map_err(websocket_error)
        })
        .await?;

        let mut io = StreamingSessionIo::new(stream, timeout);
        io.expect_ready().await?;

        Ok(StreamingSession { io })
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
    pub duration: f64,
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
#[derive(Clone)]
pub struct StreamingStartRequest {
    message_type: &'static str,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub api_key: Option<String>,
    pub mode: Option<StreamingMode>,
    pub endpointing: Option<CaptionEndpointing>,
    pub response_format: &'static str,
    pub input_audio_format: StreamingInputAudioFormat,
    pub sample_rate: Option<u32>,
    pub chunk_size_sec: Option<f32>,
    pub unfixed_chunk_num: Option<usize>,
    pub unfixed_token_num: Option<usize>,
    pub max_new_tokens_streaming: Option<usize>,
    pub max_new_tokens_final: Option<usize>,
}

impl Serialize for StreamingStartRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_protocol_message().serialize(serializer)
    }
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
            .field("mode", &self.mode)
            .field("endpointing", &self.endpointing)
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
            mode: None,
            endpointing: None,
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

    /// Enables caption mode with server-side endpointing defaults.
    #[must_use]
    pub const fn with_caption_mode(mut self) -> Self {
        self.mode = Some(StreamingMode::Caption);
        self.endpointing = None;
        self
    }

    /// Enables caption mode and sends explicit endpointing options.
    #[must_use]
    pub const fn with_caption_endpointing(mut self, endpointing: CaptionEndpointing) -> Self {
        self.mode = Some(StreamingMode::Caption);
        self.endpointing = Some(endpointing);
        self
    }

    fn to_protocol_message(&self) -> AsrStreamStartMessage {
        AsrStreamStartMessage {
            message_type: Some(self.message_type.to_string()),
            mode: self.mode.map(|mode| mode.as_str().to_string()),
            model: Some(self.model.clone()),
            language: self.language.clone(),
            prompt: self.prompt.clone(),
            api_key: self.api_key.clone(),
            response_format: Some(self.response_format.to_string()),
            input_audio_format: Some(self.input_audio_format.as_str().to_string()),
            endpointing: self.endpointing.map(Into::into),
            sample_rate: self.sample_rate,
            chunk_size_sec: self.chunk_size_sec,
            unfixed_chunk_num: self.unfixed_chunk_num,
            unfixed_token_num: self.unfixed_token_num,
            max_new_tokens_streaming: self.max_new_tokens_streaming,
            max_new_tokens_final: self.max_new_tokens_final,
        }
    }
}

fn validate_caption_endpointing(endpointing: CaptionEndpointing) -> Result<(), ClientError> {
    endpointing
        .validate()
        .map_err(|error: CaptionEndpointingValidationError| {
            ClientError::build_request(error.to_string())
        })
}

/// Event received from the ASR streaming WebSocket endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingEvent {
    Ready,
    Partial {
        text: String,
    },
    CaptionPartial {
        segment_id: u64,
        text: String,
    },
    Final {
        text: String,
    },
    SegmentFinal {
        segment_id: u64,
        text: String,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
    },
    Completed,
    Error {
        error: ServerErrorObject,
    },
}

impl StreamingEvent {
    /// Decodes a streaming event from a WebSocket text message.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the event JSON is invalid or contains an unsupported event.
    pub fn from_text(text: &str) -> Result<Self, ClientError> {
        AsrStreamEvent::from_text(text)
            .map(Into::into)
            .map_err(|error| ClientError::decode(format!("invalid streaming event: {error}")))
    }
}

impl From<AsrStreamEvent> for StreamingEvent {
    fn from(event: AsrStreamEvent) -> Self {
        match event {
            AsrStreamEvent::Ready => Self::Ready,
            AsrStreamEvent::Partial {
                text,
                segment_id: Some(segment_id),
            } => Self::CaptionPartial { segment_id, text },
            AsrStreamEvent::Partial {
                text,
                segment_id: None,
            } => Self::Partial { text },
            AsrStreamEvent::Final { text } => Self::Final { text },
            AsrStreamEvent::SegmentFinal {
                segment_id,
                text,
                start_ms,
                end_ms,
            } => Self::SegmentFinal {
                segment_id,
                text,
                start_ms,
                end_ms,
            },
            AsrStreamEvent::Completed => Self::Completed,
            AsrStreamEvent::Error { error } => Self::Error { error },
        }
    }
}

/// Active ASR streaming WebSocket session.
pub struct StreamingSession {
    io: StreamingSessionIo<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingSessionState {
    Active,
    Finishing,
    Completed,
    Terminal { failed_operation: &'static str },
}

struct StreamingSessionIo<S> {
    stream: S,
    timeout: Duration,
    state: StreamingSessionState,
}

impl StreamingSession {
    /// Sends audio bytes to the streaming session.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket send fails or the operation times out. Either
    /// failure makes the session terminal; later send, finish, and receive operations return
    /// [`ClientError::StreamingSessionTerminated`].
    pub async fn send_audio(&mut self, audio: impl Into<Vec<u8>>) -> Result<(), ClientError> {
        self.io.send_audio(audio.into()).await
    }

    /// Signals that no more audio will be sent.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the WebSocket send fails or the operation times out. Either
    /// failure makes the session terminal; later send, finish, and receive operations return
    /// [`ClientError::StreamingSessionTerminated`].
    pub async fn finish(&mut self) -> Result<(), ClientError> {
        self.io.finish().await
    }

    /// Receives the next streaming event.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the session is terminal, the operation times out, the
    /// WebSocket receives an unsupported message, or the event cannot be decoded.
    pub async fn next_event(&mut self) -> Result<Option<StreamingEvent>, ClientError> {
        self.io.next_event().await
    }

    /// Closes the streaming WebSocket session.
    ///
    /// # Errors
    ///
    /// This still attempts to close a session made terminal by a failed or timed-out send.
    /// Returns [`ClientError`] when the WebSocket close fails or the operation times out.
    pub async fn close(mut self) -> Result<(), ClientError> {
        self.io.close().await
    }
}

impl<S> StreamingSessionIo<S>
where
    S: futures_util::Sink<Message, Error = TungsteniteError>
        + futures_util::Stream<Item = Result<Message, TungsteniteError>>
        + Unpin,
{
    const fn new(stream: S, timeout: Duration) -> Self {
        Self {
            stream,
            timeout,
            state: StreamingSessionState::Active,
        }
    }

    async fn send_audio(&mut self, audio: Vec<u8>) -> Result<(), ClientError> {
        self.send_message(
            "send_audio",
            "websocket send_audio",
            Message::Binary(audio.into()),
            StreamingSessionState::Active,
        )
        .await
    }

    async fn finish(&mut self) -> Result<(), ClientError> {
        self.ensure_writable()?;
        let end_message = AsrStreamControlMessage::end().to_text().map_err(|error| {
            ClientError::decode(format!("invalid streaming end message: {error}"))
        })?;
        self.send_message(
            "finish",
            "websocket finish",
            Message::Text(end_message.into()),
            StreamingSessionState::Finishing,
        )
        .await
    }

    async fn send_message(
        &mut self,
        failed_operation: &'static str,
        timeout_operation: &'static str,
        message: Message,
        success_state: StreamingSessionState,
    ) -> Result<(), ClientError> {
        self.ensure_writable()?;
        self.state = StreamingSessionState::Terminal { failed_operation };
        let result = websocket_with_timeout(self.timeout, timeout_operation, async {
            self.stream.send(message).await.map_err(websocket_error)
        })
        .await;

        if result.is_ok() {
            self.state = success_state;
        }

        result
    }

    async fn next_event(&mut self) -> Result<Option<StreamingEvent>, ClientError> {
        if self.state == StreamingSessionState::Completed {
            return Ok(None);
        }
        self.ensure_readable()?;
        let Some(event) = self.receive_event("websocket next_event").await? else {
            self.state = StreamingSessionState::Terminal {
                failed_operation: "next_event",
            };
            return Err(ClientError::UnexpectedEof {
                stream: "asr_streaming",
            });
        };
        Ok(Some(event))
    }

    async fn expect_ready(&mut self) -> Result<(), ClientError> {
        match self.receive_event("websocket ready event").await? {
            Some(StreamingEvent::Ready) => Ok(()),
            Some(StreamingEvent::Error { error }) => Err(ClientError::StreamingServer { error }),
            Some(event) => Err(ClientError::decode(format!(
                "expected ready as the first streaming event, received {event:?}"
            ))),
            None => Err(ClientError::WebSocket {
                message: "streaming WebSocket closed before the ready event".to_string(),
            }),
        }
    }

    async fn receive_event(
        &mut self,
        timeout_operation: &'static str,
    ) -> Result<Option<StreamingEvent>, ClientError> {
        let result = websocket_with_timeout(self.timeout, timeout_operation, async {
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
        })
        .await;
        if matches!(
            result,
            Ok(Some(
                StreamingEvent::Final { .. }
                    | StreamingEvent::Completed
                    | StreamingEvent::Error { .. }
            ))
        ) {
            self.state = StreamingSessionState::Completed;
        }
        result
    }

    async fn close(&mut self) -> Result<(), ClientError> {
        websocket_with_timeout(self.timeout, "websocket close", async {
            self.stream.close().await.map_err(websocket_error)
        })
        .await
    }

    const fn ensure_writable(&self) -> Result<(), ClientError> {
        match self.state {
            StreamingSessionState::Active => Ok(()),
            StreamingSessionState::Finishing | StreamingSessionState::Completed => {
                Err(ClientError::streaming_session_terminated("finish"))
            }
            StreamingSessionState::Terminal { failed_operation } => {
                Err(ClientError::streaming_session_terminated(failed_operation))
            }
        }
    }

    const fn ensure_readable(&self) -> Result<(), ClientError> {
        match self.state {
            StreamingSessionState::Active
            | StreamingSessionState::Finishing
            | StreamingSessionState::Completed => Ok(()),
            StreamingSessionState::Terminal { failed_operation } => {
                Err(ClientError::streaming_session_terminated(failed_operation))
            }
        }
    }
}

fn websocket_error(error: impl std::fmt::Display) -> ClientError {
    ClientError::WebSocket {
        message: error.to_string(),
    }
}

async fn websocket_with_timeout<T>(
    timeout: Duration,
    operation: &'static str,
    future: impl Future<Output = Result<T, ClientError>>,
) -> Result<T, ClientError> {
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| ClientError::timeout(operation, timeout))?
}

fn websocket_connect_error(error: TungsteniteError) -> ClientError {
    let TungsteniteError::Http(response) = error else {
        return websocket_error(error);
    };
    let status = response.status();
    let body = response
        .body()
        .as_deref()
        .map(String::from_utf8_lossy)
        .unwrap_or_default();

    if let Ok(server_error) = serde_json::from_str::<crate::ServerErrorBody>(&body) {
        let message = server_error.error.message.clone();
        return ClientError::Http {
            status,
            message,
            error: Some(server_error.error),
        };
    }

    let message = if body.trim().is_empty() {
        status
            .canonical_reason()
            .unwrap_or("WebSocket handshake failed")
            .to_string()
    } else {
        body.trim().to_string()
    };
    ClientError::Http {
        status,
        message,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CaptionEndpointing, StreamingEvent, StreamingInputAudioFormat, StreamingSessionIo,
        StreamingSessionState, StreamingStartRequest,
    };
    use crate::{Client, ClientError, ServerErrorObject};
    use futures_util::{Sink, Stream};
    use orchion_protocol::{AsrStreamEvent, AsrStreamStartMessage};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::Error as TungsteniteError;
    use tokio_tungstenite::tungstenite::protocol::Message;

    const CONTROLLED_IO_TIMEOUT: Duration = Duration::from_millis(10);

    enum ControlledBehavior {
        Complete,
        StallBinarySend,
        StallTextSend,
        FailBinarySend,
        FailTextSend,
        StallClose,
    }

    struct ControlledWebSocket {
        behavior: ControlledBehavior,
        send_stalled: bool,
        close_polled: Arc<AtomicBool>,
    }

    impl ControlledWebSocket {
        fn new(behavior: ControlledBehavior) -> (Self, Arc<AtomicBool>) {
            let close_polled = Arc::new(AtomicBool::new(false));
            (
                Self {
                    behavior,
                    send_stalled: false,
                    close_polled: Arc::clone(&close_polled),
                },
                close_polled,
            )
        }
    }

    impl Sink<Message> for ControlledWebSocket {
        type Error = TungsteniteError;

        fn poll_ready(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(mut self: Pin<&mut Self>, message: Message) -> Result<(), Self::Error> {
            if matches!(
                (&self.behavior, &message),
                (ControlledBehavior::FailBinarySend, Message::Binary(_))
                    | (ControlledBehavior::FailTextSend, Message::Text(_))
            ) {
                return Err(TungsteniteError::ConnectionClosed);
            }

            self.send_stalled = matches!(
                (&self.behavior, message),
                (ControlledBehavior::StallBinarySend, Message::Binary(_))
                    | (ControlledBehavior::StallTextSend, Message::Text(_))
            );
            Ok(())
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            if self.send_stalled {
                Poll::Pending
            } else {
                Poll::Ready(Ok(()))
            }
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            self.close_polled.store(true, Ordering::Relaxed);
            if matches!(self.behavior, ControlledBehavior::StallClose) {
                Poll::Pending
            } else {
                Poll::Ready(Ok(()))
            }
        }
    }

    impl Stream for ControlledWebSocket {
        type Item = Result<Message, TungsteniteError>;

        fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    fn assert_controlled_timeout(error: &ClientError, operation: &'static str) {
        assert!(matches!(
            error,
            ClientError::Timeout {
                operation: actual_operation,
                timeout: CONTROLLED_IO_TIMEOUT,
            } if *actual_operation == operation
        ));
    }

    fn assert_terminal(error: &ClientError, operation: &'static str) {
        assert!(matches!(
            error,
            ClientError::StreamingSessionTerminated {
                operation: actual_operation,
            } if *actual_operation == operation
        ));
    }

    #[tokio::test]
    async fn send_audio_timeout_makes_session_terminal_but_close_is_still_attempted() {
        let (stream, close_polled) = ControlledWebSocket::new(ControlledBehavior::StallBinarySend);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        let error = session.send_audio(vec![1, 2, 3]).await.unwrap_err();
        assert_controlled_timeout(&error, "websocket send_audio");

        let send_error = session.send_audio(vec![4]).await.unwrap_err();
        let finish_error = session.finish().await.unwrap_err();
        let event_error = session.next_event().await.unwrap_err();
        assert_terminal(&send_error, "send_audio");
        assert_terminal(&finish_error, "send_audio");
        assert_terminal(&event_error, "send_audio");

        session.close().await.unwrap();
        assert!(close_polled.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn finish_timeout_makes_session_terminal_with_a_stable_error() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::StallTextSend);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        let error = session.finish().await.unwrap_err();
        assert_controlled_timeout(&error, "websocket finish");

        let send_error = session.send_audio(vec![1]).await.unwrap_err();
        let finish_error = session.finish().await.unwrap_err();
        let event_error = session.next_event().await.unwrap_err();
        assert_terminal(&send_error, "finish");
        assert_terminal(&finish_error, "finish");
        assert_terminal(&event_error, "finish");
    }

    #[tokio::test]
    async fn send_error_makes_session_terminal() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::FailBinarySend);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        let error = session.send_audio(vec![1]).await.unwrap_err();
        assert!(matches!(error, ClientError::WebSocket { .. }));
        assert_terminal(&session.next_event().await.unwrap_err(), "send_audio");
    }

    #[tokio::test]
    async fn finish_send_error_makes_session_terminal() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::FailTextSend);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        let error = session.finish().await.unwrap_err();
        assert!(matches!(error, ClientError::WebSocket { .. }));
        assert_terminal(&session.next_event().await.unwrap_err(), "finish");
    }

    #[tokio::test]
    async fn close_uses_its_own_timeout() {
        let (stream, close_polled) = ControlledWebSocket::new(ControlledBehavior::StallClose);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        let error = session.close().await.unwrap_err();

        assert_controlled_timeout(&error, "websocket close");
        assert!(close_polled.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn controlled_stream_can_complete_sends() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::Complete);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        session.send_audio(vec![1]).await.unwrap();
        session.finish().await.unwrap();
        assert_terminal(&session.send_audio(vec![2]).await.unwrap_err(), "finish");
        assert_terminal(&session.finish().await.unwrap_err(), "finish");
    }

    #[tokio::test]
    async fn caller_cancelling_send_makes_session_terminal() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::StallBinarySend);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);

        tokio::time::timeout(Duration::from_millis(1), session.send_audio(vec![1]))
            .await
            .unwrap_err();

        assert_terminal(
            &session.send_audio(vec![2]).await.unwrap_err(),
            "send_audio",
        );
        assert_terminal(&session.next_event().await.unwrap_err(), "send_audio");
    }

    #[tokio::test]
    async fn completed_session_returns_end_of_stream_on_later_reads() {
        let (stream, _) = ControlledWebSocket::new(ControlledBehavior::Complete);
        let mut session = StreamingSessionIo::new(stream, CONTROLLED_IO_TIMEOUT);
        session.state = StreamingSessionState::Completed;

        assert_eq!(session.next_event().await.unwrap(), None);
        assert_terminal(&session.send_audio(vec![1]).await.unwrap_err(), "finish");
    }

    #[test]
    fn stream_start_serializes_server_protocol_fields() {
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000)
        .with_language("zh")
        .with_prompt("context")
        .with_chunk_size_sec(2.0);

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["type"], "start");
        assert_eq!(value["model"], "alibaba/qwen3-asr-flash");
        assert_eq!(value["input_audio_format"], "pcm_s16le");
        assert_eq!(value["sample_rate"], 16000);
        assert_eq!(value["language"], "zh");
        assert_eq!(value["prompt"], "context");
        assert_eq!(value["chunk_size_sec"], 2.0);
    }

    #[test]
    fn public_stream_start_wrapper_round_trips_through_shared_wire_dto() {
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000)
        .with_caption_endpointing(CaptionEndpointing::default());

        let text = serde_json::to_string(&request).unwrap();
        let wire = AsrStreamStartMessage::from_text(&text).unwrap();

        assert_eq!(wire.message_type.as_deref(), Some("start"));
        assert_eq!(wire.mode.as_deref(), Some("caption"));
        assert_eq!(wire.input_audio_format.as_deref(), Some("pcm_s16le"));
        assert_eq!(wire.sample_rate, Some(16_000));
        assert_eq!(
            wire.endpointing.unwrap().min_silence_ms,
            Some(CaptionEndpointing::default().min_silence_ms)
        );
    }

    #[test]
    fn stream_start_serializes_caption_mode_with_default_endpointing() {
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000)
        .with_caption_mode();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["mode"], "caption");
        assert!(value.get("endpointing").is_none());
    }

    #[test]
    fn caption_endpointing_default_matches_server_tuned_defaults() {
        assert_eq!(
            CaptionEndpointing::default(),
            CaptionEndpointing {
                min_speech_ms: 300,
                min_silence_ms: 500,
                speech_padding_ms: 200,
            }
        );
    }

    #[test]
    fn stream_start_serializes_caption_endpointing_overrides() {
        let endpointing = CaptionEndpointing {
            min_speech_ms: 250,
            min_silence_ms: 650,
            speech_padding_ms: 120,
        };
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000)
        .with_caption_endpointing(endpointing);

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["mode"], "caption");
        assert_eq!(value["endpointing"]["min_speech_ms"], 250);
        assert_eq!(value["endpointing"]["min_silence_ms"], 650);
        assert!(value["endpointing"].get("max_segment_ms").is_none());
        assert_eq!(value["endpointing"]["speech_padding_ms"], 120);
    }

    #[test]
    fn stream_start_caption_mode_after_endpointing_omits_endpointing() {
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000)
        .with_caption_endpointing(CaptionEndpointing::default())
        .with_caption_mode();

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["mode"], "caption");
        assert!(value.get("endpointing").is_none());
    }

    #[tokio::test]
    async fn stream_start_rejects_endpointing_without_caption_mode() {
        let client = Client::new("http://localhost:8080").unwrap();
        let mut request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(16_000);
        request.endpointing = Some(CaptionEndpointing::default());

        let Err(error) = client.asr().start_streaming(request).await else {
            panic!("streaming request unexpectedly succeeded");
        };

        assert!(matches!(
            error,
            ClientError::BuildRequest { message }
                if message == "endpointing is only supported when streaming mode is caption"
        ));
    }

    #[tokio::test]
    async fn stream_start_rejects_caption_pcm_s16le_without_16000_hz() {
        let client = Client::new("http://localhost:8080").unwrap();
        let request = StreamingStartRequest::new(
            "alibaba/qwen3-asr-flash",
            StreamingInputAudioFormat::PcmS16Le,
        )
        .with_sample_rate(44_100)
        .with_caption_mode();

        let Err(error) = client.asr().start_streaming(request).await else {
            panic!("streaming request unexpectedly succeeded");
        };

        assert!(matches!(
            error,
            ClientError::BuildRequest { message }
                if message == "caption mode requires pcm_s16le input audio at 16000 Hz"
        ));
    }

    #[tokio::test]
    async fn stream_start_rejects_invalid_caption_endpointing_before_network() {
        let cases = [
            (
                CaptionEndpointing {
                    min_speech_ms: 0,
                    min_silence_ms: 500,
                    speech_padding_ms: 200,
                },
                "endpointing.min_speech_ms must be greater than zero",
            ),
            (
                CaptionEndpointing {
                    min_speech_ms: 300,
                    min_silence_ms: 0,
                    speech_padding_ms: 200,
                },
                "endpointing.min_silence_ms must be greater than zero",
            ),
            (
                CaptionEndpointing {
                    min_speech_ms: 1,
                    min_silence_ms: 500,
                    speech_padding_ms: u32::MAX,
                },
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms is too large",
            ),
            (
                CaptionEndpointing {
                    min_speech_ms: 300,
                    min_silence_ms: 500,
                    speech_padding_ms: 60_000,
                },
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms must not exceed 60000",
            ),
            (
                CaptionEndpointing {
                    min_speech_ms: 21,
                    min_silence_ms: 500,
                    speech_padding_ms: 0,
                },
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms must hold one rounded VAD speech window",
            ),
        ];

        for (endpointing, expected_message) in cases {
            let client = Client::new("http://localhost:8080").unwrap();
            let request = StreamingStartRequest::new(
                "alibaba/qwen3-asr-flash",
                StreamingInputAudioFormat::PcmS16Le,
            )
            .with_sample_rate(16_000)
            .with_caption_endpointing(endpointing);

            let Err(error) = client.asr().start_streaming(request).await else {
                panic!("streaming request unexpectedly succeeded");
            };

            assert!(matches!(
                error,
                ClientError::BuildRequest { message } if message == expected_message
            ));
        }
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

    #[test]
    fn stream_event_decodes_caption_events() {
        let partial = StreamingEvent::from_text(
            r#"{"type":"partial","segment_id":7,"text":"caption draft"}"#,
        )
        .unwrap();
        let final_event = StreamingEvent::from_text(
            r#"{
                "type":"segment_final",
                "segment_id":7,
                "text":"caption final",
                "start_ms":1000,
                "end_ms":2300
            }"#,
        )
        .unwrap();
        let completed = StreamingEvent::from_text(r#"{"type":"completed"}"#).unwrap();

        assert_eq!(
            partial,
            StreamingEvent::CaptionPartial {
                segment_id: 7,
                text: "caption draft".to_string(),
            }
        );
        assert_eq!(
            final_event,
            StreamingEvent::SegmentFinal {
                segment_id: 7,
                text: "caption final".to_string(),
                start_ms: Some(1000),
                end_ms: Some(2300),
            }
        );
        assert_eq!(completed, StreamingEvent::Completed);
    }

    #[test]
    fn shared_wire_events_preserve_public_sdk_variants() {
        let legacy = AsrStreamEvent::Partial {
            text: "legacy".to_string(),
            segment_id: None,
        };
        let caption = AsrStreamEvent::Partial {
            text: "caption".to_string(),
            segment_id: Some(4),
        };

        assert_eq!(
            StreamingEvent::from_text(&legacy.to_text().unwrap()).unwrap(),
            StreamingEvent::Partial {
                text: "legacy".to_string(),
            }
        );
        assert_eq!(
            StreamingEvent::from_text(&caption.to_text().unwrap()).unwrap(),
            StreamingEvent::CaptionPartial {
                segment_id: 4,
                text: "caption".to_string(),
            }
        );
    }

    #[test]
    fn legacy_partial_event_still_decodes_to_legacy_variant() {
        let partial = StreamingEvent::from_text(r#"{"type":"partial","text":"hel"}"#).unwrap();

        assert_eq!(
            partial,
            StreamingEvent::Partial {
                text: "hel".to_string(),
            }
        );
    }
}
