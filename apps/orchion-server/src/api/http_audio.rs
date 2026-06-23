use crate::api::http_shared::{
    authorize, is_multipart, parse_multipart_value, read_text_field, required_multipart_field,
    write_multipart_file_to_temp_file,
};
use crate::api::openai::{
    ApiError, ErrorObject, SpeechFormat, SpeechRequest, TranscriptionFormat, TranscriptionJson,
    TranscriptionVerboseJson, content_type_for,
};
use crate::api::srt::format_srt;
use crate::infrastructure::orchion::AppState;
use crate::settings::{parse_asr_model, parse_tts_model};
use axum::Json;
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use orchion::{
    AsrOptions, AsrStreamingOptions, AsrTimestampGranularity, AudioInputFormat, AudioOutputFormat,
    OrchionError, StreamingAudioDecoder, TtsAudio, decode_audio_bytes, encode_tts_audio,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

pub(super) async fn create_speech(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    if is_multipart(&headers) {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| {
                ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
            })?;
        return create_speech_multipart(state, multipart).await;
    }

    let Json(request) = Json::<SpeechRequest>::from_request(request, &state)
        .await
        .map_err(|error| {
            ApiError::invalid_request(error.to_string(), None, Some("invalid_json"))
        })?;
    if request.is_voice_clone() {
        return Err(ApiError::invalid_request(
            "voice clone requires multipart/form-data with a reference_audio file upload",
            Some("voice"),
            Some("unsupported_voice_input"),
        ));
    }
    create_speech_from_request(state, request).await
}

async fn create_speech_multipart(
    state: Arc<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut model = None;
    let mut input = None;
    let mut voice = None;
    let mut response_format = None;
    let mut speed = None;
    let mut language = None;
    let mut reference_audio = None;
    let mut reference_text = None;
    let mut seed = None;
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;
    let mut repetition_penalty = None;
    let mut max_length = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "reference_audio" => {
                reference_audio = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| {
                            ApiError::invalid_request(
                                error.to_string(),
                                Some("reference_audio"),
                                Some("invalid_file"),
                            )
                        })?
                        .to_vec(),
                );
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "input" => input = Some(read_text_field(field, "input").await?),
            "voice" => voice = Some(read_text_field(field, "voice").await?),
            "response_format" => {
                response_format = Some(SpeechFormat::try_from(
                    read_text_field(field, "response_format").await?.as_str(),
                )?);
            }
            "speed" => speed = Some(parse_multipart_value(field, "speed").await?),
            "language" => language = Some(read_text_field(field, "language").await?),
            "reference_text" => {
                reference_text = Some(read_text_field(field, "reference_text").await?);
            }
            "seed" => seed = Some(parse_multipart_value(field, "seed").await?),
            "temperature" => temperature = Some(parse_multipart_value(field, "temperature").await?),
            "top_k" => top_k = Some(parse_multipart_value(field, "top_k").await?),
            "top_p" => top_p = Some(parse_multipart_value(field, "top_p").await?),
            "repetition_penalty" => {
                repetition_penalty =
                    Some(parse_multipart_value(field, "repetition_penalty").await?);
            }
            "max_length" => max_length = Some(parse_multipart_value(field, "max_length").await?),
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let voice = required_multipart_field(voice, "voice")?;
    if voice.trim().to_ascii_lowercase() != "clone" {
        return Err(ApiError::invalid_request(
            "multipart speech is only supported for voice clone requests",
            Some("voice"),
            Some("unsupported_voice_input"),
        ));
    }
    let reference_audio = reference_audio.ok_or_else(|| {
        ApiError::invalid_request(
            "voice clone requires `reference_audio` file upload",
            Some("reference_audio"),
            Some("missing_required_parameter"),
        )
    })?;
    let model = required_multipart_field(model, "model")?;
    let input = required_multipart_field(input, "input")?;
    let request = SpeechRequest {
        model,
        input,
        voice,
        response_format,
        speed: speed.unwrap_or(1.0),
        language,
        reference_audio: Some(String::new()),
        reference_text,
        voice_prompt: None,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_length,
    };
    request.validate()?;
    parse_tts_model(&request.model).map_err(|_| ApiError::model_not_available(&request.model))?;
    let reference_file = transcode_reference_audio_to_wav_file(reference_audio).await?;
    let reference_path = reference_file
        .path()
        .to_str()
        .ok_or_else(|| ApiError::internal("reference audio path is not valid UTF-8"))?
        .to_string();
    let request = SpeechRequest {
        reference_audio: Some(reference_path),
        ..request
    };
    let response = create_speech_from_request(state, request).await;
    drop(reference_file);
    response
}

async fn transcode_reference_audio_to_wav_file(
    reference_audio: Vec<u8>,
) -> Result<NamedTempFile, ApiError> {
    let decoded = decode_audio_bytes(reference_audio)
        .await
        .map_err(reference_audio_error)?;
    let audio = TtsAudio::new(decoded.samples, decoded.sample_rate);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .map_err(reference_audio_error)?;
    let reference_file = TempFileBuilder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    tokio::fs::write(reference_file.path(), encoded.bytes)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(reference_file)
}

fn reference_audio_error(error: OrchionError) -> ApiError {
    match error {
        OrchionError::InvalidAudio { reason } => {
            ApiError::invalid_request(reason, Some("reference_audio"), Some("invalid_audio"))
        }
        other => ApiError::internal(other.to_string()),
    }
}

async fn create_speech_from_request(
    state: Arc<AppState>,
    request: SpeechRequest,
) -> Result<Response, ApiError> {
    tracing::debug!(
        model = %request.model,
        voice = %request.voice,
        format = ?request.response_format,
        has_language = request.language.is_some(),
        "speech request received"
    );
    request.validate()?;
    let requested = parse_tts_model(&request.model)
        .map_err(|_| ApiError::model_not_available(&request.model))?;
    let tts = state
        .tts(requested)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::model_not_available(&request.model))?;
    let format = request
        .response_format
        .map(Ok)
        .unwrap_or_else(|| SpeechFormat::try_from(state.config().services.tts.format.as_str()))?;
    let voice = request.to_tts_voice()?;
    let synthesis_started = Instant::now();
    tracing::debug!("speech synthesis started");
    let options = request.to_tts_options();
    let audio = tts.synthesize_with(request.input, voice, options).await?;
    let synthesis_elapsed = synthesis_started.elapsed();
    tracing::debug!(
        samples = audio.samples.len(),
        sample_rate = audio.sample_rate,
        elapsed_ms = synthesis_elapsed.as_millis(),
        "speech synthesis completed"
    );
    let encode_started = Instant::now();
    tracing::debug!(format = %format, "speech audio encode started");
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::from(format)).await?;
    let encode_elapsed = encode_started.elapsed();
    tracing::info!(format = %format, "speech request completed");
    tracing::debug!(
        bytes = encoded.bytes.len(),
        format = %format,
        elapsed_ms = encode_elapsed.as_millis(),
        "speech response encoded"
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static(content_type_for(format)),
        )
        .body(Body::from(encoded.bytes))
        .map_err(|error| ApiError::internal(error.to_string()))
}

pub(super) async fn create_transcription(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let mut audio_file = None;
    let mut model = None;
    let mut language = None;
    let mut response_format = TranscriptionFormat::default();
    let mut timestamp_granularities = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                audio_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "language" => language = Some(read_text_field(field, "language").await?),
            "response_format" => {
                let value = read_text_field(field, "response_format").await?;
                response_format = TranscriptionFormat::try_from(value.as_str())?;
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                timestamp_granularities
                    .push(read_text_field(field, "timestamp_granularities").await?);
            }
            "prompt" | "temperature" => {
                let _ = field.text().await;
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let segment_timestamps = parse_timestamp_granularities(&timestamp_granularities)?;
    let use_segments = segment_timestamps || matches!(response_format, TranscriptionFormat::Srt);

    let model = model.ok_or_else(|| {
        ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )
    })?;
    let requested = parse_asr_model(&model).map_err(|_| ApiError::model_not_available(&model))?;
    let asr = state
        .asr(requested)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let (audio_file, audio_bytes) = audio_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if audio_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded audio file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }
    let audio_path = audio_file.path().to_path_buf();
    tracing::debug!(
        model = %model,
        language = ?language,
        response_format = ?response_format,
        audio_bytes,
        "transcription request received"
    );
    let options = AsrOptions {
        language,
        ..Default::default()
    };
    let transcript = if use_segments {
        asr.transcribe_audio_file_with_segments(audio_path, options)
            .await?
    } else {
        asr.transcribe_audio_file_with(audio_path, options).await?
    };
    tracing::info!(format = ?response_format, "transcription request completed");

    Ok(match response_format {
        TranscriptionFormat::Json => Json(TranscriptionJson {
            text: transcript.text,
        })
        .into_response(),
        TranscriptionFormat::VerboseJson => Json(TranscriptionVerboseJson {
            text: transcript.text,
            language: transcript.language,
            raw_output: transcript.raw_output,
            segments: segment_timestamps.then_some(transcript.segments),
        })
        .into_response(),
        TranscriptionFormat::Srt => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            format_srt(&transcript),
        )
            .into_response(),
        TranscriptionFormat::Text => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            transcript.text,
        )
            .into_response(),
    })
}

pub(super) async fn create_transcription_ws(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let required_api_key = state.config().auth.api_key.clone();
    let header_authorized =
        transcription_stream_header_authorized(required_api_key.as_deref(), &headers);
    Ok(ws.on_upgrade(move |socket| {
        handle_transcription_ws(socket, state, required_api_key, header_authorized)
    }))
}

async fn handle_transcription_ws(
    mut socket: WebSocket,
    state: Arc<AppState>,
    required_api_key: Option<String>,
    header_authorized: bool,
) {
    let start = match receive_transcription_stream_start(&mut socket).await {
        Ok(start) => start,
        Err(error) => {
            let _ = send_stream_error(&mut socket, error).await;
            return;
        }
    };
    if let Err(error) = validate_transcription_stream_api_key(
        required_api_key.as_deref(),
        start.api_key.as_deref(),
        header_authorized,
    ) {
        let _ = send_stream_error(&mut socket, error).await;
        return;
    }
    let model = start.model.clone();
    let requested = match parse_asr_model(&model) {
        Ok(model) => model,
        Err(_) => {
            let _ = send_stream_error(&mut socket, ApiError::model_not_available(&model)).await;
            return;
        }
    };
    let asr = match state.asr(requested).await {
        Ok(Some(asr)) => asr,
        Ok(None) => {
            let _ = send_stream_error(&mut socket, ApiError::model_not_available(&model)).await;
            return;
        }
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::internal(error.to_string())).await;
            return;
        }
    };
    let streaming_options =
        start.to_streaming_options(state.config().services.asr.stream_chunk_size);
    let chunk_size_sec = streaming_options.chunk_size_sec;
    let mut stream = match asr.start_streaming_with(streaming_options).await {
        Ok(stream) => stream,
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
            return;
        }
    };
    let mut decoder = match start.audio_decoder().await {
        Ok(decoder) => decoder,
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
            return;
        }
    };
    let mut pcm_buffer = AsrPcmBuffer::new(chunk_size_sec);

    if send_stream_ready(&mut socket).await.is_err() {
        return;
    }

    while let Some(message) = socket.recv().await {
        let message = match message {
            Ok(message) => message,
            Err(error) => {
                tracing::debug!(error = %error, "transcription websocket receive failed");
                return;
            }
        };
        match message {
            Message::Binary(bytes) => {
                let decoded = match decoder.push(&bytes).await {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
                        return;
                    }
                };
                let chunks = match pcm_buffer.push(&decoded.samples, decoded.sample_rate) {
                    Ok(chunks) => chunks,
                    Err(error) => {
                        let _ = send_stream_error(&mut socket, error).await;
                        return;
                    }
                };
                if let Err(error) =
                    feed_transcription_stream_chunks(&mut socket, &mut stream, chunks).await
                {
                    let _ = send_stream_error(&mut socket, error).await;
                    return;
                }
            }
            Message::Text(text) => match parse_transcription_stream_control(text.as_str()) {
                Ok(TranscriptionStreamControl::End) => {
                    finish_transcription_stream(&mut socket, decoder, stream, pcm_buffer).await;
                    return;
                }
                Ok(TranscriptionStreamControl::Start) => {
                    let _ = send_stream_error(
                        &mut socket,
                        ApiError::invalid_request(
                            "transcription stream has already started",
                            Some("type"),
                            Some("invalid_stream_state"),
                        ),
                    )
                    .await;
                    return;
                }
                Err(error) => {
                    let _ = send_stream_error(&mut socket, error).await;
                    return;
                }
            },
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

fn transcription_stream_header_authorized(
    required_api_key: Option<&str>,
    headers: &HeaderMap,
) -> bool {
    let Some(required_api_key) = required_api_key else {
        return true;
    };
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| token == required_api_key)
}

fn validate_transcription_stream_api_key(
    required_api_key: Option<&str>,
    message_api_key: Option<&str>,
    header_authorized: bool,
) -> Result<(), ApiError> {
    let Some(required_api_key) = required_api_key else {
        return Ok(());
    };
    if header_authorized || message_api_key == Some(required_api_key) {
        Ok(())
    } else {
        Err(ApiError::invalid_api_key())
    }
}

async fn receive_transcription_stream_start(
    socket: &mut WebSocket,
) -> Result<TranscriptionStreamStart, ApiError> {
    match socket.recv().await {
        Some(Ok(Message::Text(text))) => parse_transcription_stream_start(text.as_str()),
        Some(Ok(_)) => Err(ApiError::invalid_request(
            "first websocket message must be a JSON start message",
            Some("type"),
            Some("missing_start_message"),
        )),
        Some(Err(error)) => Err(ApiError::invalid_request(
            error.to_string(),
            None,
            Some("invalid_websocket_message"),
        )),
        None => Err(ApiError::invalid_request(
            "websocket closed before start message",
            Some("type"),
            Some("missing_start_message"),
        )),
    }
}

async fn finish_transcription_stream(
    socket: &mut WebSocket,
    decoder: StreamingAudioDecoder,
    mut stream: orchion::AsrStream,
    mut pcm_buffer: AsrPcmBuffer,
) {
    let decoded = match decoder.finish().await {
        Ok(decoded) => decoded,
        Err(error) => {
            let _ = send_stream_error(socket, ApiError::from(error)).await;
            return;
        }
    };
    let chunks = match pcm_buffer.push(&decoded.samples, decoded.sample_rate) {
        Ok(chunks) => chunks,
        Err(error) => {
            let _ = send_stream_error(socket, error).await;
            return;
        }
    };
    if let Err(error) = feed_transcription_stream_chunks(socket, &mut stream, chunks).await {
        let _ = send_stream_error(socket, error).await;
        return;
    }
    if let Some((samples, sample_rate)) = pcm_buffer.drain_remaining() {
        match stream.feed(&samples, sample_rate).await {
            Ok(Some(transcript)) => {
                if send_stream_transcript(socket, "partial", &transcript)
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Ok(None) => {}
            Err(error) => {
                let _ = send_stream_error(socket, ApiError::from(error)).await;
                return;
            }
        }
    }
    match stream.finish().await {
        Ok(transcript) => {
            let _ = send_stream_transcript(socket, "final", &transcript).await;
        }
        Err(error) => {
            let _ = send_stream_error(socket, ApiError::from(error)).await;
        }
    }
}

async fn feed_transcription_stream_chunks(
    socket: &mut WebSocket,
    stream: &mut orchion::AsrStream,
    chunks: Vec<(Vec<f32>, u32)>,
) -> Result<(), ApiError> {
    for (samples, sample_rate) in chunks {
        match stream.feed(&samples, sample_rate).await {
            Ok(Some(transcript)) => send_stream_transcript(socket, "partial", &transcript)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?,
            Ok(None) => {}
            Err(error) => return Err(ApiError::from(error)),
        }
    }
    Ok(())
}

struct AsrPcmBuffer {
    chunk_size_sec: f32,
    sample_rate: Option<u32>,
    samples: Vec<f32>,
}

impl AsrPcmBuffer {
    fn new(chunk_size_sec: f32) -> Self {
        Self {
            chunk_size_sec,
            sample_rate: None,
            samples: Vec::new(),
        }
    }

    fn push(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<(Vec<f32>, u32)>, ApiError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if sample_rate == 0 || !self.chunk_size_sec.is_finite() || self.chunk_size_sec <= 0.0 {
            return Err(ApiError::invalid_request(
                "invalid streaming audio chunk size",
                Some("chunk_size_sec"),
                Some("invalid_chunk_size"),
            ));
        }
        if let Some(current_sample_rate) = self.sample_rate {
            if current_sample_rate != sample_rate {
                return Err(ApiError::invalid_request(
                    "streaming decoded audio sample rate changed",
                    Some("sample_rate"),
                    Some("invalid_sample_rate"),
                ));
            }
        } else {
            self.sample_rate = Some(sample_rate);
        }

        let chunk_samples = (sample_rate as f32 * self.chunk_size_sec) as usize;
        if chunk_samples == 0 {
            return Err(ApiError::invalid_request(
                "invalid streaming audio chunk size",
                Some("chunk_size_sec"),
                Some("invalid_chunk_size"),
            ));
        }

        self.samples.extend_from_slice(samples);
        let mut chunks = Vec::new();
        while self.samples.len() >= chunk_samples {
            chunks.push((self.samples.drain(..chunk_samples).collect(), sample_rate));
        }
        Ok(chunks)
    }

    fn drain_remaining(&mut self) -> Option<(Vec<f32>, u32)> {
        if self.samples.is_empty() {
            return None;
        }
        self.sample_rate
            .map(|sample_rate| (std::mem::take(&mut self.samples), sample_rate))
    }
}

async fn send_stream_ready(socket: &mut WebSocket) -> Result<(), axum::Error> {
    send_stream_event(
        socket,
        &TranscriptionStreamReady {
            event_type: "ready",
        },
    )
    .await
}

async fn send_stream_transcript(
    socket: &mut WebSocket,
    event_type: &'static str,
    transcript: &orchion::AsrTranscript,
) -> Result<(), axum::Error> {
    let event = stream_transcript_event(event_type, transcript);
    send_stream_event(socket, &event).await
}

fn stream_transcript_event<'a>(
    event_type: &'static str,
    transcript: &'a orchion::AsrTranscript,
) -> TranscriptionStreamTranscriptEvent<'a> {
    TranscriptionStreamTranscriptEvent {
        event_type,
        text: &transcript.text,
    }
}

async fn send_stream_error(socket: &mut WebSocket, error: ApiError) -> Result<(), axum::Error> {
    send_stream_event(
        socket,
        &TranscriptionStreamErrorEvent {
            event_type: "error",
            error: error.error,
        },
    )
    .await
}

async fn send_stream_event<T: Serialize>(
    socket: &mut WebSocket,
    event: &T,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(event).map_err(axum::Error::new)?;
    socket.send(Message::Text(text.into())).await
}

pub(super) fn parse_timestamp_granularities(values: &[String]) -> Result<bool, ApiError> {
    let mut wants_segments = false;
    for value in values {
        match value.parse::<AsrTimestampGranularity>().map_err(|error| {
            ApiError::invalid_request(
                error,
                Some("timestamp_granularities"),
                Some("unsupported_timestamp_granularity"),
            )
        })? {
            AsrTimestampGranularity::Segment => wants_segments = true,
            AsrTimestampGranularity::Word => {
                return Err(ApiError::invalid_request(
                    "word timestamp granularity is not supported",
                    Some("timestamp_granularities"),
                    Some("unsupported_timestamp_granularity"),
                ));
            }
        }
    }
    Ok(wants_segments)
}

#[derive(Debug, Clone, PartialEq)]
struct TranscriptionStreamStart {
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    api_key: Option<String>,
    response_format: TranscriptionFormat,
    input_audio_format: TranscriptionStreamInputFormat,
    sample_rate: Option<u32>,
    chunk_size_sec: Option<f32>,
    unfixed_chunk_num: Option<usize>,
    unfixed_token_num: Option<usize>,
    max_new_tokens_streaming: Option<usize>,
    max_new_tokens_final: Option<usize>,
}

impl TranscriptionStreamStart {
    fn to_streaming_options(&self, default_chunk_size_sec: f32) -> AsrStreamingOptions {
        let defaults = AsrStreamingOptions::default();
        AsrStreamingOptions {
            language: self.language.clone(),
            chunk_size_sec: self.chunk_size_sec.unwrap_or(default_chunk_size_sec),
            unfixed_chunk_num: self.unfixed_chunk_num.unwrap_or(defaults.unfixed_chunk_num),
            unfixed_token_num: self.unfixed_token_num.unwrap_or(defaults.unfixed_token_num),
            max_new_tokens_streaming: self
                .max_new_tokens_streaming
                .unwrap_or(defaults.max_new_tokens_streaming),
            max_new_tokens_final: self
                .max_new_tokens_final
                .unwrap_or(defaults.max_new_tokens_final),
            initial_text: self.prompt.clone(),
        }
    }

    async fn audio_decoder(&self) -> orchion::Result<StreamingAudioDecoder> {
        StreamingAudioDecoder::new_for_asr(self.input_audio_format.into(), self.sample_rate).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionStreamInputFormat {
    Auto,
    PcmS16Le,
    WebmOpus,
    Mp3,
    Wav,
    M4a,
    Aac,
    Flac,
    Ogg,
}

impl TryFrom<&str> for TranscriptionStreamInputFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "pcm_s16le" | "pcm" => Ok(Self::PcmS16Le),
            "webm_opus" | "webm" => Ok(Self::WebmOpus),
            "mp3" => Ok(Self::Mp3),
            "wav" => Ok(Self::Wav),
            "m4a" => Ok(Self::M4a),
            "aac" => Ok(Self::Aac),
            "flac" => Ok(Self::Flac),
            "ogg" | "opus" => Ok(Self::Ogg),
            _ => Err(ApiError::invalid_request(
                "unsupported input_audio_format; supported formats are auto, pcm_s16le, webm_opus, mp3, wav, m4a, aac, flac, ogg, and opus",
                Some("input_audio_format"),
                Some("unsupported_audio_format"),
            )),
        }
    }
}

impl From<TranscriptionStreamInputFormat> for AudioInputFormat {
    fn from(format: TranscriptionStreamInputFormat) -> Self {
        match format {
            TranscriptionStreamInputFormat::Auto => Self::Auto,
            TranscriptionStreamInputFormat::PcmS16Le => Self::PcmS16Le,
            TranscriptionStreamInputFormat::WebmOpus => Self::WebmOpus,
            TranscriptionStreamInputFormat::Mp3 => Self::Mp3,
            TranscriptionStreamInputFormat::Wav => Self::Wav,
            TranscriptionStreamInputFormat::M4a => Self::M4a,
            TranscriptionStreamInputFormat::Aac => Self::Aac,
            TranscriptionStreamInputFormat::Flac => Self::Flac,
            TranscriptionStreamInputFormat::Ogg => Self::Ogg,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionStreamControl {
    Start,
    End,
}

#[derive(Debug, Deserialize)]
struct RawTranscriptionStreamStart {
    #[serde(rename = "type")]
    message_type: Option<String>,
    model: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    input_audio_format: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    chunk_size_sec: Option<f32>,
    #[serde(default)]
    unfixed_chunk_num: Option<usize>,
    #[serde(default)]
    unfixed_token_num: Option<usize>,
    #[serde(default)]
    max_new_tokens_streaming: Option<usize>,
    #[serde(default)]
    max_new_tokens_final: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawTranscriptionStreamControl {
    #[serde(rename = "type")]
    message_type: Option<String>,
}

#[derive(Serialize)]
struct TranscriptionStreamReady {
    #[serde(rename = "type")]
    event_type: &'static str,
}

#[derive(Serialize)]
struct TranscriptionStreamTranscriptEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct TranscriptionStreamErrorEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    error: ErrorObject,
}

fn parse_transcription_stream_start(text: &str) -> Result<TranscriptionStreamStart, ApiError> {
    let raw = serde_json::from_str::<RawTranscriptionStreamStart>(text).map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_json"))
    })?;
    if raw.message_type.as_deref() != Some("start") {
        return Err(ApiError::invalid_request(
            "first websocket message must have type `start`",
            Some("type"),
            Some("missing_start_message"),
        ));
    }
    let model = raw.model.ok_or_else(|| {
        ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )
    })?;
    let input_audio_format = raw.input_audio_format.ok_or_else(|| {
        ApiError::invalid_request(
            "`input_audio_format` is required",
            Some("input_audio_format"),
            Some("missing_required_parameter"),
        )
    })?;
    let input_audio_format = TranscriptionStreamInputFormat::try_from(input_audio_format.as_str())?;
    if matches!(input_audio_format, TranscriptionStreamInputFormat::PcmS16Le)
        && raw.sample_rate.is_none()
    {
        return Err(ApiError::invalid_request(
            "`sample_rate` is required for pcm_s16le input",
            Some("sample_rate"),
            Some("missing_required_parameter"),
        ));
    }
    if raw.sample_rate.is_some_and(|value| value == 0) {
        return Err(ApiError::invalid_request(
            "`sample_rate` must be greater than zero",
            Some("sample_rate"),
            Some("invalid_sample_rate"),
        ));
    }
    let response_format = raw
        .response_format
        .as_deref()
        .map(TranscriptionFormat::try_from)
        .transpose()?
        .unwrap_or_default();
    if !matches!(response_format, TranscriptionFormat::Json) {
        return Err(ApiError::invalid_request(
            "streaming transcription supports response_format json only",
            Some("response_format"),
            Some("unsupported_response_format"),
        ));
    }
    Ok(TranscriptionStreamStart {
        model,
        language: raw.language,
        prompt: raw.prompt,
        api_key: raw.api_key,
        response_format,
        input_audio_format,
        sample_rate: raw.sample_rate,
        chunk_size_sec: raw.chunk_size_sec,
        unfixed_chunk_num: raw.unfixed_chunk_num,
        unfixed_token_num: raw.unfixed_token_num,
        max_new_tokens_streaming: raw.max_new_tokens_streaming,
        max_new_tokens_final: raw.max_new_tokens_final,
    })
}

fn parse_transcription_stream_control(text: &str) -> Result<TranscriptionStreamControl, ApiError> {
    let raw = serde_json::from_str::<RawTranscriptionStreamControl>(text).map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_json"))
    })?;
    match raw.message_type.as_deref() {
        Some("start") => Ok(TranscriptionStreamControl::Start),
        Some("end") => Ok(TranscriptionStreamControl::End),
        _ => Err(ApiError::invalid_request(
            "websocket control message must have type `end`",
            Some("type"),
            Some("unsupported_message_type"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_start_accepts_openai_fields_and_stream_audio_format() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "language":"zh",
                "prompt":"previous context",
                "api_key":"secret-key",
                "response_format":"json",
                "input_audio_format":"mp3"
            }"#,
        )
        .unwrap();

        assert_eq!(start.model, "Qwen/Qwen3-ASR-Flash");
        assert_eq!(start.language.as_deref(), Some("zh"));
        assert_eq!(start.prompt.as_deref(), Some("previous context"));
        assert_eq!(start.api_key.as_deref(), Some("secret-key"));
        assert_eq!(start.response_format, TranscriptionFormat::Json);
        assert_eq!(
            start.input_audio_format,
            TranscriptionStreamInputFormat::Mp3
        );
        assert_eq!(start.sample_rate, None);
    }

    #[test]
    fn stream_start_accepts_wav_input_format() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"wav"
            }"#,
        )
        .unwrap();

        assert_eq!(
            start.input_audio_format,
            TranscriptionStreamInputFormat::Wav
        );
        assert_eq!(start.sample_rate, None);
    }

    #[test]
    fn stream_start_accepts_auto_input_format() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"auto"
            }"#,
        )
        .unwrap();

        assert_eq!(
            start.input_audio_format,
            TranscriptionStreamInputFormat::Auto
        );
        assert_eq!(start.sample_rate, None);
    }

    #[test]
    fn stream_start_accepts_additional_file_input_formats() {
        let cases = [
            ("m4a", TranscriptionStreamInputFormat::M4a),
            ("aac", TranscriptionStreamInputFormat::Aac),
            ("flac", TranscriptionStreamInputFormat::Flac),
            ("ogg", TranscriptionStreamInputFormat::Ogg),
            ("opus", TranscriptionStreamInputFormat::Ogg),
        ];

        for (format, expected) in cases {
            let start = parse_transcription_stream_start(&format!(
                r#"{{"type":"start","model":"Qwen/Qwen3-ASR-Flash","input_audio_format":"{format}"}}"#
            ))
            .unwrap();

            assert_eq!(start.input_audio_format, expected);
            assert_eq!(start.sample_rate, None);
        }
    }

    #[test]
    fn stream_start_api_key_authenticates_when_header_is_missing() {
        validate_transcription_stream_api_key(Some("secret"), Some("secret"), false).unwrap();
    }

    #[test]
    fn stream_start_api_key_rejects_missing_key_when_required() {
        let error = validate_transcription_stream_api_key(Some("secret"), None, false).unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn stream_start_api_key_skips_message_key_after_header_auth() {
        validate_transcription_stream_api_key(Some("secret"), None, true).unwrap();
    }

    #[test]
    fn stream_start_requires_sample_rate_for_pcm_s16le() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le"
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("sample_rate"));
        assert_eq!(
            error.error.code.as_deref(),
            Some("missing_required_parameter")
        );
    }

    #[test]
    fn stream_start_rejects_text_response_format() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "response_format":"text",
                "input_audio_format":"mp3"
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("response_format"));
        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_response_format")
        );
    }

    #[test]
    fn stream_start_rejects_verbose_json_response_format() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "response_format":"verbose_json",
                "input_audio_format":"mp3"
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("response_format"));
        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_response_format")
        );
    }

    #[test]
    fn stream_transcript_event_contains_only_type_and_text() {
        let transcript = orchion::AsrTranscript {
            text: "hello".to_string(),
            language: "en".to_string(),
            raw_output: "internal".to_string(),
            segments: Vec::new(),
        };

        let event = stream_transcript_event("partial", &transcript);
        let json = serde_json::to_value(event).unwrap();

        assert_eq!(json["type"], "partial");
        assert_eq!(json["text"], "hello");
        assert!(json.get("is_final").is_none());
        assert!(json.get("language").is_none());
        assert!(json.get("raw_output").is_none());
    }

    #[test]
    fn stream_control_accepts_end_message() {
        let control = parse_transcription_stream_control(r#"{"type":"end"}"#).unwrap();

        assert_eq!(control, TranscriptionStreamControl::End);
    }

    #[test]
    fn asr_pcm_buffer_merges_decoder_outputs_until_stream_chunk() {
        let mut buffer = AsrPcmBuffer::new(2.0);

        let first_chunks = buffer.push(&vec![0.0; 10_000], 16_000).unwrap();
        let second_chunks = buffer.push(&vec![0.0; 22_000], 16_000).unwrap();

        assert!(first_chunks.is_empty());
        assert_eq!(second_chunks.len(), 1);
        assert_eq!(second_chunks[0].0.len(), 32_000);
        assert_eq!(second_chunks[0].1, 16_000);
        assert!(buffer.drain_remaining().is_none());
    }

    #[test]
    fn asr_pcm_buffer_keeps_tail_for_finish() {
        let mut buffer = AsrPcmBuffer::new(2.0);

        let chunks = buffer.push(&vec![0.0; 80_000], 16_000).unwrap();
        let tail = buffer.drain_remaining().unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0.len(), 32_000);
        assert_eq!(chunks[1].0.len(), 32_000);
        assert_eq!(tail.0.len(), 16_000);
        assert_eq!(tail.1, 16_000);
    }
}
