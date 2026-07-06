use crate::api::caption_boundary::{CaptionTextSplitter, CaptionTextUpdate};
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
use crate::settings::{DEFAULT_ASR_STREAM_MAX_SEGMENT, parse_asr_model, parse_tts_model};
use axum::Json;
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use orchion::{
    ASR_SAMPLE_RATE, AsrOptions, AsrStreamingOptions, AsrTimestampGranularity, AudioInputFormat,
    AudioOutputFormat, AudioVadMode, AudioVadStreamingConfig, AudioVadStreamingEndpoint,
    AudioVadStreamingEvent, OrchionError, StreamingAudioDecoder, TtsAudio, decode_audio_file,
    encode_tts_audio,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};
use tokio::time::timeout;

const TRANSCRIPTION_STREAM_START_TIMEOUT: Duration = Duration::from_secs(10);

fn duration_to_millis_u32(duration: Duration, field: &'static str) -> u32 {
    u32::try_from(duration.as_millis())
        .unwrap_or_else(|_| panic!("validated ASR {field} must fit in u32 milliseconds"))
}

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
    let mut reference_audio_file = None;
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
                reference_audio_file =
                    Some(write_multipart_file_to_temp_file(field, "reference_audio").await?);
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
    let (reference_audio_file, reference_audio_size) = reference_audio_file.ok_or_else(|| {
        ApiError::invalid_request(
            "voice clone requires `reference_audio` file upload",
            Some("reference_audio"),
            Some("missing_required_parameter"),
        )
    })?;
    if reference_audio_size == 0 {
        return Err(ApiError::invalid_request(
            "uploaded reference audio is empty",
            Some("reference_audio"),
            Some("invalid_file"),
        ));
    }
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
    let reference_file = transcode_reference_audio_to_wav_file(reference_audio_file.path()).await?;
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
    reference_audio: &Path,
) -> Result<NamedTempFile, ApiError> {
    let decoded = decode_audio_file(reference_audio.to_path_buf())
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
    authorize(&state, &headers)?;
    Ok(ws.on_upgrade(move |socket| handle_transcription_ws(socket, state)))
}

async fn handle_transcription_ws(mut socket: WebSocket, state: Arc<AppState>) {
    let stream_target_segment_millis = duration_to_millis_u32(
        state.config().services.asr.stream_target_segment,
        "stream_target_segment",
    );
    let stream_max_segment_millis = duration_to_millis_u32(
        state.config().services.asr.stream_max_segment,
        "stream_max_segment",
    );
    let start =
        match receive_transcription_stream_start(&mut socket, stream_max_segment_millis).await {
            Ok(start) => start,
            Err(error) => {
                let _ = send_stream_error(&mut socket, error).await;
                return;
            }
        };
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
    let default_chunk_size_sec = state.config().services.asr.stream_chunk_size;
    match start.mode {
        TranscriptionStreamMode::Legacy => {
            run_legacy_transcription_stream(socket, start, asr, default_chunk_size_sec).await;
        }
        TranscriptionStreamMode::Caption => {
            run_caption_transcription_stream(
                socket,
                start,
                asr,
                default_chunk_size_sec,
                stream_target_segment_millis,
            )
            .await;
        }
    }
}

async fn run_legacy_transcription_stream(
    mut socket: WebSocket,
    start: TranscriptionStreamStart,
    asr: orchion::Asr,
    default_chunk_size_sec: f32,
) {
    let streaming_options = start.to_streaming_options(default_chunk_size_sec);
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

async fn run_caption_transcription_stream(
    mut socket: WebSocket,
    start: TranscriptionStreamStart,
    asr: orchion::Asr,
    default_chunk_size_sec: f32,
    stream_target_segment_millis: u32,
) {
    let streaming_options = start.to_streaming_options(default_chunk_size_sec);
    if let Err(error) = validate_caption_streaming_options(&streaming_options) {
        let _ = send_stream_error(&mut socket, error).await;
        return;
    }
    let mut decoder = match start.audio_decoder().await {
        Ok(decoder) => decoder,
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
            return;
        }
    };
    let mut endpoint = match AudioVadStreamingEndpoint::new(start.endpointing.to_vad_config()) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
            return;
        }
    };
    let mut current_segment = None;
    let mut next_segment_id = 0;

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
                let events = match endpoint.push(&decoded.samples, decoded.sample_rate) {
                    Ok(events) => events,
                    Err(error) => {
                        let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
                        return;
                    }
                };
                if let Err(error) = handle_caption_vad_events(
                    &mut socket,
                    &asr,
                    &streaming_options,
                    &mut current_segment,
                    &mut next_segment_id,
                    events,
                    decoded.sample_rate,
                    stream_target_segment_millis,
                )
                .await
                {
                    let _ = send_stream_error(&mut socket, error).await;
                    return;
                }
            }
            Message::Text(text) => match parse_transcription_stream_control(text.as_str()) {
                Ok(TranscriptionStreamControl::End) => {
                    if let Err(error) = finish_caption_transcription_stream(
                        &mut socket,
                        decoder,
                        endpoint,
                        &asr,
                        &streaming_options,
                        &mut current_segment,
                        &mut next_segment_id,
                        stream_target_segment_millis,
                    )
                    .await
                    {
                        let _ = send_stream_error(&mut socket, error).await;
                    }
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

#[cfg(test)]
fn validate_transcription_stream_api_key(
    required_api_key: Option<&str>,
    _message_api_key: Option<&str>,
    header_authorized: bool,
) -> Result<(), ApiError> {
    if required_api_key.is_none() {
        return Ok(());
    };
    if header_authorized {
        Ok(())
    } else {
        Err(ApiError::invalid_api_key())
    }
}

async fn receive_transcription_stream_start(
    socket: &mut WebSocket,
    stream_max_segment_millis: u32,
) -> Result<TranscriptionStreamStart, ApiError> {
    let message = timeout(TRANSCRIPTION_STREAM_START_TIMEOUT, socket.recv())
        .await
        .map_err(|_| transcription_stream_start_timeout_error())?;
    match message {
        Some(Ok(Message::Text(text))) => parse_transcription_stream_start_with_stream_max_segment(
            text.as_str(),
            stream_max_segment_millis,
        ),
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

fn transcription_stream_start_timeout_error() -> ApiError {
    ApiError::invalid_request(
        "websocket start message timed out",
        Some("type"),
        Some("start_message_timeout"),
    )
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

async fn handle_caption_vad_events(
    socket: &mut WebSocket,
    asr: &orchion::Asr,
    streaming_options: &AsrStreamingOptions,
    current_segment: &mut Option<CaptionSegmentStream>,
    next_segment_id: &mut u64,
    events: Vec<AudioVadStreamingEvent>,
    sample_rate: u32,
    stream_target_segment_millis: u32,
) -> Result<(), ApiError> {
    for event in events {
        match event {
            AudioVadStreamingEvent::SegmentStarted {
                start_sample,
                samples,
            } => {
                if let Some(segment) = current_segment.take() {
                    finalize_caption_segment(socket, segment, start_sample).await?;
                }

                let segment = start_caption_segment(
                    asr,
                    streaming_options,
                    next_segment_id,
                    start_sample,
                    sample_rate,
                    stream_target_segment_millis,
                )
                .await?;
                let mut segment = segment;
                feed_caption_segment_audio(
                    socket,
                    &mut segment,
                    &samples,
                    sample_rate,
                    next_segment_id,
                )
                .await?;
                *current_segment = Some(segment);
            }
            AudioVadStreamingEvent::Audio { samples } => {
                if let Some(segment) = current_segment.as_mut() {
                    feed_caption_segment_audio(
                        socket,
                        segment,
                        &samples,
                        sample_rate,
                        next_segment_id,
                    )
                    .await?;
                }
            }
            AudioVadStreamingEvent::SegmentFinal { end_sample, .. } => {
                if let Some(segment) = current_segment.take() {
                    finalize_caption_segment(socket, segment, end_sample).await?;
                }
            }
        }
    }
    Ok(())
}

async fn start_caption_segment(
    asr: &orchion::Asr,
    streaming_options: &AsrStreamingOptions,
    next_segment_id: &mut u64,
    start_sample: usize,
    sample_rate: u32,
    stream_target_segment_millis: u32,
) -> Result<CaptionSegmentStream, ApiError> {
    let segment_id = allocate_caption_segment_id(next_segment_id)?;
    let stream = asr
        .start_streaming_with(streaming_options.clone())
        .await
        .map_err(ApiError::from)?;
    Ok(CaptionSegmentStream {
        segment_id,
        start_sample,
        subtitle_start_sample: start_sample,
        last_sample: start_sample,
        sample_rate,
        stream,
        pcm_buffer: AsrPcmBuffer::new(streaming_options.chunk_size_sec),
        text_splitter: CaptionTextSplitter::new(stream_target_segment_millis),
    })
}

fn allocate_caption_segment_id(next_segment_id: &mut u64) -> Result<u64, ApiError> {
    let segment_id = *next_segment_id;
    *next_segment_id = next_segment_id
        .checked_add(1)
        .ok_or_else(|| ApiError::internal("caption segment id overflowed"))?;
    Ok(segment_id)
}

async fn feed_caption_segment_audio(
    socket: &mut WebSocket,
    segment: &mut CaptionSegmentStream,
    samples: &[f32],
    sample_rate: u32,
    next_segment_id: &mut u64,
) -> Result<(), ApiError> {
    if sample_rate != segment.sample_rate {
        return Err(ApiError::invalid_request(
            "caption segment audio sample rate changed",
            Some("sample_rate"),
            Some("invalid_sample_rate"),
        ));
    }
    let last_sample = segment
        .last_sample
        .checked_add(samples.len())
        .ok_or_else(|| ApiError::internal("caption segment sample index overflowed"))?;
    let chunks = segment.pcm_buffer.push(samples, sample_rate)?;
    segment.last_sample = last_sample;
    for (chunk_samples, chunk_sample_rate) in chunks {
        match segment.stream.feed(&chunk_samples, chunk_sample_rate).await {
            Ok(Some(transcript)) => {
                send_caption_text_update(socket, segment, &transcript.text, next_segment_id)
                    .await?;
            }
            Ok(None) => {}
            Err(error) => return Err(ApiError::from(error)),
        }
    }
    Ok(())
}

async fn send_caption_text_update(
    socket: &mut WebSocket,
    segment: &mut CaptionSegmentStream,
    transcript_text: &str,
    next_segment_id: &mut u64,
) -> Result<(), ApiError> {
    let CaptionTextUpdate {
        segment_final,
        partial,
    } = segment
        .text_splitter
        .observe_partial(transcript_text, segment.duration_ms());
    if let Some(final_text) = segment_final {
        send_caption_text_final(
            socket,
            segment.segment_id,
            segment.subtitle_start_sample,
            segment.sample_rate,
            final_text,
            segment.last_sample,
        )
        .await?;
        segment.segment_id = allocate_caption_segment_id(next_segment_id)?;
        segment.subtitle_start_sample = segment.last_sample;
    }
    if !partial.trim().is_empty() {
        send_stream_event(socket, &caption_partial_event(segment.segment_id, partial))
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    Ok(())
}

async fn send_caption_text_final(
    socket: &mut WebSocket,
    segment_id: u64,
    start_sample: usize,
    sample_rate: u32,
    text: &str,
    end_sample: usize,
) -> Result<(), ApiError> {
    send_stream_event(
        socket,
        &caption_segment_final_event(
            segment_id,
            text,
            Some(sample_index_to_ms(start_sample, sample_rate)),
            Some(sample_index_to_ms(end_sample, sample_rate)),
        ),
    )
    .await
    .map_err(|error| ApiError::internal(error.to_string()))
}

async fn finalize_caption_segment(
    socket: &mut WebSocket,
    segment: CaptionSegmentStream,
    end_sample: usize,
) -> Result<(), ApiError> {
    let CaptionSegmentStream {
        segment_id,
        start_sample: _,
        subtitle_start_sample,
        last_sample: _,
        sample_rate,
        mut stream,
        mut pcm_buffer,
        mut text_splitter,
    } = segment;

    if let Some((samples, sample_rate)) = pcm_buffer.drain_remaining() {
        stream
            .feed(&samples, sample_rate)
            .await
            .map_err(ApiError::from)?;
    }

    let transcript = stream.finish().await.map_err(ApiError::from)?;
    if let Some(text) = text_splitter.flush(&transcript.text) {
        send_caption_text_final(
            socket,
            segment_id,
            subtitle_start_sample,
            sample_rate,
            text,
            end_sample,
        )
        .await?;
    }
    Ok(())
}

async fn finish_caption_transcription_stream(
    socket: &mut WebSocket,
    decoder: StreamingAudioDecoder,
    mut endpoint: AudioVadStreamingEndpoint,
    asr: &orchion::Asr,
    streaming_options: &AsrStreamingOptions,
    current_segment: &mut Option<CaptionSegmentStream>,
    next_segment_id: &mut u64,
    stream_target_segment_millis: u32,
) -> Result<(), ApiError> {
    let decoded = decoder.finish().await.map_err(ApiError::from)?;
    let events = endpoint
        .push(&decoded.samples, decoded.sample_rate)
        .map_err(ApiError::from)?;
    handle_caption_vad_events(
        socket,
        asr,
        streaming_options,
        current_segment,
        next_segment_id,
        events,
        decoded.sample_rate,
        stream_target_segment_millis,
    )
    .await?;

    let events = endpoint.finish();
    handle_caption_vad_events(
        socket,
        asr,
        streaming_options,
        current_segment,
        next_segment_id,
        events,
        decoded.sample_rate,
        stream_target_segment_millis,
    )
    .await?;

    if let Some(segment) = current_segment.take() {
        let end_sample = segment.last_sample;
        finalize_caption_segment(socket, segment, end_sample).await?;
    }

    send_stream_event(socket, &caption_completed_event())
        .await
        .map_err(|error| ApiError::internal(error.to_string()))
}

struct CaptionSegmentStream {
    segment_id: u64,
    start_sample: usize,
    subtitle_start_sample: usize,
    last_sample: usize,
    sample_rate: u32,
    stream: orchion::AsrStream,
    pcm_buffer: AsrPcmBuffer,
    text_splitter: CaptionTextSplitter,
}

impl CaptionSegmentStream {
    fn subtitle_start_ms(&self) -> u64 {
        sample_index_to_ms(self.subtitle_start_sample, self.sample_rate)
    }

    fn duration_ms(&self) -> u64 {
        sample_index_to_ms(
            self.last_sample.saturating_sub(self.start_sample),
            self.sample_rate,
        )
    }
}

fn sample_index_to_ms(sample_index: usize, sample_rate: u32) -> u64 {
    assert!(sample_rate > 0, "sample_rate must be greater than zero");
    (sample_index as u64 * 1_000) / u64::from(sample_rate)
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

fn caption_partial_event(
    segment_id: u64,
    text: &str,
) -> TranscriptionStreamCaptionPartialEvent<'_> {
    TranscriptionStreamCaptionPartialEvent {
        event_type: "partial",
        segment_id,
        text,
    }
}

fn caption_segment_final_event(
    segment_id: u64,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
) -> TranscriptionStreamCaptionSegmentFinalEvent<'_> {
    TranscriptionStreamCaptionSegmentFinalEvent {
        event_type: "segment_final",
        segment_id,
        text,
        start_ms,
        end_ms,
    }
}

fn caption_completed_event() -> TranscriptionStreamCaptionCompletedEvent {
    TranscriptionStreamCaptionCompletedEvent {
        event_type: "completed",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionStreamMode {
    Legacy,
    Caption,
}

const CAPTION_VAD_FRAME_DURATION_MS: u32 = 30;
const CAPTION_VAD_MAX_CANDIDATE_MS: u32 = 60_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CaptionEndpointingOptions {
    min_speech_ms: u32,
    min_silence_ms: u32,
    max_segment_ms: u32,
    speech_padding_ms: u32,
}

impl Default for CaptionEndpointingOptions {
    fn default() -> Self {
        Self::default_with_stream_max_segment_millis(duration_to_millis_u32(
            DEFAULT_ASR_STREAM_MAX_SEGMENT,
            "stream_max_segment",
        ))
    }
}

impl CaptionEndpointingOptions {
    fn default_with_stream_max_segment_millis(stream_max_segment_millis: u32) -> Self {
        Self {
            min_speech_ms: 300,
            min_silence_ms: 500,
            max_segment_ms: stream_max_segment_millis,
            speech_padding_ms: 200,
        }
    }

    fn to_vad_config(self) -> AudioVadStreamingConfig {
        AudioVadStreamingConfig {
            frame_duration_ms: 30,
            min_speech_ms: self.min_speech_ms,
            min_silence_ms: self.min_silence_ms,
            max_segment_ms: self.max_segment_ms,
            speech_padding_ms: self.speech_padding_ms,
            mode: AudioVadMode::Quality.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TranscriptionStreamStart {
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    api_key: Option<String>,
    response_format: TranscriptionFormat,
    mode: TranscriptionStreamMode,
    endpointing: CaptionEndpointingOptions,
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
#[serde(deny_unknown_fields)]
struct RawCaptionEndpointingOptions {
    #[serde(default)]
    min_speech_ms: Option<u32>,
    #[serde(default)]
    min_silence_ms: Option<u32>,
    #[serde(default)]
    speech_padding_ms: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawTranscriptionStreamStart {
    #[serde(rename = "type")]
    message_type: Option<String>,
    #[serde(default)]
    mode: Option<String>,
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
    endpointing: Option<RawCaptionEndpointingOptions>,
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
struct TranscriptionStreamCaptionPartialEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    segment_id: u64,
    text: &'a str,
}

#[derive(Serialize)]
struct TranscriptionStreamCaptionSegmentFinalEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    segment_id: u64,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_ms: Option<u64>,
}

#[derive(Serialize)]
struct TranscriptionStreamCaptionCompletedEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
}

#[derive(Serialize)]
struct TranscriptionStreamErrorEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    error: ErrorObject,
}

fn parse_transcription_stream_start(text: &str) -> Result<TranscriptionStreamStart, ApiError> {
    parse_transcription_stream_start_with_stream_max_segment(
        text,
        duration_to_millis_u32(DEFAULT_ASR_STREAM_MAX_SEGMENT, "stream_max_segment"),
    )
}

fn parse_transcription_stream_start_with_stream_max_segment(
    text: &str,
    stream_max_segment_millis: u32,
) -> Result<TranscriptionStreamStart, ApiError> {
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
    let mode = parse_transcription_stream_mode(raw.mode.as_deref())?;
    validate_caption_pcm_sample_rate(mode, input_audio_format, raw.sample_rate)?;
    let endpointing =
        parse_caption_endpointing_options(mode, raw.endpointing, stream_max_segment_millis)?;
    Ok(TranscriptionStreamStart {
        model,
        language: raw.language,
        prompt: raw.prompt,
        api_key: raw.api_key,
        response_format,
        mode,
        endpointing,
        input_audio_format,
        sample_rate: raw.sample_rate,
        chunk_size_sec: raw.chunk_size_sec,
        unfixed_chunk_num: raw.unfixed_chunk_num,
        unfixed_token_num: raw.unfixed_token_num,
        max_new_tokens_streaming: raw.max_new_tokens_streaming,
        max_new_tokens_final: raw.max_new_tokens_final,
    })
}

fn parse_transcription_stream_mode(
    mode: Option<&str>,
) -> Result<TranscriptionStreamMode, ApiError> {
    let Some(mode) = mode.map(str::trim) else {
        return Ok(TranscriptionStreamMode::Legacy);
    };
    if mode.is_empty() {
        return Ok(TranscriptionStreamMode::Legacy);
    }
    if mode.eq_ignore_ascii_case("caption") {
        return Ok(TranscriptionStreamMode::Caption);
    }
    Err(ApiError::invalid_request(
        "unsupported streaming transcription mode",
        Some("mode"),
        Some("unsupported_stream_mode"),
    ))
}

fn validate_caption_pcm_sample_rate(
    mode: TranscriptionStreamMode,
    input_audio_format: TranscriptionStreamInputFormat,
    sample_rate: Option<u32>,
) -> Result<(), ApiError> {
    if matches!(mode, TranscriptionStreamMode::Caption)
        && matches!(input_audio_format, TranscriptionStreamInputFormat::PcmS16Le)
        && sample_rate != Some(ASR_SAMPLE_RATE)
    {
        return Err(ApiError::invalid_request(
            format!("caption pcm_s16le input requires {ASR_SAMPLE_RATE} Hz sample_rate"),
            Some("sample_rate"),
            Some("unsupported_sample_rate"),
        ));
    }

    Ok(())
}

fn parse_caption_endpointing_options(
    mode: TranscriptionStreamMode,
    raw: Option<RawCaptionEndpointingOptions>,
    stream_max_segment_millis: u32,
) -> Result<CaptionEndpointingOptions, ApiError> {
    if !matches!(mode, TranscriptionStreamMode::Caption) && raw.is_some() {
        return Err(ApiError::invalid_request(
            "endpointing is only supported when mode is caption",
            Some("endpointing"),
            Some("unsupported_stream_option"),
        ));
    }

    let defaults = CaptionEndpointingOptions::default_with_stream_max_segment_millis(
        stream_max_segment_millis,
    );
    let Some(raw) = raw else {
        return Ok(defaults);
    };
    let options = CaptionEndpointingOptions {
        min_speech_ms: raw.min_speech_ms.unwrap_or(defaults.min_speech_ms),
        min_silence_ms: raw.min_silence_ms.unwrap_or(defaults.min_silence_ms),
        max_segment_ms: defaults.max_segment_ms,
        speech_padding_ms: raw.speech_padding_ms.unwrap_or(defaults.speech_padding_ms),
    };
    validate_caption_endpointing_options(options)?;
    Ok(options)
}

fn validate_caption_endpointing_options(
    options: CaptionEndpointingOptions,
) -> Result<(), ApiError> {
    if options.min_speech_ms == 0 {
        return Err(ApiError::invalid_request(
            "endpointing.min_speech_ms must be greater than zero",
            Some("endpointing.min_speech_ms"),
            Some("invalid_endpointing"),
        ));
    }
    if options.min_silence_ms == 0 {
        return Err(ApiError::invalid_request(
            "endpointing.min_silence_ms must be greater than zero",
            Some("endpointing.min_silence_ms"),
            Some("invalid_endpointing"),
        ));
    }
    if options.max_segment_ms < options.min_speech_ms {
        return Err(ApiError::invalid_request(
            "endpointing.min_speech_ms must not exceed configured stream_max_segment",
            Some("endpointing.min_speech_ms"),
            Some("invalid_endpointing"),
        ));
    }
    let candidate_ms = options
        .speech_padding_ms
        .checked_add(options.min_speech_ms)
        .ok_or_else(|| {
            ApiError::invalid_request(
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms is too large",
                Some("endpointing.speech_padding_ms"),
                Some("invalid_endpointing"),
            )
        })?;
    if candidate_ms > CAPTION_VAD_MAX_CANDIDATE_MS {
        return Err(ApiError::invalid_request(
            "endpointing.speech_padding_ms plus endpointing.min_speech_ms must not exceed 60000",
            Some("endpointing.speech_padding_ms"),
            Some("invalid_endpointing"),
        ));
    }
    let rounded_min_speech_ms = options
        .min_speech_ms
        .div_ceil(CAPTION_VAD_FRAME_DURATION_MS)
        .checked_mul(CAPTION_VAD_FRAME_DURATION_MS)
        .ok_or_else(|| {
            ApiError::invalid_request(
                "endpointing.min_speech_ms is too large",
                Some("endpointing.min_speech_ms"),
                Some("invalid_endpointing"),
            )
        })?;
    if candidate_ms < rounded_min_speech_ms {
        return Err(ApiError::invalid_request(
            "endpointing.speech_padding_ms plus endpointing.min_speech_ms must hold one rounded VAD speech window",
            Some("endpointing.speech_padding_ms"),
            Some("invalid_endpointing"),
        ));
    }
    Ok(())
}

fn validate_caption_streaming_options(options: &AsrStreamingOptions) -> Result<(), ApiError> {
    if !options.chunk_size_sec.is_finite() || options.chunk_size_sec <= 0.0 {
        return Err(ApiError::invalid_request(
            "streaming chunk_size_sec must be finite and greater than zero",
            Some("chunk_size_sec"),
            Some("invalid_chunk_size"),
        ));
    }

    let chunk_size_samples = (options.chunk_size_sec * ASR_SAMPLE_RATE as f32) as usize;
    if chunk_size_samples == 0 {
        return Err(ApiError::invalid_request(
            "streaming chunk_size_sec must produce at least one sample",
            Some("chunk_size_sec"),
            Some("invalid_chunk_size"),
        ));
    }

    if options.max_new_tokens_streaming == 0 {
        return Err(ApiError::invalid_request(
            "streaming max_new_tokens_streaming must be greater than zero",
            Some("max_new_tokens_streaming"),
            Some("invalid_stream_option"),
        ));
    }

    if options.max_new_tokens_final == 0 {
        return Err(ApiError::invalid_request(
            "streaming max_new_tokens_final must be greater than zero",
            Some("max_new_tokens_final"),
            Some("invalid_stream_option"),
        ));
    }

    Ok(())
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
    fn stream_start_defaults_to_legacy_mode() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"mp3"
            }"#,
        )
        .unwrap();

        assert_eq!(start.mode, TranscriptionStreamMode::Legacy);
        assert_eq!(start.endpointing, CaptionEndpointingOptions::default());
    }

    #[test]
    fn stream_start_accepts_caption_mode_with_endpointing_defaults() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000
            }"#,
        )
        .unwrap();

        assert_eq!(start.mode, TranscriptionStreamMode::Caption);
        assert_eq!(
            start.endpointing,
            CaptionEndpointingOptions {
                min_speech_ms: 300,
                min_silence_ms: 500,
                max_segment_ms: 120_000,
                speech_padding_ms: 200,
            }
        );
    }

    #[test]
    fn stream_start_accepts_caption_endpointing_overrides() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{
                    "min_speech_ms":250,
                    "min_silence_ms":700,
                    "speech_padding_ms":160
                }
            }"#,
        )
        .unwrap();

        assert_eq!(start.mode, TranscriptionStreamMode::Caption);
        assert_eq!(start.endpointing.min_speech_ms, 250);
        assert_eq!(start.endpointing.min_silence_ms, 700);
        assert_eq!(start.endpointing.max_segment_ms, 120_000);
        assert_eq!(start.endpointing.speech_padding_ms, 160);
    }

    #[test]
    fn stream_start_rejects_caption_pcm_sample_rate_that_vad_cannot_endpoint() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":48000
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("sample_rate"));
        assert_eq!(error.error.code.as_deref(), Some("unsupported_sample_rate"));
    }

    #[test]
    fn caption_streaming_options_reject_invalid_chunk_size_before_ready() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "chunk_size_sec":0.0
            }"#,
        )
        .unwrap();
        let options = start.to_streaming_options(2.0);

        let error = validate_caption_streaming_options(&options).unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("chunk_size_sec"));
        assert_eq!(error.error.code.as_deref(), Some("invalid_chunk_size"));
    }

    #[test]
    fn caption_streaming_options_reject_zero_streaming_tokens_before_ready() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "max_new_tokens_streaming":0
            }"#,
        )
        .unwrap();
        let options = start.to_streaming_options(2.0);

        let error = validate_caption_streaming_options(&options).unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("max_new_tokens_streaming")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_stream_option"));
    }

    #[test]
    fn caption_streaming_options_reject_zero_final_tokens_before_ready() {
        let start = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "max_new_tokens_final":0
            }"#,
        )
        .unwrap();
        let options = start.to_streaming_options(2.0);

        let error = validate_caption_streaming_options(&options).unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("max_new_tokens_final"));
        assert_eq!(error.error.code.as_deref(), Some("invalid_stream_option"));
    }

    #[test]
    fn stream_start_rejects_unknown_mode() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"sentence",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"mp3"
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("mode"));
        assert_eq!(error.error.code.as_deref(), Some("unsupported_stream_mode"));
    }

    #[test]
    fn stream_start_rejects_endpointing_without_caption_mode() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"mp3",
                "endpointing":{"min_silence_ms":700}
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.param.as_deref(), Some("endpointing"));
        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_stream_option")
        );
    }

    #[test]
    fn stream_start_rejects_zero_endpointing_min_speech() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"min_speech_ms":0}
            }"#,
        )
        .unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("endpointing.min_speech_ms")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_endpointing"));
    }

    #[test]
    fn stream_start_rejects_zero_endpointing_min_silence() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"min_silence_ms":0}
            }"#,
        )
        .unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("endpointing.min_silence_ms")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_endpointing"));
    }

    #[test]
    fn stream_start_rejects_min_speech_above_configured_max_segment() {
        let error = parse_transcription_stream_start_with_stream_max_segment(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"min_speech_ms":300}
            }"#,
            299,
        )
        .unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("endpointing.min_speech_ms")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_endpointing"));
    }

    #[test]
    fn stream_start_rejects_endpointing_max_segment_field() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"max_segment_ms":60001}
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("invalid_json"));
    }

    #[test]
    fn stream_start_rejects_unknown_endpointing_field() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"min_silence":700}
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("invalid_json"));
    }

    #[test]
    fn stream_start_rejects_oversized_endpointing_candidate_window() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"speech_padding_ms":60000}
            }"#,
        )
        .unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("endpointing.speech_padding_ms")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_endpointing"));
    }

    #[test]
    fn stream_start_rejects_endpointing_candidate_that_cannot_hold_rounded_speech_frame() {
        let error = parse_transcription_stream_start(
            r#"{
                "type":"start",
                "mode":"caption",
                "model":"Qwen/Qwen3-ASR-Flash",
                "input_audio_format":"pcm_s16le",
                "sample_rate":16000,
                "endpointing":{"min_speech_ms":21,"speech_padding_ms":0}
            }"#,
        )
        .unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("endpointing.speech_padding_ms")
        );
        assert_eq!(error.error.code.as_deref(), Some("invalid_endpointing"));
    }

    #[test]
    fn stream_start_api_key_does_not_authenticate_without_header() {
        let error = validate_transcription_stream_api_key(Some("secret"), Some("secret"), false)
            .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("invalid_api_key"));
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
    fn stream_start_timeout_error_uses_stable_code() {
        let error = transcription_stream_start_timeout_error();

        assert_eq!(error.error.param.as_deref(), Some("type"));
        assert_eq!(error.error.code.as_deref(), Some("start_message_timeout"));
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
    fn caption_partial_event_contains_segment_id_and_text_only() {
        let event = caption_partial_event(7, "hello");
        let json = serde_json::to_value(event).unwrap();

        assert_eq!(json["type"], "partial");
        assert_eq!(json["segment_id"], 7);
        assert_eq!(json["text"], "hello");
        assert!(json.get("language").is_none());
        assert!(json.get("raw_output").is_none());
    }

    #[test]
    fn caption_segment_final_event_contains_segment_id_text_and_optional_times() {
        let event = caption_segment_final_event(3, "stable text", Some(120), Some(980));
        let json = serde_json::to_value(event).unwrap();

        assert_eq!(json["type"], "segment_final");
        assert_eq!(json["segment_id"], 3);
        assert_eq!(json["text"], "stable text");
        assert_eq!(json["start_ms"], 120);
        assert_eq!(json["end_ms"], 980);
    }

    #[test]
    fn caption_segment_final_event_omits_absent_times() {
        let event = caption_segment_final_event(3, "stable text", None, None);
        let json = serde_json::to_value(event).unwrap();

        assert_eq!(json["type"], "segment_final");
        assert_eq!(json["segment_id"], 3);
        assert_eq!(json["text"], "stable text");
        assert!(json.get("start_ms").is_none());
        assert!(json.get("end_ms").is_none());
    }

    #[test]
    fn caption_completed_event_has_no_transcript_text() {
        let json = serde_json::to_value(caption_completed_event()).unwrap();

        assert_eq!(json["type"], "completed");
        assert!(json.get("text").is_none());
        assert!(json.get("segment_id").is_none());
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

    #[test]
    fn sample_index_to_ms_uses_integer_sample_time() {
        assert_eq!(sample_index_to_ms(16_000, 16_000), 1_000);
        assert_eq!(sample_index_to_ms(16_001, 16_000), 1_000);
        assert_eq!(sample_index_to_ms(47_999, 16_000), 2_999);
    }
}
