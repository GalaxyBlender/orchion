use crate::api::http_shared::authorize;
use crate::api::openai::ApiError;
use crate::application::streaming_transcription::{
    self, AsrPcmBuffer, LeasedAsrModel, LeasedAsrStream, TranscriptionStreamBudget,
    TranscriptionStreamLimits,
};
use crate::application::{RuntimeError, ServerApplication, UseCaseError};
use crate::settings::parse_asr_model;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use axum::response::Response;
use orchion::{AudioVadStreamingEndpoint, StreamingAudioDecoder};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

#[cfg(test)]
use crate::api::openai::TranscriptionFormat;
#[cfg(test)]
use orchion_protocol::AsrStreamInputAudioFormat;

mod caption;
mod legacy;
mod protocol;

#[allow(
    clippy::wildcard_imports,
    reason = "the protocol module defines this stream implementation's internal vocabulary"
)]
use protocol::*;

const TRANSCRIPTION_STREAM_START_TIMEOUT: Duration = Duration::from_secs(10);

async fn await_stream_operation<T>(
    budget: &TranscriptionStreamBudget,
    limits: TranscriptionStreamLimits,
    operation: impl std::future::Future<Output = T>,
) -> Result<T, ApiError> {
    streaming_transcription::await_stream_operation(budget, limits, operation)
        .await
        .map_err(ApiError::from)
}

async fn await_stream_finish<T>(
    remaining: Result<Duration, UseCaseError>,
    limits: TranscriptionStreamLimits,
    mode: &'static str,
    operation: impl std::future::Future<Output = Result<T, ApiError>>,
) -> Result<T, ApiError> {
    let deadline_was_exhausted = remaining.is_err();
    match streaming_transcription::await_stream_finish(remaining, operation).await {
        Ok(result) => result,
        Err(error) => {
            if deadline_was_exhausted {
                tracing::warn!(
                    mode,
                    max_duration_ms = limits.max_duration.as_millis(),
                    "transcription websocket finish deadline was already exhausted"
                );
            } else {
                tracing::warn!(
                    mode,
                    max_duration_ms = limits.max_duration.as_millis(),
                    "transcription websocket finish timed out"
                );
            }
            Err(ApiError::from(error))
        }
    }
}

pub(in crate::api) async fn create_transcription_ws<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let header_authorized = if headers.contains_key(AUTHORIZATION) {
        authorize(state.as_ref(), &headers)?;
        true
    } else {
        state.api_policy().api_key.is_none()
    };
    let pending_connection_permit = state
        .try_acquire_pending_websocket()
        .ok_or_else(|| ApiError::resource_exhausted("pending websocket connection"))?;
    let max_message_size = state.api_policy().max_websocket_message_size;
    Ok(ws
        .max_message_size(max_message_size)
        .max_frame_size(max_message_size)
        .on_upgrade(move |socket| {
            handle_transcription_ws(socket, state, header_authorized, pending_connection_permit)
        }))
}

#[allow(
    clippy::too_many_lines,
    reason = "authentication and resource transitions are one websocket state machine"
)]
async fn handle_transcription_ws<S>(
    mut socket: WebSocket,
    state: Arc<S>,
    header_authorized: bool,
    pending_connection_permit: tokio::sync::OwnedSemaphorePermit,
) where
    S: ServerApplication,
{
    let api_key = state.api_policy().api_key.clone();
    let max_input_bytes = state.api_policy().max_upload_size;
    let asr_policy = state
        .api_policy()
        .asr
        .clone()
        .expect("ASR websocket route requires an active ASR policy");
    let stream_target_segment_millis =
        duration_to_millis_u32(asr_policy.stream_target_segment, "stream_target_segment");
    let stream_max_segment_millis =
        duration_to_millis_u32(asr_policy.stream_max_segment, "stream_max_segment");
    let start =
        match receive_transcription_stream_start(&mut socket, stream_max_segment_millis).await {
            Ok(start) => start,
            Err(error) => {
                let _ = send_stream_error(&mut socket, error).await;
                return;
            }
        };
    if let Err(error) = validate_transcription_stream_api_key(
        api_key.as_deref(),
        start.api_key.as_deref(),
        header_authorized,
    ) {
        let _ = send_stream_error(&mut socket, error).await;
        return;
    }
    let Some(_connection_permit) = state.try_acquire_websocket() else {
        let _ = send_stream_error(
            &mut socket,
            ApiError::resource_exhausted("websocket connection"),
        )
        .await;
        return;
    };
    drop(pending_connection_permit);
    let limits = TranscriptionStreamLimits {
        idle_timeout: asr_policy.stream_idle_timeout,
        max_duration: asr_policy.stream_max_duration,
        max_input_bytes,
    };
    let budget = TranscriptionStreamBudget::new();
    let default_chunk_size_sec = asr_policy.stream_chunk_size;
    if let Err(error) = validate_transcription_streaming_options(
        &start.to_streaming_options(default_chunk_size_sec),
    ) {
        let _ = send_stream_error(&mut socket, error).await;
        return;
    }
    let model = start.model.clone();
    let Ok(requested) = parse_asr_model(&model) else {
        let _ = send_stream_error(&mut socket, ApiError::model_not_available(&model)).await;
        return;
    };
    let asr = match await_stream_operation(
        &budget,
        limits,
        load_stream_model(Arc::clone(&state), requested),
    )
    .await
    {
        Ok(Ok(Some(asr))) => asr,
        Ok(Ok(None)) => {
            let _ = send_stream_error(&mut socket, ApiError::model_not_available(&model)).await;
            return;
        }
        Ok(Err(error)) => {
            let _ = send_stream_error(&mut socket, ApiError::from(UseCaseError::from(error))).await;
            return;
        }
        Err(error) => {
            let _ = send_stream_error(&mut socket, error).await;
            return;
        }
    };
    match start.mode {
        TranscriptionStreamMode::Legacy => {
            legacy::run(socket, start, asr, default_chunk_size_sec, limits, budget).await;
        }
        TranscriptionStreamMode::Caption => {
            caption::run(
                socket,
                start,
                asr,
                default_chunk_size_sec,
                stream_target_segment_millis,
                limits,
                budget,
            )
            .await;
        }
    }
}

async fn load_stream_model<S>(
    state: Arc<S>,
    requested: orchion::AsrModel,
) -> Result<Option<LeasedAsrModel>, RuntimeError>
where
    S: ServerApplication,
{
    tokio::spawn(async move { state.lease_streaming_model(requested).await })
        .await
        .map_err(|error| {
            RuntimeError::Internal(format!("stream model load task failed: {error:#}"))
        })?
}

async fn receive_transcription_stream_message(
    socket: &mut WebSocket,
    budget: &mut TranscriptionStreamBudget,
    limits: TranscriptionStreamLimits,
) -> Result<Option<Message>, ApiError> {
    let wait = budget.next_wait(limits)?;
    let message = timeout(wait, socket.recv())
        .await
        .map_err(|_| budget.timeout_error(limits))?;
    let Some(message) = message else {
        return Ok(None);
    };
    let message = message.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_websocket_message"))
    })?;
    if let Message::Binary(bytes) = &message {
        budget.record_binary_input(bytes.len(), limits)?;
    }
    Ok(Some(message))
}

fn transcription_stream_decoder_error(
    error: orchion::OrchionError,
    limits: TranscriptionStreamLimits,
) -> ApiError {
    ApiError::from(streaming_transcription::stream_decoder_error(error, limits))
}

fn validate_transcription_stream_api_key(
    required_api_key: Option<&str>,
    message_api_key: Option<&str>,
    header_authorized: bool,
) -> Result<(), ApiError> {
    if required_api_key.is_none() {
        return Ok(());
    }
    if header_authorized || message_api_key == required_api_key {
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
        assert_eq!(start.input_audio_format, AsrStreamInputAudioFormat::Mp3);
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

        assert_eq!(start.input_audio_format, AsrStreamInputAudioFormat::Wav);
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

        assert_eq!(start.input_audio_format, AsrStreamInputAudioFormat::Auto);
        assert_eq!(start.sample_rate, None);
    }

    #[test]
    fn stream_start_accepts_additional_file_input_formats() {
        let cases = [
            ("m4a", AsrStreamInputAudioFormat::M4a),
            ("aac", AsrStreamInputAudioFormat::Aac),
            ("flac", AsrStreamInputAudioFormat::Flac),
            ("ogg", AsrStreamInputAudioFormat::Ogg),
            ("opus", AsrStreamInputAudioFormat::Ogg),
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

        let error = validate_transcription_streaming_options(&options).unwrap_err();

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

        let error = validate_transcription_streaming_options(&options).unwrap_err();

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

        let error = validate_transcription_streaming_options(&options).unwrap_err();

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
    fn stream_start_api_key_authenticates_without_header() {
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
    fn stream_start_timeout_error_uses_stable_code() {
        let error = transcription_stream_start_timeout_error();

        assert_eq!(error.error.param.as_deref(), Some("type"));
        assert_eq!(error.error.code.as_deref(), Some("start_message_timeout"));
    }

    #[test]
    fn internal_stream_error_event_does_not_expose_runtime_detail() {
        let event = stream_error_event(&ApiError::internal(
            "sensitive runtime detail: /private/model/path",
        ));
        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains("internal server error"));
        assert!(json.contains("internal_error"));
        assert!(!json.contains("sensitive runtime detail"));
        assert!(!json.contains("/private/model/path"));
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
}
