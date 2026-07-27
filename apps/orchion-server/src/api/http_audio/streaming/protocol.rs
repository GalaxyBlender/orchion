use crate::api::openai::{ApiError, TranscriptionFormat};
use crate::application::streaming_transcription::{self, StreamingTranscriptionEvent};
use crate::settings::DEFAULT_ASR_STREAM_MAX_SEGMENT;
use axum::extract::ws::{Message, WebSocket};
use orchion::{
    ASR_SAMPLE_RATE, AsrStreamingOptions, AudioInputFormat, AudioVadMode, AudioVadStreamingConfig,
    StreamingAudioDecoder,
};
use orchion_protocol::{
    AsrStreamControlMessage, AsrStreamEvent, AsrStreamInputAudioFormat, AsrStreamMode,
    AsrStreamStartMessage, CAPTION_VAD_FRAME_DURATION_MS, CaptionEndpointing,
    CaptionEndpointingOverrides, CaptionEndpointingValidationError,
    ErrorObject as ProtocolErrorObject,
};
use std::time::Duration;
use tokio::time::timeout;

const STREAM_ERROR_SEND_TIMEOUT: Duration = Duration::from_secs(1);

pub(super) fn duration_to_millis_u32(duration: Duration, field: &'static str) -> u32 {
    u32::try_from(duration.as_millis())
        .unwrap_or_else(|_| panic!("validated ASR {field} must fit in u32 milliseconds"))
}

pub(super) async fn send_stream_ready(socket: &mut WebSocket) -> Result<(), axum::Error> {
    send_stream_event(socket, &AsrStreamEvent::Ready).await
}

pub(super) async fn send_stream_transcript(
    socket: &mut WebSocket,
    event_type: &'static str,
    transcript: &orchion::AsrTranscript,
) -> Result<(), axum::Error> {
    let event = stream_transcript_event(event_type, transcript);
    send_stream_event(socket, &event).await
}

pub(super) fn stream_transcript_event(
    event_type: &'static str,
    transcript: &orchion::AsrTranscript,
) -> AsrStreamEvent {
    match event_type {
        "partial" => AsrStreamEvent::Partial {
            text: transcript.text.clone(),
            segment_id: None,
        },
        "final" => AsrStreamEvent::Final {
            text: transcript.text.clone(),
        },
        _ => unreachable!("unsupported transcript event type"),
    }
}

pub(super) fn caption_partial_event(segment_id: u64, text: &str) -> AsrStreamEvent {
    AsrStreamEvent::Partial {
        segment_id: Some(segment_id),
        text: text.to_string(),
    }
}

pub(super) fn caption_segment_final_event(
    segment_id: u64,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
) -> AsrStreamEvent {
    AsrStreamEvent::SegmentFinal {
        segment_id,
        text: text.to_string(),
        start_ms,
        end_ms,
    }
}

pub(super) fn caption_completed_event() -> AsrStreamEvent {
    AsrStreamEvent::Completed
}

pub(super) async fn send_stream_error(
    socket: &mut WebSocket,
    error: ApiError,
) -> Result<(), axum::Error> {
    let event = stream_error_event(&error);
    let error_type = error.error.error_type;
    let code = error.error.code.as_deref();
    let param = error.error.param.as_deref();

    if code == Some("internal_error") {
        tracing::error!(
            error_type,
            code,
            param,
            detail = ?error,
            "transcription websocket stream failed"
        );
    }

    match timeout(STREAM_ERROR_SEND_TIMEOUT, send_stream_event(socket, &event)).await {
        Ok(result) => result,
        Err(elapsed) => {
            tracing::warn!(
                error_type,
                code,
                param,
                timeout_ms = STREAM_ERROR_SEND_TIMEOUT.as_millis(),
                "transcription websocket error event send timed out"
            );
            Err(axum::Error::new(elapsed))
        }
    }
}

pub(super) fn stream_error_event(error: &ApiError) -> AsrStreamEvent {
    AsrStreamEvent::Error {
        error: ProtocolErrorObject {
            message: error.error.message.clone(),
            error_type: error.error.error_type.to_string(),
            param: error.error.param.clone(),
            code: error.error.code.clone(),
        },
    }
}

pub(super) async fn send_stream_event(
    socket: &mut WebSocket,
    event: &AsrStreamEvent,
) -> Result<(), axum::Error> {
    let event_type = stream_event_type(event);
    let text = event.to_text().map_err(|error| {
        tracing::error!(
            event_type,
            error = %error,
            "failed to serialize transcription websocket event"
        );
        axum::Error::new(error)
    })?;
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|error| {
            tracing::warn!(
                event_type,
                error = %error,
                "failed to send transcription websocket event"
            );
            error
        })
}

pub(super) async fn send_application_stream_events(
    socket: &mut WebSocket,
    events: Vec<StreamingTranscriptionEvent>,
) -> Result<(), ApiError> {
    for event in events {
        let wire = match event {
            StreamingTranscriptionEvent::Partial { segment_id, text } => {
                caption_partial_event(segment_id, &text)
            }
            StreamingTranscriptionEvent::SegmentFinal {
                segment_id,
                text,
                start_ms,
                end_ms,
            } => caption_segment_final_event(segment_id, &text, Some(start_ms), Some(end_ms)),
            StreamingTranscriptionEvent::Completed => caption_completed_event(),
        };
        send_stream_event(socket, &wire)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    Ok(())
}

fn stream_event_type(event: &AsrStreamEvent) -> &'static str {
    match event {
        AsrStreamEvent::Ready => "ready",
        AsrStreamEvent::Partial { .. } => "partial",
        AsrStreamEvent::Final { .. } => "final",
        AsrStreamEvent::SegmentFinal { .. } => "segment_final",
        AsrStreamEvent::Completed => "completed",
        AsrStreamEvent::Error { .. } => "error",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TranscriptionStreamMode {
    Legacy,
    Caption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_field_names,
    reason = "the ms suffix distinguishes serialized millisecond settings"
)]
pub(super) struct CaptionEndpointingOptions {
    pub(super) min_speech_ms: u32,
    pub(super) min_silence_ms: u32,
    pub(super) max_segment_ms: u32,
    pub(super) speech_padding_ms: u32,
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
        let defaults = CaptionEndpointing::default();
        Self {
            min_speech_ms: defaults.min_speech_ms,
            min_silence_ms: defaults.min_silence_ms,
            max_segment_ms: stream_max_segment_millis,
            speech_padding_ms: defaults.speech_padding_ms,
        }
    }

    pub(super) fn to_vad_config(self) -> AudioVadStreamingConfig {
        AudioVadStreamingConfig {
            frame_duration_ms: CAPTION_VAD_FRAME_DURATION_MS,
            min_speech_ms: self.min_speech_ms,
            min_silence_ms: self.min_silence_ms,
            max_segment_ms: self.max_segment_ms,
            speech_padding_ms: self.speech_padding_ms,
            mode: AudioVadMode::Quality.into(),
        }
    }
}

fn max_audio_samples_at_rate(duration: Duration, sample_rate: u32) -> Result<usize, ApiError> {
    let samples = duration
        .as_millis()
        .checked_mul(u128::from(sample_rate))
        .and_then(|samples| samples.checked_add(999))
        .map(|samples| samples / 1000)
        .ok_or_else(|| ApiError::internal("configured audio duration is too large"))?;
    usize::try_from(samples)
        .map_err(|_| ApiError::internal("configured audio duration is too large"))
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TranscriptionStreamStart {
    pub(super) model: String,
    pub(super) language: Option<String>,
    pub(super) prompt: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) response_format: TranscriptionFormat,
    pub(super) mode: TranscriptionStreamMode,
    pub(super) endpointing: CaptionEndpointingOptions,
    pub(super) input_audio_format: AsrStreamInputAudioFormat,
    pub(super) sample_rate: Option<u32>,
    chunk_size_sec: Option<f32>,
    unfixed_chunk_num: Option<usize>,
    unfixed_token_num: Option<usize>,
    max_new_tokens_streaming: Option<usize>,
    max_new_tokens_final: Option<usize>,
}

impl TranscriptionStreamStart {
    pub(super) fn to_streaming_options(&self, default_chunk_size_sec: f32) -> AsrStreamingOptions {
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

    pub(super) async fn audio_decoder(
        &self,
        max_duration: Duration,
    ) -> Result<StreamingAudioDecoder, ApiError> {
        let output_sample_rate = if self.input_audio_format == AsrStreamInputAudioFormat::PcmS16Le {
            self.sample_rate.unwrap_or(ASR_SAMPLE_RATE)
        } else {
            ASR_SAMPLE_RATE
        };
        let max_output_samples = max_audio_samples_at_rate(max_duration, output_sample_rate)?;
        StreamingAudioDecoder::new_for_asr_with_max_samples(
            to_runtime_audio_input_format(self.input_audio_format),
            self.sample_rate,
            max_output_samples,
        )
        .await
        .map_err(ApiError::from)
    }
}

fn to_runtime_audio_input_format(format: AsrStreamInputAudioFormat) -> AudioInputFormat {
    match format {
        AsrStreamInputAudioFormat::Auto => AudioInputFormat::Auto,
        AsrStreamInputAudioFormat::PcmS16Le => AudioInputFormat::PcmS16Le,
        AsrStreamInputAudioFormat::WebmOpus => AudioInputFormat::WebmOpus,
        AsrStreamInputAudioFormat::Mp3 => AudioInputFormat::Mp3,
        AsrStreamInputAudioFormat::Wav => AudioInputFormat::Wav,
        AsrStreamInputAudioFormat::M4a => AudioInputFormat::M4a,
        AsrStreamInputAudioFormat::Aac => AudioInputFormat::Aac,
        AsrStreamInputAudioFormat::Flac => AudioInputFormat::Flac,
        AsrStreamInputAudioFormat::Ogg => AudioInputFormat::Ogg,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TranscriptionStreamControl {
    Start,
    End,
}

#[cfg(test)]
pub(super) fn parse_transcription_stream_start(
    text: &str,
) -> Result<TranscriptionStreamStart, ApiError> {
    parse_transcription_stream_start_with_stream_max_segment(
        text,
        duration_to_millis_u32(DEFAULT_ASR_STREAM_MAX_SEGMENT, "stream_max_segment"),
    )
}

pub(super) fn parse_transcription_stream_start_with_stream_max_segment(
    text: &str,
    stream_max_segment_millis: u32,
) -> Result<TranscriptionStreamStart, ApiError> {
    let raw = AsrStreamStartMessage::from_text(text).map_err(|error| {
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
    let input_audio_format = input_audio_format
        .parse::<AsrStreamInputAudioFormat>()
        .map_err(|error| {
            ApiError::invalid_request(
                error.to_string(),
                Some("input_audio_format"),
                Some("unsupported_audio_format"),
            )
        })?;
    if matches!(input_audio_format, AsrStreamInputAudioFormat::PcmS16Le)
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
    mode.parse::<AsrStreamMode>()
        .map(|AsrStreamMode::Caption| TranscriptionStreamMode::Caption)
        .map_err(|error| {
            ApiError::invalid_request(
                error.to_string(),
                Some("mode"),
                Some("unsupported_stream_mode"),
            )
        })
}

fn validate_caption_pcm_sample_rate(
    mode: TranscriptionStreamMode,
    input_audio_format: AsrStreamInputAudioFormat,
    sample_rate: Option<u32>,
) -> Result<(), ApiError> {
    if matches!(mode, TranscriptionStreamMode::Caption)
        && matches!(input_audio_format, AsrStreamInputAudioFormat::PcmS16Le)
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
    raw: Option<CaptionEndpointingOverrides>,
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
    if options.max_segment_ms < options.min_speech_ms {
        return Err(ApiError::invalid_request(
            "endpointing.min_speech_ms must not exceed configured stream_max_segment",
            Some("endpointing.min_speech_ms"),
            Some("invalid_endpointing"),
        ));
    }
    CaptionEndpointing {
        min_speech_ms: options.min_speech_ms,
        min_silence_ms: options.min_silence_ms,
        speech_padding_ms: options.speech_padding_ms,
    }
    .validate()
    .map_err(caption_endpointing_validation_error)
}

fn caption_endpointing_validation_error(error: CaptionEndpointingValidationError) -> ApiError {
    let param = match error {
        CaptionEndpointingValidationError::ZeroMinSpeech
        | CaptionEndpointingValidationError::MinSpeechOverflow => "endpointing.min_speech_ms",
        CaptionEndpointingValidationError::ZeroMinSilence => "endpointing.min_silence_ms",
        CaptionEndpointingValidationError::CandidateOverflow
        | CaptionEndpointingValidationError::CandidateTooLong
        | CaptionEndpointingValidationError::CandidateTooShort => "endpointing.speech_padding_ms",
    };
    ApiError::invalid_request(error.to_string(), Some(param), Some("invalid_endpointing"))
}

pub(super) fn validate_transcription_streaming_options(
    options: &AsrStreamingOptions,
) -> Result<(), ApiError> {
    streaming_transcription::validate_streaming_options(options).map_err(ApiError::from)
}

pub(super) fn parse_transcription_stream_control(
    text: &str,
) -> Result<TranscriptionStreamControl, ApiError> {
    let raw = AsrStreamControlMessage::from_text(text).map_err(|error| {
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
