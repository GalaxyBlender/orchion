use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub const CAPTION_VAD_FRAME_DURATION_MS: u32 = 30;
pub const CAPTION_VAD_MAX_CANDIDATE_MS: u32 = 60_000;
pub const DEFAULT_CAPTION_MIN_SPEECH_MS: u32 = 300;
pub const DEFAULT_CAPTION_MIN_SILENCE_MS: u32 = 500;
pub const DEFAULT_CAPTION_SPEECH_PADDING_MS: u32 = 200;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrStreamMode {
    Caption,
}

impl AsrStreamMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Caption => "caption",
        }
    }
}

impl FromStr for AsrStreamMode {
    type Err = ParseAsrStreamModeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.trim().eq_ignore_ascii_case("caption") {
            Ok(Self::Caption)
        } else {
            Err(ParseAsrStreamModeError)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseAsrStreamModeError;

impl fmt::Display for ParseAsrStreamModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("unsupported streaming transcription mode")
    }
}

impl std::error::Error for ParseAsrStreamModeError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrStreamInputAudioFormat {
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

impl AsrStreamInputAudioFormat {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::PcmS16Le => "pcm_s16le",
            Self::WebmOpus => "webm_opus",
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::M4a => "m4a",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
        }
    }
}

impl FromStr for AsrStreamInputAudioFormat {
    type Err = ParseAsrStreamInputAudioFormatError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
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
            _ => Err(ParseAsrStreamInputAudioFormatError),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseAsrStreamInputAudioFormatError;

impl fmt::Display for ParseAsrStreamInputAudioFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "unsupported input_audio_format; supported formats are auto, pcm_s16le, webm_opus, mp3, wav, m4a, aac, flac, ogg, and opus",
        )
    }
}

impl std::error::Error for ParseAsrStreamInputAudioFormatError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CaptionEndpointing {
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_padding_ms: u32,
}

impl Default for CaptionEndpointing {
    fn default() -> Self {
        Self {
            min_speech_ms: DEFAULT_CAPTION_MIN_SPEECH_MS,
            min_silence_ms: DEFAULT_CAPTION_MIN_SILENCE_MS,
            speech_padding_ms: DEFAULT_CAPTION_SPEECH_PADDING_MS,
        }
    }
}

impl CaptionEndpointing {
    /// Validates endpointing constraints that are independent of server configuration.
    ///
    /// # Errors
    ///
    /// Returns the violated endpointing constraint.
    pub fn validate(self) -> Result<(), CaptionEndpointingValidationError> {
        if self.min_speech_ms == 0 {
            return Err(CaptionEndpointingValidationError::ZeroMinSpeech);
        }
        if self.min_silence_ms == 0 {
            return Err(CaptionEndpointingValidationError::ZeroMinSilence);
        }
        let candidate_ms = self
            .speech_padding_ms
            .checked_add(self.min_speech_ms)
            .ok_or(CaptionEndpointingValidationError::CandidateOverflow)?;
        if candidate_ms > CAPTION_VAD_MAX_CANDIDATE_MS {
            return Err(CaptionEndpointingValidationError::CandidateTooLong);
        }
        let rounded_min_speech_ms = self
            .min_speech_ms
            .div_ceil(CAPTION_VAD_FRAME_DURATION_MS)
            .checked_mul(CAPTION_VAD_FRAME_DURATION_MS)
            .ok_or(CaptionEndpointingValidationError::MinSpeechOverflow)?;
        if candidate_ms < rounded_min_speech_ms {
            return Err(CaptionEndpointingValidationError::CandidateTooShort);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionEndpointingValidationError {
    ZeroMinSpeech,
    ZeroMinSilence,
    CandidateOverflow,
    CandidateTooLong,
    MinSpeechOverflow,
    CandidateTooShort,
}

impl fmt::Display for CaptionEndpointingValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroMinSpeech => "endpointing.min_speech_ms must be greater than zero",
            Self::ZeroMinSilence => "endpointing.min_silence_ms must be greater than zero",
            Self::CandidateOverflow => {
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms is too large"
            }
            Self::CandidateTooLong => {
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms must not exceed 60000"
            }
            Self::MinSpeechOverflow => "endpointing.min_speech_ms is too large",
            Self::CandidateTooShort => {
                "endpointing.speech_padding_ms plus endpointing.min_speech_ms must hold one rounded VAD speech window"
            }
        })
    }
}

impl std::error::Error for CaptionEndpointingValidationError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptionEndpointingOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_speech_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_silence_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_padding_ms: Option<u32>,
}

impl From<CaptionEndpointing> for CaptionEndpointingOverrides {
    fn from(value: CaptionEndpointing) -> Self {
        Self {
            min_speech_ms: Some(value.min_speech_ms),
            min_silence_ms: Some(value.min_silence_ms),
            speech_padding_ms: Some(value.speech_padding_ms),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrStreamStartMessage {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpointing: Option<CaptionEndpointingOverrides>,
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

impl AsrStreamStartMessage {
    #[must_use]
    pub fn new(model: impl Into<String>, input_audio_format: AsrStreamInputAudioFormat) -> Self {
        Self {
            message_type: Some("start".to_string()),
            mode: None,
            model: Some(model.into()),
            language: None,
            prompt: None,
            api_key: None,
            response_format: Some("json".to_string()),
            input_audio_format: Some(input_audio_format.as_str().to_string()),
            endpointing: None,
            sample_rate: None,
            chunk_size_sec: None,
            unfixed_chunk_num: None,
            unfixed_token_num: None,
            max_new_tokens_streaming: None,
            max_new_tokens_final: None,
        }
    }

    /// Decodes a WebSocket start message while retaining missing fields for boundary validation.
    ///
    /// # Errors
    ///
    /// Returns a JSON decoding error when the message is malformed.
    pub fn from_text(text: &str) -> serde_json::Result<Self> {
        serde_json::from_str(text)
    }

    /// Encodes the start message as WebSocket text.
    ///
    /// # Errors
    ///
    /// Returns a JSON encoding error.
    pub fn to_text(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsrStreamControlMessage {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<String>,
}

impl AsrStreamControlMessage {
    #[must_use]
    pub fn end() -> Self {
        Self {
            message_type: Some("end".to_string()),
        }
    }

    /// Decodes a WebSocket control message while retaining a missing type for boundary validation.
    ///
    /// # Errors
    ///
    /// Returns a JSON decoding error when the message is malformed.
    pub fn from_text(text: &str) -> serde_json::Result<Self> {
        serde_json::from_str(text)
    }

    /// Encodes the control message as WebSocket text.
    ///
    /// # Errors
    ///
    /// Returns a JSON encoding error.
    pub fn to_text(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrStreamEvent {
    Ready,
    Partial {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        segment_id: Option<u64>,
    },
    Final {
        text: String,
    },
    SegmentFinal {
        segment_id: u64,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end_ms: Option<u64>,
    },
    Completed,
    Error {
        error: ErrorObject,
    },
}

impl AsrStreamEvent {
    /// Decodes an ASR event from WebSocket text.
    ///
    /// # Errors
    ///
    /// Returns a JSON decoding error for malformed or unsupported events.
    pub fn from_text(text: &str) -> serde_json::Result<Self> {
        serde_json::from_str(text)
    }

    /// Encodes an ASR event as WebSocket text.
    ///
    /// # Errors
    ///
    /// Returns a JSON encoding error.
    pub fn to_text(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_message_round_trips_all_wire_fields() {
        let mut message =
            AsrStreamStartMessage::new("Qwen/Qwen3-ASR-Flash", AsrStreamInputAudioFormat::PcmS16Le);
        message.mode = Some(AsrStreamMode::Caption.as_str().to_string());
        message.language = Some("zh".to_string());
        message.prompt = Some("context".to_string());
        message.api_key = Some("secret".to_string());
        message.endpointing = Some(CaptionEndpointing::default().into());
        message.sample_rate = Some(16_000);
        message.chunk_size_sec = Some(2.0);
        message.unfixed_chunk_num = Some(2);
        message.unfixed_token_num = Some(5);
        message.max_new_tokens_streaming = Some(32);
        message.max_new_tokens_final = Some(512);

        let text = message.to_text().unwrap();
        let decoded = AsrStreamStartMessage::from_text(&text).unwrap();

        assert_eq!(decoded, message);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap()["type"],
            "start"
        );
    }

    #[test]
    fn control_and_every_event_shape_round_trip() {
        let control = AsrStreamControlMessage::end();
        assert_eq!(
            AsrStreamControlMessage::from_text(&control.to_text().unwrap()).unwrap(),
            control
        );

        let events = [
            AsrStreamEvent::Ready,
            AsrStreamEvent::Partial {
                text: "draft".to_string(),
                segment_id: None,
            },
            AsrStreamEvent::Partial {
                text: "caption".to_string(),
                segment_id: Some(7),
            },
            AsrStreamEvent::Final {
                text: "final".to_string(),
            },
            AsrStreamEvent::SegmentFinal {
                segment_id: 7,
                text: "caption final".to_string(),
                start_ms: Some(100),
                end_ms: Some(900),
            },
            AsrStreamEvent::Completed,
            AsrStreamEvent::Error {
                error: ErrorObject {
                    message: "bad request".to_string(),
                    error_type: "invalid_request_error".to_string(),
                    param: Some("model".to_string()),
                    code: Some("model_not_available".to_string()),
                },
            },
        ];

        for event in events {
            assert_eq!(
                AsrStreamEvent::from_text(&event.to_text().unwrap()).unwrap(),
                event
            );
        }
    }

    #[test]
    fn accepted_audio_format_aliases_normalize_to_canonical_wire_values() {
        let cases = [
            ("PCM", AsrStreamInputAudioFormat::PcmS16Le, "pcm_s16le"),
            ("webm", AsrStreamInputAudioFormat::WebmOpus, "webm_opus"),
            ("opus", AsrStreamInputAudioFormat::Ogg, "ogg"),
        ];

        for (input, format, canonical) in cases {
            assert_eq!(input.parse::<AsrStreamInputAudioFormat>().unwrap(), format);
            assert_eq!(format.as_str(), canonical);
        }
    }
}
