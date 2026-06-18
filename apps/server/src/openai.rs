use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use orchion::{TtsLanguage, TtsSpeaker, TtsVoice};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;
use utoipa::ToSchema;

#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    pub error: ErrorObject,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorObject,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ApiError {
    #[must_use]
    pub fn invalid_request(
        message: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error: ErrorObject {
                message: message.into(),
                error_type: "invalid_request_error",
                param: param.map(ToOwned::to_owned),
                code: code.map(ToOwned::to_owned),
            },
        }
    }

    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error: ErrorObject {
                message: message.into(),
                error_type: "server_error",
                param: None,
                code: Some("internal_error".to_string()),
            },
        }
    }

    #[must_use]
    pub fn model_not_loaded(model: &str) -> Self {
        Self::invalid_request(
            format!("model `{model}` is not loaded by this server"),
            Some("model"),
            Some("model_not_loaded"),
        )
    }

    #[must_use]
    pub fn into_status_body(self) -> (StatusCode, ErrorBody) {
        (self.status, ErrorBody { error: self.error })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, body) = self.into_status_body();
        (status, Json(body)).into_response()
    }
}

impl From<orchion::OrchionError> for ApiError {
    fn from(error: orchion::OrchionError) -> Self {
        match error {
            orchion::OrchionError::UnsupportedCapability { capability, .. } => {
                Self::invalid_request(
                    format!("selected model does not support {capability}"),
                    Some("voice"),
                    Some("unsupported_voice"),
                )
            }
            other => Self::internal(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFormat {
    Wav,
}

impl Default for SpeechFormat {
    fn default() -> Self {
        Self::Wav
    }
}

impl TryFrom<&str> for SpeechFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wav" => Ok(Self::Wav),
            _ => Err(ApiError::invalid_request(
                "unsupported audio format; only `wav` is currently supported",
                Some("response_format"),
                Some("unsupported_audio_format"),
            )),
        }
    }
}

impl<'de> Deserialize<'de> for SpeechFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_from(value.as_str()).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.error.message)
    }
}

impl std::error::Error for ApiError {}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(default)]
    pub response_format: SpeechFormat,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub reference_audio: Option<String>,
    #[serde(default)]
    pub reference_text: Option<String>,
    #[serde(default)]
    pub voice_prompt: Option<String>,
}

impl SpeechRequest {
    pub fn to_tts_voice(&self) -> Result<TtsVoice, ApiError> {
        let language = self
            .language
            .as_deref()
            .map(parse_language)
            .transpose()?
            .unwrap_or(TtsLanguage::English);
        match normalize_identifier(&self.voice).as_str() {
            "clone" => {
                let reference_audio = self.reference_audio.as_ref().ok_or_else(|| {
                    ApiError::invalid_request(
                        "voice clone requires `reference_audio`",
                        Some("reference_audio"),
                        Some("missing_required_parameter"),
                    )
                })?;
                let reference_text = self.reference_text.as_ref().ok_or_else(|| {
                    ApiError::invalid_request(
                        "voice clone requires `reference_text`",
                        Some("reference_text"),
                        Some("missing_required_parameter"),
                    )
                })?;
                Ok(TtsVoice::Clone {
                    reference_audio: PathBuf::from(reference_audio),
                    reference_text: reference_text.clone(),
                    language,
                })
            }
            "design" => {
                let prompt = self.voice_prompt.as_ref().ok_or_else(|| {
                    ApiError::invalid_request(
                        "voice design requires `voice_prompt`",
                        Some("voice_prompt"),
                        Some("missing_required_parameter"),
                    )
                })?;
                Ok(TtsVoice::Design {
                    prompt: prompt.clone(),
                    language,
                })
            }
            voice => Ok(TtsVoice::Preset {
                speaker: parse_speaker(voice)?,
                language,
            }),
        }
    }

    pub fn validate(&self) -> Result<(), ApiError> {
        if self.input.trim().is_empty() {
            return Err(ApiError::invalid_request(
                "`input` must not be empty",
                Some("input"),
                Some("empty_input"),
            ));
        }
        if (self.speed - 1.0).abs() > f32::EPSILON {
            return Err(ApiError::invalid_request(
                "`speed` values other than 1.0 are not currently supported",
                Some("speed"),
                Some("unsupported_speed"),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionFormat {
    Json,
    Text,
    VerboseJson,
}

impl Default for TranscriptionFormat {
    fn default() -> Self {
        Self::Json
    }
}

impl TryFrom<&str> for TranscriptionFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            _ => Err(ApiError::invalid_request(
                "unsupported transcription response format; supported formats are json, text, and verbose_json",
                Some("response_format"),
                Some("unsupported_response_format"),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TranscriptionJson {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TranscriptionVerboseJson {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

#[must_use]
pub fn content_type_for(format: SpeechFormat) -> &'static str {
    match format {
        SpeechFormat::Wav => "audio/wav",
    }
}

fn default_speed() -> f32 {
    1.0
}

fn parse_speaker(value: &str) -> Result<TtsSpeaker, ApiError> {
    match normalize_identifier(value).as_str() {
        "serena" => Ok(TtsSpeaker::Serena),
        "vivian" => Ok(TtsSpeaker::Vivian),
        "uncle-fu" | "unclefu" => Ok(TtsSpeaker::UncleFu),
        "ryan" => Ok(TtsSpeaker::Ryan),
        "aiden" => Ok(TtsSpeaker::Aiden),
        "ono-anna" | "onoanna" => Ok(TtsSpeaker::OnoAnna),
        "sohee" => Ok(TtsSpeaker::Sohee),
        "eric" => Ok(TtsSpeaker::Eric),
        "dylan" => Ok(TtsSpeaker::Dylan),
        _ => Err(ApiError::invalid_request(
            format!("unsupported voice `{value}`"),
            Some("voice"),
            Some("unsupported_voice"),
        )),
    }
}

fn parse_language(value: &str) -> Result<TtsLanguage, ApiError> {
    match normalize_identifier(value).as_str() {
        "english" | "en" => Ok(TtsLanguage::English),
        "chinese" | "zh" => Ok(TtsLanguage::Chinese),
        "japanese" | "ja" => Ok(TtsLanguage::Japanese),
        "korean" | "ko" => Ok(TtsLanguage::Korean),
        "german" | "de" => Ok(TtsLanguage::German),
        "french" | "fr" => Ok(TtsLanguage::French),
        "russian" | "ru" => Ok(TtsLanguage::Russian),
        "portuguese" | "pt" => Ok(TtsLanguage::Portuguese),
        "spanish" | "es" => Ok(TtsLanguage::Spanish),
        "italian" | "it" => Ok(TtsLanguage::Italian),
        _ => Err(ApiError::invalid_request(
            format!("unsupported language `{value}`"),
            Some("language"),
            Some("unsupported_language"),
        )),
    }
}

fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}
