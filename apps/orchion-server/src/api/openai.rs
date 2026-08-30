use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use orchion::{
    AsrSegment, AudioOutputFormat, ModelCapabilities, OcrResponseFormat, TtsOptions, TtsVoice,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;
use utoipa::ToSchema;

#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    pub error: ErrorObject,
    log_message: Option<String>,
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
    #[schema(required = true, nullable)]
    pub param: Option<String>,
    #[schema(required = true, nullable)]
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
            log_message: None,
        }
    }

    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error: ErrorObject {
                message: "internal server error".to_string(),
                error_type: "server_error",
                param: None,
                code: Some("internal_error".to_string()),
            },
            log_message: Some(message.into()),
        }
    }

    #[must_use]
    pub fn invalid_api_key() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            error: ErrorObject {
                message: "invalid API key".to_string(),
                error_type: "invalid_request_error",
                param: None,
                code: Some("invalid_api_key".to_string()),
            },
            log_message: None,
        }
    }

    #[must_use]
    pub fn forbidden_origin() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            error: ErrorObject {
                message: "origin is not allowed".to_string(),
                error_type: "invalid_request_error",
                param: None,
                code: Some("cors_origin_forbidden".to_string()),
            },
            log_message: None,
        }
    }

    #[must_use]
    pub fn resource_exhausted(resource: &'static str) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            error: ErrorObject {
                message: format!("{resource} capacity is currently exhausted"),
                error_type: "rate_limit_error",
                param: None,
                code: Some("resource_exhausted".to_string()),
            },
            log_message: None,
        }
    }

    #[must_use]
    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::REQUEST_TIMEOUT,
            error: ErrorObject {
                message: message.into(),
                error_type: "invalid_request_error",
                param: None,
                code: Some("request_timeout".to_string()),
            },
            log_message: None,
        }
    }

    #[must_use]
    pub fn shutting_down() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            error: ErrorObject {
                message: "server is shutting down".to_string(),
                error_type: "server_error",
                param: None,
                code: Some("server_shutdown".to_string()),
            },
            log_message: None,
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
    pub fn model_not_available(model: &str) -> Self {
        Self::invalid_request(
            format!("model `{model}` is not available on this server"),
            Some("model"),
            Some("model_not_available"),
        )
    }

    #[must_use]
    pub fn into_status_body(self) -> (StatusCode, ErrorBody) {
        (self.status, ErrorBody { error: self.error })
    }

    #[must_use]
    pub(crate) const fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) fn activity_error(&self) -> crate::api::activity::ActivityError {
        crate::api::activity::ActivityError {
            code: self.error.code.clone(),
            message: self
                .status
                .is_server_error()
                .then(|| self.error.message.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Asr,
    Tts,
    Ocr,
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    AsrTranscription,
    AsrStreaming,
    TtsVoiceCloning,
    TtsPresetSpeakers,
    TtsVoiceDesign,
    OcrText,
    OcrLayout,
    OcrTableStructure,
    OcrVisionLanguage,
    OcrMarkdown,
    OcrHtml,
    LlmChat,
    LlmResponses,
    LlmStreaming,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    #[serde(rename = "type")]
    pub model_type: ModelType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub capabilities: Vec<ModelCapability>,
}

impl ModelObject {
    pub fn new(
        id: impl Into<String>,
        name: Option<String>,
        model_type: ModelType,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            id: id.into(),
            object: "model",
            created: 0,
            owned_by: "orchion",
            model_type,
            name,
            capabilities: ModelCapability::from_core(capabilities),
        }
    }
}

impl ModelCapability {
    fn from_core(capabilities: ModelCapabilities) -> Vec<Self> {
        [
            (ModelCapabilities::ASR_TRANSCRIPTION, Self::AsrTranscription),
            (ModelCapabilities::ASR_STREAMING, Self::AsrStreaming),
            (ModelCapabilities::TTS_VOICE_CLONING, Self::TtsVoiceCloning),
            (
                ModelCapabilities::TTS_PRESET_SPEAKERS,
                Self::TtsPresetSpeakers,
            ),
            (ModelCapabilities::TTS_VOICE_DESIGN, Self::TtsVoiceDesign),
            (ModelCapabilities::OCR_TEXT, Self::OcrText),
            (ModelCapabilities::OCR_LAYOUT, Self::OcrLayout),
            (
                ModelCapabilities::OCR_TABLE_STRUCTURE,
                Self::OcrTableStructure,
            ),
            (
                ModelCapabilities::OCR_VISION_LANGUAGE,
                Self::OcrVisionLanguage,
            ),
            (ModelCapabilities::OCR_MARKDOWN, Self::OcrMarkdown),
            (ModelCapabilities::OCR_HTML, Self::OcrHtml),
            (ModelCapabilities::LLM_CHAT, Self::LlmChat),
            (ModelCapabilities::LLM_RESPONSES, Self::LlmResponses),
            (ModelCapabilities::LLM_STREAMING, Self::LlmStreaming),
        ]
        .into_iter()
        .filter_map(|(capability, value)| capabilities.contains(capability).then_some(value))
        .collect()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let activity_error = self.activity_error();
        let log_message = self.log_message.clone();
        let (status, body) = self.into_status_body();
        if status.is_server_error() {
            tracing::error!(
                %status,
                error_type = body.error.error_type,
                code = ?body.error.code,
                param = ?body.error.param,
                message = %body.error.message,
                detail = ?log_message,
                "request failed"
            );
        } else {
            tracing::debug!(
                %status,
                error_type = body.error.error_type,
                code = ?body.error.code,
                param = ?body.error.param,
                message = %body.error.message,
                "request rejected"
            );
        }
        let mut response = (status, Json(body)).into_response();
        response.extensions_mut().insert(activity_error);
        response
    }
}

impl From<orchion::OrchionError> for ApiError {
    fn from(error: orchion::OrchionError) -> Self {
        match error {
            orchion::OrchionError::UnsupportedCapability { capability, .. } => {
                Self::invalid_request(
                    format!("selected model does not support {capability}"),
                    Some("model"),
                    Some("unsupported_capability"),
                )
            }
            orchion::OrchionError::InvalidAudio { reason } => {
                Self::invalid_request(reason, None, Some("invalid_audio"))
            }
            orchion::OrchionError::InvalidImage { reason }
            | orchion::OrchionError::InvalidDocument { reason } => {
                Self::invalid_request(reason, Some("file"), Some("invalid_file"))
            }
            error @ orchion::OrchionError::LlmContextLimit { .. } => Self::invalid_request(
                error.to_string(),
                Some("input"),
                Some("context_length_exceeded"),
            ),
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<crate::application::UseCaseError> for ApiError {
    fn from(error: crate::application::UseCaseError) -> Self {
        use crate::application::UseCaseError;

        match error {
            UseCaseError::InvalidRequest {
                message,
                param,
                code,
            } => Self::invalid_request(message, param, Some(code)),
            UseCaseError::ModelNotAvailable(model) => Self::model_not_available(&model),
            UseCaseError::ResourceExhausted(resource) => Self::resource_exhausted(resource),
            UseCaseError::Timeout(message) => Self::timeout(message),
            UseCaseError::ShuttingDown => Self::shutting_down(),
            UseCaseError::Core(error) => Self::from(error),
            UseCaseError::ReferenceAudio(orchion::OrchionError::InvalidAudio { reason }) => {
                Self::invalid_request(reason, Some("reference_audio"), Some("invalid_audio"))
            }
            UseCaseError::ReferenceAudio(error) => Self::internal(error.to_string()),
            UseCaseError::Internal(message) => Self::internal(message),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFormat {
    #[default]
    Wav,
    Mp3,
    Aac,
    Opus,
    Flac,
    Pcm,
}

impl std::fmt::Display for SpeechFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Opus => "opus",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        })
    }
}

impl TryFrom<&str> for SpeechFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "aac" => Ok(Self::Aac),
            "opus" => Ok(Self::Opus),
            "flac" => Ok(Self::Flac),
            "pcm" => Ok(Self::Pcm),
            _ => Err(ApiError::invalid_request(
                "unsupported audio format; supported formats are wav, mp3, aac, opus, flac, and pcm",
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
    pub response_format: Option<SpeechFormat>,
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
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub max_length: Option<usize>,
}

impl SpeechRequest {
    #[must_use]
    pub fn to_tts_options(&self) -> TtsOptions {
        crate::application::speech::to_tts_options(&self.application_command())
    }

    /// # Errors
    ///
    /// Returns [`ApiError`] when the requested voice configuration is invalid.
    pub fn to_tts_voice(&self) -> Result<TtsVoice, ApiError> {
        crate::application::speech::to_tts_voice(&self.application_command())
            .map_err(ApiError::from)
    }

    #[must_use]
    pub fn is_voice_clone(&self) -> bool {
        normalize_identifier(&self.voice) == "clone"
    }

    /// # Errors
    ///
    /// Returns [`ApiError`] when any speech request parameter is invalid.
    pub fn validate(&self) -> Result<(), ApiError> {
        crate::application::speech::validate(&self.application_command()).map_err(ApiError::from)
    }

    fn application_command(&self) -> crate::application::speech::SpeechCommand {
        crate::application::speech::SpeechCommand {
            model: self.model.clone(),
            input: self.input.clone(),
            voice: self.voice.clone(),
            output_format: self.response_format.map(AudioOutputFormat::from),
            speed: self.speed,
            language: self.language.clone(),
            reference_audio: self.reference_audio.as_deref().map(PathBuf::from),
            reference_audio_output: None,
            reference_text: self.reference_text.clone(),
            voice_prompt: self.voice_prompt.clone(),
            seed: self.seed,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            max_length: self.max_length,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranscriptionFormat {
    #[default]
    Json,
    Text,
    VerboseJson,
    Srt,
}

impl TryFrom<&str> for TranscriptionFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            _ => Err(ApiError::invalid_request(
                "unsupported transcription response format; supported formats are json, text, verbose_json, and srt",
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
    pub duration: f64,
    pub raw_output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<AsrSegment>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OcrApiFormat {
    #[default]
    Json,
    Text,
    Markdown,
    Html,
}

impl TryFrom<&str> for OcrApiFormat {
    type Error = ApiError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "markdown" => Ok(Self::Markdown),
            "html" => Ok(Self::Html),
            _ => Err(ApiError::invalid_request(
                "unsupported OCR response format; supported formats are json, text, markdown, and html",
                Some("response_format"),
                Some("unsupported_response_format"),
            )),
        }
    }
}

impl From<OcrApiFormat> for OcrResponseFormat {
    fn from(format: OcrApiFormat) -> Self {
        match format {
            OcrApiFormat::Json => Self::Json,
            OcrApiFormat::Text => Self::Text,
            OcrApiFormat::Markdown => Self::Markdown,
            OcrApiFormat::Html => Self::Html,
        }
    }
}

impl From<OcrResponseFormat> for OcrApiFormat {
    fn from(format: OcrResponseFormat) -> Self {
        match format {
            OcrResponseFormat::Json => Self::Json,
            OcrResponseFormat::Text => Self::Text,
            OcrResponseFormat::Markdown => Self::Markdown,
            OcrResponseFormat::Html => Self::Html,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OcrJsonResponse {
    pub model: String,
    pub format: OcrApiFormat,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub html: Option<String>,
    pub regions: Vec<orchion::OcrRegion>,
    pub layout_blocks: Vec<orchion::OcrLayoutBlock>,
    pub usage: orchion::OcrUsage,
}

#[must_use]
pub fn content_type_for(format: SpeechFormat) -> &'static str {
    AudioOutputFormat::from(format).content_type()
}

impl From<SpeechFormat> for AudioOutputFormat {
    fn from(format: SpeechFormat) -> Self {
        match format {
            SpeechFormat::Wav => Self::Wav,
            SpeechFormat::Mp3 => Self::Mp3,
            SpeechFormat::Aac => Self::Aac,
            SpeechFormat::Opus => Self::Opus,
            SpeechFormat::Flac => Self::Flac,
            SpeechFormat::Pcm => Self::Pcm,
        }
    }
}

fn default_speed() -> f32 {
    1.0
}

fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocr_response_format_accepts_json_text_markdown_html() {
        assert_eq!(OcrApiFormat::try_from("json").unwrap(), OcrApiFormat::Json);
        assert_eq!(OcrApiFormat::try_from("text").unwrap(), OcrApiFormat::Text);
        assert_eq!(
            OcrApiFormat::try_from("markdown").unwrap(),
            OcrApiFormat::Markdown
        );
        assert_eq!(OcrApiFormat::try_from("html").unwrap(), OcrApiFormat::Html);
    }

    #[test]
    fn ocr_response_format_rejects_unknown_values() {
        let error = OcrApiFormat::try_from("verbose_json").unwrap_err();
        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_response_format")
        );
    }

    #[test]
    fn transcription_format_accepts_srt() {
        assert_eq!(
            TranscriptionFormat::try_from("srt").unwrap(),
            TranscriptionFormat::Srt
        );
    }

    #[test]
    fn verbose_json_serializes_segments_when_present() {
        let response = TranscriptionVerboseJson {
            text: "hello".to_string(),
            language: "en".to_string(),
            duration: 3.5,
            raw_output: "raw".to_string(),
            segments: Some(vec![AsrSegment {
                id: 0,
                start: 1.0,
                end: 2.0,
                text: "hello".to_string(),
            }]),
        };

        let value = serde_json::to_value(response).unwrap();

        assert_eq!(value["segments"][0]["start"], 1.0);
        assert_eq!(value["segments"][0]["end"], 2.0);
        assert_eq!(value["segments"][0]["text"], "hello");
        assert_eq!(value["duration"], 3.5);
    }

    #[test]
    fn internal_errors_return_generic_message() {
        let (_status, body) =
            ApiError::internal("local path /tmp/orchion/model.bin").into_status_body();

        assert_eq!(body.error.message, "internal server error");
        assert!(!body.error.message.contains("/tmp/orchion"));
        assert_eq!(body.error.code.as_deref(), Some("internal_error"));
    }

    #[test]
    fn invalid_image_and_document_errors_use_invalid_file_shape() {
        for error in [
            orchion::OrchionError::InvalidImage {
                reason: "bad image".to_string(),
            },
            orchion::OrchionError::InvalidDocument {
                reason: "bad document".to_string(),
            },
        ] {
            let (status, body) = ApiError::from(error).into_status_body();

            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(body.error.param.as_deref(), Some("file"));
            assert_eq!(body.error.code.as_deref(), Some("invalid_file"));
        }
    }

    #[test]
    fn llm_context_limit_is_a_pre_header_invalid_request() {
        let (status, body) = ApiError::from(orchion::OrchionError::LlmContextLimit {
            prompt_tokens: 100,
            max_tokens: 50,
            context_size: 128,
        })
        .into_status_body();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.error.param.as_deref(), Some("input"));
        assert_eq!(body.error.code.as_deref(), Some("context_length_exceeded"));
    }
}
