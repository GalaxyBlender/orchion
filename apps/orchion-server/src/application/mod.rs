pub mod model_cache;
pub mod ocr;
pub mod resource_policy;
pub mod speech;
pub mod streaming_transcription;
pub mod transcription;

use crate::application::ocr::OcrRuntime;
use crate::application::resource_policy::InferenceGuard;
use crate::application::speech::SpeechRuntime;
use crate::application::streaming_transcription::StreamingTranscriptionRuntime;
use crate::application::transcription::TranscriptionRuntime;
use orchion::{AsrModel, ModelId, TtsModel};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::OwnedSemaphorePermit;

pub type InferenceGuardFuture<'a> = Pin<Box<dyn Future<Output = InferenceGuard> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct AsrApiPolicy {
    pub available_models: Vec<AsrModel>,
    pub stream_target_segment: Duration,
    pub stream_max_segment: Duration,
    pub stream_idle_timeout: Duration,
    pub stream_max_duration: Duration,
    pub stream_chunk_size: f32,
}

#[derive(Debug, Clone)]
pub struct OcrApiModels {
    pub models: Vec<ModelId>,
    pub layout_models: Vec<ModelId>,
}

#[derive(Debug, Clone)]
pub struct ApiPolicy {
    pub api_key: Option<String>,
    pub cors_allowed_origins: Vec<String>,
    pub max_upload_size: usize,
    pub max_pdf_pages: usize,
    pub max_pdf_pixels: u64,
    pub max_pdf_output_size: usize,
    pub max_websocket_message_size: usize,
    pub asr: Option<AsrApiPolicy>,
    pub tts_models: Option<Vec<TtsModel>>,
    pub ocr: Option<OcrApiModels>,
    pub ocr_vl: Option<OcrApiModels>,
}

pub trait ServerApplication:
    TranscriptionRuntime + SpeechRuntime + OcrRuntime + StreamingTranscriptionRuntime + 'static
{
    fn api_policy(&self) -> &ApiPolicy;
    fn acquire_inference(&self) -> InferenceGuardFuture<'_>;
    fn try_acquire_websocket(&self) -> Option<OwnedSemaphorePermit>;
    fn try_acquire_pending_websocket(&self) -> Option<OwnedSemaphorePermit>;
}

#[derive(Debug, thiserror::Error)]
pub enum UseCaseError {
    #[error("{message}")]
    InvalidRequest {
        message: String,
        param: Option<&'static str>,
        code: &'static str,
    },
    #[error("model `{0}` is not available")]
    ModelNotAvailable(String),
    #[error("{0} capacity is currently exhausted")]
    ResourceExhausted(&'static str),
    #[error(transparent)]
    Core(#[from] orchion::OrchionError),
    #[error("invalid reference audio: {0}")]
    ReferenceAudio(orchion::OrchionError),
    #[error("{0}")]
    Internal(String),
}

impl UseCaseError {
    pub fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: &'static str,
    ) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param,
            code,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("{0}")]
    Internal(String),
    #[error(transparent)]
    Core(#[from] orchion::OrchionError),
    #[error("{0} capacity is currently exhausted")]
    ResourceExhausted(&'static str),
}

impl From<RuntimeError> for UseCaseError {
    fn from(error: RuntimeError) -> Self {
        match error {
            RuntimeError::Internal(message) => Self::Internal(message),
            RuntimeError::Core(error) => Self::Core(error),
            RuntimeError::ResourceExhausted(resource) => Self::ResourceExhausted(resource),
        }
    }
}
