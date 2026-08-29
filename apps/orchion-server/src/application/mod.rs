pub mod model_cache;
pub mod model_lifecycle;
pub mod ocr;
pub mod resource_policy;
pub mod speech;
pub mod streaming_transcription;
pub mod transcription;

use crate::application::model_lifecycle::{ModelLifecycleRuntime, ModelService};
use crate::application::ocr::OcrRuntime;
use crate::application::resource_policy::InferenceGuard;
use crate::application::speech::SpeechRuntime;
use crate::application::streaming_transcription::StreamingTranscriptionRuntime;
use crate::application::transcription::TranscriptionRuntime;
use orchion::{AsrModel, ModelCapabilities, ModelId, TtsModel};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;
use tokio::sync::OwnedSemaphorePermit;

pub(crate) struct OwnedOperationState {
    state: AtomicU8,
}

impl OwnedOperationState {
    const CANCELLED: u8 = 1;
    const PROTECTED: u8 = 2;

    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
        }
    }

    pub(crate) fn cancel(&self) -> bool {
        self.state.fetch_or(Self::CANCELLED, Ordering::AcqRel) & Self::PROTECTED == 0
    }

    fn protect(&self) -> bool {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & Self::CANCELLED != 0 {
                return false;
            }
            match self.state.compare_exchange_weak(
                state,
                state | Self::PROTECTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(current) => state = current,
            }
        }
    }

    fn finish_protected(&self) -> bool {
        self.state.fetch_and(!Self::PROTECTED, Ordering::AcqRel) & Self::CANCELLED != 0
    }
}

tokio::task_local! {
    static OWNED_OPERATION: Arc<OwnedOperationState>;
}

pub(crate) async fn track_owned_operation<F>(
    state: Arc<OwnedOperationState>,
    operation: F,
) -> F::Output
where
    F: Future,
{
    OWNED_OPERATION.scope(state, operation).await
}

pub(crate) fn mark_owned_operation_dispatched() -> bool {
    OWNED_OPERATION
        .try_with(|state| state.protect())
        .unwrap_or(true)
}

pub(crate) fn protect_owned_file_operation() -> bool {
    mark_owned_operation_dispatched()
}

pub(crate) fn finish_owned_file_operation() -> bool {
    OWNED_OPERATION
        .try_with(|state| state.finish_protected())
        .unwrap_or(false)
}

pub type InferenceGuardFuture<'a> = Pin<Box<dyn Future<Output = InferenceGuard> + Send + 'a>>;
pub type ModelCatalogFuture<'a> = Pin<Box<dyn Future<Output = Vec<ApiModel>> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct AsrApiPolicy {
    pub models: Vec<AsrModel>,
    pub stream_target_segment: Duration,
    pub stream_max_segment: Duration,
    pub stream_idle_timeout: Duration,
    pub stream_max_duration: Duration,
    pub stream_chunk_size: f32,
}

#[derive(Debug, Clone)]
pub struct ApiModel {
    pub id: ModelId,
    pub name: Option<String>,
    pub service: ModelService,
    pub capabilities: ModelCapabilities,
}

#[derive(Debug, Clone, Copy)]
pub struct ActivityPolicy {
    pub enabled: bool,
    pub history_capacity: usize,
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
    pub activity: ActivityPolicy,
    pub models: Vec<ApiModel>,
    pub asr: Option<AsrApiPolicy>,
    pub tts_models: Option<Vec<TtsModel>>,
    pub ocr_enabled: bool,
}

pub trait ServerApplication:
    TranscriptionRuntime
    + SpeechRuntime
    + OcrRuntime
    + StreamingTranscriptionRuntime
    + ModelLifecycleRuntime
    + 'static
{
    fn api_policy(&self) -> &ApiPolicy;
    fn model_catalog(&self) -> ModelCatalogFuture<'_> {
        Box::pin(async move { self.api_policy().models.clone() })
    }
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

#[cfg(test)]
mod owned_operation_tests {
    use super::OwnedOperationState;

    #[test]
    fn cancellation_prevents_later_protection() {
        let state = OwnedOperationState::new();

        assert!(state.cancel());
        assert!(!state.protect());
    }

    #[test]
    fn protection_defers_cancellation_until_the_operation_finishes() {
        let state = OwnedOperationState::new();

        assert!(state.protect());
        assert!(!state.cancel());
        assert!(state.finish_protected());
    }
}
