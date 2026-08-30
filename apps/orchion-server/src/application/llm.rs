use crate::application::RuntimeError;
use orchion::{GenerationEvent, LlmMessage};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Notify;
use tokio::sync::mpsc;

pub type LlmGenerationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ManagedGeneration>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq)]
pub struct LlmCommand {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub options: LlmGenerationOverrides,
    pub max_tokens_param: &'static str,
    pub queue_timeout: Option<std::time::Duration>,
    pub generation_timeout: Option<std::time::Duration>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlmGenerationOverrides {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop: Vec<String>,
}

pub struct ManagedGeneration {
    events: mpsc::Receiver<Result<GenerationEvent, RuntimeError>>,
    terminal: Option<tokio::sync::oneshot::Receiver<Result<GenerationEvent, RuntimeError>>>,
    cancelled: Arc<AtomicBool>,
    cancellation: Arc<Notify>,
}

impl ManagedGeneration {
    pub(crate) fn new(
        events: mpsc::Receiver<Result<GenerationEvent, RuntimeError>>,
        terminal: tokio::sync::oneshot::Receiver<Result<GenerationEvent, RuntimeError>>,
        cancelled: Arc<AtomicBool>,
        cancellation: Arc<Notify>,
    ) -> Self {
        Self {
            events,
            terminal: Some(terminal),
            cancelled,
            cancellation,
        }
    }

    pub async fn next(&mut self) -> Option<Result<GenerationEvent, RuntimeError>> {
        if let Some(event) = self.events.recv().await {
            return Some(event);
        }
        let terminal = self.terminal.take()?;
        Some(terminal.await.unwrap_or(Err(RuntimeError::Internal(
            "LLM owner stopped without a terminal result".to_string(),
        ))))
    }

    pub fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            self.cancellation.notify_one();
        }
    }
}

impl Drop for ManagedGeneration {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub trait LlmRuntime: Send + Sync {
    fn start_generation(&self, command: LlmCommand) -> LlmGenerationFuture<'_>;
}
