use crate::application::RuntimeError;
use crate::application::metrics::InferenceOperation;
use orchion::{
    GenerationEvent, LlmAdvancedRequest, LlmChoiceEvent, LlmMessage, LlmSemanticTokenCountRequest,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use tokio::sync::Notify;
use tokio::sync::mpsc;

pub type LlmGenerationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ManagedGeneration>, RuntimeError>> + Send + 'a>>;
pub type LlmChoiceGenerationFuture<'a> = Pin<
    Box<dyn Future<Output = Result<Option<ManagedChoiceGeneration>, RuntimeError>> + Send + 'a>,
>;
pub type LlmEmbeddingFuture<'a> = Pin<
    Box<dyn Future<Output = Result<Option<orchion::LlmEmbeddingResult>, RuntimeError>> + Send + 'a>,
>;
pub type LlmTokenCountFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<usize>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmInput {
    Messages(Vec<LlmMessage>),
    Prompt(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmCommand {
    pub model: String,
    pub input: LlmInput,
    pub options: LlmGenerationOverrides,
    pub max_tokens_param: &'static str,
    pub queue_timeout: Option<std::time::Duration>,
    pub generation_timeout: Option<std::time::Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmEmbeddingCommand {
    pub model: String,
    pub inputs: Vec<orchion::LlmEmbeddingInput>,
    pub dimensions: Option<usize>,
    pub queue_timeout: Option<std::time::Duration>,
    pub embedding_timeout: Option<std::time::Duration>,
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

pub struct ManagedChoiceGeneration {
    events: mpsc::Receiver<Result<LlmChoiceEvent, RuntimeError>>,
    cancelled: Arc<AtomicBool>,
    cancellation: Arc<Notify>,
    cancellation_cause: Arc<AtomicU8>,
    reasoning_control: Option<orchion::LlmReasoningControl>,
}

#[derive(Clone)]
pub struct ManagedChoiceCancellation {
    cancelled: Arc<AtomicBool>,
    cancellation: Arc<Notify>,
    cancellation_cause: Arc<AtomicU8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceCancellationCause {
    ClientDisconnect,
    UserDeleted,
    ServerShutdown,
    ResourceExhausted,
    StreamBufferExceeded,
}

impl ChoiceCancellationCause {
    const fn encoded(self) -> u8 {
        match self {
            Self::ClientDisconnect => 1,
            Self::UserDeleted => 2,
            Self::ServerShutdown => 3,
            Self::ResourceExhausted => 4,
            Self::StreamBufferExceeded => 5,
        }
    }

    pub(crate) fn decode(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::ClientDisconnect),
            2 => Some(Self::UserDeleted),
            3 => Some(Self::ServerShutdown),
            4 => Some(Self::ResourceExhausted),
            5 => Some(Self::StreamBufferExceeded),
            _ => None,
        }
    }
}

impl ManagedChoiceCancellation {
    pub fn cancel(&self) {
        self.cancel_with(ChoiceCancellationCause::ClientDisconnect);
    }

    pub fn cancel_with(&self, cause: ChoiceCancellationCause) {
        if self
            .cancellation_cause
            .compare_exchange(0, cause.encoded(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            && !self.cancelled.swap(true, Ordering::AcqRel)
        {
            self.cancellation.notify_one();
        }
    }

    #[cfg(test)]
    pub(crate) fn cause(&self) -> Option<ChoiceCancellationCause> {
        ChoiceCancellationCause::decode(self.cancellation_cause.load(Ordering::Acquire))
    }
}

impl ManagedChoiceGeneration {
    #[allow(dead_code, reason = "test and non-control construction seam")]
    pub(crate) fn new(
        events: mpsc::Receiver<Result<LlmChoiceEvent, RuntimeError>>,
        cancelled: Arc<AtomicBool>,
        cancellation: Arc<Notify>,
    ) -> Self {
        Self::new_with_control(
            events,
            cancelled,
            cancellation,
            Arc::new(AtomicU8::new(0)),
            None,
        )
    }

    pub(crate) fn new_with_control(
        events: mpsc::Receiver<Result<LlmChoiceEvent, RuntimeError>>,
        cancelled: Arc<AtomicBool>,
        cancellation: Arc<Notify>,
        cancellation_cause: Arc<AtomicU8>,
        reasoning_control: Option<orchion::LlmReasoningControl>,
    ) -> Self {
        Self {
            events,
            cancelled,
            cancellation,
            cancellation_cause,
            reasoning_control,
        }
    }

    pub async fn next(&mut self) -> Option<Result<LlmChoiceEvent, RuntimeError>> {
        self.events.recv().await
    }

    pub fn cancel(&self) {
        self.cancellation_handle().cancel();
    }

    #[must_use]
    pub fn cancellation_handle(&self) -> ManagedChoiceCancellation {
        ManagedChoiceCancellation {
            cancelled: Arc::clone(&self.cancelled),
            cancellation: Arc::clone(&self.cancellation),
            cancellation_cause: Arc::clone(&self.cancellation_cause),
        }
    }

    #[must_use]
    pub fn reasoning_control(&self) -> Option<orchion::LlmReasoningControl> {
        self.reasoning_control.clone()
    }
}

impl Drop for ManagedChoiceGeneration {
    fn drop(&mut self) {
        self.cancel();
    }
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
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the validated generation command boundary"
    )]
    fn start_choice_generation(
        &self,
        operation: InferenceOperation,
        model: String,
        request: LlmAdvancedRequest,
        overrides: LlmGenerationOverrides,
        max_tokens_param: &'static str,
        queue_timeout: Option<std::time::Duration>,
        generation_timeout: Option<std::time::Duration>,
    ) -> LlmChoiceGenerationFuture<'_>;
    fn create_embeddings(&self, command: LlmEmbeddingCommand) -> LlmEmbeddingFuture<'_>;
    fn count_input_tokens(
        &self,
        model: String,
        messages: Vec<LlmMessage>,
    ) -> LlmTokenCountFuture<'_>;
    fn count_semantic_input_tokens(
        &self,
        model: String,
        request: LlmSemanticTokenCountRequest,
    ) -> LlmTokenCountFuture<'_>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation_and_cancellation() -> (ManagedChoiceGeneration, ManagedChoiceCancellation) {
        let (_sender, receiver) = mpsc::channel(1);
        let generation = ManagedChoiceGeneration::new(
            receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        );
        let cancellation = generation.cancellation_handle();
        (generation, cancellation)
    }

    #[test]
    fn choice_cancellation_preserves_the_first_typed_cause() {
        for cause in [
            ChoiceCancellationCause::UserDeleted,
            ChoiceCancellationCause::ServerShutdown,
            ChoiceCancellationCause::ResourceExhausted,
            ChoiceCancellationCause::StreamBufferExceeded,
        ] {
            let (generation, cancellation) = generation_and_cancellation();
            cancellation.cancel_with(cause);
            cancellation.cancel_with(ChoiceCancellationCause::ClientDisconnect);
            drop(generation);
            assert_eq!(cancellation.cause(), Some(cause));
        }

        let (generation, cancellation) = generation_and_cancellation();
        drop(generation);
        assert_eq!(
            cancellation.cause(),
            Some(ChoiceCancellationCause::ClientDisconnect)
        );
    }
}
