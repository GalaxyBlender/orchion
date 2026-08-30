use orchion_llama_cpp as backend;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct LlmBackendGuard {
    _inner: Arc<backend::BackendOwner>,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct LlmScriptedControl(backend::ScriptedControl);

#[doc(hidden)]
impl LlmScriptedControl {
    pub fn wait_started(&self) {
        self.0.wait_started();
    }

    pub fn release_ready(&self) {
        self.0.release_ready();
    }

    pub fn release_cleanup(&self) {
        self.0.release_cleanup();
    }

    pub fn wait_preparation_started(&self) {
        self.0.wait_preparation_started();
    }

    pub fn release_preparation(&self) {
        self.0.release_preparation();
    }

    pub fn has_executed(&self) -> bool {
        self.0.has_executed()
    }

    pub fn has_started(&self) -> bool {
        self.0.has_started()
    }
}

#[doc(hidden)]
pub fn initialize_llm_backend() -> crate::Result<LlmBackendGuard> {
    backend::BackendOwner::acquire()
        .map(|inner| LlmBackendGuard { _inner: inner })
        .map_err(|error| crate::OrchionError::ModelLoad {
            message: error.to_string(),
        })
}

#[doc(hidden)]
#[must_use]
pub fn llm_build_metadata_json() -> String {
    backend::build_metadata_json()
}

#[doc(hidden)]
pub fn scripted_llm_engine(script: Vec<GenerationEvent>) -> (LlmEngine, LlmScriptedControl) {
    let script = script
        .into_iter()
        .map(|event| match event {
            GenerationEvent::ContentDelta(content) => backend::Event::Content(content),
            GenerationEvent::Finished { reason, usage } => backend::Event::Finished {
                reason: match reason {
                    GenerationFinishReason::Stop => backend::FinishReason::Stop,
                    GenerationFinishReason::Length => backend::FinishReason::Length,
                    GenerationFinishReason::Cancelled => backend::FinishReason::Cancelled,
                },
                usage: backend::Usage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                },
            },
        })
        .collect();
    let (inner, control) = backend::scripted_engine(script, 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
pub fn scripted_context_limit_llm_engine(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: usize,
) -> LlmEngine {
    LlmEngine {
        inner: backend::scripted_context_limit_engine(prompt_tokens, max_tokens, context_size),
        event_queue_capacity: 1,
    }
}

#[doc(hidden)]
pub fn scripted_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_engine(
        vec![backend::Event::Failed("__orchion_test_panic__".to_string())],
        1,
    );
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
pub fn scripted_preparation_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_preparation_panicking_engine(1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
pub fn scripted_slow_preparation_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_slow_preparation_engine(Vec::new(), 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
        },
        LlmScriptedControl(control),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmRole {
    System,
    Developer,
    User,
    Assistant,
}

impl LlmRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            seed: u32::MAX,
            stop: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub messages: Vec<LlmMessage>,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub queue_time_ms: Option<u64>,
    pub eval_time_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationEvent {
    ContentDelta(String),
    Finished {
        reason: GenerationFinishReason,
        usage: LlmUsage,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmComplete {
    pub text: String,
    pub finish_reason: GenerationFinishReason,
    pub usage: LlmUsage,
}

#[derive(Debug, Clone)]
pub struct LlmEngineConfig {
    pub context_size: Option<NonZeroU32>,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub threads: i32,
    pub gpu_layers: u32,
    pub parallel_sequences: u32,
    pub request_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub chat_template: Option<String>,
    pub template_engine: LlmTemplateEngine,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmTemplateEngine {
    LlamaCpp,
    Jinja,
}

#[derive(Debug, Clone)]
pub struct LlmEngine {
    inner: backend::Engine,
    event_queue_capacity: usize,
}

pub struct LlmGeneration {
    inner: backend::Generation,
    terminal_received: bool,
}

#[doc(hidden)]
pub struct LlmReservation {
    inner: backend::Reservation,
}

impl LlmReservation {
    pub async fn commit(mut self) -> crate::Result<LlmGeneration> {
        self.commit_reserved().await
    }

    #[doc(hidden)]
    pub async fn commit_reserved(&mut self) -> crate::Result<LlmGeneration> {
        self.inner
            .commit()
            .await
            .map(|inner| LlmGeneration {
                inner,
                terminal_received: false,
            })
            .map_err(map_backend_error)
    }

    #[doc(hidden)]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }

    pub fn abort(self) {
        self.inner.abort();
    }
}

impl LlmEngine {
    pub fn load(model: PathBuf, config: LlmEngineConfig) -> crate::Result<Self> {
        let event_queue_capacity = config.event_queue_capacity;
        let inner = backend::Engine::load(
            model,
            backend::RuntimeConfig {
                context_size: config.context_size,
                batch_size: config.batch_size,
                micro_batch_size: config.micro_batch_size,
                threads: config.threads,
                gpu_layers: config.gpu_layers,
                parallel_sequences: config.parallel_sequences,
                request_queue_capacity: config.request_queue_capacity,
                event_queue_capacity,
                chat_template: config.chat_template,
                template_engine: match config.template_engine {
                    LlmTemplateEngine::LlamaCpp => backend::TemplateEngine::LlamaCpp,
                    LlmTemplateEngine::Jinja => backend::TemplateEngine::Jinja,
                },
                enable_thinking: config.enable_thinking,
            },
        )
        .map_err(|error| crate::OrchionError::Inference {
            message: error.to_string(),
        })?;
        Ok(Self {
            inner,
            event_queue_capacity,
        })
    }

    pub async fn stream(&self, request: GenerationRequest) -> crate::Result<LlmGeneration> {
        self.reserve(request).await?.commit().await
    }

    #[doc(hidden)]
    pub async fn reserve(&self, request: GenerationRequest) -> crate::Result<LlmReservation> {
        let request = backend::Request {
            messages: request
                .messages
                .into_iter()
                .map(|message| backend::Message {
                    role: message.role.as_str().to_string(),
                    content: message.content,
                })
                .collect(),
            options: backend::GenerationOptions {
                max_tokens: request.options.max_tokens,
                temperature: request.options.temperature,
                top_p: request.options.top_p,
                top_k: request.options.top_k,
                min_p: request.options.min_p,
                presence_penalty: request.options.presence_penalty,
                frequency_penalty: request.options.frequency_penalty,
                repeat_penalty: request.options.repeat_penalty,
                seed: request.options.seed,
                stop: request.options.stop,
            },
        };
        let inner = self
            .inner
            .reserve(request, self.event_queue_capacity)
            .await
            .map_err(map_backend_error)?;
        Ok(LlmReservation { inner })
    }

    pub async fn complete(&self, request: GenerationRequest) -> crate::Result<LlmComplete> {
        collect_generation(self.stream(request).await?).await
    }

    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    #[doc(hidden)]
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
}

async fn collect_generation(mut generation: LlmGeneration) -> crate::Result<LlmComplete> {
    let mut text = String::new();
    while let Some(event) = generation.next().await? {
        match event {
            GenerationEvent::ContentDelta(delta) => text.push_str(&delta),
            GenerationEvent::Finished { reason, usage } => {
                return Ok(LlmComplete {
                    text,
                    finish_reason: reason,
                    usage,
                });
            }
        }
    }
    Err(crate::OrchionError::Inference {
        message: "LLM generation ended without a terminal event".to_string(),
    })
}

impl LlmGeneration {
    pub async fn next(&mut self) -> crate::Result<Option<GenerationEvent>> {
        if self.terminal_received {
            return Ok(None);
        }
        match self.inner.events.recv().await {
            Some(backend::Event::Content(content)) => {
                Ok(Some(GenerationEvent::ContentDelta(content)))
            }
            Some(backend::Event::Finished { reason, usage }) => {
                self.terminal_received = true;
                Ok(Some(map_terminal(reason, usage)))
            }
            Some(backend::Event::Failed(message)) => {
                self.terminal_received = true;
                Err(crate::OrchionError::Inference { message })
            }
            None => {
                self.terminal_received = true;
                match self
                    .inner
                    .recv_terminal()
                    .await
                    .map_err(map_backend_error)?
                {
                    backend::Event::Finished { reason, usage } => {
                        Ok(Some(map_terminal(reason, usage)))
                    }
                    backend::Event::Failed(message) => {
                        Err(crate::OrchionError::Inference { message })
                    }
                    backend::Event::Content(content) => {
                        Ok(Some(GenerationEvent::ContentDelta(content)))
                    }
                }
            }
        }
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }
}

fn map_terminal(reason: backend::FinishReason, usage: backend::Usage) -> GenerationEvent {
    GenerationEvent::Finished {
        reason: match reason {
            backend::FinishReason::Stop => GenerationFinishReason::Stop,
            backend::FinishReason::Length => GenerationFinishReason::Length,
            backend::FinishReason::Cancelled => GenerationFinishReason::Cancelled,
        },
        usage: LlmUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.prompt_tokens + usage.completion_tokens,
            queue_time_ms: None,
            eval_time_ms: None,
        },
    }
}

fn map_backend_error(error: backend::Error) -> crate::OrchionError {
    match error {
        backend::Error::ContextLimit {
            prompt_tokens,
            max_tokens,
            context_size,
        } => crate::OrchionError::LlmContextLimit {
            prompt_tokens,
            max_tokens,
            context_size,
        },
        backend::Error::WorkerPanic(message) => crate::OrchionError::LlmWorkerFailed { message },
        backend::Error::WorkerUnavailable => crate::OrchionError::LlmWorkerFailed {
            message: "worker is unavailable".to_string(),
        },
        other => crate::OrchionError::Inference {
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn complete_collects_the_same_deltas_and_terminal_as_stream() {
        let usage = backend::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        let script = vec![
            backend::Event::Content("hel".to_string()),
            backend::Event::Content("lo".to_string()),
            backend::Event::Finished {
                reason: backend::FinishReason::Length,
                usage,
            },
        ];
        let mut streamed = LlmGeneration {
            inner: backend::deterministic_generation(script.clone()),
            terminal_received: false,
        };
        let mut stream_text = String::new();
        let mut terminal = None;
        while let Some(event) = streamed.next().await.unwrap() {
            match event {
                GenerationEvent::ContentDelta(delta) => stream_text.push_str(&delta),
                GenerationEvent::Finished { reason, usage } => terminal = Some((reason, usage)),
            }
        }
        let complete = collect_generation(LlmGeneration {
            inner: backend::deterministic_generation(script),
            terminal_received: false,
        })
        .await
        .unwrap();
        assert_eq!(complete.text, stream_text);
        assert_eq!(Some((complete.finish_reason, complete.usage)), terminal);
    }

    #[tokio::test]
    async fn deterministic_generation_cancel_is_idempotent() {
        let generation = LlmGeneration {
            inner: backend::deterministic_generation([]),
            terminal_received: false,
        };
        generation.cancel();
        generation.cancel();
    }
}
