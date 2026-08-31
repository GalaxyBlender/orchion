use orchion_llama_cpp as backend;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
#[cfg(feature = "server-support")]
use std::sync::Arc;

use crate::{LlmModel, OrchionError, Result};

/// A text-only LLM deployment backed by one exact local GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmDeployment {
    model: LlmModel,
    path: PathBuf,
}

impl LlmDeployment {
    pub fn from_file(model: LlmModel, path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let metadata = std::fs::metadata(&path).map_err(|error| OrchionError::ModelLoad {
            message: format!("cannot access LLM GGUF `{}`: {error}", path.display()),
        })?;
        if !metadata.is_file() {
            return Err(OrchionError::ModelLoad {
                message: format!("LLM GGUF path `{}` is not a file", path.display()),
            });
        }
        if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
        {
            return Err(OrchionError::ModelLoad {
                message: format!("LLM model `{}` is not a GGUF file", path.display()),
            });
        }
        Ok(Self { model, path })
    }

    #[must_use]
    pub const fn model(&self) -> &LlmModel {
        &self.model
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(feature = "download-all")]
    pub async fn provision(
        model: LlmModel,
        source: crate::ModelUrl,
        cache_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::provision_with_downloader(
            model,
            source,
            cache_dir,
            &crate::ModelDownloader::default(),
        )
        .await
    }

    #[cfg(feature = "download-all")]
    pub async fn provision_with_downloader(
        model: LlmModel,
        source: crate::ModelUrl,
        cache_dir: impl AsRef<Path>,
        downloader: &crate::ModelDownloader,
    ) -> Result<Self> {
        let plan = llm_deployment_artifact_plan(&model, &source)?;
        let publication = downloader
            .provision_logical_deployment(model.id(), crate::ModelCategory::Llm, &plan, cache_dir)
            .await?;
        let path = publication
            .artifact_file(crate::ArtifactRole::LlmModel)
            .ok_or_else(|| OrchionError::ModelLoad {
                message: format!("published LLM deployment `{model}` has no model artifact"),
            })?
            .to_path_buf();
        Self::from_file(model, path)
    }
}

#[doc(hidden)]
#[cfg(feature = "server-support")]
#[derive(Debug, Clone)]
pub struct LlmBackendGuard {
    _inner: Arc<backend::BackendOwner>,
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
#[derive(Clone)]
pub struct LlmScriptedControl(backend::ScriptedControl);

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
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
#[cfg(feature = "server-support")]
pub fn initialize_llm_backend() -> crate::Result<LlmBackendGuard> {
    backend::BackendOwner::acquire()
        .map(|inner| LlmBackendGuard { _inner: inner })
        .map_err(|error| crate::OrchionError::ModelLoad {
            message: error.to_string(),
        })
}

#[doc(hidden)]
#[cfg(feature = "server-support")]
#[must_use]
pub fn llm_build_metadata_json() -> String {
    backend::build_metadata_json()
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
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
                    timings: backend::Timings {
                        cache_n: usage.timings.cache_n,
                        prompt_n: usage.timings.prompt_n,
                        prompt_ms: usage.timings.prompt_ms,
                        prompt_per_token_ms: usage.timings.prompt_per_token_ms,
                        prompt_per_second: usage.timings.prompt_per_second,
                        predicted_n: usage.timings.predicted_n,
                        predicted_ms: usage.timings.predicted_ms,
                        predicted_per_token_ms: usage.timings.predicted_per_token_ms,
                        predicted_per_second: usage.timings.predicted_per_second,
                    },
                },
            },
        })
        .collect();
    let (inner, control) = backend::scripted_engine(script, 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_context_limit_llm_engine(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: usize,
) -> LlmEngine {
    LlmEngine {
        inner: backend::scripted_context_limit_engine(prompt_tokens, max_tokens, context_size),
        event_queue_capacity: 1,
        model: None,
    }
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_engine(
        vec![backend::Event::Failed("__orchion_test_panic__".to_string())],
        1,
    );
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_preparation_panicking_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_preparation_panicking_engine(1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
        },
        LlmScriptedControl(control),
    )
}

#[doc(hidden)]
#[cfg(feature = "llm-test-support")]
pub fn scripted_slow_preparation_llm_engine() -> (LlmEngine, LlmScriptedControl) {
    let (inner, control) = backend::scripted_slow_preparation_engine(Vec::new(), 1);
    (
        LlmEngine {
            inner,
            event_queue_capacity: 1,
            model: None,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmTimings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

impl Default for LlmTimings {
    fn default() -> Self {
        Self {
            cache_n: 0,
            prompt_n: 0,
            prompt_ms: 0.0,
            prompt_per_token_ms: 0.0,
            prompt_per_second: 0.0,
            predicted_n: 0,
            predicted_ms: 0.0,
            predicted_per_token_ms: 0.0,
            predicted_per_second: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub queue_time_ms: Option<u64>,
    pub eval_time_ms: Option<u64>,
    pub timings: LlmTimings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationEvent {
    ContentDelta(String),
    Finished {
        reason: GenerationFinishReason,
        usage: LlmUsage,
    },
}

#[derive(Debug, Clone, PartialEq)]
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

impl Default for LlmEngineConfig {
    fn default() -> Self {
        Self {
            context_size: None,
            batch_size: 512,
            micro_batch_size: 512,
            threads: 0,
            gpu_layers: u32::MAX,
            parallel_sequences: 1,
            request_queue_capacity: 8,
            event_queue_capacity: 16,
            chat_template: None,
            template_engine: LlmTemplateEngine::LlamaCpp,
            enable_thinking: false,
        }
    }
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
    model: Option<LlmModel>,
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
        self.commit_inner().await
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn commit_reserved(&mut self) -> crate::Result<LlmGeneration> {
        self.commit_inner().await
    }

    async fn commit_inner(&mut self) -> crate::Result<LlmGeneration> {
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
    #[cfg(feature = "server-support")]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn wait_for_ack(&mut self) -> crate::Result<()> {
        self.inner.wait_for_ack().await.map_err(map_backend_error)
    }

    #[cfg(feature = "server-support")]
    pub fn abort(self) {
        self.inner.abort();
    }
}

impl LlmEngine {
    /// Loads a GGUF model synchronously.
    ///
    /// This method performs blocking model initialization. Async callers should use
    /// [`Self::load_deployment`], which offloads initialization to Tokio's blocking pool.
    /// Engines created through this compatibility API do not expose a typed model identity.
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
        .map_err(|error| crate::OrchionError::ModelLoad {
            message: error.to_string(),
        })?;
        Ok(Self {
            inner,
            event_queue_capacity,
            model: None,
        })
    }

    /// Loads a typed deployment synchronously on the current thread.
    pub fn load_deployment_blocking(
        deployment: LlmDeployment,
        config: LlmEngineConfig,
    ) -> crate::Result<Self> {
        let LlmDeployment { model, path } = deployment;
        let mut engine = Self::load(path, config)?;
        engine.model = Some(model);
        Ok(engine)
    }

    /// Loads a typed deployment on Tokio's blocking thread pool.
    pub async fn load_deployment(
        deployment: LlmDeployment,
        config: LlmEngineConfig,
    ) -> crate::Result<Self> {
        tokio::task::spawn_blocking(move || Self::load_deployment_blocking(deployment, config))
            .await
            .map_err(|error| crate::OrchionError::BlockingTask {
                message: error.to_string(),
            })?
    }

    /// Returns the typed identity when the engine was loaded from an [`LlmDeployment`].
    #[must_use]
    pub const fn model(&self) -> Option<&LlmModel> {
        self.model.as_ref()
    }

    pub async fn stream(&self, request: GenerationRequest) -> crate::Result<LlmGeneration> {
        self.reserve_generation(request).await?.commit().await
    }

    #[doc(hidden)]
    #[cfg(feature = "server-support")]
    pub async fn reserve(&self, request: GenerationRequest) -> crate::Result<LlmReservation> {
        self.reserve_generation(request).await
    }

    async fn reserve_generation(
        &self,
        request: GenerationRequest,
    ) -> crate::Result<LlmReservation> {
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
            timings: LlmTimings {
                cache_n: usage.timings.cache_n,
                prompt_n: usage.timings.prompt_n,
                prompt_ms: usage.timings.prompt_ms,
                prompt_per_token_ms: usage.timings.prompt_per_token_ms,
                prompt_per_second: usage.timings.prompt_per_second,
                predicted_n: usage.timings.predicted_n,
                predicted_ms: usage.timings.predicted_ms,
                predicted_per_token_ms: usage.timings.predicted_per_token_ms,
                predicted_per_second: usage.timings.predicted_per_second,
            },
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

#[cfg(feature = "download-all")]
fn llm_deployment_artifact_plan(
    model: &LlmModel,
    source: &crate::ModelUrl,
) -> Result<crate::DeploymentArtifactPlan> {
    use crate::{
        ArtifactRole, DeploymentArtifactRequest, DeploymentArtifactSource, DownloadSource,
        ModelUrlSource,
    };

    let artifact = match source.source() {
        ModelUrlSource::File => {
            let path = PathBuf::from(source.path().expect("validated file URL has a path"));
            if path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
            {
                return Err(OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` is not a GGUF file"),
                });
            }
            let file_name = path
                .file_name()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` does not identify a file"),
                })?
                .to_string_lossy()
                .to_string();
            DeploymentArtifactRequest {
                role: ArtifactRole::LlmModel,
                source: DeploymentArtifactSource::File(path),
                repository: None,
                files: vec![file_name],
                required_source: None,
            }
        }
        ModelUrlSource::Neutral | ModelUrlSource::HuggingFace | ModelUrlSource::ModelScope => {
            let path = source.path().ok_or_else(|| OrchionError::ModelLoad {
                message: format!("LLM source `{source}` must identify an exact GGUF file"),
            })?;
            if !path.to_ascii_lowercase().ends_with(".gguf") {
                return Err(OrchionError::ModelLoad {
                    message: format!("LLM source `{source}` is not a GGUF file"),
                });
            }
            DeploymentArtifactRequest {
                role: ArtifactRole::LlmModel,
                source: match source.source() {
                    ModelUrlSource::Neutral => DeploymentArtifactSource::Neutral,
                    ModelUrlSource::HuggingFace => DeploymentArtifactSource::HuggingFace,
                    ModelUrlSource::ModelScope => DeploymentArtifactSource::ModelScope,
                    ModelUrlSource::File => unreachable!("file source handled above"),
                },
                repository: Some(format!(
                    "{}/{}",
                    source.owner().expect("validated hub URL has an owner"),
                    source
                        .repository()
                        .expect("validated hub URL has a repository")
                )),
                files: vec![path.to_string()],
                required_source: None,
            }
        }
    };
    let neutral_candidates = vec![DownloadSource::HuggingFace, DownloadSource::ModelScope];
    let neutral_suffix = if source.source() == ModelUrlSource::Neutral {
        "|neutral-policy=huggingface,modelscope"
    } else {
        ""
    };
    Ok(crate::DeploymentArtifactPlan {
        deployment_id: model.id().clone(),
        category: crate::ModelCategory::Llm,
        source_intent: format!("model={source}|mmproj=none{neutral_suffix}"),
        artifacts: vec![artifact],
        neutral_candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> LlmModel {
        LlmModel::new(crate::ModelId::parse("acme/test-llm").unwrap())
    }

    fn temporary_gguf(contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "orchion-llm-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[tokio::test]
    async fn complete_collects_the_same_deltas_and_terminal_as_stream() {
        let usage = backend::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            timings: backend::Timings::default(),
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

    #[test]
    fn engine_config_default_matches_server_runtime_defaults() {
        let config = LlmEngineConfig::default();
        assert_eq!(config.context_size, None);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.micro_batch_size, 512);
        assert_eq!(config.threads, 0);
        assert_eq!(config.gpu_layers, u32::MAX);
        assert_eq!(config.parallel_sequences, 1);
        assert_eq!(config.request_queue_capacity, 8);
        assert_eq!(config.event_queue_capacity, 16);
        assert_eq!(config.chat_template, None);
        assert_eq!(config.template_engine, LlmTemplateEngine::LlamaCpp);
        assert!(!config.enable_thinking);
    }

    #[test]
    fn deployment_checks_file_and_preserves_identity_and_path() {
        let path = temporary_gguf(b"not a real model");
        let model = test_model();
        let deployment = LlmDeployment::from_file(model.clone(), path.clone()).unwrap();
        assert_eq!(deployment.model(), &model);
        assert_eq!(deployment.path(), path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(
            LlmDeployment::from_file(test_model(), "/definitely/missing/model.gguf"),
            Err(OrchionError::ModelLoad { .. })
        ));
    }

    #[tokio::test]
    async fn typed_load_maps_invalid_gguf_to_model_load_without_nested_runtime() {
        let path = temporary_gguf(b"not a real model");
        let deployment = LlmDeployment::from_file(test_model(), path.clone()).unwrap();
        let result = LlmEngine::load_deployment(deployment, LlmEngineConfig::default()).await;
        assert!(matches!(result, Err(OrchionError::ModelLoad { .. })));
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "download-all")]
    #[test]
    fn llm_plan_preserves_exact_sources_roles_and_identity() {
        use crate::{
            ArtifactRole, DeploymentArtifactSource, DownloadSource, ModelCategory, ModelUrl,
        };

        let cases = [
            (
                "hf://owner/repo/models/main.gguf",
                DeploymentArtifactSource::HuggingFace,
                Some("owner/repo"),
                "models/main.gguf",
            ),
            (
                "ms://owner/repo/main.gguf",
                DeploymentArtifactSource::ModelScope,
                Some("owner/repo"),
                "main.gguf",
            ),
            (
                "//owner/repo/main.gguf",
                DeploymentArtifactSource::Neutral,
                Some("owner/repo"),
                "main.gguf",
            ),
            (
                "file:///tmp/main.gguf",
                DeploymentArtifactSource::File(PathBuf::from("/tmp/main.gguf")),
                None,
                "main.gguf",
            ),
        ];
        for (url, expected_source, repository, file) in cases {
            let model = test_model();
            let source = ModelUrl::parse(url).unwrap();
            let plan = llm_deployment_artifact_plan(&model, &source).unwrap();
            assert_eq!(plan.deployment_id, *model.id());
            assert_eq!(plan.category, ModelCategory::Llm);
            assert!(plan.source_intent.contains("mmproj=none"));
            assert_eq!(
                plan.neutral_candidates,
                vec![DownloadSource::HuggingFace, DownloadSource::ModelScope]
            );
            assert_eq!(plan.artifacts.len(), 1);
            let artifact = &plan.artifacts[0];
            assert_eq!(artifact.role, ArtifactRole::LlmModel);
            assert_eq!(artifact.source, expected_source);
            assert_eq!(artifact.repository.as_deref(), repository);
            assert_eq!(artifact.files, [file]);
        }
    }

    #[cfg(feature = "download-all")]
    #[test]
    fn llm_plan_rejects_repository_only_and_non_gguf_hub_sources() {
        for source in [
            "//owner/repo",
            "hf://owner/repo/model.bin",
            "file:///tmp/model.bin",
        ] {
            let source = crate::ModelUrl::parse(source).unwrap();
            assert!(matches!(
                llm_deployment_artifact_plan(&test_model(), &source),
                Err(OrchionError::ModelLoad { .. })
            ));
        }
    }
}
