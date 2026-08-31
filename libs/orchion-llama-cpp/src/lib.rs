#![allow(clippy::missing_errors_doc)]

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;
use tokio::sync::mpsc;

pub const BINDING_REVISION: &str = "3d5f424f7cfb7cd3d1f9039440bd0286e29ca050";
pub const LLAMA_CPP_REVISION: &str = "e79e4bf660e19f2ad851e06c6913f7a8c5852621";

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BuildMetadata {
    pub binding_revision: &'static str,
    pub llama_cpp_revision: &'static str,
    pub binding_features: &'static str,
    pub cargo_features: &'static str,
    pub rustc_version: &'static str,
    pub rustc_verbose_version: &'static str,
    pub toolchain: &'static str,
    pub target: &'static str,
    pub profile: &'static str,
    pub cmake_input: CmakeInputMetadata,
    pub cmake_resolved: ResolvedCmakeBuildMetadata,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CmakeInputMetadata {
    pub ggml_metal: &'static str,
    pub ggml_cuda: &'static str,
    pub ggml_openmp: &'static str,
    pub build_type: &'static str,
    pub generator: &'static str,
    pub osx_deployment_target: &'static str,
    pub macosx_deployment_target: &'static str,
    pub toolchain_file: &'static str,
    pub cuda_compute_cap: &'static str,
    pub llama_build_shared_libs: &'static str,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ResolvedCmakeBuildMetadata {
    pub cache_path_relative: &'static str,
    pub cache_sha256: &'static str,
    pub build_type: &'static str,
    pub generator: &'static str,
    pub osx_deployment_target: &'static str,
    pub build_shared_libs: &'static str,
    pub ggml_metal: &'static str,
    pub ggml_openmp: &'static str,
    pub ggml_cuda: &'static str,
    pub ggml_vulkan: &'static str,
    pub ggml_native: &'static str,
    pub c_compiler: CompilerBuildMetadata,
    pub cxx_compiler: CompilerBuildMetadata,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CompilerBuildMetadata {
    pub basename: &'static str,
    pub id: &'static str,
    pub version: &'static str,
}

#[must_use]
pub const fn build_metadata() -> BuildMetadata {
    BuildMetadata {
        binding_revision: BINDING_REVISION,
        llama_cpp_revision: LLAMA_CPP_REVISION,
        binding_features: "common,mtmd",
        cargo_features: env!("ORCHION_LLAMA_CARGO_FEATURES"),
        rustc_version: env!("ORCHION_RUSTC_VERSION"),
        rustc_verbose_version: env!("ORCHION_RUSTC_VERBOSE_VERSION"),
        toolchain: env!("ORCHION_RUST_TOOLCHAIN"),
        target: env!("TARGET"),
        profile: env!("PROFILE"),
        cmake_input: CmakeInputMetadata {
            ggml_metal: env!("ORCHION_BUILD_INPUT_GGML_METAL"),
            ggml_cuda: env!("ORCHION_BUILD_INPUT_GGML_CUDA"),
            ggml_openmp: env!("ORCHION_BUILD_INPUT_GGML_OPENMP"),
            build_type: env!("ORCHION_BUILD_INPUT_CMAKE_BUILD_TYPE"),
            generator: env!("ORCHION_BUILD_INPUT_CMAKE_GENERATOR"),
            osx_deployment_target: env!("ORCHION_BUILD_INPUT_CMAKE_OSX_DEPLOYMENT_TARGET"),
            macosx_deployment_target: env!("ORCHION_BUILD_INPUT_MACOSX_DEPLOYMENT_TARGET"),
            toolchain_file: env!("ORCHION_BUILD_INPUT_CMAKE_TOOLCHAIN_FILE"),
            cuda_compute_cap: env!("ORCHION_BUILD_INPUT_CUDA_COMPUTE_CAP"),
            llama_build_shared_libs: env!("ORCHION_BUILD_INPUT_LLAMA_BUILD_SHARED_LIBS"),
        },
        cmake_resolved: ResolvedCmakeBuildMetadata {
            cache_path_relative: env!("ORCHION_BUILD_CMAKE_CACHE_RELATIVE_PATH"),
            cache_sha256: env!("ORCHION_BUILD_CMAKE_CACHE_SHA256"),
            build_type: env!("ORCHION_BUILD_RESOLVED_CMAKE_BUILD_TYPE"),
            generator: env!("ORCHION_BUILD_RESOLVED_CMAKE_GENERATOR"),
            osx_deployment_target: env!("ORCHION_BUILD_RESOLVED_CMAKE_OSX_DEPLOYMENT_TARGET"),
            build_shared_libs: env!("ORCHION_BUILD_RESOLVED_BUILD_SHARED_LIBS"),
            ggml_metal: env!("ORCHION_BUILD_RESOLVED_GGML_METAL"),
            ggml_openmp: env!("ORCHION_BUILD_RESOLVED_GGML_OPENMP"),
            ggml_cuda: env!("ORCHION_BUILD_RESOLVED_GGML_CUDA"),
            ggml_vulkan: env!("ORCHION_BUILD_RESOLVED_GGML_VULKAN"),
            ggml_native: env!("ORCHION_BUILD_RESOLVED_GGML_NATIVE"),
            c_compiler: CompilerBuildMetadata {
                basename: env!("ORCHION_BUILD_C_COMPILER"),
                id: env!("ORCHION_BUILD_C_COMPILER_ID"),
                version: env!("ORCHION_BUILD_C_COMPILER_VERSION"),
            },
            cxx_compiler: CompilerBuildMetadata {
                basename: env!("ORCHION_BUILD_CXX_COMPILER"),
                id: env!("ORCHION_BUILD_CXX_COMPILER_ID"),
                version: env!("ORCHION_BUILD_CXX_COMPILER_VERSION"),
            },
        },
    }
}

#[must_use]
pub fn build_metadata_json() -> String {
    serde_json::to_string_pretty(&build_metadata()).unwrap_or_else(|error| {
        serde_json::json!({"metadata_error": error.to_string()}).to_string()
    })
}

static BACKEND: OnceLock<Mutex<Weak<BackendOwner>>> = OnceLock::new();

#[derive(Debug)]
pub struct BackendOwner {
    backend: LlamaBackend,
}

impl BackendOwner {
    pub fn acquire() -> Result<Arc<Self>, Error> {
        let mut owner = BACKEND
            .get_or_init(|| Mutex::new(Weak::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(owner) = owner.upgrade() {
            return Ok(owner);
        }
        let backend = LlamaBackend::init().map_err(|error| Error::Backend(error.to_string()))?;
        let devices = llama_cpp_2::list_llama_ggml_backend_devices();
        let current = Arc::new(Self { backend });
        *owner = Arc::downgrade(&current);
        tracing::debug!(
            binding_revision = BINDING_REVISION,
            llama_cpp_revision = LLAMA_CPP_REVISION,
            features = "common,mtmd",
            cpu_openmp = cfg!(feature = "cpu-openmp"),
            cuda = cfg!(feature = "cuda"),
            vulkan = cfg!(feature = "vulkan"),
            ggml_metal = option_env!("GGML_METAL").unwrap_or("cmake-default"),
            supports_gpu_offload = current.backend.supports_gpu_offload(),
            supports_mmap = current.backend.supports_mmap(),
            supports_mlock = current.backend.supports_mlock(),
            device_count = devices.len(),
            build_metadata = %build_metadata_json(),
            "initialized llama.cpp process backend"
        );
        for device in devices {
            tracing::info!(
                index = device.index,
                name = %device.name,
                description = %device.description,
                backend = %device.backend,
                device_type = ?device.device_type,
                memory_total = device.memory_total,
                memory_free = device.memory_free,
                "llama.cpp backend device"
            );
        }
        Ok(current)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Message {
    pub role: String,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub messages: Vec<Message>,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Timings {
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

impl Default for Timings {
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

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub timings: Timings,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Content(String),
    Finished { reason: FinishReason, usage: Usage },
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub context_size: Option<NonZeroU32>,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub threads: i32,
    pub gpu_layers: u32,
    pub parallel_sequences: u32,
    pub request_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub chat_template: Option<String>,
    pub template_engine: TemplateEngine,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateEngine {
    LlamaCpp,
    Jinja,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("failed to initialize llama.cpp backend: {0}")]
    Backend(String),
    #[error("failed to start llama.cpp model worker: {0}")]
    WorkerStart(String),
    #[error("llama.cpp model worker is unavailable")]
    WorkerUnavailable,
    #[error("invalid llama.cpp runtime configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "prompt ({prompt_tokens} tokens) plus completion ({max_tokens} tokens) exceeds context size {context_size}"
    )]
    ContextLimit {
        prompt_tokens: usize,
        max_tokens: usize,
        context_size: usize,
    },
    #[error("generation was cancelled before worker commit")]
    Cancelled,
    #[error("llama.cpp model worker panicked: {0}")]
    WorkerPanic(String),
    #[error("llama.cpp generation failed: {0}")]
    Generation(String),
}

#[derive(Debug)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

#[derive(Debug)]
struct EngineInner {
    commands: mpsc::Sender<Command>,
    join: Mutex<Option<JoinHandle<()>>>,
    active: Mutex<Vec<Weak<AtomicBool>>>,
    health: Arc<AtomicU8>,
}

#[derive(Debug)]
enum Command {
    Reserve {
        cancelled: Arc<AtomicBool>,
        events: mpsc::Sender<Event>,
        reserved: tokio::sync::oneshot::Sender<Result<(), Error>>,
        readiness: tokio::sync::oneshot::Sender<Result<(), Error>>,
        acknowledged: tokio::sync::oneshot::Sender<Result<(), Error>>,
        terminal: tokio::sync::oneshot::Sender<Result<Event, Error>>,
        decision: tokio::sync::oneshot::Receiver<ReservationDecision>,
    },
    Shutdown,
}

#[derive(Debug)]
enum ReservationDecision {
    Commit(Request),
    Abort,
}

pub struct Generation {
    pub events: mpsc::Receiver<Event>,
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    terminal: Option<tokio::sync::oneshot::Receiver<Result<Event, Error>>>,
}

pub struct Reservation {
    request: Option<Request>,
    events: Option<mpsc::Receiver<Event>>,
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    readiness: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    terminal: Option<tokio::sync::oneshot::Receiver<Result<Event, Error>>>,
    decision: Option<tokio::sync::oneshot::Sender<ReservationDecision>>,
    transferred: bool,
}

impl Reservation {
    pub async fn commit(&mut self) -> Result<Generation, Error> {
        let decision = self.decision.take().ok_or(Error::WorkerUnavailable)?;
        let request = self.request.take().ok_or(Error::WorkerUnavailable)?;
        decision
            .send(ReservationDecision::Commit(request))
            .map_err(|_| Error::WorkerUnavailable)?;
        let readiness = self.readiness.take().ok_or(Error::WorkerUnavailable)?;
        if let Err(error) = readiness.await.unwrap_or(Err(Error::WorkerUnavailable)) {
            if let Some(acknowledged) = self.acknowledged.take() {
                let _ = acknowledged.await;
            }
            return Err(error);
        }
        self.transferred = true;
        Ok(Generation {
            events: self.events.take().ok_or(Error::WorkerUnavailable)?,
            cancelled: Arc::clone(&self.cancelled),
            acknowledged: self.acknowledged.take(),
            terminal: self.terminal.take(),
        })
    }

    pub fn abort(mut self) {
        self.cancel();
        if let Some(decision) = self.decision.take() {
            let _ = decision.send(ReservationDecision::Abort);
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub async fn wait_for_ack(&mut self) -> Result<(), Error> {
        let Some(acknowledged) = self.acknowledged.take() else {
            return Ok(());
        };
        acknowledged.await.unwrap_or(Err(Error::WorkerUnavailable))
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.transferred {
            self.cancelled.store(true, Ordering::Release);
        }
        if let Some(decision) = self.decision.take() {
            let _ = decision.send(ReservationDecision::Abort);
        }
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct ScriptedControl {
    ready: Arc<(Mutex<bool>, std::sync::Condvar)>,
    preparation: Arc<(Mutex<bool>, std::sync::Condvar)>,
    preparation_started: Arc<(Mutex<bool>, std::sync::Condvar)>,
    cleanup: Arc<(Mutex<bool>, std::sync::Condvar)>,
    started: Arc<(Mutex<bool>, std::sync::Condvar)>,
    executed: Arc<AtomicBool>,
    panic_preparation: bool,
}

impl ScriptedControl {
    pub fn wait_started(&self) {
        wait_gate(&self.started, None);
    }

    pub fn release_ready(&self) {
        open_gate(&self.ready);
    }

    pub fn release_cleanup(&self) {
        open_gate(&self.cleanup);
    }

    pub fn wait_preparation_started(&self) {
        wait_gate(&self.preparation_started, None);
    }

    pub fn release_preparation(&self) {
        open_gate(&self.preparation);
    }

    #[must_use]
    pub fn has_executed(&self) -> bool {
        self.executed.load(Ordering::Acquire)
    }

    pub fn has_started(&self) -> bool {
        let (started, _) = &*self.started;
        *started
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Generation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub async fn wait_for_ack(&mut self) -> Result<(), Error> {
        let Some(acknowledged) = self.acknowledged.take() else {
            return Ok(());
        };
        acknowledged.await.unwrap_or(Err(Error::WorkerUnavailable))
    }

    pub async fn recv_terminal(&mut self) -> Result<Event, Error> {
        let terminal = self.terminal.take().ok_or(Error::WorkerUnavailable)?;
        terminal.await.unwrap_or(Err(Error::WorkerUnavailable))
    }
}

#[doc(hidden)]
pub fn deterministic_generation(events: impl IntoIterator<Item = Event>) -> Generation {
    let events = events.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = mpsc::channel(events.len().max(1));
    let (terminal_sender, terminal_receiver) = tokio::sync::oneshot::channel();
    let mut terminal = None;
    for event in events {
        match event {
            Event::Content(_) => sender
                .try_send(event)
                .expect("deterministic channel is sized for its script"),
            Event::Finished { .. } => terminal = Some(Ok(event)),
            Event::Failed(message) => terminal = Some(Err(Error::Generation(message))),
        }
    }
    drop(sender);
    if let Some(terminal) = terminal {
        let _ = terminal_sender.send(terminal);
    }
    Generation {
        events: receiver,
        cancelled: Arc::new(AtomicBool::new(false)),
        acknowledged: None,
        terminal: Some(terminal_receiver),
    }
}

#[doc(hidden)]
#[must_use]
pub fn scripted_engine(script: Vec<Event>, command_capacity: usize) -> (Engine, ScriptedControl) {
    scripted_engine_with_preparation(script, command_capacity, true, false)
}

#[doc(hidden)]
#[must_use]
pub fn scripted_slow_preparation_engine(
    script: Vec<Event>,
    command_capacity: usize,
) -> (Engine, ScriptedControl) {
    scripted_engine_with_preparation(script, command_capacity, false, false)
}

#[doc(hidden)]
#[must_use]
pub fn scripted_preparation_panicking_engine(command_capacity: usize) -> (Engine, ScriptedControl) {
    scripted_engine_with_preparation(Vec::new(), command_capacity, true, true)
}

fn scripted_engine_with_preparation(
    script: Vec<Event>,
    command_capacity: usize,
    preparation_ready: bool,
    panic_preparation: bool,
) -> (Engine, ScriptedControl) {
    let (commands, receiver) = mpsc::channel(command_capacity);
    let control = ScriptedControl {
        ready: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        preparation: Arc::new((Mutex::new(preparation_ready), std::sync::Condvar::new())),
        preparation_started: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        cleanup: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        started: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        executed: Arc::new(AtomicBool::new(false)),
        panic_preparation,
    };
    let worker_control = control.clone();
    let health = Arc::new(AtomicU8::new(0));
    let worker_health = Arc::clone(&health);
    let join = std::thread::spawn(move || {
        scripted_worker(receiver, script, worker_control, worker_health);
    });
    (
        Engine {
            inner: Arc::new(EngineInner {
                commands,
                join: Mutex::new(Some(join)),
                active: Mutex::new(Vec::new()),
                health,
            }),
        },
        control,
    )
}

#[doc(hidden)]
#[must_use]
pub fn scripted_context_limit_engine(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: usize,
) -> Engine {
    let (commands, mut receiver) = mpsc::channel(1);
    let health = Arc::new(AtomicU8::new(0));
    let join = std::thread::spawn(move || {
        while let Some(command) = receiver.blocking_recv() {
            match command {
                Command::Reserve {
                    cancelled,
                    reserved,
                    readiness,
                    acknowledged,
                    terminal,
                    decision,
                    ..
                } => {
                    if reserved.send(Ok(())).is_err() {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    }
                    if !matches!(
                        wait_reservation_decision(decision, &cancelled),
                        ReservationDecision::Commit(_)
                    ) {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    }
                    let error = Error::ContextLimit {
                        prompt_tokens,
                        max_tokens,
                        context_size,
                    };
                    let _ = terminal.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error.clone()));
                    let _ = readiness.send(Err(error));
                }
                Command::Shutdown => break,
            }
        }
    });
    Engine {
        inner: Arc::new(EngineInner {
            commands,
            join: Mutex::new(Some(join)),
            active: Mutex::new(Vec::new()),
            health,
        }),
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the dedicated thread owns its script and synchronization control"
)]
fn scripted_worker(
    mut commands: mpsc::Receiver<Command>,
    script: Vec<Event>,
    control: ScriptedControl,
    health: Arc<AtomicU8>,
) {
    while let Some(command) = commands.blocking_recv() {
        match command {
            Command::Reserve {
                cancelled,
                events,
                reserved,
                readiness,
                acknowledged,
                terminal,
                decision,
                ..
            } => {
                open_gate(&control.started);
                wait_gate(&control.ready, Some(&cancelled));
                if cancelled.load(Ordering::Acquire) {
                    let _ = reserved.send(Err(Error::Cancelled));
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                if reserved.send(Ok(())).is_err() {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let ReservationDecision::Commit(_request) =
                    wait_reservation_decision(decision, &cancelled)
                else {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                };
                open_gate(&control.preparation_started);
                wait_gate(&control.preparation, Some(&cancelled));
                if cancelled.load(Ordering::Acquire) {
                    let error = Error::Cancelled;
                    let _ = terminal.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error.clone()));
                    let _ = readiness.send(Err(error));
                    continue;
                }
                if control.panic_preparation {
                    let error = Error::WorkerPanic("scripted preparation panic".to_string());
                    health.store(1, Ordering::Release);
                    let _ = terminal.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error.clone()));
                    let _ = readiness.send(Err(error));
                    break;
                }
                control.executed.store(true, Ordering::Release);
                if readiness.send(Ok(())).is_err() {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut result = Ok(Event::Finished {
                        reason: FinishReason::Stop,
                        usage: Usage {
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            timings: Timings::default(),
                        },
                    });
                    for event in script.iter().cloned() {
                        if matches!(&event, Event::Failed(message) if message == "__orchion_test_panic__")
                        {
                            panic!("scripted worker panic");
                        }
                        match event {
                            Event::Content(_) => {
                                if let Err(error) = send_event(&events, event, &cancelled) {
                                    result = Err(error);
                                    break;
                                }
                            }
                            Event::Finished { .. } => result = Ok(event),
                            Event::Failed(message) => result = Err(Error::Generation(message)),
                        }
                    }
                    result
                }));
                wait_gate(&control.cleanup, None);
                let result = execution
                    .unwrap_or_else(|payload| Err(Error::WorkerPanic(panic_message(&payload))));
                if matches!(result, Err(Error::WorkerPanic(_))) {
                    health.store(1, Ordering::Release);
                }
                let _ = terminal.send(result.clone());
                let _ = acknowledged.send(result.map(|_| ()));
            }
            Command::Shutdown => break,
        }
    }
}

fn wait_gate(gate: &(Mutex<bool>, std::sync::Condvar), cancelled: Option<&AtomicBool>) {
    let (open, wake) = gate;
    let mut open = open
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while !*open && !cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Acquire)) {
        let (next, _) = wake
            .wait_timeout(open, std::time::Duration::from_millis(1))
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        open = next;
    }
}

fn open_gate(gate: &(Mutex<bool>, std::sync::Condvar)) {
    let (open, wake) = gate;
    *open
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    wake.notify_all();
}

fn wait_reservation_decision(
    mut decision: tokio::sync::oneshot::Receiver<ReservationDecision>,
    cancelled: &AtomicBool,
) -> ReservationDecision {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return ReservationDecision::Abort;
        }
        match decision.try_recv() {
            Ok(decision) => return decision,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                return ReservationDecision::Abort;
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Engine {
    pub fn load(path: PathBuf, config: RuntimeConfig) -> Result<Self, Error> {
        validate_config(&config)?;
        let backend = BackendOwner::acquire()?;
        let (commands, receiver) = mpsc::channel(config.request_queue_capacity);
        let health = Arc::new(AtomicU8::new(0));
        let worker_health = Arc::clone(&health);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let join = std::thread::Builder::new()
            .name("orchion-llama-model".to_string())
            .spawn(move || {
                worker_main(
                    &backend,
                    &path,
                    &config,
                    receiver,
                    &ready_tx,
                    &worker_health,
                );
            })
            .map_err(|error| Error::WorkerStart(error.to_string()))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                inner: Arc::new(EngineInner {
                    commands,
                    join: Mutex::new(Some(join)),
                    active: Mutex::new(Vec::new()),
                    health,
                }),
            }),
            Ok(Err(error)) => {
                let _ = join.join();
                Err(Error::WorkerStart(error))
            }
            Err(error) => {
                let _ = join.join();
                Err(Error::WorkerStart(error.to_string()))
            }
        }
    }

    pub async fn reserve(
        &self,
        request: Request,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        let cancelled = Arc::new(AtomicBool::new(false));
        {
            let mut active = self
                .inner
                .active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            active.retain(|generation| generation.strong_count() > 0);
            active.push(Arc::downgrade(&cancelled));
        }
        let (events, receiver) = mpsc::channel(event_capacity);
        let (reserved, reservation_ack) = tokio::sync::oneshot::channel();
        let (readiness, worker_readiness) = tokio::sync::oneshot::channel();
        let (acknowledged, acknowledgement) = tokio::sync::oneshot::channel();
        let (terminal, worker_terminal) = tokio::sync::oneshot::channel();
        let (decision, worker_decision) = tokio::sync::oneshot::channel();
        let mut pending = PendingGeneration {
            cancelled: Arc::clone(&cancelled),
            committed: false,
        };
        self.inner
            .commands
            .send(Command::Reserve {
                cancelled: Arc::clone(&cancelled),
                events,
                reserved,
                readiness,
                acknowledged,
                terminal,
                decision: worker_decision,
            })
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        reservation_ack
            .await
            .unwrap_or(Err(Error::WorkerUnavailable))?;
        pending.committed = true;
        Ok(Reservation {
            request: Some(request),
            events: Some(receiver),
            cancelled,
            acknowledged: Some(acknowledgement),
            readiness: Some(worker_readiness),
            terminal: Some(worker_terminal),
            decision: Some(decision),
            transferred: false,
        })
    }

    pub async fn generate(
        &self,
        request: Request,
        event_capacity: usize,
    ) -> Result<Generation, Error> {
        let mut reservation = self.reserve(request, event_capacity).await?;
        reservation.commit().await
    }

    pub fn shutdown(&self) {
        self.inner.cancel_active();
        send_shutdown(&self.inner.commands);
        if let Some(join) = self
            .inner
            .join
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = join.join();
        }
    }

    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.inner.health.load(Ordering::Acquire) == 0
    }
}

struct PendingGeneration {
    cancelled: Arc<AtomicBool>,
    committed: bool,
}

impl Drop for PendingGeneration {
    fn drop(&mut self) {
        if !self.committed {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.cancel_active();
        send_shutdown(&self.commands);
        if let Some(join) = self
            .join
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = join.join();
        }
    }
}

impl EngineInner {
    fn cancel_active(&self) {
        for generation in self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter_map(Weak::upgrade)
        {
            generation.store(true, Ordering::Release);
        }
    }
}

fn send_shutdown(commands: &mpsc::Sender<Command>) {
    let mut command = Command::Shutdown;
    loop {
        match commands.try_send(command) {
            Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => return,
            Err(mpsc::error::TrySendError::Full(pending)) => {
                command = pending;
                std::thread::yield_now();
            }
        }
    }
}

fn validate_config(config: &RuntimeConfig) -> Result<(), Error> {
    if config.parallel_sequences != 1 {
        return Err(Error::InvalidConfig(
            "parallel_sequences must be 1 in the text tracer".to_string(),
        ));
    }
    if config.batch_size == 0 || config.micro_batch_size == 0 {
        return Err(Error::InvalidConfig(
            "batch sizes must be nonzero".to_string(),
        ));
    }
    if config.request_queue_capacity == 0 || config.event_queue_capacity == 0 {
        return Err(Error::InvalidConfig(
            "queue capacities must be nonzero".to_string(),
        ));
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "owns native model/context lifecycle plus supervised request cleanup"
)]
fn worker_main(
    backend: &BackendOwner,
    path: &Path,
    config: &RuntimeConfig,
    mut commands: mpsc::Receiver<Command>,
    ready: &std::sync::mpsc::SyncSender<Result<(), String>>,
    health: &AtomicU8,
) {
    let params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
    let model = match LlamaModel::load_from_file(&backend.backend, path, &params) {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let template = match effective_template(&model, config) {
        Ok(template) => template,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(config.context_size)
        .with_n_batch(config.batch_size)
        .with_n_ubatch(config.micro_batch_size)
        .with_n_seq_max(1);
    if config.threads > 0 {
        context_params = context_params
            .with_n_threads(config.threads)
            .with_n_threads_batch(config.threads);
    }
    let mut context = match model.new_context(&backend.backend, context_params) {
        Ok(context) => context,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    while let Some(command) = commands.blocking_recv() {
        match command {
            Command::Reserve {
                cancelled,
                events,
                reserved,
                readiness,
                acknowledged,
                terminal,
                decision,
            } => {
                if reserved.send(Ok(())).is_err() {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let ReservationDecision::Commit(request) =
                    wait_reservation_decision(decision, &cancelled)
                else {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                };
                if cancelled.load(Ordering::Acquire) {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let preparation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prepare_generation(&model, &template, config, &request)
                }));
                let tokens = match preparation {
                    Ok(Ok(tokens)) if !cancelled.load(Ordering::Acquire) => tokens,
                    Ok(Ok(_)) => {
                        let error = Error::Cancelled;
                        let _ = terminal.send(Err(error.clone()));
                        let _ = acknowledged.send(Err(error.clone()));
                        let _ = readiness.send(Err(error));
                        continue;
                    }
                    Ok(Err(error)) => {
                        context.clear_kv_cache();
                        let _ = terminal.send(Err(error.clone()));
                        let _ = acknowledged.send(Err(error.clone()));
                        let _ = readiness.send(Err(error));
                        continue;
                    }
                    Err(payload) => {
                        health.store(1, Ordering::Release);
                        let error = Error::WorkerPanic(panic_message(&payload));
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            context.clear_kv_cache();
                        }));
                        let _ = terminal.send(Err(error.clone()));
                        let _ = acknowledged.send(Err(error.clone()));
                        let _ = readiness.send(Err(error));
                        break;
                    }
                };
                if cancelled.load(Ordering::Acquire) {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                if readiness.send(Ok(())).is_err() {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_generation(
                        &model,
                        &mut context,
                        config,
                        &request,
                        &tokens,
                        &cancelled,
                        &events,
                    )
                }));
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    context.clear_kv_cache();
                }));
                let result = match (execution, cleanup) {
                    (Ok(result), Ok(())) => result,
                    (Err(payload), _) | (_, Err(payload)) => {
                        Err(Error::WorkerPanic(panic_message(&payload)))
                    }
                };
                let worker_panicked = matches!(result, Err(Error::WorkerPanic(_)));
                if worker_panicked {
                    health.store(1, Ordering::Release);
                }
                let _ = terminal.send(result.clone());
                let _ = acknowledged.send(result.map(|_| ()));
                if worker_panicked {
                    break;
                }
            }
            Command::Shutdown => break,
        }
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "unknown panic payload".to_string())
        },
        |message| (*message).to_string(),
    )
}

enum EffectiveTemplate {
    LlamaCpp(LlamaChatTemplate),
    Jinja {
        source: String,
        enable_thinking: bool,
    },
}

fn effective_template(
    model: &LlamaModel,
    config: &RuntimeConfig,
) -> Result<EffectiveTemplate, String> {
    let source = match config.chat_template.as_deref() {
        Some(template) => template.to_string(),
        None => model
            .chat_template(None)
            .map_err(|error| error.to_string())?
            .to_string()
            .map_err(|error| error.to_string())?,
    };
    let template = match config.template_engine {
        TemplateEngine::LlamaCpp => EffectiveTemplate::LlamaCpp(
            LlamaChatTemplate::new(&source).map_err(|error| error.to_string())?,
        ),
        TemplateEngine::Jinja => EffectiveTemplate::Jinja {
            source,
            enable_thinking: config.enable_thinking,
        },
    };
    let canary = [
        Message {
            role: "system".to_string(),
            content: "system".to_string(),
        },
        Message {
            role: "developer".to_string(),
            content: "developer".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "user".to_string(),
        },
        Message {
            role: "assistant".to_string(),
            content: "assistant".to_string(),
        },
    ];
    apply_effective_template(model, &template, &canary, true)
        .map_err(|error| format!("effective chat template cannot be applied: {error}"))?;
    Ok(template)
}

fn apply_effective_template(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    messages: &[Message],
    add_generation_prompt: bool,
) -> Result<String, String> {
    let messages = normalize_text_messages(messages);
    match template {
        EffectiveTemplate::LlamaCpp(template) => {
            let messages = messages
                .iter()
                .map(|message| {
                    LlamaChatMessage::new(message.role.clone(), message.content.clone())
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            model
                .apply_chat_template(template, &messages, add_generation_prompt)
                .map_err(|error| error.to_string())
        }
        EffectiveTemplate::Jinja {
            source,
            enable_thinking,
        } => {
            let bos_token = special_token_text(model, model.token_bos())?;
            let eos_token = special_token_text(model, model.token_eos())?;
            render_jinja_template(
                source,
                &messages,
                add_generation_prompt,
                &bos_token,
                &eos_token,
                *enable_thinking,
            )
        }
    }
}

fn normalize_text_messages(messages: &[Message]) -> Vec<Message> {
    let mut instructions = Vec::new();
    let mut conversation = Vec::new();
    for message in messages {
        if matches!(message.role.as_str(), "system" | "developer") {
            instructions.push(message.content.as_str());
        } else {
            conversation.push(message.clone());
        }
    }
    if !instructions.is_empty() {
        conversation.insert(
            0,
            Message {
                role: "system".to_string(),
                content: instructions.join("\n"),
            },
        );
    }
    conversation
}

fn special_token_text(model: &LlamaModel, token: LlamaToken) -> Result<String, String> {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(|error| error.to_string())
}

fn render_jinja_template(
    source: &str,
    messages: &[Message],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
    enable_thinking: bool,
) -> Result<String, String> {
    let mut environment = minijinja::Environment::new();
    environment.set_unknown_method_callback(|_state, value, method, args| {
        use minijinja::value::{Value, from_args};
        let text = value
            .as_str()
            .ok_or_else(|| minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))?;
        match method {
            "startswith" => {
                let (needle,): (String,) = from_args(args)?;
                Ok(Value::from(text.starts_with(&needle)))
            }
            "endswith" => {
                let (needle,): (String,) = from_args(args)?;
                Ok(Value::from(text.ends_with(&needle)))
            }
            "split" => {
                let (separator,): (String,) = from_args(args)?;
                Ok(Value::from_serialize(
                    text.split(&separator).collect::<Vec<_>>(),
                ))
            }
            "rstrip" => {
                let (characters,): (String,) = from_args(args)?;
                Ok(Value::from(
                    text.trim_end_matches(|ch| characters.contains(ch)),
                ))
            }
            "lstrip" => {
                let (characters,): (String,) = from_args(args)?;
                Ok(Value::from(
                    text.trim_start_matches(|ch| characters.contains(ch)),
                ))
            }
            _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
        }
    });
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment.add_function("strftime_now", |format: String| -> String {
        chrono::Utc::now().format(&format).to_string()
    });
    environment
        .add_template("chat", source)
        .map_err(|error| error.to_string())?;
    environment
        .get_template("chat")
        .map_err(|error| error.to_string())?
        .render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token,
            eos_token => eos_token,
            tools => Vec::<String>::new(),
            enable_thinking => enable_thinking,
            add_vision_id => false,
        })
        .map_err(|error| error.to_string())
}

fn prepare_generation(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    request: &Request,
) -> Result<Vec<LlamaToken>, Error> {
    let prompt = apply_effective_template(model, template, &request.messages, true)
        .map_err(Error::Generation)?;
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| Error::Generation(error.to_string()))?;
    let context_size = effective_context_size(model, config);
    if tokens.len() >= context_size || request.options.max_tokens > context_size - tokens.len() {
        return Err(Error::ContextLimit {
            prompt_tokens: tokens.len(),
            max_tokens: request.options.max_tokens,
            context_size,
        });
    }
    Ok(tokens)
}

fn effective_context_size(model: &LlamaModel, config: &RuntimeConfig) -> usize {
    config
        .context_size
        .map_or_else(|| model.n_ctx_train() as usize, |size| size.get() as usize)
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps one native generation transaction and its cleanup checkpoints together"
)]
fn run_generation(
    model: &LlamaModel,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    config: &RuntimeConfig,
    request: &Request,
    tokens: &[LlamaToken],
    cancelled: &AtomicBool,
    events: &mpsc::Sender<Event>,
) -> Result<Event, Error> {
    if cancelled.load(Ordering::Acquire) {
        return Err(Error::Cancelled);
    }
    context.reset_timings();
    let context_size = effective_context_size(model, config);
    let batch_size =
        usize::try_from(config.batch_size).map_err(|error| Error::Generation(error.to_string()))?;
    let mut batch = LlamaBatch::new(batch_size, 1);
    for (chunk_index, chunk) in tokens.chunks(batch_size).enumerate() {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let base = chunk_index * batch_size;
        for (offset, token) in chunk.iter().copied().enumerate() {
            let index = base + offset;
            batch
                .add(
                    token,
                    i32::try_from(index).map_err(|error| Error::Generation(error.to_string()))?,
                    &[0],
                    index + 1 == tokens.len(),
                )
                .map_err(|error| Error::Generation(error.to_string()))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| Error::Generation(error.to_string()))?;
        batch.clear();
    }
    // Native decode may be asynchronous; reading logits synchronizes before
    // taking the prompt-phase performance snapshot.
    let _ = context.get_logits();
    let prompt_timings = context.timings();
    let mut samplers = vec![LlamaSampler::penalties(
        model.n_vocab(),
        i32::try_from(context_size).map_err(|error| Error::Generation(error.to_string()))?,
        request.options.repeat_penalty,
        request.options.frequency_penalty,
        request.options.presence_penalty,
    )];
    if request.options.temperature > 0.0 {
        samplers.extend([
            LlamaSampler::top_k(request.options.top_k),
            LlamaSampler::top_p(request.options.top_p, 1),
            LlamaSampler::min_p(request.options.min_p, 1),
            LlamaSampler::temp(request.options.temperature),
            LlamaSampler::dist(request.options.seed),
        ]);
    } else {
        samplers.push(LlamaSampler::greedy());
    }
    let mut sampler = LlamaSampler::chain_simple(samplers);
    sampler.accept_many(tokens);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut stop_filter = StopFilter::new(request.options.stop.clone());
    let mut completion_tokens = 0;
    for position in tokens.len()..tokens.len() + request.options.max_tokens {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let token = sampler.sample(context, -1);
        if model.is_eog_token(token) {
            stop_filter.flush(events, cancelled)?;
            return Ok(finish(
                context,
                prompt_timings,
                FinishReason::Stop,
                Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens,
                    timings: Timings::default(),
                },
            ));
        }
        completion_tokens += 1;
        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .map_err(|error| Error::Generation(error.to_string()))?;
        if stop_filter.push(&piece, events, cancelled)? {
            return Ok(finish(
                context,
                prompt_timings,
                FinishReason::Stop,
                Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens,
                    timings: Timings::default(),
                },
            ));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if completion_tokens == request.options.max_tokens {
            break;
        }
        batch
            .add(
                token,
                i32::try_from(position).map_err(|error| Error::Generation(error.to_string()))?,
                &[0],
                true,
            )
            .map_err(|error| Error::Generation(error.to_string()))?;
        context
            .decode(&mut batch)
            .map_err(|error| Error::Generation(error.to_string()))?;
        batch.clear();
    }
    stop_filter.flush(events, cancelled)?;
    Ok(finish(
        context,
        prompt_timings,
        FinishReason::Length,
        Usage {
            prompt_tokens: tokens.len(),
            completion_tokens,
            timings: Timings::default(),
        },
    ))
}

fn finish(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    prompt_timings: llama_cpp_2::timing::LlamaTimings,
    reason: FinishReason,
    mut usage: Usage,
) -> Event {
    usage.timings = timings_from_native(
        prompt_timings,
        context.timings(),
        usage.prompt_tokens,
        usage.completion_tokens,
    );
    Event::Finished { reason, usage }
}

fn timings_from_native(
    prompt: llama_cpp_2::timing::LlamaTimings,
    completed: llama_cpp_2::timing::LlamaTimings,
    prompt_n: usize,
    predicted_n: usize,
) -> Timings {
    // llama.cpp classifies one-token batches as generation evals. Snapshotting at
    // the phase boundary keeps a final one-token prompt batch in prefill timing.
    let prompt_ms = finite_nonnegative(
        finite_nonnegative(prompt.t_p_eval_ms()) + finite_nonnegative(prompt.t_eval_ms()),
    );
    let predicted_ms = finite_nonnegative(
        timing_delta(completed.t_p_eval_ms(), prompt.t_p_eval_ms())
            + timing_delta(completed.t_eval_ms(), prompt.t_eval_ms()),
    );
    let (prompt_per_token_ms, prompt_per_second) = timing_rates(prompt_n, prompt_ms);
    let (predicted_per_token_ms, predicted_per_second) =
        timing_rates(predicted_n.saturating_sub(1), predicted_ms);
    Timings {
        cache_n: 0,
        prompt_n,
        prompt_ms,
        prompt_per_token_ms,
        prompt_per_second,
        predicted_n,
        predicted_ms,
        predicted_per_token_ms,
        predicted_per_second,
    }
}

fn timing_delta(completed: f64, prompt: f64) -> f64 {
    finite_nonnegative(completed - prompt)
}

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

fn timing_rates(tokens: usize, milliseconds: f64) -> (f64, f64) {
    if tokens == 0 || milliseconds <= 0.0 {
        return (0.0, 0.0);
    }
    let Ok(tokens) = u32::try_from(tokens) else {
        return (0.0, 0.0);
    };
    let tokens = f64::from(tokens);
    (
        finite_nonnegative(milliseconds / tokens),
        finite_nonnegative(tokens * 1_000.0 / milliseconds),
    )
}

fn send_event(
    events: &mpsc::Sender<Event>,
    mut event: Event,
    cancelled: &AtomicBool,
) -> Result<(), Error> {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        match events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(Error::Cancelled);
            }
            Err(mpsc::error::TrySendError::Full(pending)) => {
                event = pending;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

struct StopFilter {
    stops: Vec<String>,
    pending: String,
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
        }
    }

    fn push(
        &mut self,
        piece: &str,
        events: &mpsc::Sender<Event>,
        cancelled: &AtomicBool,
    ) -> Result<bool, Error> {
        self.pending.push_str(piece);
        if let Some(index) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            self.emit_prefix(index, events, cancelled)?;
            self.pending.clear();
            return Ok(true);
        }
        let retained = longest_stop_prefix_suffix(&self.pending, &self.stops);
        let emit_len = self.pending.len() - retained;
        self.emit_prefix(emit_len, events, cancelled)?;
        Ok(false)
    }

    fn flush(&mut self, events: &mpsc::Sender<Event>, cancelled: &AtomicBool) -> Result<(), Error> {
        let len = self.pending.len();
        self.emit_prefix(len, events, cancelled)
    }

    fn emit_prefix(
        &mut self,
        len: usize,
        events: &mpsc::Sender<Event>,
        cancelled: &AtomicBool,
    ) -> Result<(), Error> {
        if len == 0 {
            return Ok(());
        }
        let suffix = self.pending.split_off(len);
        let output = std::mem::replace(&mut self.pending, suffix);
        send_event(events, Event::Content(output), cancelled)
    }
}

fn longest_stop_prefix_suffix(text: &str, stops: &[String]) -> usize {
    stops
        .iter()
        .flat_map(|stop| {
            stop.char_indices()
                .map(|(index, _)| &stop[..index])
                .chain(std::iter::once(stop.as_str()))
        })
        .filter(|prefix| !prefix.is_empty() && text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_metadata_records_pins_toolchain_features_and_native_environment() {
        let metadata = build_metadata();
        assert_eq!(metadata.binding_revision, BINDING_REVISION);
        assert_eq!(metadata.llama_cpp_revision, LLAMA_CPP_REVISION);
        assert_eq!(metadata.binding_features, "common,mtmd");
        assert!(metadata.rustc_version.starts_with("rustc "));
        assert!(!metadata.toolchain.is_empty());
        assert!(!metadata.target.is_empty());
        let resolved = metadata.cmake_resolved;
        for value in [
            resolved.build_type,
            resolved.generator,
            resolved.build_shared_libs,
            resolved.ggml_metal,
            resolved.ggml_openmp,
            resolved.ggml_cuda,
            resolved.ggml_vulkan,
            resolved.ggml_native,
        ] {
            assert!(!matches!(value, "unset" | "unavailable" | ""));
        }
        assert_eq!(resolved.cache_sha256.len(), 64);
        assert!(!Path::new(resolved.cache_path_relative).is_absolute());
        for compiler in [resolved.c_compiler, resolved.cxx_compiler] {
            assert!(!compiler.basename.contains(['/', '\\']));
            assert!(!matches!(compiler.id, "unset" | "unavailable" | ""));
            assert!(!matches!(compiler.version, "unset" | "unavailable" | ""));
        }
        assert!(serde_json::from_str::<serde_json::Value>(&build_metadata_json()).is_ok());
    }

    #[test]
    fn stop_filter_retains_cross_piece_prefix_without_leaking_it() {
        let (sender, mut receiver) = mpsc::channel(4);
        let cancelled = AtomicBool::new(false);
        let mut filter = StopFilter::new(vec!["END".to_string()]);
        assert!(!filter.push("hello E", &sender, &cancelled).unwrap());
        assert!(filter.push("ND ignored", &sender, &cancelled).unwrap());
        assert_eq!(
            receiver.try_recv().unwrap(),
            Event::Content("hello ".to_string())
        );
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn config_rejects_parallel_sequences_above_one() {
        let config = RuntimeConfig {
            context_size: NonZeroU32::new(128),
            batch_size: 32,
            micro_batch_size: 32,
            threads: 1,
            gpu_layers: 0,
            parallel_sequences: 2,
            request_queue_capacity: 1,
            event_queue_capacity: 1,
            chat_template: None,
            template_engine: TemplateEngine::LlamaCpp,
            enable_thinking: false,
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn native_timings_map_to_llama_server_fields_and_finite_rates() {
        let prompt = llama_cpp_2::timing::LlamaTimings::new(0.0, 0.0, 18.0, 2.0, 9, 1, 0);
        let completed = llama_cpp_2::timing::LlamaTimings::new(0.0, 0.0, 18.0, 10.0, 9, 5, 0);
        let timings = timings_from_native(prompt, completed, 10, 5);
        assert_eq!(timings.cache_n, 0);
        assert_eq!(timings.prompt_n, 10);
        assert!((timings.prompt_ms - 20.0).abs() < f64::EPSILON);
        assert!((timings.prompt_per_token_ms - 2.0).abs() < f64::EPSILON);
        assert!((timings.prompt_per_second - 500.0).abs() < f64::EPSILON);
        assert_eq!(timings.predicted_n, 5);
        assert!((timings.predicted_ms - 8.0).abs() < f64::EPSILON);
        assert!((timings.predicted_per_token_ms - 2.0).abs() < f64::EPSILON);
        assert!((timings.predicted_per_second - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn timing_rates_are_zero_for_invalid_or_zero_elapsed_values() {
        for milliseconds in [0.0, f64::NAN, f64::INFINITY, -1.0] {
            let (per_token, per_second) = timing_rates(4, finite_nonnegative(milliseconds));
            assert_eq!((per_token, per_second), (0.0, 0.0));
            assert!(per_token.is_finite());
            assert!(per_second.is_finite());
        }
        assert_eq!(timing_rates(0, 4.0), (0.0, 0.0));
    }

    #[test]
    fn jinja_text_subset_renders_qwen_style_roles_and_generation_prompt() {
        let template = "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}";
        let rendered = render_jinja_template(
            template,
            &[
                Message {
                    role: "system".to_string(),
                    content: "policy".to_string(),
                },
                Message {
                    role: "developer".to_string(),
                    content: "format".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "prior".to_string(),
                },
            ],
            true,
            "<s>",
            "</s>",
            true,
        )
        .unwrap();
        assert!(rendered.starts_with("<s><|im_start|>system\npolicy"));
        assert!(rendered.contains("<|im_start|>developer\nformat"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn jinja_raise_exception_is_a_render_error_without_fallback() {
        let error = render_jinja_template(
            "{{ raise_exception('bad role') }}",
            &[],
            true,
            "",
            "",
            false,
        )
        .unwrap_err();
        assert!(error.contains("bad role"));
    }

    #[test]
    fn jinja_text_subset_supports_qwen_namespace_macro_and_reverse_scan() {
        let template = "
{%- macro render_content(content) %}{{- content|trim }}{%- endmacro %}
{%- set ns = namespace(last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
  {%- if message.role == 'user' %}{%- set ns.last_query_index = (messages|length - 1) - loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + render_content(message.content) + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n' }}{%- endif %}
";
        let rendered = render_jinja_template(
            template,
            &[Message {
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            true,
            "",
            "",
            true,
        )
        .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn qwen35_template_text_branch_passes_four_role_canary_after_instruction_normalization() {
        let source = include_str!("../tests/fixtures/qwen35_chat_template.jinja");
        let messages = normalize_text_messages(&[
            Message {
                role: "system".to_string(),
                content: "system".to_string(),
            },
            Message {
                role: "developer".to_string(),
                content: "developer".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "prior".to_string(),
            },
        ]);
        let thinking = render_jinja_template(source, &messages, true, "", "", true).unwrap();
        assert!(thinking.contains("<|im_start|>system\nsystem\ndeveloper<|im_end|>"));
        assert!(thinking.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"));

        let no_thinking = render_jinja_template(source, &messages, true, "", "", false).unwrap();
        assert!(no_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn bounded_event_queue_applies_worker_backpressure() {
        let (sender, mut receiver) = mpsc::channel(1);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let worker = std::thread::spawn(move || {
            sender
                .blocking_send(Event::Content("one".to_string()))
                .unwrap();
            sender
                .blocking_send(Event::Content("two".to_string()))
                .unwrap();
            worker_finished.store(true, Ordering::Release);
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!finished.load(Ordering::Acquire));
        assert_eq!(
            receiver.blocking_recv(),
            Some(Event::Content("one".to_string()))
        );
        assert_eq!(
            receiver.blocking_recv(),
            Some(Event::Content("two".to_string()))
        );
        worker.join().unwrap();
        assert!(finished.load(Ordering::Acquire));
    }

    #[test]
    fn cancellation_interrupts_a_full_event_queue() {
        let (sender, _receiver) = mpsc::channel(1);
        sender.try_send(Event::Content("full".to_string())).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = std::thread::spawn(move || {
            send_event(
                &sender,
                Event::Content("blocked".to_string()),
                &worker_cancelled,
            )
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        cancelled.store(true, Ordering::Release);
        assert!(matches!(
            worker.join().unwrap().unwrap_err(),
            Error::Cancelled
        ));
    }

    #[test]
    fn cancellation_stops_before_another_decode_step() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let decodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_cancelled = Arc::clone(&cancelled);
        let worker_decodes = Arc::clone(&decodes);
        let worker = std::thread::spawn(move || {
            while !worker_cancelled.load(Ordering::Acquire) {
                worker_decodes.fetch_add(1, Ordering::AcqRel);
                std::thread::park();
            }
        });
        while decodes.load(Ordering::Acquire) == 0 {
            std::thread::yield_now();
        }
        cancelled.store(true, Ordering::Release);
        worker.thread().unpark();
        worker.join().unwrap();
        assert_eq!(decodes.load(Ordering::Acquire), 1);
    }

    #[test]
    fn explicit_shutdown_command_joins_worker() {
        let (sender, mut receiver) = mpsc::channel(1);
        let worker = std::thread::spawn(move || {
            while let Some(command) = receiver.blocking_recv() {
                if matches!(command, Command::Shutdown) {
                    break;
                }
            }
        });
        send_shutdown(&sender);
        worker.join().unwrap();
    }

    #[test]
    fn shutdown_cancels_registered_generations_before_join() {
        let (commands, _receiver) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let inner = EngineInner {
            commands,
            join: Mutex::new(None),
            active: Mutex::new(vec![Arc::downgrade(&cancelled)]),
            health: Arc::new(AtomicU8::new(0)),
        };
        inner.cancel_active();
        assert!(cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn worker_panic_is_forwarded_through_cleanup_ack() {
        let (engine, control) =
            scripted_engine(vec![Event::Failed("__orchion_test_panic__".to_string())], 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let start = engine.generate(test_request(), 1);
            tokio::pin!(start);
            let release = control.clone();
            std::thread::spawn(move || {
                release.wait_started();
                release.release_ready();
            });
            let mut generation = start.await.unwrap();
            control.release_cleanup();
            assert!(matches!(
                generation.wait_for_ack().await,
                Err(Error::WorkerPanic(message)) if message == "scripted worker panic"
            ));
            assert!(!engine.is_healthy());
        });
        engine.shutdown();
    }

    #[test]
    fn preparation_panic_marks_worker_unhealthy_before_commit_error() {
        let (engine, control) = scripted_preparation_panicking_engine(1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let reserve = engine.reserve(test_request(), 1);
            tokio::pin!(reserve);
            let release = control.clone();
            std::thread::spawn(move || {
                release.wait_started();
                release.release_ready();
            });
            let mut reservation = reserve.await.unwrap();
            assert!(matches!(
                reservation.commit().await,
                Err(Error::WorkerPanic(message)) if message == "scripted preparation panic"
            ));
            assert!(!engine.is_healthy());
        });
        engine.shutdown();
    }

    #[test]
    fn dropped_reservation_aborts_without_executing_script() {
        let (engine, control) =
            scripted_engine(vec![Event::Content("must not execute".to_string())], 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let reserve = engine.reserve(test_request(), 1);
            tokio::pin!(reserve);
            let release = control.clone();
            std::thread::spawn(move || {
                release.wait_started();
                release.release_ready();
            });
            let reservation = reserve.await.unwrap();
            drop(reservation);
            std::thread::sleep(std::time::Duration::from_millis(10));
            assert!(!control.has_executed());
        });
        engine.shutdown();
    }

    #[test]
    fn shutdown_aborts_an_uncommitted_reservation_without_hanging() {
        let (engine, control) = scripted_engine(Vec::new(), 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let reservation = runtime.block_on(async {
            let reserve = engine.reserve(test_request(), 1);
            tokio::pin!(reserve);
            let release = control.clone();
            std::thread::spawn(move || {
                release.wait_started();
                release.release_ready();
            });
            reserve.await.unwrap()
        });
        engine.shutdown();
        assert!(!control.has_executed());
        drop(reservation);
    }

    fn test_request() -> Request {
        Request {
            messages: vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            options: GenerationOptions {
                max_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repeat_penalty: 1.0,
                seed: 1,
                stop: Vec::new(),
            },
        }
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_load_generate_and_shutdown() {
        let path = PathBuf::from(
            std::env::var("ORCHION_TEST_GGUF").expect("ORCHION_TEST_GGUF must be set"),
        );
        let engine = Engine::load(
            path,
            RuntimeConfig {
                context_size: NonZeroU32::new(512),
                batch_size: 128,
                micro_batch_size: 128,
                threads: 2,
                gpu_layers: 0,
                parallel_sequences: 1,
                request_queue_capacity: 1,
                event_queue_capacity: 2,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: false,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut generation = engine
                .generate(
                    Request {
                        messages: vec![Message {
                            role: "user".to_string(),
                            content: "Reply with OK.".to_string(),
                        }],
                        options: GenerationOptions {
                            max_tokens: 8,
                            temperature: 0.0,
                            top_p: 1.0,
                            top_k: 0,
                            min_p: 0.0,
                            presence_penalty: 0.0,
                            frequency_penalty: 0.0,
                            repeat_penalty: 1.0,
                            seed: 1,
                            stop: Vec::new(),
                        },
                    },
                    2,
                )
                .await
                .unwrap();
            while let Some(event) = generation.events.recv().await {
                if matches!(event, Event::Finished { .. }) {
                    break;
                }
            }
        });
        engine.shutdown();
    }
}
