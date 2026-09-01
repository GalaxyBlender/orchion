use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;
use tokio::sync::mpsc;

use crate::common_chat::{Preparation, PreparedChat, ReasoningControl};
#[allow(
    clippy::wildcard_imports,
    reason = "the private scheduler consumes most of the facade contract"
)]
use crate::contract::*;
use crate::multimodal::{PreparedMedia, Projector};
use crate::prefix_cache::{Compatibility, InputCompatibility, PrefixCache};
use crate::slot::{
    ActiveSlot, DrainingSlot, Lifecycle, OperationCapability, ReservedSlot, Slot, SlotState,
    StopFilter, create_sampler, finish_event, logits_targets, plan_batch, send_event, timing_rates,
    try_flush_content, try_flush_draining,
};
use crate::template::{
    EffectiveTemplate, common_chat_template_parts, effective_template,
    legacy_request_from_semantic, prepare_legacy_messages, tokenize_prompt,
};

#[cfg(test)]
use crate::slot::finite_nonnegative;
#[cfg(test)]
use crate::template::{normalize_text_messages, render_jinja_template, validate_and_flatten};

static BACKEND: OnceLock<Mutex<Weak<BackendOwner>>> = OnceLock::new();
static LOAD_EPOCH: AtomicUsize = AtomicUsize::new(1);

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

#[derive(Debug)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

#[derive(Debug)]
struct EngineInner {
    commands: mpsc::Sender<Command>,
    controls: mpsc::Sender<ControlCommand>,
    join: Mutex<Option<JoinHandle<()>>>,
    active: Mutex<Vec<Weak<AtomicBool>>>,
    health: Arc<AtomicU8>,
    parallel_sequences: usize,
    next_operation: AtomicU64,
    group_admission: tokio::sync::Mutex<()>,
}

#[derive(Debug)]
enum Command {
    CountTokens {
        request: CountInput,
        cancelled: Arc<AtomicBool>,
        result: tokio::sync::oneshot::Sender<Result<usize, Error>>,
    },
    ReserveGeneration {
        operation: Option<OperationCapability>,
        cancelled: Arc<AtomicBool>,
        events: mpsc::Sender<Event>,
        reserved: tokio::sync::oneshot::Sender<Result<(), Error>>,
        readiness: tokio::sync::oneshot::Sender<Result<(), Error>>,
        acknowledged: tokio::sync::oneshot::Sender<Result<(), Error>>,
        terminal: tokio::sync::oneshot::Sender<Result<Event, Error>>,
        decision: tokio::sync::oneshot::Receiver<ReservationDecision>,
    },
    ReserveEmbedding {
        cancelled: Arc<AtomicBool>,
        reserved: tokio::sync::oneshot::Sender<Result<(), Error>>,
        readiness: tokio::sync::oneshot::Sender<Result<(), Error>>,
        acknowledged: tokio::sync::oneshot::Sender<Result<(), Error>>,
        result: tokio::sync::oneshot::Sender<Result<EmbeddingOutput, Error>>,
        decision: tokio::sync::oneshot::Receiver<EmbeddingReservationDecision>,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningControlResult {
    Success,
    NotFound,
    NotReasoning,
    Disabled,
}

#[derive(Debug)]
struct ControlCommand {
    operation: OperationCapability,
    state: Arc<AtomicU8>,
    result: tokio::sync::oneshot::Sender<Result<ReasoningControlResult, Error>>,
}

const CONTROL_PENDING: u8 = 0;
const CONTROL_APPLYING: u8 = 1;
const CONTROL_CANCELLED: u8 = 2;
const CONTROL_COMPLETED: u8 = 3;

pub struct ReasoningControlAttempt {
    state: Arc<AtomicU8>,
    receiver: Option<tokio::sync::oneshot::Receiver<Result<ReasoningControlResult, Error>>>,
}

#[derive(Clone)]
pub struct ReasoningControlCancellation(Arc<AtomicU8>);

impl ReasoningControlCancellation {
    #[must_use]
    pub fn cancel_pending(&self) -> bool {
        self.0
            .compare_exchange(
                CONTROL_PENDING,
                CONTROL_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

impl ReasoningControlAttempt {
    #[must_use]
    pub fn cancellation_handle(&self) -> ReasoningControlCancellation {
        ReasoningControlCancellation(Arc::clone(&self.state))
    }

    pub async fn result(mut self) -> Result<ReasoningControlResult, Error> {
        let Some(receiver) = self.receiver.take() else {
            return Err(Error::WorkerUnavailable);
        };
        receiver.await.unwrap_or(Err(Error::WorkerUnavailable))
    }
}

impl Drop for ReasoningControlAttempt {
    fn drop(&mut self) {
        let _ = self.state.compare_exchange(
            CONTROL_PENDING,
            CONTROL_CANCELLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Debug, Clone)]
pub struct ReasoningControlHandle {
    operation: OperationCapability,
    controls: mpsc::Sender<ControlCommand>,
}

impl ReasoningControlHandle {
    pub async fn reasoning_end(&self) -> Result<ReasoningControlResult, Error> {
        self.begin_reasoning_end()?.result().await
    }

    pub fn begin_reasoning_end(&self) -> Result<ReasoningControlAttempt, Error> {
        let (result, receiver) = tokio::sync::oneshot::channel();
        let state = Arc::new(AtomicU8::new(CONTROL_PENDING));
        self.controls
            .try_send(ControlCommand {
                operation: self.operation,
                state: Arc::clone(&state),
                result,
            })
            .map_err(|_| Error::WorkerUnavailable)?;
        Ok(ReasoningControlAttempt {
            state,
            receiver: Some(receiver),
        })
    }
}

#[derive(Debug)]
enum CountInput {
    Legacy(TokenCountRequest),
    Semantic(SemanticTokenCountRequest),
}

#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "the worker owns committed request state inline until slot activation"
)]
pub(crate) enum ReservationDecision {
    Commit(AdvancedRequest),
    Abort,
}

#[derive(Debug)]
enum EmbeddingReservationDecision {
    Commit(EmbeddingRequest),
    Abort,
}

static DECODE_CALLS: AtomicUsize = AtomicUsize::new(0);
static MULTI_SLOT_DECODE_CALLS: AtomicUsize = AtomicUsize::new(0);
static PREFILL_DECODE_OVERLAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static MAX_BUSY_SLOTS_PER_DECODE: AtomicUsize = AtomicUsize::new(0);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerInstrumentation {
    pub decode_calls: usize,
    pub multi_slot_decode_calls: usize,
    pub prefill_decode_overlap_calls: usize,
    pub max_busy_slots_per_decode: usize,
}

#[doc(hidden)]
#[must_use]
pub fn scheduler_instrumentation() -> SchedulerInstrumentation {
    SchedulerInstrumentation {
        decode_calls: DECODE_CALLS.load(Ordering::Acquire),
        multi_slot_decode_calls: MULTI_SLOT_DECODE_CALLS.load(Ordering::Acquire),
        prefill_decode_overlap_calls: PREFILL_DECODE_OVERLAP_CALLS.load(Ordering::Acquire),
        max_busy_slots_per_decode: MAX_BUSY_SLOTS_PER_DECODE.load(Ordering::Acquire),
    }
}

#[doc(hidden)]
pub fn reset_scheduler_instrumentation() {
    DECODE_CALLS.store(0, Ordering::Release);
    MULTI_SLOT_DECODE_CALLS.store(0, Ordering::Release);
    PREFILL_DECODE_OVERLAP_CALLS.store(0, Ordering::Release);
    MAX_BUSY_SLOTS_PER_DECODE.store(0, Ordering::Release);
}

pub struct Generation {
    pub events: mpsc::Receiver<Event>,
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    terminal: Option<tokio::sync::oneshot::Receiver<Result<Event, Error>>>,
    control: Option<ReasoningControlHandle>,
}

pub struct Reservation {
    request: Option<AdvancedRequest>,
    events: Option<mpsc::Receiver<Event>>,
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    readiness: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    terminal: Option<tokio::sync::oneshot::Receiver<Result<Event, Error>>>,
    decision: Option<tokio::sync::oneshot::Sender<ReservationDecision>>,
    transferred: bool,
    control: Option<ReasoningControlHandle>,
}

pub struct ChoiceReservation {
    reservations: Vec<Reservation>,
    event_capacity: usize,
}

pub struct ChoiceGeneration {
    pub events: mpsc::Receiver<ChoiceEvent>,
    cancelled: Vec<Arc<AtomicBool>>,
    tasks: Vec<tokio::task::JoinHandle<Result<(), Error>>>,
    control: Option<ReasoningControlHandle>,
}

pub struct Embedding {
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    result: Option<tokio::sync::oneshot::Receiver<Result<EmbeddingOutput, Error>>>,
}

pub struct EmbeddingReservation {
    request: Option<EmbeddingRequest>,
    cancelled: Arc<AtomicBool>,
    acknowledged: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    readiness: Option<tokio::sync::oneshot::Receiver<Result<(), Error>>>,
    result: Option<tokio::sync::oneshot::Receiver<Result<EmbeddingOutput, Error>>>,
    decision: Option<tokio::sync::oneshot::Sender<EmbeddingReservationDecision>>,
    transferred: bool,
}

impl EmbeddingReservation {
    pub async fn commit(&mut self) -> Result<Embedding, Error> {
        let decision = self.decision.take().ok_or(Error::WorkerUnavailable)?;
        let request = self.request.take().ok_or(Error::WorkerUnavailable)?;
        decision
            .send(EmbeddingReservationDecision::Commit(request))
            .map_err(|_| Error::WorkerUnavailable)?;
        let readiness = self.readiness.take().ok_or(Error::WorkerUnavailable)?;
        if let Err(error) = readiness.await.unwrap_or(Err(Error::WorkerUnavailable)) {
            if let Some(acknowledged) = self.acknowledged.take() {
                let _ = acknowledged.await;
            }
            return Err(error);
        }
        self.transferred = true;
        Ok(Embedding {
            cancelled: Arc::clone(&self.cancelled),
            acknowledged: self.acknowledged.take(),
            result: self.result.take(),
        })
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

    pub fn abort(mut self) {
        self.cancel();
        if let Some(decision) = self.decision.take() {
            let _ = decision.send(EmbeddingReservationDecision::Abort);
        }
    }
}

impl Drop for EmbeddingReservation {
    fn drop(&mut self) {
        if !self.transferred {
            self.cancel();
        }
        if let Some(decision) = self.decision.take() {
            let _ = decision.send(EmbeddingReservationDecision::Abort);
        }
    }
}

impl Embedding {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub async fn result(&mut self) -> Result<EmbeddingOutput, Error> {
        let result = self.result.take().ok_or(Error::WorkerUnavailable)?;
        result.await.unwrap_or(Err(Error::WorkerUnavailable))
    }

    pub async fn wait_for_ack(&mut self) -> Result<(), Error> {
        let Some(acknowledged) = self.acknowledged.take() else {
            return Ok(());
        };
        acknowledged.await.unwrap_or(Err(Error::WorkerUnavailable))
    }
}

impl Drop for Embedding {
    fn drop(&mut self) {
        self.cancel();
    }
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
            control: self.control.take(),
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

impl ChoiceReservation {
    #[allow(
        clippy::too_many_lines,
        reason = "keeps grouped commit, forwarding, and aggregate terminal ownership together"
    )]
    pub async fn commit(&mut self) -> Result<ChoiceGeneration, Error> {
        let mut generations = Vec::with_capacity(self.reservations.len());
        for reservation in &mut self.reservations {
            match reservation.commit().await {
                Ok(generation) => generations.push(generation),
                Err(error) => {
                    for generation in &generations {
                        generation.cancel();
                    }
                    for reservation in &mut self.reservations {
                        reservation.request_abort();
                    }
                    for generation in &mut generations {
                        let _ = generation.wait_for_ack().await;
                    }
                    for reservation in &mut self.reservations {
                        let _ = reservation.wait_for_ack().await;
                    }
                    return Err(error);
                }
            }
        }
        let capacity = self
            .event_capacity
            .saturating_mul(generations.len())
            .clamp(1, MAX_EVENT_CAPACITY);
        let (events, receiver) = mpsc::channel(capacity);
        let aggregate = Arc::new(Mutex::new(ChoiceAggregate {
            remaining: generations.len(),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                reasoning_tokens: 0,
                timings: Timings::default(),
            },
            prompt_tokens: None,
            failure: None,
        }));
        let cancelled = generations
            .iter()
            .map(|generation| Arc::clone(&generation.cancelled))
            .collect::<Vec<_>>();
        let group_cancelled = Arc::new(cancelled.clone());
        let control = generations
            .first()
            .and_then(|generation| generation.control.clone());
        let tasks = generations
            .into_iter()
            .enumerate()
            .map(|(index, mut generation)| {
                let events = events.clone();
                let aggregate = Arc::clone(&aggregate);
                let group_cancelled = Arc::clone(&group_cancelled);
                tokio::spawn(async move {
                    let mut forwarding = true;
                    while let Some(event) = generation.events.recv().await {
                        let event = match event {
                            Event::Content(text) => ChoiceEvent::Delta {
                                index,
                                text,
                                logprobs: None,
                            },
                            Event::Token { text, logprobs } => ChoiceEvent::Delta {
                                index,
                                text,
                                logprobs: Some(logprobs),
                            },
                            Event::Semantic(delta) => ChoiceEvent::SemanticDelta { index, delta },
                            Event::Finished { reason, usage } => ChoiceEvent::Finished {
                                index,
                                reason,
                                usage,
                            },
                            Event::Failed(message) => ChoiceEvent::Failed {
                                index: Some(index),
                                message,
                            },
                        };
                        if forwarding && events.send(event).await.is_err() {
                            generation.cancel();
                            forwarding = false;
                        }
                    }
                    let terminal = generation.recv_terminal().await;
                    let (final_event, usage, failure) = match terminal {
                        Ok(Event::Finished { reason, usage }) => (
                            ChoiceEvent::Finished {
                                index,
                                reason,
                                usage,
                            },
                            Some(usage),
                            None,
                        ),
                        Ok(Event::Failed(message)) | Err(Error::Generation(message)) => (
                            ChoiceEvent::Failed {
                                index: Some(index),
                                message: message.clone(),
                            },
                            None,
                            Some(message),
                        ),
                        Ok(Event::Content(_) | Event::Token { .. } | Event::Semantic(_)) => (
                            ChoiceEvent::Failed {
                                index: Some(index),
                                message: "choice ended with a non-terminal event".to_string(),
                            },
                            None,
                            Some("choice ended with a non-terminal event".to_string()),
                        ),
                        Err(error) => (
                            ChoiceEvent::Failed {
                                index: Some(index),
                                message: error.to_string(),
                            },
                            None,
                            Some(error.to_string()),
                        ),
                    };
                    let first_failure = failure.as_ref().is_some_and(|message| {
                        let mut aggregate = aggregate
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if aggregate.failure.is_some() {
                            false
                        } else {
                            aggregate.failure = Some(message.clone());
                            true
                        }
                    });
                    if first_failure {
                        for cancelled in group_cancelled.iter() {
                            cancelled.store(true, Ordering::Release);
                        }
                    }
                    let acknowledgement = generation.wait_for_ack().await;
                    if forwarding && events.send(final_event).await.is_err() {
                        forwarding = false;
                    }
                    let parent = {
                        let mut aggregate = aggregate
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if let Some(usage) = usage
                            && let Err(message) = merge_usage(&mut aggregate, usage)
                            && aggregate.failure.is_none()
                        {
                            aggregate.failure = Some(message);
                        }
                        if aggregate.failure.is_none() {
                            aggregate.failure = failure;
                        }
                        aggregate.remaining = aggregate.remaining.saturating_sub(1);
                        if aggregate.remaining == 0 {
                            aggregate.failure.as_ref().map_or_else(
                                || {
                                    Some(ChoiceEvent::FinishedAll {
                                        usage: aggregate.usage,
                                    })
                                },
                                |message| {
                                    Some(ChoiceEvent::Failed {
                                        index: None,
                                        message: message.clone(),
                                    })
                                },
                            )
                        } else {
                            None
                        }
                    };
                    if forwarding && let Some(event) = parent {
                        let _ = events.send(event).await;
                    }
                    acknowledgement
                })
            })
            .collect();
        drop(events);
        Ok(ChoiceGeneration {
            events: receiver,
            cancelled,
            tasks,
            control,
        })
    }

    pub fn cancel(&self) {
        for reservation in &self.reservations {
            reservation.cancel();
        }
    }

    pub async fn cancel_and_wait(&mut self) -> Result<(), Error> {
        for reservation in &mut self.reservations {
            reservation.request_abort();
        }
        let mut first_error = None;
        for reservation in &mut self.reservations {
            if let Err(error) = reservation.wait_for_ack().await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

struct ChoiceAggregate {
    remaining: usize,
    usage: Usage,
    prompt_tokens: Option<usize>,
    failure: Option<String>,
}

fn merge_usage(aggregate: &mut ChoiceAggregate, choice: Usage) -> Result<(), String> {
    if let Some(expected) = aggregate.prompt_tokens {
        if choice.prompt_tokens != expected {
            return Err(format!(
                "choice prompt token usage differed: expected {expected}, received {}",
                choice.prompt_tokens
            ));
        }
    } else {
        aggregate.prompt_tokens = Some(choice.prompt_tokens);
        aggregate.usage.prompt_tokens = choice.prompt_tokens;
    }
    let total = &mut aggregate.usage;
    total.completion_tokens = total
        .completion_tokens
        .saturating_add(choice.completion_tokens);
    total.reasoning_tokens = total
        .reasoning_tokens
        .saturating_add(choice.reasoning_tokens);
    total.timings.cache_n = total.timings.cache_n.saturating_add(choice.timings.cache_n);
    total.timings.prompt_n = total
        .timings
        .prompt_n
        .saturating_add(choice.timings.prompt_n);
    total.timings.prompt_ms += choice.timings.prompt_ms;
    total.timings.predicted_n = total
        .timings
        .predicted_n
        .saturating_add(choice.timings.predicted_n);
    total.timings.predicted_ms += choice.timings.predicted_ms;
    (
        total.timings.prompt_per_token_ms,
        total.timings.prompt_per_second,
    ) = timing_rates(total.timings.prompt_n, total.timings.prompt_ms);
    (
        total.timings.predicted_per_token_ms,
        total.timings.predicted_per_second,
    ) = timing_rates(total.timings.predicted_n, total.timings.predicted_ms);
    Ok(())
}

impl ChoiceGeneration {
    #[must_use]
    pub fn reasoning_control(&self) -> Option<ReasoningControlHandle> {
        self.control.clone()
    }

    pub fn cancel(&self) {
        for cancelled in &self.cancelled {
            cancelled.store(true, Ordering::Release);
        }
    }

    pub async fn wait_for_ack(&mut self) -> Result<(), Error> {
        self.events.close();
        while self.events.recv().await.is_some() {}
        let mut first_error = None;
        for task in std::mem::take(&mut self.tasks) {
            let result = task.await.unwrap_or(Err(Error::WorkerUnavailable));
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for ChoiceGeneration {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Reservation {
    fn request_abort(&mut self) {
        self.cancel();
        if let Some(decision) = self.decision.take() {
            let _ = decision.send(ReservationDecision::Abort);
        }
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
            Event::Content(_) | Event::Token { .. } | Event::Semantic(_) => sender
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
        control: None,
    }
}

#[doc(hidden)]
pub fn deterministic_choice_generation(
    events: impl IntoIterator<Item = ChoiceEvent>,
) -> ChoiceGeneration {
    let events = events.into_iter().collect::<Vec<_>>();
    let (sender, receiver) = mpsc::channel(events.len().max(1));
    for event in events {
        sender
            .try_send(event)
            .expect("deterministic channel is sized for its script");
    }
    drop(sender);
    ChoiceGeneration {
        events: receiver,
        cancelled: Vec::new(),
        tasks: Vec::new(),
        control: None,
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

#[doc(hidden)]
#[must_use]
pub fn scripted_embedding_engine(
    output: EmbeddingOutput,
    command_capacity: usize,
) -> (Engine, ScriptedControl) {
    let (commands, mut receiver) = mpsc::channel(command_capacity);
    let (controls, _control_receiver) = mpsc::channel(1);
    let control = ScriptedControl {
        ready: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        preparation: Arc::new((Mutex::new(true), std::sync::Condvar::new())),
        preparation_started: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        cleanup: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        started: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
        executed: Arc::new(AtomicBool::new(false)),
        panic_preparation: false,
    };
    let worker_control = control.clone();
    let health = Arc::new(AtomicU8::new(0));
    let join = std::thread::spawn(move || {
        while let Some(command) = receiver.blocking_recv() {
            match command {
                Command::CountTokens { result, .. } => {
                    let _ = result.send(Err(Error::InvalidConfig(
                        "scripted embedding engine does not support token counting".to_string(),
                    )));
                }
                Command::ReserveEmbedding {
                    cancelled,
                    reserved,
                    readiness,
                    acknowledged,
                    result,
                    decision,
                } => {
                    open_gate(&worker_control.started);
                    wait_gate(&worker_control.ready, Some(&cancelled));
                    if cancelled.load(Ordering::Acquire) || reserved.send(Ok(())).is_err() {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    }
                    let EmbeddingReservationDecision::Commit(_) =
                        wait_embedding_reservation_decision(decision, &cancelled)
                    else {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    };
                    open_gate(&worker_control.preparation_started);
                    wait_gate(&worker_control.preparation, Some(&cancelled));
                    if cancelled.load(Ordering::Acquire) {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    }
                    worker_control.executed.store(true, Ordering::Release);
                    if readiness.send(Ok(())).is_err() {
                        let _ = acknowledged.send(Err(Error::Cancelled));
                        continue;
                    }
                    wait_gate(&worker_control.cleanup, None);
                    let completed = if cancelled.load(Ordering::Acquire) {
                        Err(Error::Cancelled)
                    } else {
                        Ok(output.clone())
                    };
                    let _ = result.send(completed.clone());
                    let _ = acknowledged.send(completed.map(|_| ()));
                }
                Command::ReserveGeneration {
                    reserved,
                    readiness,
                    acknowledged,
                    terminal,
                    ..
                } => {
                    let error = Error::InvalidConfig(
                        "scripted embedding engine does not support generation".to_string(),
                    );
                    let _ = reserved.send(Err(error.clone()));
                    let _ = readiness.send(Err(error.clone()));
                    let _ = terminal.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error));
                }
                Command::Shutdown => break,
            }
        }
    });
    (
        Engine {
            inner: Arc::new(EngineInner {
                commands,
                controls,
                join: Mutex::new(Some(join)),
                active: Mutex::new(Vec::new()),
                health,
                parallel_sequences: 1,
                next_operation: AtomicU64::new(1),
                group_admission: tokio::sync::Mutex::new(()),
            }),
        },
        control,
    )
}

fn scripted_engine_with_preparation(
    script: Vec<Event>,
    command_capacity: usize,
    preparation_ready: bool,
    panic_preparation: bool,
) -> (Engine, ScriptedControl) {
    let (commands, receiver) = mpsc::channel(command_capacity);
    let (controls, control_receiver) = mpsc::channel(1);
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
        scripted_worker(
            receiver,
            control_receiver,
            script,
            worker_control,
            worker_health,
        );
    });
    (
        Engine {
            inner: Arc::new(EngineInner {
                commands,
                controls,
                join: Mutex::new(Some(join)),
                active: Mutex::new(Vec::new()),
                health,
                parallel_sequences: 1,
                next_operation: AtomicU64::new(1),
                group_admission: tokio::sync::Mutex::new(()),
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
    let (controls, _control_receiver) = mpsc::channel(1);
    let health = Arc::new(AtomicU8::new(0));
    let join = std::thread::spawn(move || {
        while let Some(command) = receiver.blocking_recv() {
            match command {
                Command::CountTokens {
                    request, result, ..
                } => {
                    let _ = result.send(Ok(scripted_token_count(&request)));
                }
                Command::ReserveGeneration {
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
                Command::ReserveEmbedding {
                    reserved,
                    readiness,
                    acknowledged,
                    result,
                    ..
                } => {
                    let error = Error::InvalidConfig(
                        "scripted generation engine does not support embeddings".to_string(),
                    );
                    let _ = reserved.send(Err(error.clone()));
                    let _ = readiness.send(Err(error.clone()));
                    let _ = result.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error));
                }
                Command::Shutdown => break,
            }
        }
    });
    Engine {
        inner: Arc::new(EngineInner {
            commands,
            controls,
            join: Mutex::new(Some(join)),
            active: Mutex::new(Vec::new()),
            health,
            parallel_sequences: 1,
            next_operation: AtomicU64::new(1),
            group_admission: tokio::sync::Mutex::new(()),
        }),
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the dedicated thread owns its script and synchronization control"
)]
#[allow(
    clippy::too_many_lines,
    reason = "keeps scripted reservation, execution, and acknowledgement behavior together"
)]
fn scripted_worker(
    mut commands: mpsc::Receiver<Command>,
    mut controls: mpsc::Receiver<ControlCommand>,
    script: Vec<Event>,
    control: ScriptedControl,
    health: Arc<AtomicU8>,
) {
    while let Some(command) = commands.blocking_recv() {
        match command {
            Command::CountTokens {
                request,
                cancelled,
                result,
            } => {
                let value = if cancelled.load(Ordering::Acquire) {
                    Err(Error::Cancelled)
                } else {
                    Ok(scripted_token_count(&request))
                };
                let _ = result.send(value);
            }
            Command::ReserveGeneration {
                operation,
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
                            reasoning_tokens: 0,
                            timings: Timings::default(),
                        },
                    });
                    let mut reasoning = false;
                    for event in script.iter().cloned() {
                        if matches!(&event, Event::Failed(message) if message == "__orchion_test_panic__")
                        {
                            panic!("scripted worker panic");
                        }
                        reasoning = match &event {
                            Event::Semantic(SemanticDelta::Reasoning(_)) => true,
                            Event::Semantic(SemanticDelta::Text(_))
                            | Event::Content(_)
                            | Event::Token { .. } => false,
                            _ => reasoning,
                        };
                        match event {
                            Event::Content(_) | Event::Token { .. } | Event::Semantic(_) => {
                                if let Err(error) = send_event(&events, event, &cancelled) {
                                    result = Err(error);
                                    break;
                                }
                            }
                            Event::Finished { .. } => result = Ok(event),
                            Event::Failed(message) => result = Err(Error::Generation(message)),
                        }
                    }
                    (result, reasoning)
                }));
                let (result, reasoning) = execution.unwrap_or_else(|payload| {
                    (Err(Error::WorkerPanic(panic_message(&payload))), false)
                });
                wait_scripted_cleanup_with_controls(
                    &control.cleanup,
                    &mut controls,
                    operation,
                    reasoning,
                );
                if matches!(result, Err(Error::WorkerPanic(_))) {
                    health.store(1, Ordering::Release);
                }
                let _ = terminal.send(result.clone());
                let _ = acknowledged.send(result.map(|_| ()));
            }
            Command::ReserveEmbedding {
                reserved,
                readiness,
                acknowledged,
                result,
                ..
            } => {
                let error = Error::InvalidConfig(
                    "scripted generation engine does not support embeddings".to_string(),
                );
                let _ = reserved.send(Err(error.clone()));
                let _ = readiness.send(Err(error.clone()));
                let _ = result.send(Err(error.clone()));
                let _ = acknowledged.send(Err(error));
            }
            Command::Shutdown => break,
        }
    }
}

fn wait_scripted_cleanup_with_controls(
    gate: &(Mutex<bool>, std::sync::Condvar),
    controls: &mut mpsc::Receiver<ControlCommand>,
    active_operation: Option<OperationCapability>,
    mut reasoning: bool,
) {
    let (open, wake) = gate;
    let mut open = open
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while !*open {
        drop(open);
        while let Ok(command) = controls.try_recv() {
            if command
                .state
                .compare_exchange(
                    CONTROL_PENDING,
                    CONTROL_APPLYING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                let _ = command.result.send(Ok(ReasoningControlResult::NotFound));
                continue;
            }
            let result = if controls_operation(active_operation, command.operation) {
                if reasoning {
                    reasoning = false;
                    ReasoningControlResult::Success
                } else {
                    ReasoningControlResult::NotReasoning
                }
            } else {
                ReasoningControlResult::NotFound
            };
            command.state.store(CONTROL_COMPLETED, Ordering::Release);
            let _ = command.result.send(Ok(result));
        }
        open = gate
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (next, _) = wake
            .wait_timeout(open, std::time::Duration::from_millis(1))
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        open = next;
    }
}

fn controls_operation(active: Option<OperationCapability>, requested: OperationCapability) -> bool {
    active == Some(requested)
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

fn scripted_token_count(request: &CountInput) -> usize {
    match request {
        CountInput::Legacy(request) => {
            1 + request
                .messages
                .iter()
                .map(|message| message.content.split_whitespace().count())
                .sum::<usize>()
        }
        CountInput::Semantic(request) => {
            1 + request
                .messages
                .iter()
                .flat_map(|message| &message.content)
                .map(|part| match part {
                    ContentPart::Text { text } | ContentPart::Reasoning { text } => {
                        text.split_whitespace().count()
                    }
                    ContentPart::ToolResult(result) => result.content.split_whitespace().count(),
                    ContentPart::Image(_) | ContentPart::Media(_) => 0,
                })
                .sum::<usize>()
        }
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

fn wait_embedding_reservation_decision(
    mut decision: tokio::sync::oneshot::Receiver<EmbeddingReservationDecision>,
    cancelled: &AtomicBool,
) -> EmbeddingReservationDecision {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return EmbeddingReservationDecision::Abort;
        }
        match decision.try_recv() {
            Ok(decision) => return decision,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                return EmbeddingReservationDecision::Abort;
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
        let parallel_sequences = usize::try_from(config.parallel_sequences)
            .map_err(|error| Error::InvalidConfig(error.to_string()))?;
        let backend = BackendOwner::acquire()?;
        let (commands, receiver) = mpsc::channel(config.request_queue_capacity);
        let (controls, control_receiver) = mpsc::channel(32);
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
                    control_receiver,
                    &ready_tx,
                    &worker_health,
                );
            })
            .map_err(|error| Error::WorkerStart(error.to_string()))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                inner: Arc::new(EngineInner {
                    commands,
                    controls,
                    join: Mutex::new(Some(join)),
                    active: Mutex::new(Vec::new()),
                    health,
                    parallel_sequences,
                    next_operation: AtomicU64::new(1),
                    group_admission: tokio::sync::Mutex::new(()),
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
        self.reserve_advanced(request.into(), event_capacity).await
    }

    pub async fn reserve_advanced(
        &self,
        request: AdvancedRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        validate_event_capacity(event_capacity)?;
        if request.choices != 1 {
            return Err(Error::InvalidConfig(
                "a single reservation requires choices == 1; use reserve_choices".to_string(),
            ));
        }
        let operation = self
            .inner
            .reasoning_operation(request.reasoning_control_id.as_ref())?;
        let control = operation.map(|operation| ReasoningControlHandle {
            operation,
            controls: self.inner.controls.clone(),
        });
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
            .send(Command::ReserveGeneration {
                operation,
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
            control,
        })
    }

    pub async fn reserve_semantic(
        &self,
        request: SemanticRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        self.reserve(legacy_request_from_semantic(request), event_capacity)
            .await
    }

    pub async fn reserve_advanced_semantic(
        &self,
        request: AdvancedSemanticRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        let request = crate::template::advanced_request_from_semantic(request)?;
        self.reserve_advanced(request, event_capacity).await
    }

    pub async fn reserve_choices(
        &self,
        mut request: AdvancedRequest,
        event_capacity: usize,
    ) -> Result<ChoiceReservation, Error> {
        validate_event_capacity(event_capacity)?;
        if request.choices != 1 && request.reasoning_control_id.is_some() {
            return Err(Error::InvalidConfig(
                "reasoning control requires choices == 1".to_string(),
            ));
        }
        validate_choice_count(request.choices, self.inner.parallel_sequences)?;
        if request.choices != 1 && request_has_images(&request) {
            return Err(Error::InvalidConfig(
                "multimodal requests require choices == 1".to_string(),
            ));
        }
        let _admission = self.inner.group_admission.lock().await;
        let choices = request.choices;
        request.choices = 1;
        let mut reservations = Vec::with_capacity(choices);
        for index in 0..choices {
            let mut choice = request.clone();
            if choice.options.seed != u32::MAX {
                choice.options.seed = choice.options.seed.wrapping_add(
                    u32::try_from(index)
                        .map_err(|error| Error::InvalidConfig(error.to_string()))?,
                );
            }
            match self.reserve_advanced(choice, event_capacity).await {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for reservation in &mut reservations {
                        reservation.request_abort();
                    }
                    for reservation in &mut reservations {
                        let _ = reservation.wait_for_ack().await;
                    }
                    return Err(error);
                }
            }
        }
        Ok(ChoiceReservation {
            reservations,
            event_capacity,
        })
    }

    pub async fn reserve_choice_semantic(
        &self,
        request: AdvancedSemanticRequest,
        event_capacity: usize,
    ) -> Result<ChoiceReservation, Error> {
        self.reserve_choices(
            crate::template::advanced_request_from_semantic(request)?,
            event_capacity,
        )
        .await
    }

    pub async fn reserve_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingReservation, Error> {
        let cancelled = Arc::new(AtomicBool::new(false));
        {
            let mut active = self
                .inner
                .active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            active.retain(|operation| operation.strong_count() > 0);
            active.push(Arc::downgrade(&cancelled));
        }
        let (reserved, reservation_ack) = tokio::sync::oneshot::channel();
        let (readiness, worker_readiness) = tokio::sync::oneshot::channel();
        let (acknowledged, acknowledgement) = tokio::sync::oneshot::channel();
        let (result, worker_result) = tokio::sync::oneshot::channel();
        let (decision, worker_decision) = tokio::sync::oneshot::channel();
        let mut pending = PendingGeneration {
            cancelled: Arc::clone(&cancelled),
            committed: false,
        };
        self.inner
            .commands
            .send(Command::ReserveEmbedding {
                cancelled: Arc::clone(&cancelled),
                reserved,
                readiness,
                acknowledged,
                result,
                decision: worker_decision,
            })
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        reservation_ack
            .await
            .unwrap_or(Err(Error::WorkerUnavailable))?;
        pending.committed = true;
        Ok(EmbeddingReservation {
            request: Some(request),
            cancelled,
            acknowledged: Some(acknowledgement),
            readiness: Some(worker_readiness),
            result: Some(worker_result),
            decision: Some(decision),
            transferred: false,
        })
    }

    pub async fn count_tokens(&self, request: TokenCountRequest) -> Result<usize, Error> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut pending = PendingGeneration {
            cancelled: Arc::clone(&cancelled),
            committed: false,
        };
        let (result, response) = tokio::sync::oneshot::channel();
        self.inner
            .commands
            .send(Command::CountTokens {
                request: CountInput::Legacy(request),
                cancelled,
                result,
            })
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        let result = response.await.unwrap_or(Err(Error::WorkerUnavailable));
        pending.committed = true;
        result
    }

    pub async fn count_semantic_tokens(
        &self,
        request: SemanticTokenCountRequest,
    ) -> Result<usize, Error> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut pending = PendingGeneration {
            cancelled: Arc::clone(&cancelled),
            committed: false,
        };
        let (result, response) = tokio::sync::oneshot::channel();
        self.inner
            .commands
            .send(Command::CountTokens {
                request: CountInput::Semantic(request),
                cancelled,
                result,
            })
            .await
            .map_err(|_| Error::WorkerUnavailable)?;
        let result = response.await.unwrap_or(Err(Error::WorkerUnavailable));
        pending.committed = true;
        result
    }

    pub async fn generate(
        &self,
        request: Request,
        event_capacity: usize,
    ) -> Result<Generation, Error> {
        let mut reservation = self.reserve(request, event_capacity).await?;
        reservation.commit().await
    }

    pub async fn generate_semantic(
        &self,
        request: SemanticRequest,
        event_capacity: usize,
    ) -> Result<Generation, Error> {
        let mut reservation = self.reserve_semantic(request, event_capacity).await?;
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
    fn reasoning_operation(
        &self,
        external_id: Option<&String>,
    ) -> Result<Option<OperationCapability>, Error> {
        external_id.map(|_| self.allocate_operation()).transpose()
    }

    fn allocate_operation(&self) -> Result<OperationCapability, Error> {
        self.next_operation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(OperationCapability)
            .map_err(|_| Error::WorkerUnavailable)
    }

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

#[allow(
    clippy::too_many_lines,
    reason = "validates the complete native runtime contract"
)]
fn validate_config(config: &RuntimeConfig) -> Result<(), Error> {
    if config.parallel_sequences == 0 {
        return Err(Error::InvalidConfig(
            "parallel_sequences must be nonzero".to_string(),
        ));
    }
    if config.batch_size == 0 || config.micro_batch_size == 0 {
        return Err(Error::InvalidConfig(
            "batch sizes must be nonzero".to_string(),
        ));
    }
    if config.batch_size > i32::MAX as u32 || config.micro_batch_size > i32::MAX as u32 {
        return Err(Error::InvalidConfig(
            "batch sizes must fit the native signed batch limit".to_string(),
        ));
    }
    if config.request_queue_capacity == 0 {
        return Err(Error::InvalidConfig(
            "request_queue_capacity must be nonzero".to_string(),
        ));
    }
    validate_event_capacity(config.event_queue_capacity)?;
    if !(1..=64).contains(&config.prompt_cache.max_entries) {
        return Err(Error::InvalidConfig(
            "prompt_cache.max_entries must be in 1..=64".to_string(),
        ));
    }
    if config.prompt_cache.max_bytes == 0
        || u64::try_from(config.prompt_cache.max_bytes).unwrap_or(u64::MAX) > 4 * 1024 * 1024 * 1024
    {
        return Err(Error::InvalidConfig(
            "prompt_cache.max_bytes must be nonzero and at most 4 GiB".to_string(),
        ));
    }
    if config.prompt_cache.min_prefix_tokens == 0 {
        return Err(Error::InvalidConfig(
            "prompt_cache.min_prefix_tokens must be nonzero".to_string(),
        ));
    }
    if matches!(config.mode, RuntimeMode::Embeddings { .. })
        && config.prompt_cache != PromptCacheConfig::default()
    {
        return Err(Error::InvalidConfig(
            "embedding mode does not support prompt_cache".to_string(),
        ));
    }
    if matches!(config.mode, RuntimeMode::Embeddings { .. }) && config.vision.is_some() {
        return Err(Error::InvalidConfig(
            "embedding mode does not support vision".to_string(),
        ));
    }
    if let Some(vision) = &config.vision {
        validate_vision_config(vision)?;
    }
    if matches!(config.mode, RuntimeMode::Generation)
        && config.batch_size < config.parallel_sequences
    {
        return Err(Error::InvalidConfig(
            "generation batch_size must be at least parallel_sequences".to_string(),
        ));
    }
    if matches!(config.mode, RuntimeMode::Embeddings { .. }) && config.parallel_sequences != 1 {
        return Err(Error::InvalidConfig(
            "embedding mode requires parallel_sequences == 1".to_string(),
        ));
    }
    if let Some(per_slot) = config.context_size {
        let _ = per_slot
            .get()
            .checked_mul(config.parallel_sequences)
            .and_then(NonZeroU32::new)
            .ok_or_else(|| {
                Error::InvalidConfig(
                    "context_size * parallel_sequences exceeds the native context limit"
                        .to_string(),
                )
            })?;
    }
    if matches!(
        config.mode,
        RuntimeMode::Embeddings {
            max_input_tokens: 0,
            ..
        }
    ) {
        return Err(Error::InvalidConfig(
            "max_input_tokens must be nonzero".to_string(),
        ));
    }
    if let RuntimeMode::Embeddings {
        max_input_tokens, ..
    } = config.mode
    {
        if config.batch_size != config.micro_batch_size {
            return Err(Error::InvalidConfig(
                "embedding mode requires batch_size == micro_batch_size".to_string(),
            ));
        }
        if usize::try_from(config.batch_size).unwrap_or(usize::MAX) < max_input_tokens {
            return Err(Error::InvalidConfig(
                "embedding batch_size must cover max_input_tokens".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_event_capacity(event_capacity: usize) -> Result<(), Error> {
    if !(1..=MAX_EVENT_CAPACITY).contains(&event_capacity) {
        return Err(Error::InvalidConfig(format!(
            "event capacity must be in 1..={MAX_EVENT_CAPACITY}"
        )));
    }
    Ok(())
}

fn validate_vision_config(config: &LlmVisionConfig) -> Result<(), Error> {
    if !config.mmproj.is_file() {
        return Err(Error::InvalidConfig(format!(
            "mmproj `{}` is not a file",
            config.mmproj.display()
        )));
    }
    let limits = config.limits;
    let defaults = VisionLimits::default();
    if limits.max_images == 0
        || limits.max_images > 32
        || limits.max_bytes_per_image == 0
        || limits.max_bytes_per_image > 64 * 1024 * 1024
        || limits.max_total_bytes < limits.max_bytes_per_image
        || limits.max_total_bytes > 128 * 1024 * 1024
        || limits.max_side == 0
        || limits.max_side > 16_384
        || limits.max_pixels_per_image == 0
        || limits.max_pixels_per_image > 67_108_864
        || limits.max_total_pixels < limits.max_pixels_per_image
        || limits.max_total_pixels > 134_217_728
    {
        return Err(Error::InvalidConfig(format!(
            "vision limits are invalid or exceed safety bounds (defaults: {defaults:?})"
        )));
    }
    Ok(())
}

fn validate_choice_count(choices: usize, parallel_sequences: usize) -> Result<(), Error> {
    if choices == 0 || choices > parallel_sequences {
        return Err(Error::InvalidConfig(format!(
            "choices must be in 1..={parallel_sequences} for this deployment"
        )));
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
    mut controls: mpsc::Receiver<ControlCommand>,
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
    let projector = match &config.vision {
        Some(vision) => match Projector::load(
            &vision.mmproj,
            &model,
            config.threads,
            config.gpu_layers != 0,
        ) {
            Ok(projector) => Some(projector),
            Err(error) => {
                let _ = ready.send(Err(error.to_string()));
                return;
            }
        },
        None => None,
    };
    let template = match config.mode {
        RuntimeMode::Generation => match effective_template(&model, config) {
            Ok(template) => Some(template),
            Err(error) => {
                let _ = ready.send(Err(error));
                return;
            }
        },
        RuntimeMode::Embeddings { .. } => None,
    };
    let per_slot_context = config
        .context_size
        .unwrap_or_else(|| NonZeroU32::new(model.n_ctx_train()).expect("model context is nonzero"));
    if config.prompt_cache.enabled
        && config.prompt_cache.min_prefix_tokens >= per_slot_context.get() as usize
    {
        let _ = ready.send(Err(
            "prompt_cache.min_prefix_tokens must be below the per-slot context size".to_string(),
        ));
        return;
    }
    let Some(total_context) = per_slot_context
        .get()
        .checked_mul(config.parallel_sequences)
        .and_then(NonZeroU32::new)
    else {
        let _ = ready.send(Err(
            "context_size * parallel_sequences exceeds the native context limit".to_string(),
        ));
        return;
    };
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(Some(total_context))
        .with_n_batch(config.batch_size)
        .with_n_ubatch(config.micro_batch_size)
        .with_n_seq_max(config.parallel_sequences)
        .with_kv_unified(false);
    if let RuntimeMode::Embeddings { pooling, .. } = config.mode {
        context_params = context_params
            .with_embeddings(true)
            .with_pooling_type(match pooling {
                EmbeddingPooling::Last => LlamaPoolingType::Last,
            });
    }
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

    match config.mode {
        RuntimeMode::Generation => generation_worker_loop(
            &model,
            projector.as_ref(),
            &mut context,
            template
                .as_ref()
                .expect("generation mode initializes a chat template"),
            config,
            &mut commands,
            &mut controls,
            health,
        ),
        RuntimeMode::Embeddings { .. } => {
            embedding_worker_loop(&model, &mut context, config, &mut commands, health);
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "owns one worker's model, command channels, slots, and health loop"
)]
fn generation_worker_loop(
    model: &LlamaModel,
    projector: Option<&Projector>,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    commands: &mut mpsc::Receiver<Command>,
    controls: &mut mpsc::Receiver<ControlCommand>,
    health: &AtomicU8,
) {
    let slot_count = usize::try_from(config.parallel_sequences)
        .expect("validated parallel sequence count fits usize");
    let mut slots = (0..slot_count)
        .map(Slot::vacant)
        .collect::<Result<Vec<_>, _>>()
        .expect("validated slot ids fit native sequence ids");
    let mut pending_command = None;
    let mut prefill_cursor = 0;
    let load_epoch = u64::try_from(LOAD_EPOCH.fetch_add(1, Ordering::AcqRel)).unwrap_or(u64::MAX);
    let mut prefix_cache = PrefixCache::<SeqState>::new(config.prompt_cache.clone());
    let mut media_prefilled_last_tick = false;
    loop {
        let tick = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generation_tick(
                model,
                projector,
                context,
                template,
                config,
                commands,
                controls,
                &mut slots,
                &mut pending_command,
                &mut prefill_cursor,
                &mut prefix_cache,
                load_epoch,
                &mut media_prefilled_last_tick,
            )
        }));
        match tick {
            Ok(Ok(TickResult::Continue)) => {}
            Ok(Ok(TickResult::Shutdown)) => {
                cancel_all_slots(&mut slots);
                if let Err(error) = finalize_draining_slots(context, &mut slots) {
                    health.store(1, Ordering::Release);
                    fail_all_slots(&mut slots, &error);
                }
                if let Some(command) = pending_command.take() {
                    reject_command(command, Error::Cancelled);
                }
                break;
            }
            Ok(Err(error)) => {
                fatal_generation_worker(context, &mut slots, health, error);
                break;
            }
            Err(payload) => {
                fatal_generation_worker(
                    context,
                    &mut slots,
                    health,
                    Error::WorkerPanic(panic_message(&payload)),
                );
                break;
            }
        }
        if pending_command.is_none() && slots.iter().all(Slot::is_vacant) {
            match commands.blocking_recv() {
                Some(command) => pending_command = Some(command),
                None => break,
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TickResult {
    Continue,
    Shutdown,
}

#[allow(
    clippy::too_many_arguments,
    reason = "one cooperative native scheduler tick"
)]
fn generation_tick(
    model: &LlamaModel,
    projector: Option<&Projector>,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    commands: &mut mpsc::Receiver<Command>,
    controls: &mut mpsc::Receiver<ControlCommand>,
    slots: &mut [Slot],
    pending_command: &mut Option<Command>,
    prefill_cursor: &mut usize,
    prefix_cache: &mut PrefixCache<SeqState>,
    load_epoch: u64,
    media_prefilled_last_tick: &mut bool,
) -> Result<TickResult, Error> {
    poll_reasoning_controls(slots, controls);
    observe_cancellations(slots);
    flush_active_content(slots);
    observe_cancellations(slots);
    finalize_draining_slots(context, slots)?;
    let media_prefilled = poll_generation_decisions(
        model,
        projector,
        context,
        template,
        config,
        slots,
        prefix_cache,
        load_epoch,
        *media_prefilled_last_tick,
    )?;
    if media_prefilled {
        *media_prefilled_last_tick = true;
        return Ok(TickResult::Continue);
    }

    if pending_command.is_none() {
        match commands.try_recv() {
            Ok(command) => *pending_command = Some(command),
            Err(mpsc::error::TryRecvError::Disconnected) => return Ok(TickResult::Shutdown),
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
    }
    if let Some(command) = pending_command.take() {
        match admit_generation_command(model, projector, template, config, slots, command) {
            Admission::Admitted => {}
            Admission::Blocked(command) => *pending_command = Some(command),
            Admission::Shutdown => return Ok(TickResult::Shutdown),
        }
    }

    let capacity = usize::try_from(config.batch_size)
        .map_err(|error| Error::InvalidConfig(error.to_string()))?;
    let plan = plan_batch(slots, capacity, *prefill_cursor);
    *prefill_cursor = plan.next_prefill_slot;
    if plan.entries.is_empty() {
        return Ok(TickResult::Continue);
    }

    let mut batch = LlamaBatch::new(capacity, 1);
    for entry in &plan.entries {
        batch
            .add(
                entry.token,
                entry.position,
                &[slots[entry.slot].id.sequence()?],
                entry.logits,
            )
            .map_err(|error| Error::Generation(error.to_string()))?;
    }
    let decode_started = std::time::Instant::now();
    context
        .decode(&mut batch)
        .map_err(|error| Error::Generation(error.to_string()))?;
    let decode_elapsed = decode_started.elapsed();
    record_decode(&plan);
    apply_decoded_batch(model, context, slots, &plan, decode_elapsed, prefix_cache)?;
    *media_prefilled_last_tick = false;
    Ok(TickResult::Continue)
}

fn poll_reasoning_controls(slots: &mut [Slot], controls: &mut mpsc::Receiver<ControlCommand>) {
    while let Ok(command) = controls.try_recv() {
        if command
            .state
            .compare_exchange(
                CONTROL_PENDING,
                CONTROL_APPLYING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            let _ = command.result.send(Ok(ReasoningControlResult::NotFound));
            continue;
        }
        let mut result = Ok(ReasoningControlResult::NotFound);
        for slot in slots.iter_mut() {
            let SlotState::Active(active) = &mut slot.state else {
                continue;
            };
            if !controls_operation(active.lifecycle.operation, command.operation) {
                continue;
            }
            result = active.sampler.force_reasoning_end().map(|forced| {
                if forced {
                    ReasoningControlResult::Success
                } else {
                    ReasoningControlResult::NotReasoning
                }
            });
            break;
        }
        command.state.store(CONTROL_COMPLETED, Ordering::Release);
        let _ = command.result.send(result);
    }
}

enum Admission {
    Admitted,
    Blocked(Command),
    Shutdown,
}

fn admit_generation_command(
    model: &LlamaModel,
    projector: Option<&Projector>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    slots: &mut [Slot],
    command: Command,
) -> Admission {
    match command {
        Command::ReserveGeneration {
            operation,
            cancelled,
            events,
            reserved,
            readiness,
            acknowledged,
            terminal,
            decision,
        } => {
            let Some(slot) = slots.iter_mut().find(|slot| slot.is_vacant()) else {
                return Admission::Blocked(Command::ReserveGeneration {
                    operation,
                    cancelled,
                    events,
                    reserved,
                    readiness,
                    acknowledged,
                    terminal,
                    decision,
                });
            };
            if cancelled.load(Ordering::Acquire) || reserved.send(Ok(())).is_err() {
                let _ = acknowledged.send(Err(Error::Cancelled));
                return Admission::Admitted;
            }
            slot.state = SlotState::Reserved(ReservedSlot {
                lifecycle: Lifecycle {
                    operation,
                    cancelled,
                    events: Some(events),
                    readiness: Some(readiness),
                    acknowledged: Some(acknowledged),
                    terminal: Some(terminal),
                },
                decision: Some(decision),
                committed: None,
            });
            Admission::Admitted
        }
        Command::CountTokens {
            request,
            cancelled,
            result,
        } => {
            if slots.iter().any(Slot::is_occupied) {
                return Admission::Blocked(Command::CountTokens {
                    request,
                    cancelled,
                    result,
                });
            }
            let count = if cancelled.load(Ordering::Acquire) {
                Err(Error::Cancelled)
            } else {
                match &request {
                    CountInput::Legacy(request) => {
                        prepare_messages(model, template, &request.messages)
                            .map(|tokens| tokens.len())
                    }
                    CountInput::Semantic(request) => {
                        prepare_semantic_count(model, projector, template, config, request)
                    }
                }
            };
            let _ = result.send(count);
            Admission::Admitted
        }
        Command::ReserveEmbedding {
            reserved,
            readiness,
            acknowledged,
            result,
            ..
        } => {
            let error =
                Error::InvalidConfig("generation deployment cannot create embeddings".to_string());
            let _ = reserved.send(Err(error.clone()));
            let _ = readiness.send(Err(error.clone()));
            let _ = result.send(Err(error.clone()));
            let _ = acknowledged.send(Err(error));
            Admission::Admitted
        }
        Command::Shutdown => Admission::Shutdown,
    }
}

fn observe_cancellations(slots: &mut [Slot]) {
    for slot in slots {
        let cancelled = match &slot.state {
            SlotState::Reserved(reserved) => reserved.lifecycle.cancelled.load(Ordering::Acquire),
            SlotState::Active(active) => active.lifecycle.cancelled.load(Ordering::Acquire),
            SlotState::Draining(draining) => draining.lifecycle.cancelled.load(Ordering::Acquire),
            SlotState::Vacant => false,
        };
        if !cancelled {
            continue;
        }
        let state = std::mem::replace(&mut slot.state, SlotState::Vacant);
        slot.state = match state {
            SlotState::Reserved(mut reserved) => {
                let error = Error::Cancelled;
                if let Some(readiness) = reserved.lifecycle.readiness.take() {
                    let _ = readiness.send(Err(error.clone()));
                }
                SlotState::Draining(DrainingSlot {
                    lifecycle: reserved.lifecycle,
                    pending_tokens: std::collections::VecDeque::new(),
                    pending_content: None,
                    pending_semantic: std::collections::VecDeque::new(),
                    outcome: Err(error),
                })
            }
            SlotState::Active(active) => {
                SlotState::Draining(active.into_draining(Err(Error::Cancelled), false))
            }
            SlotState::Draining(mut draining) => {
                draining.pending_tokens.clear();
                draining.pending_content = None;
                draining.pending_semantic.clear();
                draining.outcome = Err(Error::Cancelled);
                SlotState::Draining(draining)
            }
            SlotState::Vacant => SlotState::Vacant,
        };
    }
}

fn flush_active_content(slots: &mut [Slot]) {
    let mut cancelled = Vec::new();
    for (index, slot) in slots.iter_mut().enumerate() {
        let SlotState::Active(active) = &mut slot.state else {
            continue;
        };
        if try_flush_content(active).is_err() {
            cancelled.push(index);
        }
    }
    for index in cancelled {
        let state = std::mem::replace(&mut slots[index].state, SlotState::Vacant);
        let SlotState::Active(active) = state else {
            unreachable!("only active slots are collected for output cancellation");
        };
        slots[index].state =
            SlotState::Draining(active.into_draining(Err(Error::Cancelled), false));
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "keeps committed request preparation and slot activation atomic"
)]
fn poll_generation_decisions(
    model: &LlamaModel,
    projector: Option<&Projector>,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    slots: &mut [Slot],
    prefix_cache: &mut PrefixCache<SeqState>,
    load_epoch: u64,
    media_prefilled_last_tick: bool,
) -> Result<bool, Error> {
    let context_size = effective_context_size(model, config);
    let ordinary_work = !plan_batch(slots, 1, 0).entries.is_empty();
    for slot in slots {
        if !matches!(slot.state, SlotState::Reserved(_)) {
            continue;
        }
        let SlotState::Reserved(mut reserved) =
            std::mem::replace(&mut slot.state, SlotState::Vacant)
        else {
            unreachable!();
        };
        let request = if let Some(request) = reserved.committed.take() {
            Some(request)
        } else {
            match reserved
                .decision
                .as_mut()
                .map(tokio::sync::oneshot::Receiver::try_recv)
            {
                Some(Ok(ReservationDecision::Commit(request))) => {
                    reserved.decision = None;
                    Some(request)
                }
                None
                | Some(
                    Ok(ReservationDecision::Abort)
                    | Err(tokio::sync::oneshot::error::TryRecvError::Closed),
                ) => None,
                Some(Err(tokio::sync::oneshot::error::TryRecvError::Empty)) => {
                    slot.state = SlotState::Reserved(reserved);
                    continue;
                }
            }
        };
        let Some(request) = request else {
            if let Some(readiness) = reserved.lifecycle.readiness.take() {
                let _ = readiness.send(Err(Error::Cancelled));
            }
            slot.state = SlotState::Draining(DrainingSlot {
                lifecycle: reserved.lifecycle,
                pending_tokens: std::collections::VecDeque::new(),
                pending_content: None,
                pending_semantic: std::collections::VecDeque::new(),
                outcome: Err(Error::Cancelled),
            });
            continue;
        };
        let has_media = request_has_images(&request);
        if defer_media_prefill(has_media, media_prefilled_last_tick, ordinary_work) {
            reserved.committed = Some(request);
            slot.state = SlotState::Reserved(reserved);
            continue;
        }
        if reserved.lifecycle.cancelled.load(Ordering::Acquire) {
            if let Some(readiness) = reserved.lifecycle.readiness.take() {
                let _ = readiness.send(Err(Error::Cancelled));
            }
            slot.state = SlotState::Draining(DrainingSlot {
                lifecycle: reserved.lifecycle,
                pending_tokens: std::collections::VecDeque::new(),
                pending_content: None,
                pending_semantic: std::collections::VecDeque::new(),
                outcome: Err(Error::Cancelled),
            });
            continue;
        }
        let mut prepared = match prepare_generation(model, projector, template, config, &request) {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Some(readiness) = reserved.lifecycle.readiness.take() {
                    let _ = readiness.send(Err(error.clone()));
                }
                slot.state = SlotState::Draining(DrainingSlot {
                    lifecycle: reserved.lifecycle,
                    pending_tokens: std::collections::VecDeque::new(),
                    pending_content: None,
                    pending_semantic: std::collections::VecDeque::new(),
                    outcome: Err(error),
                });
                continue;
            }
        };
        let tokens = &prepared.tokens;
        let prompt_tokens = prepared
            .media
            .as_ref()
            .map_or(tokens.len(), PreparedMedia::total_tokens);
        let next_position = prepared.media.as_ref().map_or(tokens.len(), |media| {
            usize::try_from(media.total_positions()).expect("prepared media positions are positive")
        });
        let cache_target = tokens.len().saturating_sub(1);
        let compatibility = Arc::new(cache_compatibility(&request, template, load_epoch));
        let mut cache_n = 0;
        if prefix_cache.enabled() && request_allows_prefix_cache(&request) {
            let restore = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let Some((prefix_len, state)) = prefix_cache.lookup(&compatibility, tokens) else {
                    return Ok(None);
                };
                let sequence = slot.id.sequence().map_err(|error| error.to_string())?;
                context
                    .state_seq_set(state, sequence)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>(Some(prefix_len))
            }));
            match restore {
                Ok(Ok(Some(prefix_len))) => cache_n = prefix_len,
                Ok(Ok(None)) => {}
                Ok(Err(error)) => {
                    tracing::warn!(%error, "prompt cache restore failed; disabling worker cache");
                    prefix_cache.disable();
                    context
                        .kv_cache_seq_rm(slot.id.sequence()?, None, None)
                        .map_err(|cleanup| {
                            Error::Generation(format!(
                                "prompt cache fallback sequence cleanup failed: {cleanup}"
                            ))
                        })?;
                }
                Err(payload) => {
                    tracing::warn!(
                        error = %panic_message(&payload),
                        "prompt cache restore panicked; disabling worker cache"
                    );
                    prefix_cache.disable();
                    context
                        .kv_cache_seq_rm(slot.id.sequence()?, None, None)
                        .map_err(|cleanup| {
                            Error::Generation(format!(
                                "prompt cache fallback sequence cleanup failed: {cleanup}"
                            ))
                        })?;
                }
            }
        }
        let reasoning_control = prepared.reasoning_control.take();
        let sampler_prompt = sampler_prompt_tokens(&prepared);
        let sampler = match create_sampler(
            model,
            &request,
            context_size,
            sampler_prompt,
            prepared
                .semantic_parser
                .as_ref()
                .map(|parser| &parser.metadata),
            reasoning_control,
        ) {
            Ok(sampler) => sampler,
            Err(error) => {
                if let Some(readiness) = reserved.lifecycle.readiness.take() {
                    let _ = readiness.send(Err(error.clone()));
                }
                slot.state = SlotState::Draining(DrainingSlot {
                    lifecycle: reserved.lifecycle,
                    pending_tokens: std::collections::VecDeque::new(),
                    pending_content: None,
                    pending_semantic: std::collections::VecDeque::new(),
                    outcome: Err(error),
                });
                continue;
            }
        };
        let tokens = prepared.tokens;
        let mut active = ActiveSlot {
            lifecycle: reserved.lifecycle,
            request: request.clone(),
            tokens,
            prefill_cursor: cache_n,
            cache_n,
            cache_capture_at: None,
            cache_compatibility: compatibility,
            pending_decode: None,
            sampler,
            decoder: encoding_rs::UTF_8.new_decoder(),
            semantic_parser: prepared
                .parse_semantic
                .then(|| prepared.semantic_parser.map(PreparedChat::into_parser))
                .flatten(),
            preserved_tokens: prepared.preserved_tokens,
            stop_filter: StopFilter::new(prepared.stops),
            pending_semantic: std::collections::VecDeque::new(),
            pending_tokens: std::collections::VecDeque::new(),
            pending_content: None,
            completion_tokens: 0,
            reasoning_tokens: 0,
            reasoning_active: prepared.reasoning_active,
            prompt_tokens,
            next_position,
            prompt_wall: std::time::Duration::ZERO,
            predicted_wall: std::time::Duration::ZERO,
        };
        if let Some(media) = prepared.media {
            let projector = projector.expect("media preparation requires configured projector");
            if active.lifecycle.cancelled.load(Ordering::Acquire) {
                slot.state =
                    SlotState::Draining(active.into_draining(Err(Error::Cancelled), false));
                continue;
            }
            let started = std::time::Instant::now();
            let sequence = match slot.id.sequence() {
                Ok(sequence) => sequence,
                Err(error) => {
                    slot.state = SlotState::Active(active);
                    return Err(error);
                }
            };
            let batch = match i32::try_from(config.micro_batch_size) {
                Ok(batch) => batch,
                Err(error) => {
                    slot.state = SlotState::Active(active);
                    return Err(Error::InvalidConfig(error.to_string()));
                }
            };
            let next = match projector.eval(&media, context, sequence, batch) {
                Ok(next) => next,
                Err(error) => {
                    slot.state = SlotState::Active(active);
                    return Err(error);
                }
            };
            active.prompt_wall = started.elapsed();
            if next != media.total_positions() {
                let error = Error::Generation(format!(
                    "multimodal prefill returned position {next}, expected {}",
                    media.total_positions()
                ));
                slot.state = SlotState::Active(active);
                return Err(error);
            }
            if active.lifecycle.cancelled.load(Ordering::Acquire) {
                slot.state =
                    SlotState::Draining(active.into_draining(Err(Error::Cancelled), false));
                continue;
            }
            if active.request.options.max_tokens > 0 {
                let top = active.request.logprobs.map(|options| options.top_logprobs);
                let candidates = PreparedMedia::last_logits(context);
                let (token, logprobs) =
                    match active.sampler.sample_candidates(model, candidates, top) {
                        Ok(sample) => sample,
                        Err(error) => {
                            slot.state =
                                SlotState::Draining(active.into_draining(Err(error), false));
                            return Ok(true);
                        }
                    };
                let reason = match accept_sampled_token(model, &mut active, token, logprobs) {
                    Ok(reason) => reason,
                    Err(error) => {
                        slot.state = SlotState::Draining(active.into_draining(Err(error), false));
                        return Ok(true);
                    }
                };
                if let Some(reason) = reason {
                    let terminal = finish_event(&active, reason);
                    slot.state = SlotState::Draining(active.into_draining(Ok(terminal), true));
                    return Ok(true);
                }
            } else {
                let terminal = finish_event(&active, FinishReason::Length);
                slot.state = SlotState::Draining(active.into_draining(Ok(terminal), true));
                return Ok(true);
            }
            if active
                .lifecycle
                .readiness
                .take()
                .is_none_or(|readiness| readiness.send(Ok(())).is_err())
            {
                slot.state =
                    SlotState::Draining(active.into_draining(Err(Error::Cancelled), false));
                continue;
            }
            slot.state = SlotState::Active(active);
            return Ok(true);
        }
        if active
            .lifecycle
            .readiness
            .take()
            .is_none_or(|readiness| readiness.send(Ok(())).is_err())
        {
            slot.state = SlotState::Draining(DrainingSlot {
                lifecycle: active.lifecycle,
                pending_tokens: std::collections::VecDeque::new(),
                pending_content: None,
                pending_semantic: std::collections::VecDeque::new(),
                outcome: Err(Error::Cancelled),
            });
            continue;
        }
        let cache_capture_at = (prefix_cache.enabled()
            && cache_target >= prefix_cache.min_prefix_tokens()
            && cache_n < cache_target)
            .then_some(cache_target);
        active.cache_capture_at = cache_capture_at;
        slot.state = SlotState::Active(active);
    }
    Ok(false)
}

fn apply_decoded_batch(
    model: &LlamaModel,
    context: &llama_cpp_2::context::LlamaContext<'_>,
    slots: &mut [Slot],
    plan: &crate::slot::BatchPlan,
    elapsed: std::time::Duration,
    prefix_cache: &mut PrefixCache<SeqState>,
) -> Result<(), Error> {
    let mut saw_prefill = vec![false; slots.len()];
    let mut saw_decode = vec![false; slots.len()];
    for entry in &plan.entries {
        let SlotState::Active(active) = &mut slots[entry.slot].state else {
            continue;
        };
        match entry.kind {
            crate::slot::BatchEntryKind::Prefill => {
                active.prefill_cursor += 1;
                saw_prefill[entry.slot] = true;
            }
            crate::slot::BatchEntryKind::Decode => {
                active.pending_decode = None;
                active.next_position = active.next_position.saturating_add(1);
                saw_decode[entry.slot] = true;
            }
        }
    }
    for (index, slot) in slots.iter_mut().enumerate() {
        let SlotState::Active(active) = &mut slot.state else {
            continue;
        };
        if saw_prefill[index] {
            active.prompt_wall += elapsed;
        }
        if saw_decode[index] {
            active.predicted_wall += elapsed;
        }
    }

    capture_prompt_prefixes(context, slots, prefix_cache);

    observe_cancellations(slots);
    let logits = logits_targets(plan)?;
    for (raw_index, slot_index) in logits {
        let finish_reason = {
            let SlotState::Active(active) = &mut slots[slot_index].state else {
                continue;
            };
            if active.lifecycle.cancelled.load(Ordering::Acquire) {
                continue;
            }
            let top_logprobs = active.request.logprobs.map(|options| options.top_logprobs);
            match active
                .sampler
                .sample(model, context, raw_index, top_logprobs)
                .and_then(|(token, logprobs)| accept_sampled_token(model, active, token, logprobs))
            {
                Ok(reason) => reason,
                Err(error) => {
                    fail_active_slot(slots, slot_index, error);
                    continue;
                }
            }
        };
        if let Some(reason) = finish_reason {
            drain_finished_slot(slots, slot_index, reason);
        }
    }

    observe_cancellations(slots);
    let zero_length = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| match &slot.state {
            SlotState::Active(active)
                if active.prefill_cursor == active.tokens.len()
                    && active.request.options.max_tokens == 0 =>
            {
                Some(index)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for index in zero_length {
        drain_finished_slot(slots, index, FinishReason::Length);
    }
    Ok(())
}

fn accept_sampled_token(
    model: &LlamaModel,
    active: &mut ActiveSlot,
    token: LlamaToken,
    logprobs: Option<TokenLogprobs>,
) -> Result<Option<FinishReason>, Error> {
    if model.is_eog_token(token) {
        return Ok(Some(FinishReason::Stop));
    }
    active.completion_tokens += 1;
    let was_reasoning = active.reasoning_active;
    if was_reasoning {
        active.reasoning_tokens += 1;
    }
    let render_special = active.preserved_tokens.contains(&token);
    let piece = model
        .token_to_piece(token, &mut active.decoder, render_special, None)
        .map_err(|error| Error::Generation(error.to_string()))?;
    let stopped = if let Some(logprobs) = logprobs {
        let (stopped, output) = active.stop_filter.push_token(piece, logprobs);
        active.pending_tokens.extend(output);
        stopped
    } else {
        let (stopped, output) = active.stop_filter.push_piece(&piece);
        if let Some(output) = output {
            if let Some(parser) = &mut active.semantic_parser {
                let deltas = parser.push(&output, true)?;
                let produced_reasoning = deltas
                    .iter()
                    .any(|delta| matches!(delta, SemanticDelta::Reasoning(_)));
                let left_reasoning = deltas.iter().any(|delta| {
                    matches!(
                        delta,
                        SemanticDelta::Text(_) | SemanticDelta::ToolCall { .. }
                    )
                });
                if produced_reasoning && !was_reasoning {
                    active.reasoning_tokens += 1;
                }
                if produced_reasoning {
                    active.reasoning_active = true;
                }
                if left_reasoning {
                    active.reasoning_active = false;
                }
                active.pending_semantic.extend(deltas);
            } else {
                active.pending_content = Some(output);
            }
        }
        stopped
    };
    if stopped {
        Ok(Some(FinishReason::Stop))
    } else if active.completion_tokens >= active.request.options.max_tokens {
        Ok(Some(FinishReason::Length))
    } else {
        active.pending_decode = Some(token);
        Ok(None)
    }
}

fn capture_prompt_prefixes(
    context: &llama_cpp_2::context::LlamaContext<'_>,
    slots: &mut [Slot],
    prefix_cache: &mut PrefixCache<SeqState>,
) {
    if !prefix_cache.enabled() {
        for slot in slots {
            if let SlotState::Active(active) = &mut slot.state {
                active.cache_capture_at = None;
            }
        }
        return;
    }
    for slot in slots.iter_mut() {
        let SlotState::Active(active) = &mut slot.state else {
            continue;
        };
        let Some(target) = active.cache_capture_at else {
            continue;
        };
        if active.prefill_cursor != target {
            continue;
        }
        let capture = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let sequence = slot.id.sequence().map_err(|error| error.to_string())?;
            context
                .state_seq_get(sequence, LlamaStateSeqFlags::empty())
                .map_err(|error| error.to_string())
        }));
        match capture {
            Ok(Ok(state)) => {
                let bytes = state.byte_len();
                prefix_cache.insert(
                    Arc::clone(&active.cache_compatibility),
                    active.tokens[..target].to_vec(),
                    state,
                    bytes,
                );
                active.cache_capture_at = None;
            }
            Ok(Err(error)) => {
                tracing::warn!(%error, "prompt cache capture failed; disabling worker cache");
                prefix_cache.disable();
                break;
            }
            Err(payload) => {
                tracing::warn!(
                    error = %panic_message(&payload),
                    "prompt cache capture panicked; disabling worker cache"
                );
                prefix_cache.disable();
                break;
            }
        }
    }
    if !prefix_cache.enabled() {
        for slot in slots {
            if let SlotState::Active(active) = &mut slot.state {
                active.cache_capture_at = None;
            }
        }
    }
}

fn drain_finished_slot(slots: &mut [Slot], index: usize, mut reason: FinishReason) {
    let state = std::mem::replace(&mut slots[index].state, SlotState::Vacant);
    let SlotState::Active(mut active) = state else {
        return;
    };
    if let Some(parser) = &mut active.semantic_parser {
        let suffix = active.stop_filter.take_flush().unwrap_or_default();
        let deltas = match parser.finish(&suffix) {
            Ok(deltas) => deltas,
            Err(error) => {
                slots[index].state = SlotState::Draining(active.into_draining(Err(error), false));
                return;
            }
        };
        active.pending_semantic.extend(deltas);
        if parser.has_tool_calls() {
            reason = FinishReason::ToolCalls;
        }
    }
    let terminal = finish_event(&active, reason);
    slots[index].state = SlotState::Draining(active.into_draining(Ok(terminal), true));
}

fn fail_active_slot(slots: &mut [Slot], index: usize, error: Error) {
    let state = std::mem::replace(&mut slots[index].state, SlotState::Vacant);
    let SlotState::Active(active) = state else {
        return;
    };
    slots[index].state = SlotState::Draining(active.into_draining(Err(error), false));
}

fn finalize_draining_slots(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    slots: &mut [Slot],
) -> Result<(), Error> {
    finalize_draining_slots_with(slots, |slot_id| {
        context
            .kv_cache_seq_rm(slot_id.sequence()?, None, None)
            .map_err(|error| Error::Generation(format!("KV sequence cleanup failed: {error}")))
    })
}

fn finalize_draining_slots_with(
    slots: &mut [Slot],
    mut clear_sequence: impl FnMut(crate::slot::SlotId) -> Result<(), Error>,
) -> Result<(), Error> {
    for slot in slots {
        let SlotState::Draining(draining) = &mut slot.state else {
            continue;
        };
        if draining.lifecycle.cancelled.load(Ordering::Acquire) {
            draining.pending_tokens.clear();
            draining.pending_content = None;
            draining.outcome = Err(Error::Cancelled);
        }
        if !try_flush_draining(draining) {
            continue;
        }
        clear_sequence(slot.id)?;
        let SlotState::Draining(draining) = &mut slot.state else {
            unreachable!();
        };
        drop(draining.lifecycle.events.take());
        if let Some(readiness) = draining.lifecycle.readiness.take() {
            let _ = readiness.send(draining.outcome.clone().map(|_| ()));
        }
        if let Some(terminal) = draining.lifecycle.terminal.take() {
            let _ = terminal.send(draining.outcome.clone());
        }
        if let Some(acknowledged) = draining.lifecycle.acknowledged.take() {
            let _ = acknowledged.send(draining.outcome.clone().map(|_| ()));
        }
        slot.state = SlotState::Vacant;
    }
    Ok(())
}

fn record_decode(plan: &crate::slot::BatchPlan) {
    let mut busy = std::collections::BTreeSet::new();
    let mut has_prefill = false;
    let mut has_decode = false;
    for entry in &plan.entries {
        busy.insert(entry.slot);
        has_prefill |= matches!(entry.kind, crate::slot::BatchEntryKind::Prefill);
        has_decode |= matches!(entry.kind, crate::slot::BatchEntryKind::Decode);
    }
    DECODE_CALLS.fetch_add(1, Ordering::AcqRel);
    if busy.len() > 1 {
        MULTI_SLOT_DECODE_CALLS.fetch_add(1, Ordering::AcqRel);
    }
    if has_prefill && has_decode {
        PREFILL_DECODE_OVERLAP_CALLS.fetch_add(1, Ordering::AcqRel);
    }
    MAX_BUSY_SLOTS_PER_DECODE.fetch_max(busy.len(), Ordering::AcqRel);
    tracing::trace!(
        batch_tokens = plan.entries.len(),
        busy_slots = busy.len(),
        has_prefill,
        has_decode,
        "llama.cpp cooperative scheduler decode"
    );
}

fn cancel_all_slots(slots: &mut [Slot]) {
    for slot in slots.iter() {
        match &slot.state {
            SlotState::Reserved(reserved) => {
                reserved.lifecycle.cancelled.store(true, Ordering::Release);
            }
            SlotState::Active(active) => {
                active.lifecycle.cancelled.store(true, Ordering::Release);
            }
            SlotState::Draining(draining) => {
                draining.lifecycle.cancelled.store(true, Ordering::Release);
            }
            SlotState::Vacant => {}
        }
    }
    observe_cancellations(slots);
}

fn fatal_generation_worker(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    slots: &mut [Slot],
    health: &AtomicU8,
    mut error: Error,
) {
    health.store(1, Ordering::Release);
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.clear_kv_cache();
    })) {
        error = Error::WorkerPanic(panic_message(&payload));
    }
    fail_all_slots(slots, &error);
}

fn fail_all_slots(slots: &mut [Slot], error: &Error) {
    for slot in slots {
        let lifecycle = match &mut slot.state {
            SlotState::Reserved(reserved) => &mut reserved.lifecycle,
            SlotState::Active(active) => &mut active.lifecycle,
            SlotState::Draining(draining) => &mut draining.lifecycle,
            SlotState::Vacant => continue,
        };
        drop(lifecycle.events.take());
        if let Some(readiness) = lifecycle.readiness.take() {
            let _ = readiness.send(Err(error.clone()));
        }
        if let Some(terminal) = lifecycle.terminal.take() {
            let _ = terminal.send(Err(error.clone()));
        }
        if let Some(acknowledged) = lifecycle.acknowledged.take() {
            let _ = acknowledged.send(Err(error.clone()));
        }
        slot.state = SlotState::Vacant;
    }
}

fn reject_command(command: Command, error: Error) {
    match command {
        Command::CountTokens { result, .. } => {
            let _ = result.send(Err(error));
        }
        Command::ReserveGeneration {
            reserved,
            readiness,
            acknowledged,
            terminal,
            ..
        } => {
            let _ = reserved.send(Err(error.clone()));
            let _ = readiness.send(Err(error.clone()));
            let _ = terminal.send(Err(error.clone()));
            let _ = acknowledged.send(Err(error));
        }
        Command::ReserveEmbedding {
            reserved,
            readiness,
            acknowledged,
            result,
            ..
        } => {
            let _ = reserved.send(Err(error.clone()));
            let _ = readiness.send(Err(error.clone()));
            let _ = result.send(Err(error.clone()));
            let _ = acknowledged.send(Err(error));
        }
        Command::Shutdown => {}
    }
}

fn embedding_worker_loop(
    model: &LlamaModel,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    config: &RuntimeConfig,
    commands: &mut mpsc::Receiver<Command>,
    health: &AtomicU8,
) {
    while let Some(command) = commands.blocking_recv() {
        match command {
            Command::ReserveEmbedding {
                cancelled,
                reserved,
                readiness,
                acknowledged,
                result,
                decision,
            } => {
                if cancelled.load(Ordering::Acquire) || reserved.send(Ok(())).is_err() {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                }
                let EmbeddingReservationDecision::Commit(request) =
                    wait_embedding_reservation_decision(decision, &cancelled)
                else {
                    let _ = acknowledged.send(Err(Error::Cancelled));
                    continue;
                };
                if cancelled.load(Ordering::Acquire) || readiness.send(Ok(())).is_err() {
                    let error = Error::Cancelled;
                    let _ = result.send(Err(error.clone()));
                    let _ = acknowledged.send(Err(error));
                    continue;
                }
                let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_embeddings(model, context, config, &request, &cancelled)
                }));
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    context.clear_kv_cache();
                }));
                let output = match (execution, cleanup) {
                    (Ok(output), Ok(())) => output,
                    (Err(payload), _) | (_, Err(payload)) => {
                        Err(Error::WorkerPanic(panic_message(&payload)))
                    }
                };
                if matches!(output, Err(Error::WorkerPanic(_))) {
                    health.store(1, Ordering::Release);
                }
                let fatal = matches!(output, Err(Error::WorkerPanic(_)));
                let _ = result.send(output.clone());
                let _ = acknowledged.send(output.map(|_| ()));
                if fatal {
                    break;
                }
            }
            Command::CountTokens { result, .. } => {
                let _ = result.send(Err(Error::InvalidConfig(
                    "embedding deployment cannot count generation input tokens".to_string(),
                )));
            }
            Command::ReserveGeneration {
                reserved,
                readiness,
                acknowledged,
                terminal,
                ..
            } => {
                let error =
                    Error::InvalidConfig("embedding deployment cannot generate text".to_string());
                let _ = reserved.send(Err(error.clone()));
                let _ = readiness.send(Err(error.clone()));
                let _ = terminal.send(Err(error.clone()));
                let _ = acknowledged.send(Err(error));
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

struct PreparedGeneration {
    tokens: Vec<LlamaToken>,
    media_prompt_tokens: Option<Vec<LlamaToken>>,
    media: Option<PreparedMedia>,
    semantic_parser: Option<PreparedChat>,
    preserved_tokens: Vec<LlamaToken>,
    stops: Vec<String>,
    parse_semantic: bool,
    reasoning_active: bool,
    reasoning_control: Option<ReasoningControl>,
}

fn sampler_prompt_tokens(prepared: &PreparedGeneration) -> &[LlamaToken] {
    prepared
        .media_prompt_tokens
        .as_deref()
        .unwrap_or(&prepared.tokens)
}

#[allow(
    clippy::too_many_lines,
    reason = "prepares one complete text or media generation contract"
)]
fn prepare_generation(
    model: &LlamaModel,
    projector: Option<&Projector>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    request: &AdvancedRequest,
) -> Result<PreparedGeneration, Error> {
    let mut semantic_parser = None;
    let mut preserved_tokens = Vec::new();
    let mut stops = request.options.stop.clone();
    let mut parse_semantic = false;
    let mut reasoning_active = false;
    let mut media = None;
    let mut media_prompt_tokens = None;
    let mut reasoning_control = None;
    let tokens = match &request.input {
        Input::Messages(messages) => prepare_messages(model, template, messages)?,
        Input::Prompt(prompt) => tokenize_prompt(model, prompt)?,
        Input::Semantic(semantic) => {
            let (source, bos, eos, template_thinking) =
                common_chat_template_parts(model, template)?;
            let mut reasoning = semantic.reasoning;
            let reasoning_enabled =
                reasoning.enabled.unwrap_or(template_thinking) || reasoning.effort.is_some();
            reasoning.enabled = Some(reasoning_enabled);
            parse_semantic = reasoning_enabled
                || reasoning.effort.is_some()
                || (!semantic.tools.is_empty() && semantic.tool_choice != ToolChoice::None);
            reasoning_active = reasoning_enabled;
            if request.logprobs.is_some() && parse_semantic {
                return Err(Error::Unsupported {
                    field: "logprobs",
                    detail: "token logprobs cannot be truthfully mapped through parsed reasoning or tool calls".to_string(),
                });
            }
            let prepared = PreparedChat::prepare(Preparation {
                template: source,
                bos: &bos,
                eos: &eos,
                messages: &semantic.messages,
                tools: &semantic.tools,
                tool_choice: &semantic.tool_choice,
                parallel_tool_calls: semantic.parallel_tool_calls,
                reasoning,
                output: &request.output,
            })?;
            if reasoning_enabled && !prepared.metadata.supports_thinking {
                return Err(Error::Unsupported {
                    field: "reasoning",
                    detail: "selected template does not expose separate reasoning".to_string(),
                });
            }
            if request.reasoning_control_id.is_some() {
                reasoning_control = Some(reasoning_control_for(model, &prepared.metadata)?);
            }
            for piece in &prepared.metadata.preserved_tokens {
                let ids = model
                    .str_to_token(piece, AddBos::Never)
                    .map_err(|error| Error::Generation(error.to_string()))?;
                if let [token] = ids.as_slice() {
                    preserved_tokens.push(*token);
                }
            }
            for stop in &prepared.metadata.additional_stops {
                if !stop.is_empty() && !stops.contains(stop) {
                    stops.push(stop.clone());
                }
            }
            let images = semantic
                .messages
                .iter()
                .flat_map(|message| &message.content)
                .filter_map(|part| match part {
                    ContentPart::Image(image) => Some(image),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let tokens = if images.is_empty() {
                tokenize_prompt(model, &prepared.metadata.prompt)?
            } else {
                let projector = projector.ok_or_else(|| Error::Unsupported {
                    field: "messages.content.image",
                    detail: "deployment has no configured mmproj".to_string(),
                })?;
                validate_images(
                    &images,
                    config.vision.as_ref().expect("projector has config"),
                )?;
                let ordered_images = prepared
                    .rendered_image_order
                    .iter()
                    .map(|&index| {
                        images.get(index).copied().ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "rendered image order references missing image {index}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let prepared_media = projector.prepare(
                    prepared.metadata.prompt.clone(),
                    &ordered_images.into_iter().cloned().collect::<Vec<_>>(),
                )?;
                media_prompt_tokens = Some(prepared_media.text_tokens().to_vec());
                media = Some(prepared_media);
                Vec::new()
            };
            semantic_parser = Some(prepared);
            tokens
        }
    };
    let context_size = effective_context_size(model, config);
    let prompt_tokens = media
        .as_ref()
        .map_or(tokens.len(), PreparedMedia::total_tokens);
    let prompt_positions = media.as_ref().map_or(tokens.len(), |prepared| {
        usize::try_from(prepared.total_positions()).expect("prepared media positions are positive")
    });
    if prompt_positions >= context_size
        || request.options.max_tokens > context_size - prompt_positions
    {
        return Err(Error::ContextLimit {
            prompt_tokens,
            max_tokens: request.options.max_tokens,
            context_size,
        });
    }
    Ok(PreparedGeneration {
        tokens,
        media_prompt_tokens,
        media,
        semantic_parser,
        preserved_tokens,
        stops,
        parse_semantic,
        reasoning_active,
        reasoning_control,
    })
}

fn request_has_images(request: &AdvancedRequest) -> bool {
    matches!(&request.input, Input::Semantic(input) if input.messages.iter().any(|message| {
        message.content.iter().any(|part| matches!(part, ContentPart::Image(_)))
    }))
}

fn reasoning_control_for(
    model: &LlamaModel,
    metadata: &crate::common_chat::PreparedMetadata,
) -> Result<ReasoningControl, Error> {
    if metadata.thinking_start_tag.is_empty() || metadata.thinking_end_tags.is_empty() {
        return Err(Error::Unsupported {
            field: "reasoning_control",
            detail: "selected template does not expose reasoning control tags".to_string(),
        });
    }
    let tokenize = |value: &str| {
        model
            .str_to_token(value, AddBos::Never)
            .map(|tokens| tokens.into_iter().map(|token| token.0).collect::<Vec<_>>())
            .map_err(|error| Error::Generation(error.to_string()))
    };
    let start = tokenize(&metadata.thinking_start_tag)?;
    let ends = metadata
        .thinking_end_tags
        .iter()
        .map(|tag| tokenize(tag))
        .collect::<Result<Vec<_>, _>>()?;
    let forced = ends.first().cloned().ok_or_else(|| Error::Unsupported {
        field: "reasoning_control",
        detail: "selected template has no forceable reasoning end tag".to_string(),
    })?;
    let prompt = tokenize(&metadata.generation_prompt)?;
    ReasoningControl::new(&start, &ends, &forced, &prompt)
}

fn request_allows_prefix_cache(request: &AdvancedRequest) -> bool {
    !request_has_images(request)
}

const fn defer_media_prefill(
    has_media: bool,
    media_prefilled_last_tick: bool,
    ordinary_work: bool,
) -> bool {
    has_media && media_prefilled_last_tick && ordinary_work
}

fn validate_images(images: &[&ImageInput], config: &LlmVisionConfig) -> Result<(), Error> {
    let limits = config.limits;
    if images.len() > limits.max_images {
        return Err(Error::InvalidConfig(format!(
            "image count exceeds configured maximum {}",
            limits.max_images
        )));
    }
    let mut total_bytes = 0usize;
    let mut total_pixels = 0u64;
    for image in images {
        let pixels = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .ok_or_else(|| Error::InvalidConfig("image dimensions overflow".to_string()))?;
        if image.bytes.len() > limits.max_bytes_per_image
            || image.width == 0
            || image.height == 0
            || image.width > limits.max_side
            || image.height > limits.max_side
            || pixels > limits.max_pixels_per_image
        {
            return Err(Error::InvalidConfig(
                "image exceeds configured byte or dimension limits".to_string(),
            ));
        }
        total_bytes = total_bytes
            .checked_add(image.bytes.len())
            .ok_or_else(|| Error::InvalidConfig("aggregate image bytes overflow".to_string()))?;
        total_pixels = total_pixels
            .checked_add(pixels)
            .ok_or_else(|| Error::InvalidConfig("aggregate image pixels overflow".to_string()))?;
    }
    if total_bytes > limits.max_total_bytes || total_pixels > limits.max_total_pixels {
        return Err(Error::InvalidConfig(
            "images exceed configured aggregate limits".to_string(),
        ));
    }
    Ok(())
}

fn cache_compatibility(
    request: &AdvancedRequest,
    template: &EffectiveTemplate,
    load_epoch: u64,
) -> Compatibility {
    let input = match &request.input {
        Input::Prompt(_) => InputCompatibility::RawPrompt,
        Input::Messages(_) | Input::Semantic(_) => match template {
            EffectiveTemplate::LlamaCpp {
                source,
                enable_thinking,
                ..
            } => InputCompatibility::Template {
                engine: TemplateEngine::LlamaCpp,
                source: source.clone(),
                enable_thinking: *enable_thinking,
            },
            EffectiveTemplate::Jinja {
                source,
                enable_thinking,
            } => InputCompatibility::Template {
                engine: TemplateEngine::Jinja,
                source: source.clone(),
                enable_thinking: *enable_thinking,
            },
        },
    };
    Compatibility { load_epoch, input }
}

fn prepare_messages(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    messages: &[Message],
) -> Result<Vec<LlamaToken>, Error> {
    prepare_legacy_messages(model, template, messages)
}

fn prepare_semantic_count(
    model: &LlamaModel,
    projector: Option<&Projector>,
    template: &EffectiveTemplate,
    config: &RuntimeConfig,
    request: &SemanticTokenCountRequest,
) -> Result<usize, Error> {
    let (source, bos, eos, template_thinking) = common_chat_template_parts(model, template)?;
    let mut reasoning = request.reasoning;
    reasoning.enabled =
        Some(reasoning.enabled.unwrap_or(template_thinking) || reasoning.effort.is_some());
    let prepared = PreparedChat::prepare(Preparation {
        template: source,
        bos: &bos,
        eos: &eos,
        messages: &request.messages,
        tools: &request.tools,
        tool_choice: &request.tool_choice,
        parallel_tool_calls: request.parallel_tool_calls,
        reasoning,
        output: &request.output,
    })?;
    let images = request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|part| match part {
            ContentPart::Image(image) => Some(image),
            _ => None,
        })
        .collect::<Vec<_>>();
    if images.is_empty() {
        return tokenize_prompt(model, &prepared.metadata.prompt).map(|tokens| tokens.len());
    }
    let projector = projector.ok_or_else(|| Error::Unsupported {
        field: "messages.content.image",
        detail: "deployment has no configured mmproj".to_string(),
    })?;
    validate_images(
        &images,
        config.vision.as_ref().expect("projector has config"),
    )?;
    let ordered_images = prepared
        .rendered_image_order
        .iter()
        .map(|&index| {
            images.get(index).copied().ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "rendered image order references missing image {index}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    projector
        .prepare(
            prepared.metadata.prompt.clone(),
            &ordered_images.into_iter().cloned().collect::<Vec<_>>(),
        )
        .map(|prepared| prepared.total_tokens())
}

fn effective_context_size(model: &LlamaModel, config: &RuntimeConfig) -> usize {
    config
        .context_size
        .map_or_else(|| model.n_ctx_train() as usize, |size| size.get() as usize)
}

fn run_embeddings(
    model: &LlamaModel,
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    config: &RuntimeConfig,
    request: &EmbeddingRequest,
    cancelled: &AtomicBool,
) -> Result<EmbeddingOutput, Error> {
    let RuntimeMode::Embeddings {
        max_input_tokens, ..
    } = config.mode
    else {
        return Err(Error::InvalidConfig(
            "generation deployment cannot create embeddings".to_string(),
        ));
    };
    if request.inputs.is_empty() || request.inputs.len() > 2048 {
        return Err(Error::Embedding("input must not be empty".to_string()));
    }
    let output_dimensions =
        usize::try_from(model.n_embd_out()).map_err(|error| Error::Embedding(error.to_string()))?;
    if output_dimensions == 0 {
        return Err(Error::Embedding(
            "model reports zero output embedding dimensions".to_string(),
        ));
    }
    let context_size = effective_context_size(model, config);
    let vocabulary_size = model.n_vocab();
    let mut tokenized_inputs = Vec::with_capacity(request.inputs.len());
    let mut prompt_tokens = 0usize;
    for input in &request.inputs {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let tokens = match input {
            EmbeddingInput::Text(text) => model
                .str_to_token(text, AddBos::Always)
                .map_err(|error| Error::Embedding(error.to_string()))?,
            EmbeddingInput::Tokens(tokens) => {
                validate_embedding_token_ids(tokens, vocabulary_size)?
            }
        };
        if tokens.is_empty() {
            return Err(Error::Embedding(
                "each embedding input must contain at least one token".to_string(),
            ));
        }
        if tokens.len() > max_input_tokens || tokens.len() > context_size {
            return Err(Error::ContextLimit {
                prompt_tokens: tokens.len(),
                max_tokens: 0,
                context_size: max_input_tokens.min(context_size),
            });
        }
        if prompt_tokens.saturating_add(tokens.len()) > 300_000 {
            return Err(Error::Embedding(
                "embedding request exceeds 300000 aggregate tokens".to_string(),
            ));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        prompt_tokens = prompt_tokens.saturating_add(tokens.len());
        tokenized_inputs.push(tokens);
    }
    let mut embeddings = Vec::with_capacity(tokenized_inputs.len());
    for tokens in tokenized_inputs {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        context.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|error| Error::Embedding(error.to_string()))?;
        context
            .decode(&mut batch)
            .map_err(|error| Error::Embedding(error.to_string()))?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let embedding = context
            .embeddings_seq_ith(0)
            .map_err(|error| Error::Embedding(error.to_string()))?;
        if embedding.len() != output_dimensions {
            return Err(Error::Embedding(format!(
                "model returned {} dimensions, expected {output_dimensions}",
                embedding.len()
            )));
        }
        embeddings.push(embedding.to_vec());
    }
    Ok(EmbeddingOutput {
        embeddings,
        prompt_tokens,
    })
}

fn validate_embedding_token_ids(
    tokens: &[i32],
    vocabulary_size: i32,
) -> Result<Vec<LlamaToken>, Error> {
    tokens
        .iter()
        .copied()
        .map(|token| {
            if token < 0 || token >= vocabulary_size {
                Err(Error::Embedding(format!(
                    "token id {token} is outside model vocabulary 0..{vocabulary_size}"
                )))
            } else {
                Ok(LlamaToken(token))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_reasoning_control_cancellation_atomically_prevents_late_apply() {
        let (sender, mut controls) = mpsc::channel(1);
        let handle = ReasoningControlHandle {
            operation: OperationCapability(7),
            controls: sender,
        };
        let attempt = handle.begin_reasoning_end().unwrap();
        let cancellation = attempt.cancellation_handle();
        assert!(cancellation.cancel_pending());
        let command = controls.try_recv().unwrap();
        assert_eq!(command.state.load(Ordering::Acquire), CONTROL_CANCELLED);
        assert!(
            command
                .state
                .compare_exchange(
                    CONTROL_PENDING,
                    CONTROL_APPLYING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
        );
    }

    #[test]
    fn applying_reasoning_control_cannot_be_reported_as_cancelled() {
        let (sender, mut controls) = mpsc::channel(1);
        let handle = ReasoningControlHandle {
            operation: OperationCapability(9),
            controls: sender,
        };
        let attempt = handle.begin_reasoning_end().unwrap();
        let cancellation = attempt.cancellation_handle();
        let command = controls.try_recv().unwrap();
        command
            .state
            .compare_exchange(
                CONTROL_PENDING,
                CONTROL_APPLYING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .unwrap();
        assert!(!cancellation.cancel_pending());
        command.state.store(CONTROL_COMPLETED, Ordering::Release);
        command
            .result
            .send(Ok(ReasoningControlResult::Success))
            .unwrap();
        drop(attempt);
    }

    fn rich_text(role: Role, text: &str) -> RichMessage {
        RichMessage {
            role,
            content: vec![ContentPart::Text {
                text: text.to_string(),
            }],
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn semantic_text_subset_preserves_legacy_message_render_input() {
        let legacy = vec![
            Message {
                role: "developer".to_string(),
                content: "format".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        ];
        let rich = legacy
            .iter()
            .map(|message| rich_text(Role::from(message.role.clone()), &message.content))
            .collect::<Vec<_>>();
        let flattened =
            validate_and_flatten(&rich, &[], &ToolChoice::None, ReasoningOptions::default())
                .unwrap();

        assert_eq!(flattened, legacy);
        assert_eq!(
            normalize_text_messages(&flattened),
            normalize_text_messages(&legacy)
        );
    }

    #[test]
    fn unsupported_semantic_fields_fail_with_typed_errors() {
        let cases = [
            validate_and_flatten(
                &[RichMessage {
                    role: Role::User,
                    content: vec![ContentPart::Media(MediaPlaceholder {
                        media_type: MediaType::Image,
                        id: "image-1".to_string(),
                        mime_type: Some("image/png".to_string()),
                    })],
                    tool_calls: Vec::new(),
                }],
                &[],
                &ToolChoice::None,
                ReasoningOptions::default(),
            ),
            validate_and_flatten(
                &[RichMessage {
                    role: Role::Assistant,
                    content: Vec::new(),
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        arguments: serde_json::json!({}),
                    }],
                }],
                &[],
                &ToolChoice::None,
                ReasoningOptions::default(),
            ),
            validate_and_flatten(
                &[RichMessage {
                    role: Role::Tool,
                    content: vec![ContentPart::ToolResult(ToolResult {
                        tool_call_id: "call-1".to_string(),
                        content: "result".to_string(),
                        is_error: false,
                    })],
                    tool_calls: Vec::new(),
                }],
                &[],
                &ToolChoice::None,
                ReasoningOptions::default(),
            ),
            validate_and_flatten(
                &[rich_text(Role::User, "hello")],
                &[ToolDefinition {
                    name: "lookup".to_string(),
                    description: None,
                    parameters: serde_json::json!({"type": "object"}),
                }],
                &ToolChoice::Auto,
                ReasoningOptions::default(),
            ),
            validate_and_flatten(
                &[rich_text(Role::User, "hello")],
                &[],
                &ToolChoice::Required,
                ReasoningOptions::default(),
            ),
            validate_and_flatten(
                &[rich_text(Role::User, "hello")],
                &[],
                &ToolChoice::None,
                ReasoningOptions {
                    enabled: Some(true),
                    effort: Some(ReasoningEffort::High),
                },
            ),
        ];

        for error in cases.map(Result::unwrap_err) {
            assert!(matches!(error, Error::Unsupported { .. }));
        }
    }

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
    fn config_accepts_parallel_generation_and_rejects_invalid_batch_or_embedding_parallelism() {
        let generation = RuntimeConfig {
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
            prompt_cache: PromptCacheConfig::default(),
            mode: RuntimeMode::Generation,
            vision: None,
        };
        assert!(validate_config(&generation).is_ok());

        let mut undersized_batch = generation.clone();
        undersized_batch.batch_size = 1;
        assert!(matches!(
            validate_config(&undersized_batch),
            Err(Error::InvalidConfig(message)) if message.contains("at least parallel_sequences")
        ));

        let mut embedding = generation;
        embedding.mode = RuntimeMode::Embeddings {
            pooling: EmbeddingPooling::Last,
            max_input_tokens: 32,
        };
        assert!(matches!(
            validate_config(&embedding),
            Err(Error::InvalidConfig(message)) if message.contains("parallel_sequences == 1")
        ));

        let mut embedding_vision = embedding.clone();
        embedding_vision.vision = Some(LlmVisionConfig {
            mmproj: PathBuf::from("missing-mmproj.gguf"),
            limits: VisionLimits::default(),
        });
        assert!(matches!(
            validate_config(&embedding_vision),
            Err(Error::InvalidConfig(message)) if message.contains("embedding mode does not support vision")
        ));

        embedding.parallel_sequences = 1;
        embedding.prompt_cache.enabled = true;
        assert!(matches!(
            validate_config(&embedding),
            Err(Error::InvalidConfig(message)) if message.contains("does not support prompt_cache")
        ));

        let mut invalid_cache = embedding;
        invalid_cache.mode = RuntimeMode::Generation;
        invalid_cache.prompt_cache = PromptCacheConfig::default();
        invalid_cache.prompt_cache.max_entries = 65;
        assert!(matches!(
            validate_config(&invalid_cache),
            Err(Error::InvalidConfig(message)) if message.contains("1..=64")
        ));

        for capacity in [0, MAX_EVENT_CAPACITY + 1, usize::MAX] {
            let mut invalid_events = invalid_cache.clone();
            invalid_events.prompt_cache = PromptCacheConfig::default();
            invalid_events.event_queue_capacity = capacity;
            assert!(matches!(
                validate_config(&invalid_events),
                Err(Error::InvalidConfig(message)) if message.contains("event capacity")
            ));
        }
    }

    #[test]
    fn media_text_tokens_are_selected_as_sampler_prompt_history() {
        let prepared = PreparedGeneration {
            tokens: Vec::new(),
            media_prompt_tokens: Some(vec![LlamaToken(7), LlamaToken(11)]),
            media: None,
            semantic_parser: None,
            preserved_tokens: Vec::new(),
            stops: Vec::new(),
            parse_semantic: false,
            reasoning_active: false,
            reasoning_control: None,
        };
        assert_eq!(
            sampler_prompt_tokens(&prepared),
            [LlamaToken(7), LlamaToken(11)]
        );
        assert!(
            prepared.tokens.is_empty(),
            "media scheduler prefill must stay empty"
        );
    }

    #[test]
    fn configured_image_aggregates_are_enforced_incrementally() {
        let config = LlmVisionConfig {
            mmproj: PathBuf::from("unused.gguf"),
            limits: VisionLimits {
                max_images: 2,
                max_bytes_per_image: 4,
                max_total_bytes: 5,
                max_side: 8,
                max_pixels_per_image: 16,
                max_total_pixels: 20,
            },
        };
        let first = ImageInput {
            bytes: vec![0; 3],
            format: ImageFormat::Png,
            width: 4,
            height: 4,
        };
        let bytes = ImageInput {
            bytes: vec![0; 3],
            ..first.clone()
        };
        assert!(validate_images(&[&first, &bytes], &config).is_err());

        let pixels = ImageInput {
            bytes: vec![0],
            width: 2,
            height: 3,
            ..first.clone()
        };
        assert!(validate_images(&[&first, &pixels], &config).is_err());
    }

    #[test]
    fn media_requests_bypass_prefix_cache_and_force_an_ordinary_fairness_tick() {
        let request = AdvancedRequest {
            input: Input::Semantic(Box::new(SemanticInput {
                messages: vec![RichMessage {
                    role: Role::User,
                    content: vec![ContentPart::Image(ImageInput {
                        bytes: vec![0x89, b'P', b'N', b'G'],
                        format: ImageFormat::Png,
                        width: 1,
                        height: 1,
                    })],
                    tool_calls: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                parallel_tool_calls: false,
                reasoning: ReasoningOptions::default(),
            })),
            options: test_request().options,
            output: OutputConstraint::Text,
            logprobs: None,
            logit_bias: Vec::new(),
            sampling: SamplingExtensions::default(),
            choices: 1,
            reasoning_control_id: None,
        };
        assert!(request_has_images(&request));
        assert!(!request_allows_prefix_cache(&request));
        assert!(defer_media_prefill(true, true, true));
        assert!(!defer_media_prefill(true, false, true));
        assert!(!defer_media_prefill(true, true, false));
        assert!(!defer_media_prefill(false, true, true));
    }

    #[test]
    fn choice_count_is_bounded_by_native_parallel_sequences() {
        assert!(validate_choice_count(1, 3).is_ok());
        assert!(validate_choice_count(3, 3).is_ok());
        assert!(validate_choice_count(0, 3).is_err());
        assert!(validate_choice_count(4, 3).is_err());
    }

    fn test_terminal(reason: FinishReason) -> Event {
        Event::Finished {
            reason,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                reasoning_tokens: 0,
                timings: Timings::default(),
            },
        }
    }

    #[test]
    fn output_backpressure_blocks_only_that_slot_and_terminal_cleanup_allows_reuse() {
        let (slow_events, mut slow_receiver) = mpsc::channel(1);
        slow_events
            .try_send(Event::Content("held".to_string()))
            .unwrap();
        let (fast_events, mut fast_receiver) = mpsc::channel(1);
        let (slow_terminal, mut slow_terminal_rx) = tokio::sync::oneshot::channel();
        let (fast_terminal, mut fast_terminal_rx) = tokio::sync::oneshot::channel();
        let (slow_ack, mut slow_ack_rx) = tokio::sync::oneshot::channel();
        let (fast_ack, mut fast_ack_rx) = tokio::sync::oneshot::channel();
        let mut slots = vec![
            Slot {
                id: crate::slot::SlotId::new(0).unwrap(),
                state: SlotState::Draining(DrainingSlot {
                    lifecycle: Lifecycle {
                        operation: None,
                        cancelled: Arc::new(AtomicBool::new(false)),
                        events: Some(slow_events),
                        readiness: None,
                        acknowledged: Some(slow_ack),
                        terminal: Some(slow_terminal),
                    },
                    pending_tokens: std::collections::VecDeque::new(),
                    pending_content: Some("blocked".to_string()),
                    pending_semantic: std::collections::VecDeque::new(),
                    outcome: Ok(test_terminal(FinishReason::Stop)),
                }),
            },
            Slot {
                id: crate::slot::SlotId::new(1).unwrap(),
                state: SlotState::Draining(DrainingSlot {
                    lifecycle: Lifecycle {
                        operation: None,
                        cancelled: Arc::new(AtomicBool::new(false)),
                        events: Some(fast_events),
                        readiness: None,
                        acknowledged: Some(fast_ack),
                        terminal: Some(fast_terminal),
                    },
                    pending_tokens: std::collections::VecDeque::new(),
                    pending_content: None,
                    pending_semantic: std::collections::VecDeque::new(),
                    outcome: Ok(test_terminal(FinishReason::Length)),
                }),
            },
        ];
        let mut cleared = Vec::new();
        finalize_draining_slots_with(&mut slots, |id| {
            cleared.push(id.sequence().unwrap());
            Ok(())
        })
        .unwrap();

        assert!(matches!(slots[0].state, SlotState::Draining(_)));
        assert!(slots[1].is_vacant());
        assert_eq!(cleared, [1]);
        assert!(matches!(
            fast_receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
        assert!(matches!(
            fast_terminal_rx.try_recv(),
            Ok(Ok(Event::Finished { .. }))
        ));
        assert!(matches!(fast_ack_rx.try_recv(), Ok(Ok(()))));
        assert!(slow_terminal_rx.try_recv().is_err());
        assert!(slow_ack_rx.try_recv().is_err());

        if let SlotState::Draining(draining) = &slots[0].state {
            draining.lifecycle.cancelled.store(true, Ordering::Release);
        }
        finalize_draining_slots_with(&mut slots, |id| {
            cleared.push(id.sequence().unwrap());
            Ok(())
        })
        .unwrap();
        assert!(slots[0].is_vacant());
        assert_eq!(cleared, [1, 0]);
        assert_eq!(
            slow_receiver.try_recv().unwrap(),
            Event::Content("held".to_string())
        );
        assert!(matches!(
            slow_terminal_rx.try_recv(),
            Ok(Err(Error::Cancelled))
        ));
        assert!(matches!(slow_ack_rx.try_recv(), Ok(Err(Error::Cancelled))));
    }

    #[test]
    fn shared_decode_error_and_panic_fan_out_to_every_slot() {
        for error in [
            Error::Generation("shared decode failed".to_string()),
            Error::WorkerPanic("shared decode panicked".to_string()),
        ] {
            let mut terminals = Vec::new();
            let mut acknowledgements = Vec::new();
            let mut slots = Vec::new();
            for index in 0..2 {
                let (events, _receiver) = mpsc::channel(1);
                let (readiness, _readiness_rx) = tokio::sync::oneshot::channel();
                let (terminal, terminal_rx) = tokio::sync::oneshot::channel();
                let (acknowledged, acknowledged_rx) = tokio::sync::oneshot::channel();
                let (_decision, decision_rx) = tokio::sync::oneshot::channel();
                terminals.push(terminal_rx);
                acknowledgements.push(acknowledged_rx);
                slots.push(Slot {
                    id: crate::slot::SlotId::new(index).unwrap(),
                    state: SlotState::Reserved(ReservedSlot {
                        lifecycle: Lifecycle {
                            operation: None,
                            cancelled: Arc::new(AtomicBool::new(false)),
                            events: Some(events),
                            readiness: Some(readiness),
                            acknowledged: Some(acknowledged),
                            terminal: Some(terminal),
                        },
                        decision: Some(decision_rx),
                        committed: None,
                    }),
                });
            }
            fail_all_slots(&mut slots, &error);
            assert!(slots.iter().all(Slot::is_vacant));
            for mut terminal in terminals {
                assert!(matches!(terminal.try_recv(), Ok(Err(_))));
            }
            for mut acknowledgement in acknowledgements {
                assert!(matches!(acknowledgement.try_recv(), Ok(Err(_))));
            }
        }
    }

    #[test]
    fn decode_instrumentation_counts_multi_slot_and_prefill_overlap() {
        reset_scheduler_instrumentation();
        let plan = crate::slot::BatchPlan {
            entries: vec![
                crate::slot::BatchEntry {
                    slot: 0,
                    token: LlamaToken(1),
                    position: 1,
                    logits: true,
                    kind: crate::slot::BatchEntryKind::Decode,
                },
                crate::slot::BatchEntry {
                    slot: 1,
                    token: LlamaToken(2),
                    position: 0,
                    logits: false,
                    kind: crate::slot::BatchEntryKind::Prefill,
                },
            ],
            next_prefill_slot: 0,
        };
        record_decode(&plan);
        assert_eq!(
            scheduler_instrumentation(),
            SchedulerInstrumentation {
                decode_calls: 1,
                multi_slot_decode_calls: 1,
                prefill_decode_overlap_calls: 1,
                max_busy_slots_per_decode: 2,
            }
        );
    }

    #[test]
    fn embedding_token_ids_must_be_inside_the_model_vocabulary() {
        assert_eq!(
            validate_embedding_token_ids(&[0, 2, 9], 10).unwrap(),
            vec![LlamaToken(0), LlamaToken(2), LlamaToken(9)]
        );
        for tokens in [vec![-1], vec![10]] {
            assert!(matches!(
                validate_embedding_token_ids(&tokens, 10),
                Err(Error::Embedding(_))
            ));
        }
    }

    #[test]
    fn wall_timings_map_to_public_fields_and_finite_rates() {
        let timings = crate::slot::timings_from_wall(
            3,
            10,
            std::time::Duration::from_millis(20),
            4,
            std::time::Duration::from_millis(8),
        );
        assert_eq!(timings.cache_n, 3);
        assert_eq!(timings.prompt_n, 10);
        assert!((timings.prompt_ms - 20.0).abs() < f64::EPSILON);
        assert!((timings.prompt_per_token_ms - 2.0).abs() < f64::EPSILON);
        assert!((timings.prompt_per_second - 500.0).abs() < f64::EPSILON);
        assert_eq!(timings.predicted_n, 4);
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
    fn reasoning_control_reports_unavailable_when_worker_channel_is_closed() {
        let (controls, receiver) = mpsc::channel(1);
        drop(receiver);
        let handle = ReasoningControlHandle {
            operation: OperationCapability(1),
            controls,
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        assert!(matches!(
            runtime.block_on(handle.reasoning_end()),
            Err(Error::WorkerUnavailable)
        ));
    }

    #[test]
    fn reasoning_capabilities_are_unique_for_concurrent_and_reused_external_ids() {
        let (engine, _) = scripted_engine(Vec::new(), 1);
        let external_id = "reused-request-id".to_string();
        let concurrent_first = engine
            .inner
            .reasoning_operation(Some(&external_id))
            .unwrap()
            .unwrap();
        let concurrent_second = engine
            .inner
            .reasoning_operation(Some(&external_id))
            .unwrap()
            .unwrap();
        assert_ne!(concurrent_first, concurrent_second);
        assert!(controls_operation(Some(concurrent_first), concurrent_first));
        assert!(!controls_operation(
            Some(concurrent_second),
            concurrent_first
        ));

        let after_finished = engine
            .inner
            .reasoning_operation(Some(&external_id))
            .unwrap()
            .unwrap();
        assert_ne!(after_finished, concurrent_first);
        assert_ne!(after_finished, concurrent_second);
        assert!(!controls_operation(Some(after_finished), concurrent_first));
        assert!(!controls_operation(None, after_finished));
        engine.shutdown();
    }

    #[test]
    fn invalid_event_capacities_return_typed_errors_without_panicking() {
        let (engine, _) = scripted_engine(Vec::new(), 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for capacity in [0, usize::MAX] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.block_on(engine.reserve(test_request(), capacity))
            }));
            assert!(matches!(result, Ok(Err(Error::InvalidConfig(_)))));
        }
        engine.shutdown();
    }

    #[test]
    fn multi_choice_reasoning_control_is_typed_invalid_config() {
        let (engine, _) = scripted_engine(Vec::new(), 1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let mut request = AdvancedRequest::from(test_request());
        request.choices = 2;
        request.reasoning_control_id = Some("external".to_string());
        let result = runtime.block_on(engine.reserve_choices(request, 1));
        assert!(matches!(
            result,
            Err(Error::InvalidConfig(message)) if message.contains("reasoning control requires choices == 1")
        ));
        engine.shutdown();
    }

    #[test]
    fn shutdown_cancels_registered_generations_before_join() {
        let (commands, _receiver) = mpsc::channel(1);
        let (controls, _control_receiver) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let inner = EngineInner {
            commands,
            controls,
            join: Mutex::new(None),
            active: Mutex::new(vec![Arc::downgrade(&cancelled)]),
            health: Arc::new(AtomicU8::new(0)),
            parallel_sequences: 1,
            next_operation: AtomicU64::new(1),
            group_admission: tokio::sync::Mutex::new(()),
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
    #[allow(
        clippy::too_many_lines,
        reason = "keeps the three constituent worker protocols visible in one lifecycle test"
    )]
    fn partial_choice_commit_cancels_and_acknowledges_committed_and_reserved_calls() {
        fn reservation(request: AdvancedRequest) -> (Reservation, ReservationWorker) {
            let cancelled = Arc::new(AtomicBool::new(false));
            let (events, event_receiver) = mpsc::channel(1);
            let (readiness, worker_readiness) = tokio::sync::oneshot::channel();
            let (acknowledged, acknowledgement) = tokio::sync::oneshot::channel();
            let (terminal, worker_terminal) = tokio::sync::oneshot::channel();
            let (decision, worker_decision) = tokio::sync::oneshot::channel();
            (
                Reservation {
                    request: Some(request),
                    events: Some(event_receiver),
                    cancelled: Arc::clone(&cancelled),
                    acknowledged: Some(acknowledgement),
                    readiness: Some(worker_readiness),
                    terminal: Some(worker_terminal),
                    decision: Some(decision),
                    transferred: false,
                    control: None,
                },
                ReservationWorker {
                    cancelled,
                    events,
                    readiness,
                    acknowledged,
                    terminal,
                    decision: worker_decision,
                },
            )
        }

        struct ReservationWorker {
            cancelled: Arc<AtomicBool>,
            events: mpsc::Sender<Event>,
            readiness: tokio::sync::oneshot::Sender<Result<(), Error>>,
            acknowledged: tokio::sync::oneshot::Sender<Result<(), Error>>,
            terminal: tokio::sync::oneshot::Sender<Result<Event, Error>>,
            decision: tokio::sync::oneshot::Receiver<ReservationDecision>,
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = AdvancedRequest::from(test_request());
            let (first, first_worker) = reservation(request.clone());
            let (second, second_worker) = reservation(request.clone());
            let (third, third_worker) = reservation(request);
            let first_acked = Arc::new(AtomicBool::new(false));
            let second_acked = Arc::new(AtomicBool::new(false));
            let third_acked = Arc::new(AtomicBool::new(false));

            let first_task = {
                let acked = Arc::clone(&first_acked);
                tokio::spawn(async move {
                    assert!(matches!(
                        first_worker.decision.await,
                        Ok(ReservationDecision::Commit(_))
                    ));
                    first_worker.readiness.send(Ok(())).unwrap();
                    while !first_worker.cancelled.load(Ordering::Acquire) {
                        tokio::task::yield_now().await;
                    }
                    drop(first_worker.events);
                    let _ = first_worker.terminal.send(Err(Error::Cancelled));
                    acked.store(true, Ordering::Release);
                    let _ = first_worker.acknowledged.send(Err(Error::Cancelled));
                })
            };
            let second_task = {
                let acked = Arc::clone(&second_acked);
                tokio::spawn(async move {
                    assert!(matches!(
                        second_worker.decision.await,
                        Ok(ReservationDecision::Commit(_))
                    ));
                    let failure = Error::Generation("second choice preparation failed".to_string());
                    let _ = second_worker.readiness.send(Err(failure.clone()));
                    drop(second_worker.events);
                    let _ = second_worker.terminal.send(Err(failure.clone()));
                    acked.store(true, Ordering::Release);
                    let _ = second_worker.acknowledged.send(Err(failure));
                })
            };
            let third_task = {
                let acked = Arc::clone(&third_acked);
                tokio::spawn(async move {
                    assert!(matches!(
                        third_worker.decision.await,
                        Ok(ReservationDecision::Abort)
                    ));
                    assert!(third_worker.cancelled.load(Ordering::Acquire));
                    let _ = third_worker.readiness.send(Err(Error::Cancelled));
                    drop(third_worker.events);
                    let _ = third_worker.terminal.send(Err(Error::Cancelled));
                    acked.store(true, Ordering::Release);
                    let _ = third_worker.acknowledged.send(Err(Error::Cancelled));
                })
            };

            let mut choices = ChoiceReservation {
                reservations: vec![first, second, third],
                event_capacity: 1,
            };
            assert!(matches!(
                choices.commit().await,
                Err(Error::Generation(message)) if message == "second choice preparation failed"
            ));
            assert!(first_acked.load(Ordering::Acquire));
            assert!(second_acked.load(Ordering::Acquire));
            assert!(third_acked.load(Ordering::Acquire));
            first_task.await.unwrap();
            second_task.await.unwrap();
            third_task.await.unwrap();
        });
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
            input: Input::Messages(vec![Message {
                role: "user".to_string(),
                content: "hello".to_string(),
            }]),
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
    fn choice_usage_counts_logical_prompt_once_and_physical_prefill_per_choice() {
        let mut aggregate = ChoiceAggregate {
            remaining: 2,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                reasoning_tokens: 0,
                timings: Timings::default(),
            },
            prompt_tokens: None,
            failure: None,
        };
        for completion_tokens in [1, 1] {
            merge_usage(
                &mut aggregate,
                Usage {
                    prompt_tokens: 10,
                    completion_tokens,
                    reasoning_tokens: 0,
                    timings: Timings {
                        prompt_n: 10,
                        predicted_n: completion_tokens,
                        ..Timings::default()
                    },
                },
            )
            .unwrap();
        }
        assert_eq!(aggregate.usage.prompt_tokens, 10);
        assert_eq!(aggregate.usage.completion_tokens, 2);
        assert_eq!(
            aggregate.usage.prompt_tokens + aggregate.usage.completion_tokens,
            12
        );
        assert_eq!(aggregate.usage.timings.prompt_n, 20);

        assert!(
            merge_usage(
                &mut aggregate,
                Usage {
                    prompt_tokens: 11,
                    completion_tokens: 0,
                    reasoning_tokens: 0,
                    timings: Timings::default(),
                },
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_two_request_parallel_canary() {
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
                parallel_sequences: 2,
                request_queue_capacity: 2,
                event_queue_capacity: 2,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: false,
                prompt_cache: PromptCacheConfig {
                    enabled: true,
                    max_entries: 4,
                    max_bytes: 268_435_456,
                    min_prefix_tokens: 8,
                },
                mode: RuntimeMode::Generation,
                vision: None,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = Request {
                input: Input::Messages(vec![Message {
                    role: "user".to_string(),
                    content: "Reply with OK.".to_string(),
                }]),
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
            };
            let mut first = engine.generate(request.clone(), 2).await.unwrap();
            let mut second = engine.generate(request, 2).await.unwrap();
            let mut first_open = true;
            let mut second_open = true;
            while first_open || second_open {
                if first_open {
                    first_open = first.events.recv().await.is_some();
                }
                if second_open {
                    second_open = second.events.recv().await.is_some();
                }
            }
            assert!(matches!(
                first.recv_terminal().await,
                Ok(Event::Finished { .. })
            ));
            assert!(matches!(
                second.recv_terminal().await,
                Ok(Event::Finished { .. })
            ));
        });
        engine.shutdown();
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_invalid_bias_fails_one_slot_without_stopping_worker() {
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
                parallel_sequences: 2,
                request_queue_capacity: 2,
                event_queue_capacity: 4,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: false,
                prompt_cache: PromptCacheConfig::default(),
                mode: RuntimeMode::Generation,
                vision: None,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut bad_request = AdvancedRequest::from(test_request());
            bad_request.logit_bias.push(LogitBias {
                token_id: i32::MAX,
                bias: 1.0,
            });
            let mut bad = engine.reserve_advanced(bad_request, 4).await.unwrap();
            let mut good = engine.reserve(test_request(), 4).await.unwrap();
            let bad_task = tokio::spawn(async move { bad.commit().await });
            let good_result = good.commit().await;
            let bad_result = bad_task.await.unwrap();
            assert!(matches!(
                bad_result,
                Err(Error::InvalidRequest {
                    field: "logit_bias",
                    ..
                })
            ));
            let mut good = good_result.unwrap();
            while good.events.recv().await.is_some() {}
            assert!(matches!(
                good.recv_terminal().await,
                Ok(Event::Finished { .. })
            ));
            assert!(good.wait_for_ack().await.is_ok());
            assert!(engine.is_healthy());
        });
        engine.shutdown();
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_semantic_reasoning_bridge_canary() {
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
                event_queue_capacity: 8,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: true,
                prompt_cache: PromptCacheConfig::default(),
                mode: RuntimeMode::Generation,
                vision: None,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = AdvancedSemanticRequest {
                messages: vec![RichMessage {
                    role: Role::User,
                    content: vec![ContentPart::Text {
                        text: "Think briefly, then answer OK.".into(),
                    }],
                    tool_calls: Vec::new(),
                }],
                options: test_request().options,
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                parallel_tool_calls: false,
                reasoning: ReasoningOptions {
                    enabled: Some(true),
                    effort: Some(ReasoningEffort::Low),
                },
                output: OutputConstraint::Text,
                logprobs: None,
                logit_bias: Vec::new(),
                sampling: SamplingExtensions::default(),
                choices: 1,
                reasoning_control_id: None,
            };
            let mut generation = engine
                .reserve_choice_semantic(request, 8)
                .await
                .unwrap()
                .commit()
                .await
                .unwrap();
            let mut terminal = false;
            let mut semantic = false;
            while let Some(event) = generation.events.recv().await {
                semantic |= matches!(
                    event,
                    ChoiceEvent::SemanticDelta {
                        delta: SemanticDelta::Reasoning(_) | SemanticDelta::Text(_),
                        ..
                    }
                );
                terminal |= matches!(event, ChoiceEvent::Finished { .. });
            }
            assert!(terminal);
            assert!(semantic);
        });
        engine.shutdown();
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_constrained_indexed_choices_logprobs_canary() {
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
                parallel_sequences: 2,
                request_queue_capacity: 2,
                event_queue_capacity: 8,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: false,
                prompt_cache: PromptCacheConfig::default(),
                mode: RuntimeMode::Generation,
                vision: None,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = AdvancedRequest {
                input: Input::Prompt("Return a JSON object with integer field n:".to_string()),
                options: GenerationOptions {
                    max_tokens: 32,
                    temperature: 0.8,
                    top_p: 0.95,
                    top_k: 20,
                    min_p: 0.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repeat_penalty: 1.0,
                    seed: 17,
                    stop: Vec::new(),
                },
                output: OutputConstraint::JsonSchema(serde_json::json!({
                    "type":"object",
                    "properties":{"n":{"type":"integer"}},
                    "required":["n"],
                    "additionalProperties":false
                })),
                logprobs: Some(LogprobsOptions { top_logprobs: 3 }),
                logit_bias: vec![LogitBias {
                    token_id: 0,
                    bias: 0.0,
                }],
                sampling: SamplingExtensions {
                    typical_p: Some(0.95),
                    top_n_sigma: Some(3.0),
                },
                choices: 2,
                reasoning_control_id: None,
            };
            let mut generation = engine
                .reserve_choices(request, 8)
                .await
                .unwrap()
                .commit()
                .await
                .unwrap();
            let mut finished = std::collections::BTreeSet::new();
            let mut saw_logprobs = false;
            let mut parent = false;
            while let Some(event) = generation.events.recv().await {
                match event {
                    ChoiceEvent::Delta {
                        index,
                        logprobs: Some(logprobs),
                        ..
                    } => {
                        assert!(index < 2);
                        assert!(logprobs.chosen.logprob.is_finite());
                        assert!(logprobs.top.len() <= 3);
                        saw_logprobs = true;
                    }
                    ChoiceEvent::Finished { index, .. } => {
                        finished.insert(index);
                    }
                    ChoiceEvent::FinishedAll { usage } => {
                        assert!(usage.completion_tokens > 0);
                        parent = true;
                    }
                    ChoiceEvent::Delta { .. } | ChoiceEvent::SemanticDelta { .. } => {}
                    ChoiceEvent::Failed { message, .. } => {
                        panic!("generation failed: {message}")
                    }
                }
            }
            assert_eq!(finished, [0, 1].into_iter().collect());
            assert!(saw_logprobs && parent);
        });
        engine.shutdown();
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_GGUF pointing to a real text GGUF"]
    fn real_gguf_prompt_cache_warm_cold_canary() {
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
                request_queue_capacity: 2,
                event_queue_capacity: 4,
                chat_template: None,
                template_engine: TemplateEngine::Jinja,
                enable_thinking: false,
                prompt_cache: PromptCacheConfig {
                    enabled: true,
                    max_entries: 4,
                    max_bytes: 268_435_456,
                    min_prefix_tokens: 1,
                },
                mode: RuntimeMode::Generation,
                vision: None,
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = Request {
                input: Input::Prompt(
                    "A repeated deterministic prompt with enough tokens. Reply OK:".to_string(),
                ),
                options: GenerationOptions {
                    max_tokens: 4,
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
            };
            let mut cold = engine.generate(request.clone(), 4).await.unwrap();
            while cold.events.recv().await.is_some() {}
            let Event::Finished {
                usage: cold_usage, ..
            } = cold.recv_terminal().await.unwrap()
            else {
                panic!("expected cold terminal usage");
            };

            let mut warm = engine.generate(request, 4).await.unwrap();
            while warm.events.recv().await.is_some() {}
            let Event::Finished {
                usage: warm_usage, ..
            } = warm.recv_terminal().await.unwrap()
            else {
                panic!("expected warm terminal usage");
            };

            assert_eq!(cold_usage.timings.cache_n, 0);
            assert!(warm_usage.timings.cache_n > 0);
            assert_eq!(cold_usage.prompt_tokens, warm_usage.prompt_tokens);
            assert_eq!(
                warm_usage.timings.prompt_n,
                warm_usage.prompt_tokens - warm_usage.timings.cache_n
            );
        });
        engine.shutdown();
    }

    #[test]
    #[ignore = "requires ORCHION_TEST_VISION_GGUF, ORCHION_TEST_MMPROJ_GGUF, and ORCHION_TEST_IMAGE"]
    fn real_gguf_multimodal_canary() {
        let model = PathBuf::from(std::env::var("ORCHION_TEST_VISION_GGUF").unwrap());
        let mmproj = PathBuf::from(std::env::var("ORCHION_TEST_MMPROJ_GGUF").unwrap());
        let image_path = PathBuf::from(std::env::var("ORCHION_TEST_IMAGE").unwrap());
        let bytes = std::fs::read(&image_path).unwrap();
        let format = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            ImageFormat::Png
        } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
            ImageFormat::Jpeg
        } else {
            panic!("ORCHION_TEST_IMAGE must be PNG or JPEG");
        };
        let image_format = match format {
            ImageFormat::Png => image::ImageFormat::Png,
            ImageFormat::Jpeg => image::ImageFormat::Jpeg,
        };
        let (width, height) =
            image::ImageReader::with_format(std::io::Cursor::new(&bytes), image_format)
                .into_dimensions()
                .unwrap();
        let engine = Engine::load(
            model,
            RuntimeConfig {
                context_size: NonZeroU32::new(4096),
                batch_size: 512,
                micro_batch_size: 512,
                threads: 2,
                gpu_layers: 0,
                parallel_sequences: 1,
                request_queue_capacity: 2,
                event_queue_capacity: 8,
                chat_template: None,
                template_engine: TemplateEngine::LlamaCpp,
                enable_thinking: false,
                prompt_cache: PromptCacheConfig::default(),
                mode: RuntimeMode::Generation,
                vision: Some(LlmVisionConfig {
                    mmproj,
                    limits: VisionLimits::default(),
                }),
            },
        )
        .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut generation = engine
                .generate_semantic(
                    SemanticRequest {
                        messages: vec![RichMessage {
                            role: Role::User,
                            content: vec![
                                ContentPart::Text {
                                    text: "Describe: ".into(),
                                },
                                ContentPart::Image(ImageInput {
                                    bytes,
                                    format,
                                    width,
                                    height,
                                }),
                            ],
                            tool_calls: Vec::new(),
                        }],
                        options: GenerationOptions {
                            max_tokens: 4,
                            ..test_request().options
                        },
                        tools: Vec::new(),
                        tool_choice: ToolChoice::None,
                        parallel_tool_calls: false,
                        reasoning: ReasoningOptions::default(),
                    },
                    8,
                )
                .await
                .unwrap();
            while generation.events.recv().await.is_some() {}
            assert!(matches!(
                generation.recv_terminal().await,
                Ok(Event::Finished { .. })
            ));
        });
        engine.shutdown();
    }
}
