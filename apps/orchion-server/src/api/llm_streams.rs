use crate::api::activity::{ActivityContext, ActivityError, ActivityOutcome};
use crate::api::chat_controls::ChatControls;
use crate::api::http::ServerShutdown;
use crate::api::sse;
use crate::application::llm::{ChoiceCancellationCause, ManagedChoiceCancellation};
use crate::application::metrics::{Metrics, StreamObservation};
use crate::application::{MIN_STREAMING_ERROR_FRAME_BYTES, StreamingPolicy};
use axum::body::{Body, Bytes};
use base64::Engine as _;
use http_body_util::BodyExt as _;
pub(crate) use orchion_protocol::{
    LlmStreamMetadata as StreamMetadata, LlmStreamProtocol as StreamProtocol,
    LlmStreamStatus as StreamStatus,
};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;

const PRODUCER_PENDING: u8 = 0;
const PRODUCER_COMPLETED: u8 = 1;
const PRODUCER_FAILED: u8 = 2;
const MAX_STREAM_ERROR_CODE_BYTES: usize = 32;
const MAX_STREAM_ERROR_MESSAGE_BYTES: usize = 128;

#[derive(Clone, Default)]
pub(crate) struct StreamTerminalSignal(Arc<AtomicU8>);

impl StreamTerminalSignal {
    pub(crate) fn is_pending(&self) -> bool {
        self.0.load(Ordering::Acquire) == PRODUCER_PENDING
    }

    pub(crate) fn complete(&self) {
        let _ = self.0.compare_exchange(
            PRODUCER_PENDING,
            PRODUCER_COMPLETED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(crate) fn fail(&self) {
        self.0.store(PRODUCER_FAILED, Ordering::Release);
    }

    fn outcome(&self) -> Option<StreamStatus> {
        match self.0.load(Ordering::Acquire) {
            PRODUCER_COMPLETED => Some(StreamStatus::Completed),
            PRODUCER_FAILED => Some(StreamStatus::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PrincipalId {
    Anonymous,
    Authenticated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessError {
    NotFound,
    ReplayLost,
    FollowersExhausted,
    ShuttingDown,
    InvalidCursor,
}

#[derive(Debug)]
pub(crate) enum StartError {
    Capacity,
    Entropy,
    ShuttingDown,
}

#[derive(Clone)]
pub(crate) struct LlmStreams {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    policy: StreamingPolicy,
    shutdown: ServerShutdown,
    metrics: Metrics,
    controls: ChatControls,
}

#[derive(Default)]
struct State {
    sessions: HashMap<String, Session>,
    total_bytes: usize,
}

struct Session {
    principal: PrincipalId,
    protocol: StreamProtocol,
    frames: VecDeque<StoredFrame>,
    bytes: usize,
    next_id: u64,
    followers: usize,
    status: StreamStatus,
    owner_active: bool,
    terminal_at: Option<Instant>,
    deleted: bool,
    revision: watch::Sender<u64>,
    cancellation: ManagedChoiceCancellation,
    completion_id: Option<String>,
}

struct StoredFrame {
    id: u64,
    bytes: Bytes,
}

impl LlmStreams {
    pub(crate) fn new(policy: StreamingPolicy, shutdown: ServerShutdown, metrics: Metrics) -> Self {
        let controls = ChatControls::new(metrics.clone(), shutdown.clone());
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                policy,
                shutdown,
                metrics,
                controls,
            }),
        }
    }

    pub(crate) fn controls(&self) -> ChatControls {
        self.inner.controls.clone()
    }

    pub(crate) fn ttl(&self) -> Duration {
        self.inner.policy.ttl
    }

    pub(crate) fn lookup_max(&self) -> usize {
        self.inner.policy.lookup_max
    }

    pub(crate) fn ensure_start_capacity(&self) -> Result<(), StartError> {
        if self.inner.shutdown.is_triggered() {
            return Err(StartError::ShuttingDown);
        }
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        self.purge_locked(&mut state);
        if state
            .sessions
            .values()
            .filter(|session| session.owner_active)
            .count()
            >= self.inner.policy.max_active
        {
            Err(StartError::Capacity)
        } else {
            Ok(())
        }
    }

    #[allow(
        dead_code,
        reason = "retained as the protocol-neutral non-control start seam"
    )]
    pub(crate) fn start(
        &self,
        principal: PrincipalId,
        protocol: StreamProtocol,
        source: Body,
        cancellation: ManagedChoiceCancellation,
        activity: Option<ActivityContext>,
        terminal: StreamTerminalSignal,
    ) -> Result<(String, Body), StartError> {
        self.start_with_completion(
            principal,
            protocol,
            source,
            cancellation,
            activity,
            None,
            terminal,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "keeps protocol ownership, cancellation, activity, and typed terminal explicit"
    )]
    pub(crate) fn start_with_completion(
        &self,
        principal: PrincipalId,
        protocol: StreamProtocol,
        source: Body,
        cancellation: ManagedChoiceCancellation,
        activity: Option<ActivityContext>,
        completion_id: Option<String>,
        terminal: StreamTerminalSignal,
    ) -> Result<(String, Body), StartError> {
        if self.inner.shutdown.is_triggered() {
            return Err(StartError::ShuttingDown);
        }
        let (stream_id, revision) = {
            let mut state = self.inner.state.lock().expect("stream state poisoned");
            self.purge_locked(&mut state);
            let active = state
                .sessions
                .values()
                .filter(|session| session.owner_active)
                .count();
            if active >= self.inner.policy.max_active {
                return Err(StartError::Capacity);
            }
            let stream_id = loop {
                let candidate = random_stream_id()?;
                if !state.sessions.contains_key(&candidate) {
                    break candidate;
                }
            };
            let (revision, follower_revision) = watch::channel(0);
            state.sessions.insert(
                stream_id.clone(),
                Session {
                    principal,
                    protocol,
                    frames: VecDeque::new(),
                    bytes: 0,
                    next_id: 1,
                    followers: 1,
                    status: StreamStatus::Active,
                    owner_active: true,
                    terminal_at: None,
                    deleted: false,
                    revision,
                    cancellation,
                    completion_id,
                },
            );
            self.inner.metrics.resumable_created();
            self.inner.metrics.resumable_attachment();
            (stream_id, follower_revision)
        };
        let body = self.follower_body(stream_id.clone(), principal, 0, true, revision);
        let owner = self.clone();
        let owner_id = stream_id.clone();
        tokio::spawn(async move {
            owner.run_owner(owner_id, source, activity, terminal).await;
        });
        Ok((stream_id, body))
    }

    pub(crate) fn attach(
        &self,
        stream_id: &str,
        principal: PrincipalId,
        after: u64,
        follow: bool,
    ) -> Result<Body, AccessError> {
        if self.inner.shutdown.is_triggered() {
            return Err(AccessError::ShuttingDown);
        }
        let revision = {
            let mut state = self.inner.state.lock().expect("stream state poisoned");
            self.purge_locked(&mut state);
            let session = state
                .sessions
                .get_mut(stream_id)
                .ok_or(AccessError::NotFound)?;
            if session.deleted || session.principal != principal {
                return Err(AccessError::NotFound);
            }
            let first = session
                .frames
                .front()
                .map_or(session.next_id, |frame| frame.id);
            if after.saturating_add(1) < first {
                return Err(AccessError::ReplayLost);
            }
            if after >= session.next_id {
                return Err(AccessError::InvalidCursor);
            }
            if session.followers >= self.inner.policy.max_followers_per_session {
                return Err(AccessError::FollowersExhausted);
            }
            session.followers += 1;
            self.inner.metrics.resumable_attachment();
            session.revision.subscribe()
        };
        Ok(self.follower_body(stream_id.to_string(), principal, after, follow, revision))
    }

    fn follower_body(
        &self,
        stream_id: String,
        principal: PrincipalId,
        after: u64,
        follow: bool,
        revision: watch::Receiver<u64>,
    ) -> Body {
        let follower = Follower {
            streams: self.clone(),
            stream_id,
            principal,
            revision,
        };
        let stream = async_stream::stream! {
            let mut follower_guard = follower;
            let mut cursor = after;
            loop {
                let action = follower_guard.next(cursor, follow);
                match action {
                    FollowerAction::Frame(frame) => {
                        cursor = frame.id;
                        yield Ok::<Bytes, Infallible>(frame.bytes);
                    }
                    FollowerAction::Wait => {
                        let _ = follower_guard.revision.changed().await;
                    }
                    FollowerAction::ReplayLost(protocol) => {
                        yield Ok(stream_error_frame(protocol, cursor.saturating_add(1), "replay_lost", "requested events are no longer retained"));
                        break;
                    }
                    FollowerAction::End => break,
                }
            }
        };
        Body::from_stream(stream)
    }

    pub(crate) fn lookup(&self, principal: PrincipalId, ids: &[String]) -> Vec<StreamMetadata> {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        self.purge_locked(&mut state);
        let now = Instant::now();
        ids.iter()
            .filter_map(|id| state.sessions.get(id).map(|session| (id, session)))
            .filter(|(_, session)| !session.deleted && session.principal == principal)
            .map(|(id, session)| StreamMetadata {
                stream_id: id.clone(),
                protocol: session.protocol,
                status: session.status,
                last_event_id: session.next_id.saturating_sub(1),
                expires_in_seconds: session.terminal_at.map_or(
                    self.inner.policy.ttl.as_secs(),
                    |at| {
                        self.inner
                            .policy
                            .ttl
                            .saturating_sub(now.saturating_duration_since(at))
                            .as_secs()
                    },
                ),
            })
            .collect()
    }

    pub(crate) fn delete(&self, stream_id: &str, principal: PrincipalId) {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        let Some(session) = state.sessions.get_mut(stream_id) else {
            return;
        };
        if session.principal != principal || session.deleted {
            return;
        }
        session.deleted = true;
        if session.status == StreamStatus::Active {
            session.status = StreamStatus::Cancelled;
            session.terminal_at = Some(Instant::now());
            self.inner.metrics.resumable_terminal();
            session
                .cancellation
                .cancel_with(ChoiceCancellationCause::UserDeleted);
        }
        if let Some(completion_id) = session.completion_id.take() {
            self.inner.controls.remove(&completion_id);
        }
        session
            .revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
    }

    async fn run_owner(
        &self,
        stream_id: String,
        mut source: Body,
        activity: Option<ActivityContext>,
        terminal: StreamTerminalSignal,
    ) {
        let mut cancelled = false;
        loop {
            let frame = if cancelled {
                source.frame().await
            } else {
                tokio::select! {
                    () = self.inner.shutdown.cancelled() => {
                        self.cancel_owner(
                            &stream_id,
                            ChoiceCancellationCause::ServerShutdown,
                            "server_shutdown",
                            "server is shutting down",
                        );
                        cancelled = true;
                        continue;
                    }
                    frame = source.frame() => frame,
                }
            };
            let Some(frame) = frame else { break };
            let Ok(frame) = frame else {
                self.fail(
                    &stream_id,
                    "stream_transport_error",
                    "stream transport failed",
                );
                break;
            };
            if let Ok(data) = frame.into_data()
                && !self.append(&stream_id, &data)
            {
                cancelled = true;
            }
        }
        let status = match terminal.outcome() {
            Some(StreamStatus::Completed) => self.finish(&stream_id, StreamStatus::Completed),
            Some(StreamStatus::Failed) => self.finish(&stream_id, StreamStatus::Failed),
            _ if cancelled => self.finish(&stream_id, StreamStatus::Cancelled),
            _ => {
                self.fail(
                    &stream_id,
                    "stream_terminal_missing",
                    "stream ended before a protocol terminal event",
                );
                self.finish(&stream_id, StreamStatus::Failed)
            }
        };
        if let Some(activity) = activity {
            match status {
                Some(StreamStatus::Cancelled) => activity.complete_stream_cancelled(),
                Some(StreamStatus::Failed) => activity.complete_stream_failure(
                    ActivityOutcome::ServerError,
                    ActivityError {
                        code: Some("stream_failed".to_string()),
                        message: Some("stream failed after response headers".to_string()),
                    },
                ),
                Some(StreamStatus::Completed) => activity.complete_http(200, None),
                _ => {}
            }
        }
    }

    fn append(&self, stream_id: &str, data: &Bytes) -> bool {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        let policy = self.inner.policy;
        let (added, removed) = {
            let Some(session) = state.sessions.get_mut(stream_id) else {
                return false;
            };
            if session.deleted || session.status != StreamStatus::Active {
                return false;
            }
            let id = session.next_id;
            let bytes = sse::number_frame(id, data);
            if bytes.len() > policy.max_bytes_per_session || bytes.len() > policy.max_total_bytes {
                session
                    .cancellation
                    .cancel_with(ChoiceCancellationCause::StreamBufferExceeded);
                drop(state);
                self.fail(
                    stream_id,
                    "stream_frame_too_large",
                    "stream frame exceeds retention capacity",
                );
                return false;
            }
            session.next_id += 1;
            let added = bytes.len();
            session.bytes += added;
            session.frames.push_back(StoredFrame { id, bytes });
            let mut removed = 0;
            while session.frames.len() > policy.max_events_per_session
                || session.bytes > policy.max_bytes_per_session
            {
                if let Some(frame) = session.frames.pop_front() {
                    session.bytes -= frame.bytes.len();
                    removed += frame.bytes.len();
                    self.inner.metrics.resumable_truncation();
                }
            }
            (added, removed)
        };
        state.total_bytes = state
            .total_bytes
            .saturating_add(added)
            .saturating_sub(removed);
        while state.total_bytes > policy.max_total_bytes {
            let Some((id, _)) = state
                .sessions
                .iter()
                .filter(|(_, value)| !value.owner_active)
                .min_by_key(|(_, value)| value.terminal_at)
                .map(|(id, value)| (id.clone(), value.terminal_at))
            else {
                let removed = {
                    let session = state
                        .sessions
                        .get_mut(stream_id)
                        .expect("owner session exists");
                    let removed = session
                        .frames
                        .pop_back()
                        .map_or(0, |frame| frame.bytes.len());
                    session.bytes = session.bytes.saturating_sub(removed);
                    session.next_id = session.next_id.saturating_sub(1);
                    session
                        .cancellation
                        .cancel_with(ChoiceCancellationCause::ResourceExhausted);
                    removed
                };
                state.total_bytes = state.total_bytes.saturating_sub(removed);
                drop(state);
                self.fail(
                    stream_id,
                    "stream_capacity_exhausted",
                    "global stream retention capacity exhausted",
                );
                return false;
            };
            remove_session(&mut state, &id);
            self.inner.metrics.resumable_eviction();
        }
        state
            .sessions
            .get(stream_id)
            .expect("owner session exists")
            .revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
        true
    }

    fn fail(&self, stream_id: &str, code: &str, message: &str) {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        let Some((protocol, id)) = state.sessions.get(stream_id).and_then(|session| {
            (session.status == StreamStatus::Active).then_some((session.protocol, session.next_id))
        }) else {
            return;
        };
        let bytes = stream_error_frame(protocol, id, code, message);
        let frame_bytes = bytes.len();
        debug_assert!(frame_bytes <= MIN_STREAMING_ERROR_FRAME_BYTES);
        debug_assert!(frame_bytes <= self.inner.policy.max_bytes_per_session);
        debug_assert!(frame_bytes <= self.inner.policy.max_total_bytes);

        loop {
            let session = state.sessions.get(stream_id).expect("owner session exists");
            let session_over = session.frames.len() >= self.inner.policy.max_events_per_session
                || session.bytes.saturating_add(frame_bytes)
                    > self.inner.policy.max_bytes_per_session;
            let global_over =
                state.total_bytes.saturating_add(frame_bytes) > self.inner.policy.max_total_bytes;
            if !session_over && !global_over {
                break;
            }
            let removal_id = if session_over {
                Some(stream_id.to_string())
            } else {
                state
                    .sessions
                    .iter()
                    .filter(|(_, session)| !session.frames.is_empty())
                    .max_by_key(|(_, session)| session.bytes)
                    .map(|(id, _)| id.clone())
            };
            let Some(removal_id) = removal_id else {
                break;
            };
            let session = state
                .sessions
                .get_mut(&removal_id)
                .expect("selected retention session exists");
            let Some(frame) = session.frames.pop_front() else {
                break;
            };
            session.bytes = session.bytes.saturating_sub(frame.bytes.len());
            state.total_bytes = state.total_bytes.saturating_sub(frame.bytes.len());
            self.inner.metrics.resumable_truncation();
        }

        {
            let session = state
                .sessions
                .get_mut(stream_id)
                .expect("owner session exists");
            session.bytes += frame_bytes;
            session.frames.push_back(StoredFrame { id, bytes });
            session.next_id += 1;
            session.status = StreamStatus::Failed;
            session.terminal_at = Some(Instant::now());
            self.inner.metrics.resumable_terminal();
            session
                .revision
                .send_modify(|revision| *revision = revision.wrapping_add(1));
        }
        state.total_bytes += frame_bytes;
    }

    fn cancel_owner(
        &self,
        stream_id: &str,
        cause: ChoiceCancellationCause,
        code: &str,
        message: &str,
    ) {
        let cancellation = {
            let state = self.inner.state.lock().expect("stream state poisoned");
            state
                .sessions
                .get(stream_id)
                .map(|session| session.cancellation.clone())
        };
        if let Some(cancellation) = cancellation {
            cancellation.cancel_with(cause);
        }
        self.fail(stream_id, code, message);
        if code == "server_shutdown" {
            let mut state = self.inner.state.lock().expect("stream state poisoned");
            if let Some(session) = state.sessions.get_mut(stream_id) {
                session.status = StreamStatus::Cancelled;
            }
        }
    }

    fn finish(&self, stream_id: &str, outcome: StreamStatus) -> Option<StreamStatus> {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        let session = state.sessions.get_mut(stream_id)?;
        session.owner_active = false;
        if session.status == StreamStatus::Active {
            session.status = outcome;
            session.terminal_at = Some(Instant::now());
            self.inner.metrics.resumable_terminal();
        }
        let status = session.status;
        session
            .revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
        self.purge_locked(&mut state);
        Some(status)
    }

    fn purge_locked(&self, state: &mut State) {
        let now = Instant::now();
        let expired = state
            .sessions
            .iter()
            .filter(|(_, session)| {
                (session.deleted && !session.owner_active)
                    || session.terminal_at.is_some_and(|at| {
                        now.saturating_duration_since(at) >= self.inner.policy.ttl
                    })
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in expired {
            remove_session(state, &id);
            self.inner.metrics.resumable_eviction();
        }
        while state
            .sessions
            .values()
            .filter(|session| !session.owner_active)
            .count()
            > self.inner.policy.max_retained
        {
            let Some(id) = state
                .sessions
                .iter()
                .filter(|(_, session)| !session.owner_active)
                .min_by_key(|(_, session)| session.terminal_at)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            remove_session(state, &id);
            self.inner.metrics.resumable_eviction();
        }
    }

    pub(crate) fn observation(&self) -> StreamObservation {
        let mut state = self.inner.state.lock().expect("stream state poisoned");
        self.purge_locked(&mut state);
        StreamObservation {
            active: state
                .sessions
                .values()
                .filter(|session| session.owner_active)
                .count(),
            retained: state
                .sessions
                .values()
                .filter(|session| !session.owner_active)
                .count(),
            followers: state
                .sessions
                .values()
                .map(|session| session.followers)
                .sum(),
            buffered_events: state
                .sessions
                .values()
                .map(|session| session.frames.len())
                .sum(),
            buffered_bytes: state.total_bytes,
        }
    }
}

struct Follower {
    streams: LlmStreams,
    stream_id: String,
    principal: PrincipalId,
    revision: watch::Receiver<u64>,
}

enum FollowerAction {
    Frame(StoredFrame),
    Wait,
    ReplayLost(StreamProtocol),
    End,
}

impl Follower {
    fn next(&self, cursor: u64, follow: bool) -> FollowerAction {
        let state = self
            .streams
            .inner
            .state
            .lock()
            .expect("stream state poisoned");
        let Some(session) = state.sessions.get(&self.stream_id) else {
            return FollowerAction::End;
        };
        if session.principal != self.principal || session.deleted {
            return FollowerAction::End;
        }
        let first = session
            .frames
            .front()
            .map_or(session.next_id, |frame| frame.id);
        if cursor.saturating_add(1) < first {
            return FollowerAction::ReplayLost(session.protocol);
        }
        if let Some(frame) = session.frames.iter().find(|frame| frame.id > cursor) {
            return FollowerAction::Frame(StoredFrame {
                id: frame.id,
                bytes: frame.bytes.clone(),
            });
        }
        if !follow || session.status != StreamStatus::Active {
            FollowerAction::End
        } else {
            FollowerAction::Wait
        }
    }
}

impl Drop for Follower {
    fn drop(&mut self) {
        let mut state = self
            .streams
            .inner
            .state
            .lock()
            .expect("stream state poisoned");
        if let Some(session) = state.sessions.get_mut(&self.stream_id) {
            session.followers = session.followers.saturating_sub(1);
        }
    }
}

fn remove_session(state: &mut State, id: &str) {
    if let Some(session) = state.sessions.remove(id) {
        state.total_bytes = state.total_bytes.saturating_sub(session.bytes);
        session
            .revision
            .send_modify(|revision| *revision = revision.wrapping_add(1));
    }
}

fn random_stream_id() -> Result<String, StartError> {
    let mut entropy = [0_u8; 24];
    getrandom::fill(&mut entropy).map_err(|_| StartError::Entropy)?;
    Ok(format!(
        "strm_{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(entropy)
    ))
}

pub(crate) fn valid_stream_id(value: &str) -> bool {
    let Some(encoded) = value.strip_prefix("strm_") else {
        return false;
    };
    encoded.len() == 32
        && encoded
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn stream_error_frame(protocol: StreamProtocol, id: u64, code: &str, message: &str) -> Bytes {
    let code = bounded_error_text(code, MAX_STREAM_ERROR_CODE_BYTES);
    let message = bounded_error_text(message, MAX_STREAM_ERROR_MESSAGE_BYTES);
    let data = match protocol {
        StreamProtocol::Responses => format!(
            "event: error\ndata: {}\n\n",
            json!({"type":"error","sequence_number":id.saturating_sub(1),"code":code,"message":message,"param":null})
        ),
        StreamProtocol::Chat | StreamProtocol::Completions => format!(
            "data: {}\n\n",
            json!({"error":{"message":message,"type":"server_error","param":null,"code":code}})
        ),
    };
    let frame = sse::number_frame(id, data.as_bytes());
    debug_assert!(frame.len() <= MIN_STREAMING_ERROR_FRAME_BYTES);
    frame
}

fn bounded_error_text(value: &str, max_bytes: usize) -> String {
    value
        .chars()
        .take(max_bytes)
        .map(|character| {
            if character.is_ascii() && !character.is_ascii_control() {
                character
            } else {
                '?'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::llm::ManagedChoiceGeneration;
    use std::sync::atomic::AtomicBool;
    use tokio::sync::{Notify, mpsc};

    fn policy() -> StreamingPolicy {
        StreamingPolicy {
            max_active: 2,
            max_retained: 4,
            max_events_per_session: 2,
            max_bytes_per_session: 4096,
            max_total_bytes: 8192,
            max_followers_per_session: 2,
            ttl: Duration::from_secs(60),
            lookup_max: 100,
            keepalive_interval: Duration::from_secs(15),
        }
    }

    fn cancellation() -> ManagedChoiceCancellation {
        let (_sender, receiver) = mpsc::channel(1);
        let generation = ManagedChoiceGeneration::new(
            receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        );
        generation.cancellation_handle()
    }

    fn completed_terminal() -> StreamTerminalSignal {
        let terminal = StreamTerminalSignal::default();
        terminal.complete();
        terminal
    }

    #[test]
    fn stream_ids_are_random_url_safe_and_prefixed() {
        let first = random_stream_id().unwrap();
        let second = random_stream_id().unwrap();
        assert!(valid_stream_id(&first));
        assert!(valid_stream_id(&second));
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn replay_is_strictly_after_cursor_and_isolated_by_principal() {
        let streams = LlmStreams::new(policy(), ServerShutdown::new(), Metrics::new());
        let source = Body::from_stream(futures_util::stream::iter([
            Ok::<_, Infallible>(Bytes::from_static(b"data: one\n\n")),
            Ok(Bytes::from_static(b"data: two\n\n")),
        ]));
        let (id, initial) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                source,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let initial = initial.collect().await.unwrap().to_bytes();
        assert!(initial.windows(5).any(|value| value == b"id: 1"));
        assert!(initial.windows(5).any(|value| value == b"id: 2"));

        let replay = streams
            .attach(&id, PrincipalId::Anonymous, 1, false)
            .unwrap()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        assert!(!replay.windows(5).any(|value| value == b"id: 1"));
        assert!(replay.windows(5).any(|value| value == b"id: 2"));
        assert!(matches!(
            streams.attach(&id, PrincipalId::Authenticated, 0, false),
            Err(AccessError::NotFound)
        ));
    }

    #[tokio::test]
    async fn prefix_loss_and_follower_capacity_are_reported() {
        let mut limits = policy();
        limits.max_followers_per_session = 1;
        let streams = LlmStreams::new(limits, ServerShutdown::new(), Metrics::new());
        let (sender, source) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(8);
        let source = Body::from_stream(async_stream::stream! {
            let mut source = source;
            while let Some(frame) = source.recv().await { yield frame; }
        });
        let (id, initial) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                source,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        assert!(matches!(
            streams.attach(&id, PrincipalId::Anonymous, 0, true),
            Err(AccessError::FollowersExhausted)
        ));
        sender
            .send(Ok(Bytes::from_static(b"data: one\n\n")))
            .await
            .unwrap();
        sender
            .send(Ok(Bytes::from_static(b"data: two\n\n")))
            .await
            .unwrap();
        sender
            .send(Ok(Bytes::from_static(b"data: three\n\n")))
            .await
            .unwrap();
        drop(sender);
        let _ = initial.collect().await.unwrap();
        assert!(matches!(
            streams.attach(&id, PrincipalId::Anonymous, 0, false),
            Err(AccessError::ReplayLost)
        ));
    }

    #[tokio::test]
    async fn delete_is_idempotent_and_hides_lookup() {
        let streams = LlmStreams::new(policy(), ServerShutdown::new(), Metrics::new());
        let source =
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>());
        let (id, initial) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Responses,
                source,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        drop(initial);
        assert_eq!(
            streams
                .lookup(PrincipalId::Anonymous, std::slice::from_ref(&id))
                .len(),
            1
        );
        streams.delete(&id, PrincipalId::Authenticated);
        assert_eq!(
            streams
                .lookup(PrincipalId::Anonymous, std::slice::from_ref(&id))
                .len(),
            1
        );
        streams.delete(&id, PrincipalId::Anonymous);
        streams.delete(&id, PrincipalId::Anonymous);
        assert!(streams.lookup(PrincipalId::Anonymous, &[id]).is_empty());
    }

    #[tokio::test]
    async fn active_capacity_ttl_and_shutdown_are_enforced() {
        let mut limits = policy();
        limits.max_active = 1;
        limits.ttl = Duration::from_millis(1);
        let shutdown = ServerShutdown::new();
        let streams = LlmStreams::new(limits, shutdown.clone(), Metrics::new());
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(1);
        let pending = Body::from_stream(async_stream::stream! {
            let mut receiver = receiver;
            while let Some(frame) = receiver.recv().await { yield frame; }
        });
        let (id, follower) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                pending,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let second =
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>());
        assert!(matches!(
            streams.start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                second,
                cancellation(),
                None,
                completed_terminal(),
            ),
            Err(StartError::Capacity)
        ));
        streams.delete(&id, PrincipalId::Anonymous);
        drop(follower);
        drop(sender);
        tokio::task::yield_now().await;

        let completed =
            Body::from_stream(futures_util::stream::empty::<Result<Bytes, Infallible>>());
        let (completed_id, completed_follower) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                completed,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let _ = completed_follower.collect().await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(
            streams
                .lookup(PrincipalId::Anonymous, &[completed_id])
                .is_empty()
        );

        shutdown.trigger();
        assert!(matches!(
            streams.attach(
                "strm_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                PrincipalId::Anonymous,
                0,
                true
            ),
            Err(AccessError::ShuttingDown)
        ));
    }

    #[tokio::test]
    async fn multiple_followers_receive_the_same_immutable_frame() {
        let mut limits = policy();
        limits.max_followers_per_session = 3;
        let streams = LlmStreams::new(limits, ServerShutdown::new(), Metrics::new());
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(1);
        let source = Body::from_stream(async_stream::stream! {
            let mut receiver = receiver;
            while let Some(frame) = receiver.recv().await { yield frame; }
        });
        let (id, first) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                source,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let second = streams
            .attach(&id, PrincipalId::Anonymous, 0, true)
            .unwrap();
        let third = streams
            .attach(&id, PrincipalId::Anonymous, 0, true)
            .unwrap();
        sender
            .send(Ok(Bytes::from_static(b"data: shared\n\n")))
            .await
            .unwrap();
        drop(sender);
        let (first, second, third) =
            tokio::join!(first.collect(), second.collect(), third.collect());
        for body in [first.unwrap(), second.unwrap(), third.unwrap()] {
            assert_eq!(body.to_bytes(), "id: 1\ndata: shared\n\n");
        }
    }

    #[tokio::test]
    async fn byte_eviction_and_oversize_frames_remain_bounded() {
        let mut limits = policy();
        limits.max_bytes_per_session = MIN_STREAMING_ERROR_FRAME_BYTES;
        limits.max_total_bytes = 2 * MIN_STREAMING_ERROR_FRAME_BYTES;
        let streams = LlmStreams::new(limits, ServerShutdown::new(), Metrics::new());
        let source = Body::from_stream(futures_util::stream::iter([
            Ok::<_, Infallible>(Bytes::from(vec![b'1'; 350])),
            Ok(Bytes::from(vec![b'2'; 350])),
        ]));
        let (id, initial) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                source,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let _ = initial.collect().await.unwrap();
        assert!(matches!(
            streams.attach(&id, PrincipalId::Anonymous, 0, false),
            Err(AccessError::ReplayLost)
        ));

        let oversized = Body::from_stream(futures_util::stream::once(async {
            Ok::<_, Infallible>(Bytes::from(vec![b'x'; 1024]))
        }));
        let (oversized_id, follower) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Responses,
                oversized,
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        let body = follower.collect().await.unwrap().to_bytes();
        assert!(
            body.windows("stream_frame_too_large".len())
                .any(|value| value == b"stream_frame_too_large")
        );
        let metadata = streams.lookup(PrincipalId::Anonymous, &[oversized_id]);
        assert_eq!(metadata[0].status, StreamStatus::Failed);
        let state = streams.inner.state.lock().unwrap();
        assert!(state.total_bytes <= limits.max_total_bytes);
        assert!(
            state
                .sessions
                .values()
                .all(|session| session.bytes <= limits.max_bytes_per_session)
        );
    }

    #[tokio::test]
    async fn global_pressure_never_exceeds_the_hard_byte_limit() {
        let mut limits = policy();
        limits.max_bytes_per_session = MIN_STREAMING_ERROR_FRAME_BYTES;
        limits.max_total_bytes = MIN_STREAMING_ERROR_FRAME_BYTES;
        let streams = LlmStreams::new(limits, ServerShutdown::new(), Metrics::new());
        let payload = Bytes::from(vec![b'a'; 350]);
        let mut followers = Vec::new();
        for protocol in [StreamProtocol::Chat, StreamProtocol::Completions] {
            let source = Body::from_stream(futures_util::stream::once({
                let payload = payload.clone();
                async move { Ok::<_, Infallible>(payload) }
            }));
            followers.push(
                streams
                    .start(
                        PrincipalId::Anonymous,
                        protocol,
                        source,
                        cancellation(),
                        None,
                        completed_terminal(),
                    )
                    .unwrap()
                    .1,
            );
        }
        for follower in followers {
            let _ = follower.collect().await.unwrap();
        }
        let state = streams.inner.state.lock().unwrap();
        assert!(state.total_bytes <= limits.max_total_bytes);
    }

    #[tokio::test]
    async fn server_retention_cancellation_keeps_typed_first_cause_and_stores_error() {
        let mut limits = policy();
        limits.max_bytes_per_session = MIN_STREAMING_ERROR_FRAME_BYTES;
        limits.max_total_bytes = MIN_STREAMING_ERROR_FRAME_BYTES;
        let streams = LlmStreams::new(limits, ServerShutdown::new(), Metrics::new());

        let (first_generation, first_cancellation) = cancellation_pair();
        let (first_id, first_body) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>()),
                first_cancellation,
                None,
                StreamTerminalSignal::default(),
            )
            .unwrap();
        assert!(streams.append(&first_id, &Bytes::from(vec![b'a'; 350])));

        let (second_generation, second_cancellation) = cancellation_pair();
        let observed = second_cancellation.clone();
        let (second_id, second_body) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Responses,
                Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>()),
                second_cancellation,
                None,
                StreamTerminalSignal::default(),
            )
            .unwrap();
        assert!(!streams.append(&second_id, &Bytes::from(vec![b'b'; 350])));
        assert_eq!(
            observed.cause(),
            Some(ChoiceCancellationCause::ResourceExhausted)
        );
        drop(second_generation);
        assert_eq!(
            observed.cause(),
            Some(ChoiceCancellationCause::ResourceExhausted)
        );
        let error = second_body.collect().await.unwrap().to_bytes();
        assert!(
            error
                .windows("stream_capacity_exhausted".len())
                .any(|value| value == b"stream_capacity_exhausted")
        );
        assert!(streams.observation().buffered_bytes <= limits.max_total_bytes);

        streams.delete(&first_id, PrincipalId::Anonymous);
        drop(first_body);
        drop(first_generation);
    }

    #[test]
    fn every_protocol_terminal_error_frame_fits_the_validated_minimum() {
        for protocol in [
            StreamProtocol::Chat,
            StreamProtocol::Completions,
            StreamProtocol::Responses,
        ] {
            let frame = stream_error_frame(
                protocol,
                u64::MAX,
                "stream_capacity_exhausted",
                "global stream retention capacity exhausted",
            );
            assert!(frame.len() <= MIN_STREAMING_ERROR_FRAME_BYTES);
            assert!(frame.starts_with(format!("id: {}\n", u64::MAX).as_bytes()));

            let escaped = stream_error_frame(
                protocol,
                u64::MAX,
                &"\"".repeat(MAX_STREAM_ERROR_CODE_BYTES + 1),
                &"\"".repeat(MAX_STREAM_ERROR_MESSAGE_BYTES + 1),
            );
            assert!(escaped.len() <= MIN_STREAMING_ERROR_FRAME_BYTES);
        }
    }

    #[tokio::test]
    async fn follower_revision_catches_append_between_check_and_wait_registration() {
        let streams = LlmStreams::new(policy(), ServerShutdown::new(), Metrics::new());
        let source =
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>());
        let (id, initial) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                source,
                cancellation(),
                None,
                StreamTerminalSignal::default(),
            )
            .unwrap();
        let revision = streams
            .inner
            .state
            .lock()
            .unwrap()
            .sessions
            .get(&id)
            .unwrap()
            .revision
            .subscribe();
        let mut follower = std::mem::ManuallyDrop::new(Follower {
            streams: streams.clone(),
            stream_id: id.clone(),
            principal: PrincipalId::Anonymous,
            revision,
        });
        assert!(matches!(follower.next(0, true), FollowerAction::Wait));
        assert!(streams.append(&id, &Bytes::from_static(b"data: raced\n\n")));
        tokio::time::timeout(Duration::from_millis(100), follower.revision.changed())
            .await
            .expect("the revision observed before the check must retain the wakeup")
            .unwrap();
        streams.delete(&id, PrincipalId::Anonymous);
        drop(initial);
    }

    #[tokio::test]
    async fn shutdown_racing_start_never_panics_or_leaves_an_active_session() {
        for _ in 0..32 {
            let shutdown = ServerShutdown::new();
            let streams = LlmStreams::new(policy(), shutdown.clone(), Metrics::new());
            let starter = tokio::spawn({
                let streams = streams.clone();
                async move {
                    tokio::task::yield_now().await;
                    streams.start(
                        PrincipalId::Anonymous,
                        StreamProtocol::Chat,
                        Body::from_stream(
                            futures_util::stream::pending::<Result<Bytes, Infallible>>(),
                        ),
                        cancellation(),
                        None,
                        StreamTerminalSignal::default(),
                    )
                }
            });
            tokio::task::yield_now().await;
            shutdown.trigger();
            if let Ok((_id, body)) = starter.await.unwrap() {
                drop(body);
            }
            tokio::time::timeout(Duration::from_secs(1), async {
                while streams.observation().active != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn delete_preserves_completed_and_failed_terminal_state_and_metrics() {
        let metrics = Metrics::new();
        let streams = LlmStreams::new(policy(), ServerShutdown::new(), metrics.clone());
        let (completed_id, completed_body) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Chat,
                Body::empty(),
                cancellation(),
                None,
                completed_terminal(),
            )
            .unwrap();
        completed_body.collect().await.unwrap();
        streams.delete(&completed_id, PrincipalId::Anonymous);
        assert_eq!(
            streams
                .inner
                .state
                .lock()
                .unwrap()
                .sessions
                .get(&completed_id)
                .unwrap()
                .status,
            StreamStatus::Completed
        );

        let (failed_id, failed_body) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Responses,
                Body::empty(),
                cancellation(),
                None,
                StreamTerminalSignal::default(),
            )
            .unwrap();
        failed_body.collect().await.unwrap();
        streams.delete(&failed_id, PrincipalId::Anonymous);
        assert_eq!(
            streams
                .inner
                .state
                .lock()
                .unwrap()
                .sessions
                .get(&failed_id)
                .unwrap()
                .status,
            StreamStatus::Failed
        );
        assert_eq!(
            metric_total(
                &metrics.encode().unwrap(),
                "orchion_resumable_terminal_total"
            ),
            2
        );
    }

    #[tokio::test]
    async fn eof_before_typed_terminal_fails_all_protocols_without_success_sentinel() {
        for protocol in [
            StreamProtocol::Chat,
            StreamProtocol::Completions,
            StreamProtocol::Responses,
        ] {
            let streams = LlmStreams::new(policy(), ServerShutdown::new(), Metrics::new());
            let (id, body) = streams
                .start(
                    PrincipalId::Anonymous,
                    protocol,
                    Body::empty(),
                    cancellation(),
                    None,
                    StreamTerminalSignal::default(),
                )
                .unwrap();
            let body =
                String::from_utf8(body.collect().await.unwrap().to_bytes().to_vec()).unwrap();
            assert!(!body.contains("[DONE]"));
            assert!(body.contains("stream_terminal_missing"));
            if protocol == StreamProtocol::Responses {
                assert!(body.contains("event: error"));
                assert!(!body.contains("response.completed"));
            }
            assert_eq!(
                streams.lookup(PrincipalId::Anonymous, &[id])[0].status,
                StreamStatus::Failed
            );
        }
    }

    #[tokio::test]
    async fn failure_terminal_eviction_counts_one_truncation() {
        let mut limits = policy();
        limits.max_events_per_session = 1;
        let metrics = Metrics::new();
        let streams = LlmStreams::new(limits, ServerShutdown::new(), metrics.clone());
        let (id, body) = streams
            .start(
                PrincipalId::Anonymous,
                StreamProtocol::Responses,
                Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>()),
                cancellation(),
                None,
                StreamTerminalSignal::default(),
            )
            .unwrap();
        assert!(streams.append(&id, &Bytes::from_static(b"event: partial\n\n")));
        streams.fail(&id, "failed", "failed");
        assert_eq!(
            metric_total(
                &metrics.encode().unwrap(),
                "orchion_resumable_truncations_total"
            ),
            1
        );
        drop(body);
        streams.delete(&id, PrincipalId::Anonymous);
    }

    fn metric_total(metrics: &str, name: &str) -> u64 {
        metrics
            .lines()
            .find_map(|line| {
                line.strip_prefix(name)
                    .and_then(|value| value.trim().parse().ok())
            })
            .unwrap_or(0)
    }

    fn cancellation_pair() -> (ManagedChoiceGeneration, ManagedChoiceCancellation) {
        let (_sender, receiver) = mpsc::channel(1);
        let generation = ManagedChoiceGeneration::new(
            receiver,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
        );
        let cancellation = generation.cancellation_handle();
        (generation, cancellation)
    }
}
