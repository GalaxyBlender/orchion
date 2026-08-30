use super::contract::{
    ActivityEntry, ActivityError, ActivityEventPayload, ActivityOperation, ActivityOutcome,
    ActivityPage, ActivityState, ActivitySummary, ActivityTransport,
};
use crate::application::ActivityPolicy;
use orchion::ModelId;
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

const EVENT_BUFFER_CAPACITY: usize = 256;
const MAX_MODEL_ID_BYTES: usize = 256;
const MAX_USER_AGENT_BYTES: usize = 512;
const INPUT_UPDATE_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivityEventKind {
    Started,
    Updated,
    Completed,
}

impl ActivityEventKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Updated => "updated",
            Self::Completed => "completed",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ActivityEvent {
    pub(crate) kind: ActivityEventKind,
    pub(crate) payload: ActivityEventPayload,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ActivityFilter {
    pub(crate) limit: usize,
    pub(crate) before: Option<u64>,
    pub(crate) operation: Option<ActivityOperation>,
    pub(crate) outcome: Option<ActivityOutcome>,
    pub(crate) model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ActivityHub {
    inner: Arc<Inner>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LiveRequestMetadata {
    address: Option<String>,
    user_agent: Option<String>,
}

impl LiveRequestMetadata {
    pub(crate) fn new(address: Option<String>, user_agent: Option<String>) -> Self {
        Self {
            address: address
                .and_then(|value| value.parse::<IpAddr>().ok())
                .map(|address| address.to_string()),
            user_agent: user_agent.filter(|value| value.len() <= MAX_USER_AGENT_BYTES),
        }
    }
}

#[derive(Debug)]
struct Inner {
    enabled: bool,
    history_capacity: usize,
    next_id: AtomicU64,
    state: Mutex<StoreState>,
    events: broadcast::Sender<ActivityEvent>,
}

#[derive(Debug, Default)]
struct StoreState {
    active: HashMap<u64, ActiveEntry>,
    history: VecDeque<ActivityEntry>,
    success_count: usize,
    sorted_durations: Vec<u64>,
    event_seq: u64,
}

#[derive(Debug, Clone)]
struct ActiveEntry {
    entry: ActivityEntry,
    started: Instant,
    pending_input_bytes: u64,
    last_input_update: Option<Instant>,
    input_flush_scheduled: bool,
}

#[derive(Debug, Clone)]
pub struct ActivityContext {
    hub: ActivityHub,
    id: u64,
    handed_off: Arc<AtomicBool>,
}

#[derive(Debug)]
pub struct WebSocketActivity {
    context: ActivityContext,
    completed: AtomicBool,
}

impl ActivityHub {
    #[must_use]
    pub fn new(policy: ActivityPolicy) -> Self {
        let (events, _) = broadcast::channel(EVENT_BUFFER_CAPACITY);
        Self {
            inner: Arc::new(Inner {
                enabled: policy.enabled,
                history_capacity: policy.history_capacity,
                next_id: AtomicU64::new(1),
                state: Mutex::new(StoreState::default()),
                events,
            }),
        }
    }

    #[must_use]
    pub fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[cfg(test)]
    pub(crate) fn start(
        &self,
        operation: ActivityOperation,
        transport: ActivityTransport,
        method: &str,
        route: &str,
        input_bytes: Option<u64>,
    ) -> Option<ActivityContext> {
        self.start_with_live_metadata(
            operation,
            transport,
            method,
            route,
            input_bytes,
            LiveRequestMetadata::default(),
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "request lifecycle fields form one Activity entry"
    )]
    pub(crate) fn start_with_live_metadata(
        &self,
        operation: ActivityOperation,
        transport: ActivityTransport,
        method: &str,
        route: &str,
        input_bytes: Option<u64>,
        live_metadata: LiveRequestMetadata,
    ) -> Option<ActivityContext> {
        if !self.enabled() {
            return None;
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = ActivityEntry {
            id: id.to_string(),
            state: ActivityState::InFlight,
            transport,
            operation,
            method: method.to_string(),
            route: route.to_string(),
            model: None,
            address: live_metadata.address,
            user_agent: live_metadata.user_agent,
            started_at_ms: unix_time_ms(),
            duration_ms: Some(0),
            http_status: None,
            outcome: None,
            input_bytes,
            prompt_tokens: None,
            completion_tokens: None,
            queue_time_ms: None,
            eval_time_ms: None,
            error_code: None,
            error_message: None,
        };
        let active = ActiveEntry {
            entry: entry.clone(),
            started: Instant::now(),
            pending_input_bytes: 0,
            last_input_update: None,
            input_flush_scheduled: false,
        };
        self.mutate(|state| {
            state.active.insert(id, active);
            (ActivityEventKind::Started, entry)
        });
        Some(ActivityContext {
            hub: self.clone(),
            id,
            handed_off: Arc::new(AtomicBool::new(false)),
        })
    }

    #[must_use]
    pub(crate) fn page(&self, filter: &ActivityFilter) -> ActivityPage {
        let state = self.inner.state.lock().expect("activity store poisoned");
        if !self.enabled() {
            return ActivityPage {
                enabled: false,
                cursor: state.event_seq.to_string(),
                active: Vec::new(),
                history: Vec::new(),
                summary: ActivitySummary {
                    active: 0,
                    retained: 0,
                    success_rate: None,
                    p95_duration_ms: None,
                },
                next_before: None,
            };
        }

        let mut active = state
            .active
            .values()
            .map(active_snapshot)
            .filter(|entry| matches_filter(entry, filter, false))
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|entry| std::cmp::Reverse(activity_id(entry)));

        let mut history = state
            .history
            .iter()
            .filter(|entry| matches_filter(entry, filter, true))
            .take(filter.limit.saturating_add(1))
            .cloned()
            .collect::<Vec<_>>();
        let has_more = history.len() > filter.limit;
        if has_more {
            history.pop();
        }
        let next_before = has_more
            .then(|| history.last().map(|entry| entry.id.clone()))
            .flatten();

        ActivityPage {
            enabled: true,
            cursor: state.event_seq.to_string(),
            active,
            history,
            summary: summary(&state),
            next_before,
        }
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<ActivityEvent> {
        self.inner.events.subscribe()
    }

    pub(crate) fn reset_subscription(
        &self,
    ) -> (broadcast::Receiver<ActivityEvent>, ActivityEventPayload) {
        let receiver = self.subscribe();
        let reset = self.reset_event();
        (receiver, reset)
    }

    #[must_use]
    pub(crate) fn snapshot_event(&self) -> ActivityEventPayload {
        let state = self.inner.state.lock().expect("activity store poisoned");
        let mut active = state
            .active
            .values()
            .map(active_snapshot)
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|entry| std::cmp::Reverse(activity_id(entry)));
        ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: None,
            active: Some(active),
            summary: Some(summary(&state)),
        }
    }

    #[must_use]
    pub(crate) fn reset_event(&self) -> ActivityEventPayload {
        let state = self.inner.state.lock().expect("activity store poisoned");
        ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: None,
            active: None,
            summary: Some(summary(&state)),
        }
    }

    fn mutate(&self, update: impl FnOnce(&mut StoreState) -> (ActivityEventKind, ActivityEntry)) {
        let mut state = self.inner.state.lock().expect("activity store poisoned");
        let (kind, entry) = update(&mut state);
        state.event_seq = state.event_seq.saturating_add(1);
        let payload = ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: Some(entry),
            active: None,
            summary: Some(summary(&state)),
        };
        let _ = self.inner.events.send(ActivityEvent { kind, payload });
    }

    fn update(&self, id: u64, update: impl FnOnce(&mut ActivityEntry)) {
        self.mutate_if_present(id, |active| {
            update(&mut active.entry);
            (ActivityEventKind::Updated, active.entry.clone())
        });
    }

    fn mutate_if_present(
        &self,
        id: u64,
        update: impl FnOnce(&mut ActiveEntry) -> (ActivityEventKind, ActivityEntry),
    ) {
        let mut state = self.inner.state.lock().expect("activity store poisoned");
        let Some(active) = state.active.get_mut(&id) else {
            return;
        };
        let (kind, mut entry) = update(active);
        entry.duration_ms = Some(duration_ms(active.started.elapsed()));
        state.event_seq = state.event_seq.saturating_add(1);
        let payload = ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: Some(entry),
            active: None,
            summary: Some(summary(&state)),
        };
        let _ = self.inner.events.send(ActivityEvent { kind, payload });
    }

    fn add_input_bytes(&self, id: u64, bytes: usize) {
        let wait = {
            let mut state = self.inner.state.lock().expect("activity store poisoned");
            let Some(active) = state.active.get_mut(&id) else {
                return;
            };
            active.pending_input_bytes = active
                .pending_input_bytes
                .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
            let now = Instant::now();
            let wait = active.last_input_update.and_then(|last_update| {
                INPUT_UPDATE_INTERVAL.checked_sub(now.duration_since(last_update))
            });
            if wait.is_some() {
                if active.input_flush_scheduled {
                    return;
                }
                active.input_flush_scheduled = true;
            }
            wait
        };

        if let Some(wait) = wait {
            let hub = self.clone();
            tokio::spawn(async move {
                tokio::time::sleep(wait).await;
                hub.flush_input_update(id);
            });
        } else {
            self.flush_input_update(id);
        }
    }

    fn flush_input_update(&self, id: u64) {
        let mut state = self.inner.state.lock().expect("activity store poisoned");
        let entry = {
            let Some(active) = state.active.get_mut(&id) else {
                return;
            };
            active.input_flush_scheduled = false;
            if active.pending_input_bytes == 0 {
                return;
            }
            flush_pending_input(active);
            active.last_input_update = Some(Instant::now());
            active_snapshot(active)
        };
        state.event_seq = state.event_seq.saturating_add(1);
        let payload = ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: Some(entry),
            active: None,
            summary: Some(summary(&state)),
        };
        let _ = self.inner.events.send(ActivityEvent {
            kind: ActivityEventKind::Updated,
            payload,
        });
    }

    fn complete(
        &self,
        id: u64,
        http_status: Option<u16>,
        outcome: ActivityOutcome,
        activity_error: Option<ActivityError>,
    ) {
        let mut state = self.inner.state.lock().expect("activity store poisoned");
        let Some(mut active) = state.active.remove(&id) else {
            return;
        };
        flush_pending_input(&mut active);
        active.entry.state = ActivityState::Completed;
        active.entry.duration_ms = Some(duration_ms(active.started.elapsed()));
        active.entry.http_status = http_status;
        active.entry.outcome = Some(outcome);
        active.entry.error_code = activity_error.as_ref().and_then(|error| error.code.clone());
        active.entry.error_message = activity_error.and_then(|error| error.message);
        active.entry.address = None;
        active.entry.user_agent = None;
        let entry = active.entry;
        if self.inner.history_capacity > 0 {
            add_history_metrics(&mut state, &entry);
            state.history.push_front(entry.clone());
            while state.history.len() > self.inner.history_capacity {
                if let Some(removed) = state.history.pop_back() {
                    remove_history_metrics(&mut state, &removed);
                }
            }
        }
        state.event_seq = state.event_seq.saturating_add(1);
        let payload = ActivityEventPayload {
            cursor: state.event_seq.to_string(),
            entry: Some(entry),
            active: None,
            summary: Some(summary(&state)),
        };
        let _ = self.inner.events.send(ActivityEvent {
            kind: ActivityEventKind::Completed,
            payload,
        });
    }
}

impl ActivityContext {
    pub fn set_model(&self, model: impl Into<String>) {
        let model = model.into();
        if model.len() <= MAX_MODEL_ID_BYTES && ModelId::parse(&model).is_ok() {
            self.hub.update(self.id, |entry| entry.model = Some(model));
        }
    }

    pub fn set_input_bytes(&self, input_bytes: u64) {
        self.hub
            .update(self.id, |entry| entry.input_bytes = Some(input_bytes));
    }

    pub fn set_llm_usage(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.hub.update(self.id, |entry| {
            entry.prompt_tokens = Some(prompt_tokens);
            entry.completion_tokens = Some(completion_tokens);
        });
    }

    pub fn set_llm_timing(&self, queue_time_ms: Option<u64>, eval_time_ms: Option<u64>) {
        self.hub.update(self.id, |entry| {
            entry.queue_time_ms = queue_time_ms;
            entry.eval_time_ms = eval_time_ms;
        });
    }

    pub(crate) fn complete_stream_failure(&self, outcome: ActivityOutcome, error: ActivityError) {
        self.hub.complete(self.id, Some(200), outcome, Some(error));
    }

    pub(crate) fn complete_http(&self, status: u16, activity_error: Option<ActivityError>) {
        let outcome = activity_error.as_ref().map_or_else(
            || http_outcome(status),
            |error| match error.code.as_deref() {
                Some("request_timeout") => ActivityOutcome::Timeout,
                Some("resource_exhausted") => ActivityOutcome::ResourceExhausted,
                Some("server_shutdown") => ActivityOutcome::Cancelled,
                _ => http_outcome(status),
            },
        );
        self.hub
            .complete(self.id, Some(status), outcome, activity_error);
    }

    pub(crate) fn cancel(&self) {
        self.hub
            .complete(self.id, None, ActivityOutcome::Cancelled, None);
    }

    pub(crate) fn disconnect(&self) {
        self.hub
            .complete(self.id, None, ActivityOutcome::Disconnected, None);
    }

    #[must_use]
    pub fn handoff_to_websocket(&self) -> WebSocketActivity {
        self.handed_off.store(true, Ordering::Release);
        self.hub.update(self.id, |entry| {
            entry.transport = ActivityTransport::Websocket;
        });
        WebSocketActivity {
            context: self.clone(),
            completed: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub(crate) fn was_handed_off(&self) -> bool {
        self.handed_off.load(Ordering::Acquire)
    }
}

impl WebSocketActivity {
    pub fn set_model(&self, model: impl Into<String>) {
        self.context.set_model(model);
    }

    pub fn add_input_bytes(&self, bytes: usize) {
        self.context.hub.add_input_bytes(self.context.id, bytes);
    }

    pub fn complete_success(&self) {
        self.complete(ActivityOutcome::Success, None);
    }

    pub(crate) fn complete_error(&self, status: u16, activity_error: Option<ActivityError>) {
        self.complete(http_outcome(status), activity_error);
    }

    pub(crate) fn complete_timeout(&self, activity_error: Option<ActivityError>) {
        self.complete(ActivityOutcome::Timeout, activity_error);
    }

    pub fn complete_disconnected(&self) {
        self.complete(ActivityOutcome::Disconnected, None);
    }

    fn complete(&self, outcome: ActivityOutcome, activity_error: Option<ActivityError>) {
        if self.completed.swap(true, Ordering::AcqRel) {
            return;
        }
        self.context
            .hub
            .complete(self.context.id, Some(101), outcome, activity_error);
    }
}

impl Drop for WebSocketActivity {
    fn drop(&mut self) {
        self.complete_disconnected();
    }
}

fn matches_filter(entry: &ActivityEntry, filter: &ActivityFilter, use_before: bool) -> bool {
    if use_before
        && filter.before.is_some_and(|before| {
            entry
                .id
                .parse::<u64>()
                .is_ok_and(|entry_id| entry_id >= before)
        })
    {
        return false;
    }
    if filter
        .operation
        .is_some_and(|operation| entry.operation != operation)
    {
        return false;
    }
    if filter
        .outcome
        .is_some_and(|outcome| entry.outcome != Some(outcome))
    {
        return false;
    }
    filter
        .model
        .as_ref()
        .is_none_or(|model| entry.model.as_deref() == Some(model.as_str()))
}

fn activity_id(entry: &ActivityEntry) -> u64 {
    entry.id.parse().unwrap_or_default()
}

fn summary(state: &StoreState) -> ActivitySummary {
    let retained = state.history.len();
    let success_rate = (retained > 0).then(|| {
        let per_mille = state.success_count.saturating_mul(1_000) / retained;
        f64::from(u16::try_from(per_mille).unwrap_or(1_000)) / 10.0
    });
    let p95_duration_ms = (!state.sorted_durations.is_empty()).then(|| {
        let index = (state.sorted_durations.len() * 95)
            .div_ceil(100)
            .saturating_sub(1);
        state.sorted_durations[index]
    });
    ActivitySummary {
        active: state.active.len(),
        retained,
        success_rate,
        p95_duration_ms,
    }
}

fn flush_pending_input(active: &mut ActiveEntry) {
    if active.pending_input_bytes == 0 {
        return;
    }
    active.entry.input_bytes = Some(
        active
            .entry
            .input_bytes
            .unwrap_or_default()
            .saturating_add(active.pending_input_bytes),
    );
    active.pending_input_bytes = 0;
}

fn active_snapshot(active: &ActiveEntry) -> ActivityEntry {
    let mut entry = active.entry.clone();
    entry.duration_ms = Some(duration_ms(active.started.elapsed()));
    entry
}

fn add_history_metrics(state: &mut StoreState, entry: &ActivityEntry) {
    if entry.outcome == Some(ActivityOutcome::Success) {
        state.success_count = state.success_count.saturating_add(1);
    }
    if let Some(duration) = entry.duration_ms {
        let index = state
            .sorted_durations
            .partition_point(|value| *value <= duration);
        state.sorted_durations.insert(index, duration);
    }
}

fn remove_history_metrics(state: &mut StoreState, entry: &ActivityEntry) {
    if entry.outcome == Some(ActivityOutcome::Success) {
        state.success_count = state.success_count.saturating_sub(1);
    }
    if let Some(duration) = entry.duration_ms
        && let Ok(index) = state.sorted_durations.binary_search(&duration)
    {
        state.sorted_durations.remove(index);
    }
}

const fn http_outcome(status: u16) -> ActivityOutcome {
    match status {
        200..=399 => ActivityOutcome::Success,
        429 => ActivityOutcome::ResourceExhausted,
        400..=499 => ActivityOutcome::ClientError,
        _ => ActivityOutcome::ServerError,
    }
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn duration_ms(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hub(capacity: usize) -> ActivityHub {
        ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: capacity,
        })
    }

    fn start(hub: &ActivityHub) -> ActivityContext {
        hub.start(
            ActivityOperation::Asr,
            ActivityTransport::Http,
            "POST",
            "/v1/audio/transcriptions",
            None,
        )
        .unwrap()
    }

    #[test]
    fn request_timeout_error_uses_timeout_outcome() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let context = start(&hub);
        context.complete_http(
            408,
            Some(ActivityError {
                code: Some("request_timeout".to_string()),
                message: None,
            }),
        );
        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Timeout));
    }

    #[test]
    fn history_evicts_only_the_oldest_completed_entry() {
        let hub = hub(2);
        for _ in 0..3 {
            start(&hub).complete_http(200, None);
        }

        let page = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });

        assert_eq!(page.history.len(), 2);
        assert_eq!(page.history[0].id, "3");
        assert_eq!(page.history[1].id, "2");
        assert_eq!(page.summary.success_rate, Some(100.0));
        assert!(page.summary.p95_duration_ms.is_some());
    }

    #[test]
    fn websocket_drop_completes_once_as_disconnected() {
        let hub = hub(10);
        let context = start(&hub);
        drop(context.handoff_to_websocket());
        context.cancel();

        let page = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });

        assert!(page.active.is_empty());
        assert_eq!(page.history.len(), 1);
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Disconnected));
    }

    #[test]
    fn websocket_completion_is_idempotent() {
        let hub = hub(10);
        let context = start(&hub);
        let activity = context.handoff_to_websocket();
        activity.complete_timeout(Some(ActivityError {
            code: Some("stream_idle_timeout".to_string()),
            message: Some("stream idle timeout".to_string()),
        }));
        activity.complete_success();
        drop(activity);

        let page = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });

        assert_eq!(page.history.len(), 1);
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Timeout));
        assert_eq!(
            page.history[0].error_code.as_deref(),
            Some("stream_idle_timeout")
        );
    }

    #[test]
    fn model_metadata_rejects_unstructured_and_oversized_values() {
        let hub = hub(10);
        let unstructured = start(&hub);
        unstructured.set_model("private request text");
        unstructured.complete_http(400, None);
        let oversized = start(&hub);
        oversized.set_model(format!("Acme/{}", "x".repeat(MAX_MODEL_ID_BYTES)));
        oversized.complete_http(400, None);

        let page = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });

        assert!(page.history.iter().all(|entry| entry.model.is_none()));
    }

    #[test]
    fn live_client_metadata_is_removed_before_history_is_retained() {
        let hub = hub(10);
        let context = hub
            .start_with_live_metadata(
                ActivityOperation::Asr,
                ActivityTransport::Http,
                "POST",
                "/v1/audio/transcriptions",
                None,
                LiveRequestMetadata::new(
                    Some("203.0.113.7".to_string()),
                    Some("orchion-test-agent/1.0".to_string()),
                ),
            )
            .unwrap();
        context.set_model("Qwen/Qwen3-ASR-0.6B");
        let active = hub.page(&ActivityFilter::default());
        assert_eq!(active.active[0].address.as_deref(), Some("203.0.113.7"));
        assert_eq!(
            active.active[0].user_agent.as_deref(),
            Some("orchion-test-agent/1.0")
        );
        assert_eq!(
            active.active[0].model.as_deref(),
            Some("Qwen/Qwen3-ASR-0.6B")
        );

        let mut events = hub.subscribe();
        context.complete_http(200, None);
        let completed = events.try_recv().unwrap();
        let completed_entry = completed.payload.entry.unwrap();
        assert!(completed_entry.address.is_none());
        assert!(completed_entry.user_agent.is_none());

        let history = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });
        assert!(history.history[0].address.is_none());
        assert!(history.history[0].user_agent.is_none());
        assert_eq!(
            history.history[0].model.as_deref(),
            Some("Qwen/Qwen3-ASR-0.6B")
        );
    }

    #[tokio::test]
    async fn websocket_input_updates_are_versioned_and_flushed_after_throttling() {
        let hub = hub(10);
        let context = start(&hub);
        let activity = context.handoff_to_websocket();
        let cursor_before_input = hub.page(&ActivityFilter::default()).cursor;

        activity.add_input_bytes(4);
        activity.add_input_bytes(6);
        let active = hub.page(&ActivityFilter::default());
        assert!(
            active.cursor.parse::<u64>().unwrap() > cursor_before_input.parse::<u64>().unwrap()
        );
        assert_eq!(active.active[0].input_bytes, Some(4));

        let flushed = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let page = hub.page(&ActivityFilter::default());
                if page.active[0].input_bytes == Some(10) {
                    break page;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(flushed.active[0].input_bytes, Some(10));

        activity.complete_success();
        let completed = hub.page(&ActivityFilter {
            limit: 10,
            ..ActivityFilter::default()
        });
        assert_eq!(completed.history[0].input_bytes, Some(10));
    }

    #[tokio::test]
    async fn active_duration_snapshot_uses_monotonic_elapsed_time() {
        let hub = hub(10);
        let _context = start(&hub);

        tokio::time::sleep(Duration::from_millis(30)).await;
        let page = hub.page(&ActivityFilter::default());

        assert!(
            page.active[0]
                .duration_ms
                .is_some_and(|duration| duration >= 30)
        );
    }

    #[tokio::test]
    async fn reset_subscription_starts_after_its_reset_cursor() {
        let hub = hub(EVENT_BUFFER_CAPACITY + 2);
        let mut lagged = hub.subscribe();
        for _ in 0..=EVENT_BUFFER_CAPACITY {
            let _ = start(&hub);
        }
        assert!(matches!(
            lagged.recv().await,
            Err(broadcast::error::RecvError::Lagged(_))
        ));

        let (mut receiver, reset) = hub.reset_subscription();
        let _ = start(&hub);
        let event = receiver.recv().await.unwrap();

        assert!(
            event.payload.cursor.parse::<u64>().unwrap() > reset.cursor.parse::<u64>().unwrap()
        );
    }

    #[test]
    fn cursor_page_uses_stable_request_ids() {
        let hub = hub(10);
        for _ in 0..3 {
            start(&hub).complete_http(200, None);
        }
        let first = hub.page(&ActivityFilter {
            limit: 2,
            ..ActivityFilter::default()
        });
        let second = hub.page(&ActivityFilter {
            limit: 2,
            before: first
                .next_before
                .as_deref()
                .map(str::parse)
                .transpose()
                .unwrap(),
            ..ActivityFilter::default()
        });

        assert_eq!(first.history.len(), 2);
        assert_eq!(second.history.len(), 1);
        assert_eq!(second.history[0].id, "1");
    }

    #[test]
    fn concurrent_requests_receive_unique_ids() {
        let hub = hub(64);
        let workers = (0..32)
            .map(|_| {
                let hub = hub.clone();
                std::thread::spawn(move || start(&hub).complete_http(200, None))
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let page = hub.page(&ActivityFilter {
            limit: 64,
            ..ActivityFilter::default()
        });
        let ids = page
            .history
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<std::collections::HashSet<_>>();

        assert_eq!(page.history.len(), 32);
        assert_eq!(ids.len(), 32);
    }
}
