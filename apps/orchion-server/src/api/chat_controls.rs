use crate::api::http::ServerShutdown;
use crate::api::llm_streams::PrincipalId;
use crate::application::metrics::{Metrics, ReasoningControlOutcome};
use orchion::{LlmReasoningControl, LlmReasoningControlResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CONTROL_TIMEOUT: Duration = Duration::from_millis(250);
const FAILURE_MESSAGE: &str = "reasoning control was not applied";

#[derive(Clone)]
pub(crate) struct ChatControls {
    inner: Arc<Inner>,
}

struct Inner {
    entries: Mutex<HashMap<String, Entry>>,
    metrics: Metrics,
    shutdown: ServerShutdown,
}

#[derive(Clone)]
struct Entry {
    principal: PrincipalId,
    model: String,
    control: LlmReasoningControl,
}

pub(crate) struct Registration {
    controls: ChatControls,
    id: String,
    armed: bool,
}

pub(crate) enum ApplyResult {
    Applied,
    Rejected,
    Unavailable,
}

impl ChatControls {
    pub(crate) fn new(metrics: Metrics, shutdown: ServerShutdown) -> Self {
        Self {
            inner: Arc::new(Inner {
                entries: Mutex::new(HashMap::new()),
                metrics,
                shutdown,
            }),
        }
    }

    pub(crate) fn register(
        &self,
        id: String,
        principal: PrincipalId,
        model: String,
        control: LlmReasoningControl,
    ) -> Registration {
        self.inner
            .entries
            .lock()
            .expect("chat control registry poisoned")
            .insert(
                id.clone(),
                Entry {
                    principal,
                    model,
                    control,
                },
            );
        Registration {
            controls: self.clone(),
            id,
            armed: true,
        }
    }

    pub(crate) fn remove(&self, id: &str) {
        self.inner
            .entries
            .lock()
            .expect("chat control registry poisoned")
            .remove(id);
    }

    pub(crate) async fn reasoning_end(
        &self,
        id: &str,
        principal: PrincipalId,
        model: Option<&str>,
    ) -> (ApplyResult, &'static str) {
        if self.inner.shutdown.is_triggered() {
            self.inner
                .entries
                .lock()
                .expect("chat control registry poisoned")
                .clear();
            self.observe(ReasoningControlOutcome::NotFound);
            return (ApplyResult::Rejected, FAILURE_MESSAGE);
        }
        let entry = self
            .inner
            .entries
            .lock()
            .expect("chat control registry poisoned")
            .get(id)
            .filter(|entry| entry.principal == principal)
            .cloned();
        let Some(entry) = entry else {
            self.observe(ReasoningControlOutcome::NotFound);
            return (ApplyResult::Rejected, FAILURE_MESSAGE);
        };
        if model.is_some_and(|model| model != entry.model) {
            self.observe(ReasoningControlOutcome::ModelMismatch);
            return (ApplyResult::Rejected, FAILURE_MESSAGE);
        }
        let Ok(attempt) = entry.control.begin_reasoning_end() else {
            self.observe(ReasoningControlOutcome::Unavailable);
            return (
                ApplyResult::Unavailable,
                "reasoning control is temporarily unavailable",
            );
        };
        let cancellation = attempt.cancellation_handle();
        let mut result = Box::pin(attempt.result());
        let result = tokio::select! {
            result = &mut result => result,
            () = tokio::time::sleep(CONTROL_TIMEOUT) => {
                if cancellation.cancel_pending() {
                    self.observe(ReasoningControlOutcome::Unavailable);
                    return (
                        ApplyResult::Unavailable,
                        "reasoning control is temporarily unavailable",
                    );
                }
                result.await
            }
        };
        match result {
            Ok(LlmReasoningControlResult::Success) => {
                self.observe(ReasoningControlOutcome::Success);
                (ApplyResult::Applied, "")
            }
            Ok(LlmReasoningControlResult::NotReasoning) => {
                self.observe(ReasoningControlOutcome::NotReasoning);
                (ApplyResult::Rejected, FAILURE_MESSAGE)
            }
            Ok(LlmReasoningControlResult::Disabled) => {
                self.observe(ReasoningControlOutcome::Disabled);
                (ApplyResult::Rejected, FAILURE_MESSAGE)
            }
            Ok(LlmReasoningControlResult::NotFound) => {
                self.observe(ReasoningControlOutcome::NotFound);
                (ApplyResult::Rejected, FAILURE_MESSAGE)
            }
            Err(_) => {
                self.observe(ReasoningControlOutcome::Unavailable);
                (
                    ApplyResult::Unavailable,
                    "reasoning control is temporarily unavailable",
                )
            }
        }
    }

    pub(crate) fn observe_invalid(&self) {
        self.observe(ReasoningControlOutcome::Invalid);
    }

    fn observe(&self, outcome: ReasoningControlOutcome) {
        self.inner.metrics.observe_reasoning_control(outcome);
    }
}

impl Registration {
    #[must_use]
    pub(crate) fn id(&self) -> &str {
        &self.id
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if self.armed {
            self.controls.remove(&self.id);
        }
    }
}
