use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct InferenceGuard {
    _permit: Arc<OwnedSemaphorePermit>,
}

#[derive(Clone)]
pub struct ResourcePolicy {
    inference: Arc<Semaphore>,
    websocket_connections: Arc<Semaphore>,
    pending_websocket_connections: Arc<Semaphore>,
}

impl ResourcePolicy {
    #[must_use]
    pub fn new(
        max_concurrent_inference: usize,
        max_websocket_connections: usize,
        max_pending_websocket_connections: usize,
    ) -> Self {
        Self {
            inference: Arc::new(Semaphore::new(max_concurrent_inference)),
            websocket_connections: Arc::new(Semaphore::new(max_websocket_connections)),
            pending_websocket_connections: Arc::new(Semaphore::new(
                max_pending_websocket_connections,
            )),
        }
    }

    #[must_use]
    pub fn try_acquire_inference(&self) -> Option<InferenceGuard> {
        Arc::clone(&self.inference)
            .try_acquire_owned()
            .ok()
            .map(|permit| InferenceGuard {
                _permit: Arc::new(permit),
            })
    }

    #[must_use]
    pub fn try_acquire_websocket(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.websocket_connections)
            .try_acquire_owned()
            .ok()
    }

    #[must_use]
    pub fn try_acquire_pending_websocket(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.pending_websocket_connections)
            .try_acquire_owned()
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_work_beyond_each_configured_limit() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let inference = policy.try_acquire_inference().unwrap();
        let websocket = policy.try_acquire_websocket().unwrap();
        let pending_websocket = policy.try_acquire_pending_websocket().unwrap();

        assert!(policy.try_acquire_inference().is_none());
        assert!(policy.try_acquire_websocket().is_none());
        assert!(policy.try_acquire_pending_websocket().is_none());

        drop(inference);
        drop(websocket);
        drop(pending_websocket);
        assert!(policy.try_acquire_inference().is_some());
        assert!(policy.try_acquire_websocket().is_some());
        assert!(policy.try_acquire_pending_websocket().is_some());
    }

    #[test]
    fn cloned_inference_guard_holds_permit_until_last_clone_drops() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let inference = policy.try_acquire_inference().unwrap();
        let clone = inference.clone();

        drop(inference);
        assert!(policy.try_acquire_inference().is_none());

        drop(clone);
        assert!(policy.try_acquire_inference().is_some());
    }

    #[test]
    fn pending_websocket_does_not_consume_authenticated_capacity() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let _pending = policy.try_acquire_pending_websocket().unwrap();

        assert!(policy.try_acquire_websocket().is_some());
    }
}
