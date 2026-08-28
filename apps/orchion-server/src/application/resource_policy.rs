use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub struct InferenceGuard {
    _permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(crate) struct InferenceLimiter {
    permits: Arc<Semaphore>,
}

impl InferenceLimiter {
    pub(crate) async fn acquire(&self) -> InferenceGuard {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .expect("inference semaphore must remain open");
        InferenceGuard { _permit: permit }
    }
}

#[derive(Clone)]
pub struct ResourcePolicy {
    inference: InferenceLimiter,
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
            inference: InferenceLimiter {
                permits: Arc::new(Semaphore::new(max_concurrent_inference)),
            },
            websocket_connections: Arc::new(Semaphore::new(max_websocket_connections)),
            pending_websocket_connections: Arc::new(Semaphore::new(
                max_pending_websocket_connections,
            )),
        }
    }

    /// Waits until global inference capacity is available.
    ///
    /// # Panics
    ///
    /// Panics if the private inference semaphore is closed. `ResourcePolicy` never closes it.
    pub async fn acquire_inference(&self) -> InferenceGuard {
        self.inference.acquire().await
    }

    pub(crate) fn inference_limiter(&self) -> InferenceLimiter {
        self.inference.clone()
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
    use std::time::Duration;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn queues_inference_beyond_the_configured_limit() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let inference = policy.acquire_inference().await;
        let websocket = policy.try_acquire_websocket().unwrap();
        let pending_websocket = policy.try_acquire_pending_websocket().unwrap();
        let queued = policy.acquire_inference();
        tokio::pin!(queued);

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut queued)
                .await
                .is_err()
        );
        assert!(policy.try_acquire_websocket().is_none());
        assert!(policy.try_acquire_pending_websocket().is_none());

        drop(inference);
        drop(websocket);
        drop(pending_websocket);
        tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap();
        assert!(policy.try_acquire_websocket().is_some());
        assert!(policy.try_acquire_pending_websocket().is_some());
    }

    #[tokio::test]
    async fn cancelled_inference_waiter_does_not_consume_capacity() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let inference = policy.acquire_inference().await;
        let started = Arc::new(Notify::new());
        let waiter = tokio::spawn({
            let policy = policy.clone();
            let started = Arc::clone(&started);
            async move {
                started.notify_one();
                policy.acquire_inference().await
            }
        });
        started.notified().await;
        tokio::task::yield_now().await;

        waiter.abort();
        let Err(error) = waiter.await else {
            panic!("aborted inference waiter must be cancelled");
        };
        assert!(error.is_cancelled());
        drop(inference);

        tokio::time::timeout(Duration::from_secs(1), policy.acquire_inference())
            .await
            .unwrap();
    }

    #[test]
    fn pending_websocket_does_not_consume_authenticated_capacity() {
        let policy = ResourcePolicy::new(1, 1, 1);
        let _pending = policy.try_acquire_pending_websocket().unwrap();

        assert!(policy.try_acquire_websocket().is_some());
    }
}
