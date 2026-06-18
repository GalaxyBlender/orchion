use crate::{OrchionError, Result};

pub async fn run<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| OrchionError::BlockingTask {
            message: error.to_string(),
        })?
}
