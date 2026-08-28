use crate::api::openai::ApiError;
use crate::application::ServerApplication;
use crate::application::resource_policy::InferenceGuard;
use crate::application::{
    OwnedOperationState, mark_owned_operation_dispatched, track_owned_operation,
};
use axum::extract::multipart::Field;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue};
use std::future::Future;
use std::sync::Arc;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;

struct OwnedRequestTask<T> {
    task: JoinHandle<T>,
    state: Arc<OwnedOperationState>,
}

impl<T> Drop for OwnedRequestTask<T> {
    fn drop(&mut self) {
        if self.state.cancel() {
            self.task.abort();
        }
    }
}

pub(super) async fn run_inference_owned<T, F, P>(permit: P, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ApiError>> + Send + 'static,
    P: Future<Output = InferenceGuard> + Send,
{
    let permit = permit.await;
    run_owned(async move {
        let _permit = permit;
        if !mark_owned_operation_dispatched() {
            return Err(ApiError::internal("request cancelled before dispatch"));
        }
        operation.await
    })
    .await
}

pub(super) async fn run_owned<T, F>(operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ApiError>> + Send + 'static,
{
    let state = Arc::new(OwnedOperationState::new());
    let mut task = OwnedRequestTask {
        task: tokio::spawn(track_owned_operation(Arc::clone(&state), operation)),
        state,
    };
    (&mut task.task)
        .await
        .map_err(|error| ApiError::internal(format!("request task failed: {error:#}")))?
}

pub(super) fn authorize(
    state: &impl ServerApplication,
    headers: &HeaderMap,
) -> Result<(), ApiError> {
    let Some(api_key) = state.api_policy().api_key.as_deref() else {
        return Ok(());
    };
    let Some(header) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(ApiError::invalid_api_key());
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err(ApiError::invalid_api_key());
    };
    if token == api_key {
        Ok(())
    } else {
        Err(ApiError::invalid_api_key())
    }
}

pub(super) fn origin_is_allowed(allowed_origins: &[String], origin: &HeaderValue) -> bool {
    allowed_origins
        .iter()
        .any(|allowed| allowed == "*" || allowed.as_bytes() == origin.as_bytes())
}

pub(super) fn is_multipart(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        })
}

pub(super) fn required_multipart_field(
    value: Option<String>,
    param: &'static str,
) -> Result<String, ApiError> {
    value.ok_or_else(|| {
        ApiError::invalid_request(
            format!("`{param}` is required"),
            Some(param),
            Some("missing_required_parameter"),
        )
    })
}

pub(super) async fn parse_multipart_value<T>(
    field: Field<'_>,
    param: &'static str,
) -> Result<T, ApiError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = read_text_field(field, param).await?;
    value.trim().parse().map_err(|error| {
        ApiError::invalid_request(
            format!("invalid `{param}`: {error}"),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}

pub(super) async fn read_text_field(field: Field<'_>, param: &str) -> Result<String, ApiError> {
    field.text().await.map_err(|error| {
        ApiError::invalid_request(
            error.to_string(),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}

pub(super) async fn write_multipart_file_to_temp_file(
    mut field: Field<'_>,
    param: &'static str,
) -> Result<(NamedTempFile, u64), ApiError> {
    let suffix = multipart_file_suffix(field.content_type());
    let audio_file = TempFileBuilder::new()
        .prefix("orchion-upload-")
        .suffix(suffix)
        .tempfile()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut writer = tokio::fs::File::create(audio_file.path())
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut bytes_written = 0_u64;

    while let Some(chunk) = field.chunk().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), Some(param), Some("invalid_file"))
    })? {
        writer
            .write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        bytes_written += u64::try_from(chunk.len()).map_err(|error| {
            ApiError::internal(format!("uploaded file chunk size overflowed u64: {error}"))
        })?;
    }
    writer
        .flush()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;

    Ok((audio_file, bytes_written))
}

pub(super) fn multipart_file_suffix(content_type: Option<&str>) -> &'static str {
    match content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("image/png") => ".png",
        Some("image/jpeg" | "image/jpg") => ".jpg",
        Some("image/webp") => ".webp",
        Some("image/bmp") => ".bmp",
        Some("image/tiff") => ".tiff",
        Some("application/pdf") => ".pdf",
        Some("video/mp4") => ".mp4",
        Some("video/quicktime") => ".mov",
        Some("video/webm") => ".webm",
        Some("video/x-matroska") => ".mkv",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::model_cache::ModelCache;
    use crate::application::resource_policy::ResourcePolicy;
    use orchion::AsrModel;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::time::Duration;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn cancelled_waiter_does_not_release_inference_resources_early() {
        let resources = ResourcePolicy::new(1, 1, 1);
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let finished = Arc::new(Notify::new());
        let waiter = tokio::spawn({
            let resources = resources.clone();
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let finished = Arc::clone(&finished);
            async move {
                run_inference_owned(resources.acquire_inference(), async move {
                    let _file = file;
                    started.notify_one();
                    release.notified().await;
                    finished.notify_one();
                    Ok::<(), ApiError>(())
                })
                .await
            }
        });
        started.notified().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        let queued = resources.acquire_inference();
        tokio::pin!(queued);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut queued)
                .await
                .is_err()
        );
        assert!(path.exists());

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), finished.notified())
            .await
            .unwrap();
        tokio::task::yield_now().await;
        tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn cancelled_owned_request_before_dispatch_is_aborted() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let waiter = tokio::spawn({
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let completed = Arc::clone(&completed);
            async move {
                run_owned(async move {
                    let _file = file;
                    started.notify_one();
                    release.notified().await;
                    completed.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok::<(), ApiError>(())
                })
                .await
            }
        });
        started.notified().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        tokio::task::yield_now().await;
        assert!(!path.exists());

        release.notify_one();
        tokio::task::yield_now().await;
        assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancelled_request_keeps_file_until_blocking_file_operation_finishes() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let started = Arc::new(Notify::new());
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let finished = Arc::new(Notify::new());
        let cancellation_observed = Arc::new(AtomicBool::new(false));
        let waiter = tokio::spawn({
            let path = path.clone();
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let finished = Arc::clone(&finished);
            let cancellation_observed = Arc::clone(&cancellation_observed);
            async move {
                run_owned(async move {
                    let _file = file;
                    crate::application::protect_owned_file_operation();
                    tokio::task::spawn_blocking(move || {
                        started.notify_one();
                        let (released, wake) = &*release;
                        let guard = released
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        drop(
                            wake.wait_while(guard, |released| !*released)
                                .unwrap_or_else(std::sync::PoisonError::into_inner),
                        );
                        assert!(path.exists());
                    })
                    .await
                    .unwrap();
                    cancellation_observed.store(
                        crate::application::finish_owned_file_operation(),
                        Ordering::SeqCst,
                    );
                    finished.notify_one();
                    Ok::<(), ApiError>(())
                })
                .await
            }
        });
        started.notified().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(path.exists());

        let (released, wake) = &*release;
        *released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        wake.notify_all();
        tokio::time::timeout(Duration::from_secs(1), finished.notified())
            .await
            .unwrap();
        tokio::task::yield_now().await;
        assert!(cancellation_observed.load(Ordering::SeqCst));
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn disconnected_inference_waiter_never_dispatches() {
        let resources = ResourcePolicy::new(1, 1, 1);
        let global_permit = resources.acquire_inference().await;
        let model = AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap();
        let cache = ModelCache::new(
            "asr",
            vec![model.clone()],
            Duration::from_mins(1),
            1,
            PathBuf::from("models"),
        );
        let lease = cache
            .get_or_load(model, |_, _| async { Ok(()) })
            .await
            .unwrap()
            .unwrap();
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let queued = Arc::new(Notify::new());
        let dispatched = Arc::new(AtomicBool::new(false));
        let waiter = tokio::spawn({
            let resources = resources.clone();
            let queued = Arc::clone(&queued);
            let dispatched = Arc::clone(&dispatched);
            async move {
                run_owned(async move {
                    let _file = file;
                    queued.notify_one();
                    lease
                        .run_with_inference(resources.inference_limiter(), move |_| async move {
                            dispatched.store(true, Ordering::SeqCst);
                        })
                        .await
                        .unwrap();
                    Ok::<(), ApiError>(())
                })
                .await
            }
        });
        queued.notified().await;
        tokio::task::yield_now().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        tokio::task::yield_now().await;
        assert!(!path.exists());

        drop(global_permit);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!dispatched.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancelled_owned_request_keeps_temporary_files_after_dispatch() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let finished = Arc::new(Notify::new());
        let waiter = tokio::spawn({
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let finished = Arc::clone(&finished);
            async move {
                run_owned(async move {
                    let _file = file;
                    mark_owned_operation_dispatched();
                    started.notify_one();
                    release.notified().await;
                    finished.notify_one();
                    Ok::<(), ApiError>(())
                })
                .await
            }
        });
        started.notified().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(path.exists());

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), finished.notified())
            .await
            .unwrap();
        tokio::task::yield_now().await;
        assert!(!path.exists());
    }
}
