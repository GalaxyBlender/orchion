use crate::api::openai::ApiError;
use crate::application::ServerApplication;
use crate::application::resource_policy::InferenceGuard;
use axum::extract::multipart::Field;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue};
use std::future::Future;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};
use tokio::io::AsyncWriteExt;

pub(super) async fn run_inference_owned<T, F, P>(permit: P, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ApiError>> + Send + 'static,
    P: Future<Output = InferenceGuard> + Send,
{
    let permit = permit.await;
    tokio::spawn(async move {
        let _permit = permit;
        operation.await
    })
    .await
    .map_err(|error| ApiError::internal(format!("inference task failed: {error:#}")))?
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
    use crate::application::resource_policy::ResourcePolicy;
    use std::sync::Arc;
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
}
