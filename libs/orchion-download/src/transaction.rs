use orchion_core::{OrchionError, Result};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

pub(crate) struct CacheLock(std::fs::File);

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

pub(crate) async fn acquire_model_lock(cache_dir: &Path, model_key: &str) -> Result<CacheLock> {
    let lock_dir = super::cache_state_path(cache_dir, super::MODEL_LOCK_DIR);
    tokio::fs::create_dir_all(&lock_dir)
        .await
        .map_err(|error| lock_error(model_key, &error))?;
    acquire_lock(
        lock_dir.join(format!("{}.lock", model_key_digest(model_key))),
        model_key.to_string(),
    )
    .await
}

pub(crate) fn acquire_model_lock_sync(cache_dir: &Path, model_key: &str) -> Result<CacheLock> {
    let lock_dir = super::cache_state_path(cache_dir, super::MODEL_LOCK_DIR);
    std::fs::create_dir_all(&lock_dir).map_err(|error| lock_error(model_key, &error))?;
    acquire_lock_sync(
        lock_dir.join(format!("{}.lock", model_key_digest(model_key))),
        model_key,
    )
}

pub(crate) fn model_staging_prefix(model_key: &str) -> String {
    format!("{}-", model_key_digest(model_key))
}

fn model_key_digest(model_key: &str) -> String {
    super::encode_hex(&Sha256::digest(model_key.as_bytes()))
}

pub(crate) async fn acquire_publication_lock(
    cache_dir: &Path,
    model_key: &str,
) -> Result<CacheLock> {
    acquire_lock(
        super::cache_state_path(cache_dir, super::PUBLICATION_LOCK_FILE),
        model_key.to_string(),
    )
    .await
}

#[cfg(test)]
pub(crate) fn acquire_publication_lock_sync(
    cache_dir: &Path,
    model_key: &str,
) -> Result<CacheLock> {
    acquire_lock_sync(
        super::cache_state_path(cache_dir, super::PUBLICATION_LOCK_FILE),
        model_key,
    )
}

async fn acquire_lock(lock_path: PathBuf, model_key: String) -> Result<CacheLock> {
    tokio::task::spawn_blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(|error| lock_error(&model_key, &error))?;
        fs2::FileExt::lock_exclusive(&file).map_err(|error| lock_error(&model_key, &error))?;
        Ok(CacheLock(file))
    })
    .await
    .map_err(|error| OrchionError::BlockingTask {
        message: error.to_string(),
    })?
}

fn acquire_lock_sync(lock_path: PathBuf, model_key: &str) -> Result<CacheLock> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(|error| lock_error(model_key, &error))?;
    fs2::FileExt::lock_exclusive(&file).map_err(|error| lock_error(model_key, &error))?;
    Ok(CacheLock(file))
}

fn lock_error(model_key: &str, error: &std::io::Error) -> OrchionError {
    OrchionError::Download {
        source_name: "cache",
        repo: model_key.to_string(),
        message: error.to_string(),
    }
}
