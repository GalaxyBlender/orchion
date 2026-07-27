use orchion_core::{OrchionError, Result};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

const MODEL_LOCK_DIR: &str = ".orchion-download-locks";
const PUBLICATION_LOCK_FILE: &str = ".orchion-publish.lock";

pub(crate) struct CacheLock(std::fs::File);

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

pub(crate) async fn acquire_model_lock(cache_dir: &Path, model_key: &str) -> Result<CacheLock> {
    let lock_dir = cache_dir.join(MODEL_LOCK_DIR);
    tokio::fs::create_dir_all(&lock_dir)
        .await
        .map_err(|error| lock_error(model_key, &error))?;
    let digest = Sha256::digest(model_key.as_bytes());
    acquire_lock(
        lock_dir.join(format!("{digest:x}.lock")),
        model_key.to_string(),
    )
    .await
}

pub(crate) async fn acquire_publication_lock(
    cache_dir: &Path,
    model_key: &str,
) -> Result<CacheLock> {
    acquire_lock(cache_dir.join(PUBLICATION_LOCK_FILE), model_key.to_string()).await
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

fn lock_error(model_key: &str, error: &std::io::Error) -> OrchionError {
    OrchionError::Download {
        source_name: "cache",
        repo: model_key.to_string(),
        message: error.to_string(),
    }
}
