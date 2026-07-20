use orchion_core::{DownloadFailure, ModelCategory, ModelId, ModelSpec, OrchionError, Result};
use std::fs::OpenOptions;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const READY_MANIFEST_FILE: &str = ".orchion-ready.json";
const READY_MANIFEST_SCHEMA_VERSION: u64 = 1;
const READY_MANIFEST_LAYOUT: &str = "model-hub-native";
const DOWNLOAD_LOCK_FILE: &str = ".orchion-download.lock";
const PUBLISH_TRANSACTION_DIR: &str = ".orchion-publish-transaction";
const PUBLISH_TRANSACTION_MANIFEST: &str = "manifest.json";
const PUBLISH_TRANSACTION_COMMITTED: &str = "committed";

struct CacheDownloadLock(std::fs::File);

impl Drop for CacheDownloadLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadSource {
    Auto,
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DownloadEnv {
    orchion_model_source: Option<String>,
    hf_endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedSource {
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ModelHubAssetKind {
    RequiredFile,
    PaddleOcrDictionary { output_file: &'static str },
    ModelScopeFile { output_file: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ModelHubAsset {
    repo: &'static str,
    file: &'static str,
    kind: ModelHubAssetKind,
}

const PP_OCRV5_MOBILE_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_det.onnx",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_det.onnx",
        },
    },
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_rec.onnx",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_rec.onnx",
        },
    },
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "ppocrv5_dict.txt",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "ppocrv5_dict.txt",
        },
    },
];

const PP_OCRV5_SERVER_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv5_dict.txt",
        },
    },
];

const PP_OCRV6_TINY_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_tiny_dict.txt",
        },
    },
];

const PP_OCRV6_SMALL_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_dict.txt",
        },
    },
];

const PP_OCRV6_MEDIUM_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_dict.txt",
        },
    },
];

const PP_DOCLAYOUTV3_ASSETS: &[ModelHubAsset] = &[ModelHubAsset {
    repo: "PaddlePaddle/PP-DocLayoutV3_onnx",
    file: "inference.onnx",
    kind: ModelHubAssetKind::RequiredFile,
}];

impl ResolvedSource {
    const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }
}

impl DownloadEnv {
    fn current() -> Self {
        Self {
            orchion_model_source: std::env::var("ORCHION_MODEL_SOURCE").ok(),
            hf_endpoint: std::env::var("HF_ENDPOINT").ok(),
        }
    }
}

fn resolve_source(source: DownloadSource, env: &DownloadEnv) -> Result<Vec<ResolvedSource>> {
    if let Some(value) = env.orchion_model_source.as_deref() {
        return match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(vec![
                ResolvedSource::HuggingFace,
                ResolvedSource::ModelScope,
            ]),
            "huggingface" | "hf" => Ok(vec![ResolvedSource::HuggingFace]),
            "modelscope" | "ms" => Ok(vec![ResolvedSource::ModelScope]),
            _ => Err(OrchionError::InvalidModelSource {
                value: value.to_string(),
            }),
        };
    }

    match source {
        DownloadSource::Auto => Ok(vec![
            ResolvedSource::HuggingFace,
            ResolvedSource::ModelScope,
        ]),
        DownloadSource::HuggingFace => Ok(vec![ResolvedSource::HuggingFace]),
        DownloadSource::ModelScope => Ok(vec![ResolvedSource::ModelScope]),
    }
}

#[derive(Debug, Clone)]
pub struct ModelDownloader {
    source: DownloadSource,
    huggingface_available: Arc<tokio::sync::OnceCell<bool>>,
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new(DownloadSource::Auto)
    }
}

impl ModelDownloader {
    pub fn new(source: DownloadSource) -> Self {
        Self {
            source,
            huggingface_available: Arc::new(tokio::sync::OnceCell::const_new()),
        }
    }

    pub async fn download<M: ModelSpec>(
        &self,
        model: M,
        cache_dir: impl AsRef<Path>,
    ) -> Result<PathBuf> {
        let env = DownloadEnv::current();
        self.download_with_client_and_probe(
            model,
            cache_dir,
            &LibraryDownloadClient,
            &HttpSourceProbe,
            &env,
        )
        .await
    }

    #[cfg(test)]
    async fn download_with_client<M: ModelSpec, C: DownloadClient>(
        &self,
        model: M,
        cache_dir: impl AsRef<Path>,
        client: &C,
        env: &DownloadEnv,
    ) -> Result<PathBuf> {
        self.download_with_client_and_probe(model, cache_dir, client, &AlwaysAvailableProbe, env)
            .await
    }

    async fn download_with_client_and_probe<M: ModelSpec, C: DownloadClient, P: SourceProbe>(
        &self,
        model: M,
        cache_dir: impl AsRef<Path>,
        client: &C,
        probe: &P,
        env: &DownloadEnv,
    ) -> Result<PathBuf> {
        let cache_dir = cache_dir.as_ref();
        validate_repo_id(model.huggingface_repo())?;
        validate_repo_id(model.modelscope_repo())?;
        for asset in model_hub_assets(&model) {
            validate_repo_id(asset.repo)?;
        }
        let target = validated_model_cache_path(&model, cache_dir)?;

        tokio::fs::create_dir_all(cache_dir)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: error.to_string(),
            })?;
        let _cache_lock = acquire_cache_download_lock(cache_dir, model.huggingface_repo()).await?;
        for repo in std::iter::once(model.huggingface_repo())
            .chain(model_hub_assets(&model).iter().map(|asset| asset.repo))
        {
            validate_repo_cache_ancestors(cache_dir, repo)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: "cache",
                    repo: repo.to_string(),
                    message: error.to_string(),
                })?;
        }
        let publication_clean =
            recover_interrupted_publication(cache_dir)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: "cache",
                    repo: model.huggingface_repo().to_string(),
                    message: format!("failed to recover interrupted cache publication: {error}"),
                })?;

        if is_ready_cache(&model, &target).await? {
            tracing::debug!(model = ?model, path = %target.display(), "model cache ready");
            return Ok(target);
        }
        if !publication_clean {
            return Err(OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: "a committed cache publication is awaiting cleanup; refusing to replace its recovery data"
                    .to_string(),
            });
        }

        if !uses_hub_download(&model) {
            unreachable!("direct asset downloads are not implemented yet");
        }

        let assets = model_hub_assets(&model);
        let candidates = if uses_modelscope_file_assets(assets) {
            vec![ResolvedSource::ModelScope]
        } else {
            self.resolve_candidates(env, probe).await?
        };
        tracing::info!(
            model = ?model,
            path = %target.display(),
            source_count = candidates.len(),
            "ensuring model cache is available"
        );
        let mut failures = Vec::new();
        for candidate in candidates {
            let staging = tempfile::Builder::new()
                .prefix(".orchion-download-")
                .tempdir_in(cache_dir)
                .map_err(|error| OrchionError::Download {
                    source_name: candidate.label(),
                    repo: model.huggingface_repo().to_string(),
                    message: error.to_string(),
                })?;
            let staging_root = staging.path();
            let staging_target = validated_model_cache_path(&model, staging_root)?;
            if !assets.is_empty() {
                match download_hub_assets(
                    &model,
                    candidate,
                    assets,
                    staging_root,
                    &staging_target,
                    client,
                    env,
                )
                .await
                {
                    Ok(()) => {
                        prepare_cached_model(&model, &staging_target, candidate.label()).await?;
                        ensure_ready_cache_files(&model, &staging_target, candidate.label())
                            .await?;
                        write_ready_manifest(&model, &staging_target, candidate.label()).await?;
                        publish_staged_cache(
                            &model,
                            assets,
                            staging_root,
                            cache_dir,
                            candidate.label(),
                        )
                        .await?;
                        tracing::info!(
                            source = candidate.label(),
                            path = %target.display(),
                            "model asset download completed"
                        );
                        return Ok(target);
                    }
                    Err(error) => {
                        tracing::warn!(
                            source = candidate.label(),
                            path = %target.display(),
                            error = %error,
                            "model asset download failed"
                        );
                        failures.push(DownloadFailure {
                            source_name: candidate.label(),
                            message: error.to_string(),
                        });
                    }
                }
                continue;
            }

            let repo = match candidate {
                ResolvedSource::HuggingFace => model.huggingface_repo(),
                ResolvedSource::ModelScope => model.modelscope_repo(),
            };
            tracing::info!(
                source = candidate.label(),
                repo,
                path = %target.display(),
                "downloading model"
            );
            match client
                .download(candidate, repo, staging_root, &staging_target, None, env)
                .await
            {
                Ok(()) => {
                    prepare_cached_model(&model, &staging_target, candidate.label()).await?;
                    ensure_ready_cache_files(&model, &staging_target, candidate.label()).await?;
                    write_ready_manifest(&model, &staging_target, candidate.label()).await?;
                    publish_staged_cache(
                        &model,
                        assets,
                        staging_root,
                        cache_dir,
                        candidate.label(),
                    )
                    .await?;
                    tracing::info!(
                        source = candidate.label(),
                        repo,
                        path = %target.display(),
                        "model download completed"
                    );
                    return Ok(target);
                }
                Err(error) => {
                    tracing::warn!(
                        source = candidate.label(),
                        repo,
                        path = %target.display(),
                        error = %error,
                        "model download failed"
                    );
                    failures.push(DownloadFailure {
                        source_name: candidate.label(),
                        message: error.to_string(),
                    });
                }
            }
        }

        Err(OrchionError::DownloadFallbackExhausted {
            repo: model.huggingface_repo().to_string(),
            failures,
        })
    }

    async fn resolve_candidates<P: SourceProbe>(
        &self,
        env: &DownloadEnv,
        probe: &P,
    ) -> Result<Vec<ResolvedSource>> {
        let candidates = resolve_source(self.source, env)?;
        if self.source != DownloadSource::Auto || env.orchion_model_source.is_some() {
            return Ok(candidates);
        }
        if *self
            .huggingface_available
            .get_or_init(|| probe.huggingface_available(env))
            .await
        {
            Ok(candidates)
        } else {
            tracing::warn!("huggingface unavailable; using modelscope download source");
            Ok(candidates
                .into_iter()
                .filter(|source| *source != ResolvedSource::HuggingFace)
                .collect())
        }
    }
}

fn validate_repo_id(repo: &str) -> Result<()> {
    ModelId::parse(repo).map_err(|error| OrchionError::Download {
        source_name: "cache",
        repo: repo.to_string(),
        message: error.to_string(),
    })?;
    if repo
        .split('/')
        .next()
        .is_some_and(|segment| segment.to_ascii_lowercase().starts_with(".orchion-"))
    {
        return Err(OrchionError::Download {
            source_name: "cache",
            repo: repo.to_string(),
            message: "repository uses the reserved `.orchion-` cache namespace".to_string(),
        });
    }
    Ok(())
}

fn validated_model_cache_path<M: ModelSpec>(model: &M, cache_dir: &Path) -> Result<PathBuf> {
    let path = model.cache_path(cache_dir);
    let expected = repo_cache_path(cache_dir, model.huggingface_repo());
    if path != expected {
        return Err(OrchionError::Download {
            source_name: "cache",
            repo: model.huggingface_repo().to_string(),
            message: format!(
                "model cache path `{}` must match validated repository path `{}`",
                path.display(),
                expected.display()
            ),
        });
    }
    Ok(path)
}

async fn acquire_cache_download_lock(cache_dir: &Path, repo: &str) -> Result<CacheDownloadLock> {
    let lock_path = cache_dir.join(DOWNLOAD_LOCK_FILE);
    let repo = repo.to_string();
    tokio::task::spawn_blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: repo.clone(),
                message: error.to_string(),
            })?;
        fs2::FileExt::lock_exclusive(&file).map_err(|error| OrchionError::Download {
            source_name: "cache",
            repo,
            message: error.to_string(),
        })?;
        Ok(CacheDownloadLock(file))
    })
    .await
    .map_err(|error| OrchionError::BlockingTask {
        message: error.to_string(),
    })?
}

async fn publish_staged_cache<M: ModelSpec>(
    model: &M,
    assets: &[ModelHubAsset],
    staging_root: &Path,
    cache_dir: &Path,
    source_name: &'static str,
) -> Result<()> {
    let mut repos = Vec::new();
    for asset in assets {
        if asset.repo != model.huggingface_repo() && !repos.iter().any(|repo| repo == asset.repo) {
            repos.push(asset.repo.to_string());
        }
    }
    repos.push(model.huggingface_repo().to_string());
    publish_staged_repositories(staging_root, cache_dir, &repos)
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })
}

async fn publish_staged_repositories(
    staging_root: &Path,
    cache_dir: &Path,
    repos: &[String],
) -> std::io::Result<()> {
    if !recover_interrupted_publication(cache_dir).await? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::ResourceBusy,
            "a committed cache publication is awaiting cleanup",
        ));
    }
    let transaction_dir = cache_dir.join(PUBLISH_TRANSACTION_DIR);
    tokio::fs::create_dir_all(&transaction_dir).await?;

    let mut entries = Vec::with_capacity(repos.len());
    for repo in repos {
        if ModelId::parse(repo).is_err() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid repository id in publication transaction: {repo}"),
            ));
        }
        validate_repo_cache_ancestors(cache_dir, repo).await?;
        let target = repo_cache_path(cache_dir, repo);
        let had_target = path_exists(&target).await?;
        entries.push(serde_json::json!({"repo": repo, "had_target": had_target}));
    }
    let manifest = serde_json::to_vec(&serde_json::json!({"repos": entries}))
        .map_err(std::io::Error::other)?;
    let manifest_temp = transaction_dir.join("manifest.tmp");
    write_synced_file(&manifest_temp, manifest).await?;
    tokio::fs::rename(
        &manifest_temp,
        transaction_dir.join(PUBLISH_TRANSACTION_MANIFEST),
    )
    .await?;
    sync_directory(&transaction_dir).await?;
    sync_directory(cache_dir).await?;

    let publish_result = async {
        for repo in repos {
            let target = repo_cache_path(cache_dir, repo);
            if path_exists(&target).await? {
                let backup = repo_cache_path(&transaction_dir, repo);
                let parent = backup.parent().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "publication backup path has no parent",
                    )
                })?;
                tokio::fs::create_dir_all(parent).await?;
                tokio::fs::rename(&target, &backup).await?;
                sync_directory(parent).await?;
                sync_directory(&transaction_dir).await?;
                if let Some(target_parent) = target.parent() {
                    sync_directory(target_parent).await?;
                }
                sync_directory(cache_dir).await?;
            }
        }
        for repo in repos {
            let staged = repo_cache_path(staging_root, repo);
            let target = repo_cache_path(cache_dir, repo);
            let parent = target.parent().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "model cache target has no parent",
                )
            })?;
            tokio::fs::create_dir_all(parent).await?;
            let staged_parent = staged.parent().map(Path::to_path_buf);
            tokio::fs::rename(staged, &target).await?;
            sync_directory(parent).await?;
            sync_directory(cache_dir).await?;
            if let Some(staged_parent) = staged_parent {
                sync_directory(&staged_parent).await?;
            }
        }
        let commit_temp = transaction_dir.join("committed.tmp");
        write_synced_file(&commit_temp, b"committed\n".to_vec()).await?;
        tokio::fs::rename(
            commit_temp,
            transaction_dir.join(PUBLISH_TRANSACTION_COMMITTED),
        )
        .await?;
        sync_directory(&transaction_dir).await
    }
    .await;

    if let Err(error) = publish_result {
        return match recover_interrupted_publication(cache_dir).await {
            Ok(true) => Err(error),
            Ok(false) => Err(std::io::Error::other(format!(
                "cache publication failed: {error}; rollback was not completed"
            ))),
            Err(rollback_error) => Err(std::io::Error::other(format!(
                "cache publication failed: {error}; rollback failed: {rollback_error}"
            ))),
        };
    }

    if let Err(error) = tokio::fs::remove_dir_all(&transaction_dir).await {
        tracing::warn!(
            path = %transaction_dir.display(),
            %error,
            "committed cache publication cleanup deferred"
        );
    } else {
        sync_directory(cache_dir).await?;
    }
    Ok(())
}

async fn recover_interrupted_publication(cache_dir: &Path) -> std::io::Result<bool> {
    let transaction_dir = cache_dir.join(PUBLISH_TRANSACTION_DIR);
    if !path_exists(&transaction_dir).await? {
        return Ok(true);
    }
    if tokio::fs::read(transaction_dir.join(PUBLISH_TRANSACTION_COMMITTED))
        .await
        .is_ok_and(|marker| marker == b"committed\n")
    {
        if let Err(error) = tokio::fs::remove_dir_all(&transaction_dir).await {
            tracing::warn!(
                path = %transaction_dir.display(),
                %error,
                "committed cache publication cleanup remains deferred"
            );
            return Ok(false);
        }
        sync_directory(cache_dir).await?;
        return Ok(true);
    }

    let manifest_path = transaction_dir.join(PUBLISH_TRANSACTION_MANIFEST);
    let manifest = match tokio::fs::read(&manifest_path).await {
        Ok(manifest) => manifest,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tokio::fs::remove_dir_all(transaction_dir).await?;
            sync_directory(cache_dir).await?;
            return Ok(true);
        }
        Err(error) => return Err(error),
    };
    let manifest: serde_json::Value = serde_json::from_slice(&manifest)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let entries = manifest["repos"].as_array().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "publication transaction manifest has no repos",
        )
    })?;
    for entry in entries {
        let repo = entry["repo"].as_str().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "publication transaction repo is invalid",
            )
        })?;
        if ModelId::parse(repo).is_err()
            || repo
                .split('/')
                .next()
                .is_some_and(|segment| segment.to_ascii_lowercase().starts_with(".orchion-"))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("publication transaction repo is unsafe: {repo}"),
            ));
        }
        let had_target = entry["had_target"].as_bool().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "publication transaction target state is invalid",
            )
        })?;
        validate_repo_cache_ancestors(cache_dir, repo).await?;
        let target = repo_cache_path(cache_dir, repo);
        let backup = repo_cache_path(&transaction_dir, repo);
        if had_target {
            if path_exists(&backup).await? {
                remove_cache_entry(&target).await?;
                let parent = target.parent().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "model cache target has no parent",
                    )
                })?;
                tokio::fs::create_dir_all(parent).await?;
                tokio::fs::rename(backup, &target).await?;
                sync_directory(parent).await?;
            }
        } else {
            remove_cache_entry(&target).await?;
            if let Some(parent) = target.parent() {
                sync_directory(parent).await?;
            }
        }
    }
    tokio::fs::remove_dir_all(transaction_dir).await?;
    sync_directory(cache_dir).await?;
    Ok(true)
}

async fn path_exists(path: &Path) -> std::io::Result<bool> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

async fn validate_repo_cache_ancestors(cache_dir: &Path, repo: &str) -> std::io::Result<()> {
    let segments = repo.split('/').collect::<Vec<_>>();
    let mut path = cache_dir.to_path_buf();
    for segment in segments.iter().take(segments.len().saturating_sub(1)) {
        path.push(segment);
        match tokio::fs::symlink_metadata(&path).await {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("model cache ancestor is a symlink: {}", path.display()),
                ));
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotADirectory,
                    format!(
                        "model cache ancestor is not a directory: {}",
                        path.display()
                    ),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

async fn remove_cache_entry(path: &Path) -> std::io::Result<()> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            tokio::fs::remove_dir_all(path).await
        }
        Ok(_) => tokio::fs::remove_file(path).await,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

async fn write_synced_file(path: &Path, bytes: Vec<u8>) -> std::io::Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        file.sync_all()
    })
    .await
    .map_err(std::io::Error::other)?
}

#[cfg(unix)]
async fn sync_directory(path: &Path) -> std::io::Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || std::fs::File::open(path)?.sync_all())
        .await
        .map_err(std::io::Error::other)?
}

#[cfg(not(unix))]
async fn sync_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

async fn is_ready_cache<M: ModelSpec>(model: &M, target: &Path) -> Result<bool> {
    let manifest = match tokio::fs::read_to_string(target.join(READY_MANIFEST_FILE)).await {
        Ok(manifest) => manifest,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: error.to_string(),
            });
        }
    };
    let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest) else {
        return Ok(false);
    };
    if manifest["schema_version"].as_u64() != Some(READY_MANIFEST_SCHEMA_VERSION)
        || manifest["repo_id"].as_str() != Some(model.huggingface_repo())
        || manifest["layout"].as_str() != Some(READY_MANIFEST_LAYOUT)
    {
        return Ok(false);
    }

    required_cache_files_exist(model, target).await
}

async fn required_cache_files_exist<M: ModelSpec>(model: &M, target: &Path) -> Result<bool> {
    for file_name in model.required_files() {
        if !cache_file_exists(model, target, file_name).await? {
            return Ok(false);
        }
    }
    if model.category() == ModelCategory::OcrVl && !ocr_vl_weight_files_exist(model, target).await?
    {
        return Ok(false);
    }
    if !hub_asset_files_exist(model, target).await? {
        return Ok(false);
    }
    Ok(true)
}

async fn hub_asset_files_exist<M: ModelSpec>(model: &M, target: &Path) -> Result<bool> {
    let Some(cache_dir) = cache_root_from_target(target) else {
        return Ok(false);
    };
    for asset in model_hub_assets(model) {
        let asset_path = match asset.kind {
            ModelHubAssetKind::RequiredFile | ModelHubAssetKind::PaddleOcrDictionary { .. } => {
                repo_cache_path(cache_dir, asset.repo).join(asset.file)
            }
            ModelHubAssetKind::ModelScopeFile { .. } => {
                repo_cache_path(cache_dir, asset.repo).join(asset.file)
            }
        };
        if !tokio::fs::try_exists(&asset_path)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: asset.repo.to_string(),
                message: error.to_string(),
            })?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn cache_root_from_target(target: &Path) -> Option<&Path> {
    target.parent().and_then(Path::parent)
}

fn model_hub_assets<M: ModelSpec>(model: &M) -> &'static [ModelHubAsset] {
    match model.huggingface_repo() {
        "PaddlePaddle/PP-OCRv5_mobile" => PP_OCRV5_MOBILE_ASSETS,
        "PaddlePaddle/PP-OCRv5_server" => PP_OCRV5_SERVER_ASSETS,
        "PaddlePaddle/PP-OCRv6_tiny" => PP_OCRV6_TINY_ASSETS,
        "PaddlePaddle/PP-OCRv6_small" => PP_OCRV6_SMALL_ASSETS,
        "PaddlePaddle/PP-OCRv6_medium" => PP_OCRV6_MEDIUM_ASSETS,
        "PaddlePaddle/PP-DocLayoutV3" => PP_DOCLAYOUTV3_ASSETS,
        _ => &[],
    }
}

fn repo_cache_path(cache_dir: &Path, repo: &str) -> PathBuf {
    repo.split('/')
        .fold(cache_dir.to_path_buf(), |path, segment| path.join(segment))
}

async fn download_hub_assets<M: ModelSpec, C: DownloadClient>(
    model: &M,
    source: ResolvedSource,
    assets: &[ModelHubAsset],
    cache_dir: &Path,
    target: &Path,
    client: &C,
    env: &DownloadEnv,
) -> Result<()> {
    tokio::fs::create_dir_all(target)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })?;

    let mut downloaded_repos = Vec::new();
    for asset in assets {
        if downloaded_repos.contains(&asset.repo) {
            continue;
        }
        let repo_target = repo_cache_path(cache_dir, asset.repo);
        let repo_files = asset_files_for_repo(assets, asset.repo);
        tracing::info!(
            source = source.label(),
            repo = asset.repo,
            path = %repo_target.display(),
            "downloading model asset repo"
        );
        client
            .download(
                source,
                asset.repo,
                cache_dir,
                &repo_target,
                Some(&repo_files),
                env,
            )
            .await?;
        downloaded_repos.push(asset.repo);
    }

    for asset in assets {
        let source_path = repo_cache_path(cache_dir, asset.repo).join(asset.file);
        match asset.kind {
            ModelHubAssetKind::RequiredFile => {
                ensure_asset_file_exists(source, asset.repo, &source_path).await?;
            }
            ModelHubAssetKind::PaddleOcrDictionary { output_file } => {
                let dictionary =
                    build_paddle_ocr_dictionary(source, asset.repo, &source_path).await?;
                tokio::fs::write(target.join(output_file), dictionary)
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name: source.label(),
                        repo: asset.repo.to_string(),
                        message: error.to_string(),
                    })?;
            }
            ModelHubAssetKind::ModelScopeFile { output_file } => {
                if source != ResolvedSource::ModelScope {
                    return Err(OrchionError::Download {
                        source_name: source.label(),
                        repo: asset.repo.to_string(),
                        message: "asset is only available from ModelScope".to_string(),
                    });
                }
                ensure_asset_file_exists(source, asset.repo, &source_path).await?;
                let _ = output_file;
            }
        }
    }
    Ok(())
}

fn uses_modelscope_file_assets(assets: &[ModelHubAsset]) -> bool {
    assets
        .iter()
        .any(|asset| matches!(asset.kind, ModelHubAssetKind::ModelScopeFile { .. }))
}

fn asset_files_for_repo(assets: &[ModelHubAsset], repo: &'static str) -> Vec<&'static str> {
    let mut files = Vec::new();
    for asset in assets.iter().filter(|asset| asset.repo == repo) {
        if !files.contains(&asset.file) {
            files.push(asset.file);
        }
    }
    files
}

async fn ensure_ready_cache_files<M: ModelSpec>(
    model: &M,
    target: &Path,
    source_name: &'static str,
) -> Result<()> {
    if required_cache_files_exist(model, target).await? {
        return Ok(());
    }
    Err(OrchionError::Download {
        source_name,
        repo: model.huggingface_repo().to_string(),
        message: "download completed without all required cache files".to_string(),
    })
}

async fn ensure_asset_file_exists(
    source: ResolvedSource,
    repo: &'static str,
    path: &Path,
) -> Result<()> {
    if tokio::fs::try_exists(path)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo: repo.to_string(),
            message: error.to_string(),
        })?
    {
        return Ok(());
    }
    Err(OrchionError::Download {
        source_name: source.label(),
        repo: repo.to_string(),
        message: format!("missing required model asset `{}`", path.display()),
    })
}

async fn build_paddle_ocr_dictionary(
    source: ResolvedSource,
    repo: &'static str,
    path: &Path,
) -> Result<String> {
    let yaml = tokio::fs::read_to_string(path)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo: repo.to_string(),
            message: error.to_string(),
        })?;
    let characters =
        parse_paddle_ocr_character_dict(&yaml).ok_or_else(|| OrchionError::Download {
            source_name: source.label(),
            repo: repo.to_string(),
            message: format!("missing character_dict in `{}`", path.display()),
        })?;
    Ok(format!("{}\n", characters.join("\n")))
}

fn parse_paddle_ocr_character_dict(yaml: &str) -> Option<Vec<String>> {
    let mut entries = Vec::new();
    let mut in_character_dict = false;
    let mut list_indent = None;
    for line in yaml.lines() {
        let content = line.trim_start();
        if !in_character_dict {
            if content.trim_end() == "character_dict:" {
                in_character_dict = true;
            }
            continue;
        }

        if content.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let Some(value) = content.strip_prefix("- ") else {
            if !entries.is_empty() && list_indent.is_some_and(|current| indent <= current) {
                break;
            }
            continue;
        };
        let current_indent = *list_indent.get_or_insert(indent);
        if indent < current_indent {
            break;
        }
        entries.push(parse_yaml_scalar(value));
    }
    (!entries.is_empty()).then_some(entries)
}

fn parse_yaml_scalar(value: &str) -> String {
    if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
        return value[1..value.len() - 1].replace("''", "'");
    }
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        let mut parsed = String::new();
        let mut chars = value[1..value.len() - 1].chars();
        while let Some(character) = chars.next() {
            if character == '\\' {
                if let Some(escaped) = chars.next() {
                    parsed.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        other => other,
                    });
                }
            } else {
                parsed.push(character);
            }
        }
        return parsed;
    }
    value.to_string()
}

async fn cache_file_exists<M: ModelSpec>(
    model: &M,
    target: &Path,
    file_name: &str,
) -> Result<bool> {
    tokio::fs::try_exists(target.join(file_name))
        .await
        .map_err(|error| OrchionError::Download {
            source_name: "cache",
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })
}

async fn ocr_vl_weight_files_exist<M: ModelSpec>(model: &M, target: &Path) -> Result<bool> {
    let index = match tokio::fs::read_to_string(target.join("model.safetensors.index.json")).await {
        Ok(index) => index,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return cache_file_is_nonempty(model, target, "model.safetensors").await;
        }
        Err(error) => {
            return Err(OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: error.to_string(),
            });
        }
    };
    let Ok(index) = serde_json::from_str::<serde_json::Value>(&index) else {
        return Ok(false);
    };
    let Some(weight_map) = index["weight_map"].as_object() else {
        return Ok(false);
    };
    let mut weight_files = Vec::new();
    for file_name in weight_map.values() {
        let Some(file_name) = file_name.as_str() else {
            return Ok(false);
        };
        let path = Path::new(file_name);
        if path.is_absolute()
            || !path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
        {
            return Ok(false);
        }
        if !weight_files.contains(&file_name) {
            weight_files.push(file_name);
        }
    }
    if weight_files.is_empty() {
        return Ok(false);
    }
    for file_name in weight_files {
        if !cache_file_is_nonempty(model, target, file_name).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn cache_file_is_nonempty<M: ModelSpec>(
    model: &M,
    target: &Path,
    file_name: &str,
) -> Result<bool> {
    match tokio::fs::metadata(target.join(file_name)).await {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(OrchionError::Download {
            source_name: "cache",
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        }),
    }
}

async fn write_ready_manifest<M: ModelSpec>(
    model: &M,
    target: &Path,
    source_name: &'static str,
) -> Result<()> {
    let manifest = serde_json::json!({
        "schema_version": READY_MANIFEST_SCHEMA_VERSION,
        "repo_id": model.huggingface_repo(),
        "layout": READY_MANIFEST_LAYOUT,
    });
    let tmp = target.join(format!("{READY_MANIFEST_FILE}.tmp"));
    tokio::fs::write(&tmp, manifest.to_string())
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })?;
    tokio::fs::rename(&tmp, target.join(READY_MANIFEST_FILE))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })
}

fn uses_hub_download<M: ModelSpec>(_model: &M) -> bool {
    true
}

trait SourceProbe {
    fn huggingface_available<'a>(&'a self, env: &'a DownloadEnv) -> BoxFuture<'a, bool>;
}

struct HttpSourceProbe;

const HUGGINGFACE_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

impl HttpSourceProbe {
    const fn timeout(&self) -> Duration {
        HUGGINGFACE_PROBE_TIMEOUT
    }
}

impl SourceProbe for HttpSourceProbe {
    fn huggingface_available<'a>(&'a self, env: &'a DownloadEnv) -> BoxFuture<'a, bool> {
        Box::pin(async move {
            let endpoint = env
                .hf_endpoint
                .as_deref()
                .unwrap_or("https://huggingface.co")
                .trim_end_matches('/');
            let client = match reqwest::Client::builder().timeout(self.timeout()).build() {
                Ok(client) => client,
                Err(error) => {
                    tracing::warn!(error = %error, "failed to create huggingface probe client");
                    return false;
                }
            };
            match client.head(endpoint).send().await {
                Ok(response) => {
                    response.status().is_success() || response.status().is_redirection()
                }
                Err(error) => {
                    tracing::warn!(url = endpoint, error = %error, "huggingface HEAD probe failed");
                    false
                }
            }
        })
    }
}

#[cfg(test)]
struct AlwaysAvailableProbe;

#[cfg(test)]
impl SourceProbe for AlwaysAvailableProbe {
    fn huggingface_available<'a>(&'a self, _env: &'a DownloadEnv) -> BoxFuture<'a, bool> {
        Box::pin(async { true })
    }
}

trait DownloadClient {
    fn download<'a>(
        &'a self,
        source: ResolvedSource,
        repo: &'a str,
        cache_dir: &'a Path,
        target: &'a Path,
        files: Option<&'a [&'static str]>,
        env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<()>>;
}

struct LibraryDownloadClient;

impl DownloadClient for LibraryDownloadClient {
    fn download<'a>(
        &'a self,
        source: ResolvedSource,
        repo: &'a str,
        cache_dir: &'a Path,
        _target: &'a Path,
        files: Option<&'a [&'static str]>,
        _env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { download_model_hub(source, repo, cache_dir, files).await })
    }
}

async fn download_model_hub(
    source: ResolvedSource,
    repo: &str,
    cache_dir: &Path,
    files: Option<&[&'static str]>,
) -> Result<()> {
    let provider = match source {
        ResolvedSource::HuggingFace => model_hub::HubProvider::HuggingFace { token: None },
        ResolvedSource::ModelScope => model_hub::HubProvider::ModelScope { token: None },
    };
    let downloader =
        model_hub::ModelDownloader::new(provider).map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo: repo.to_string(),
            message: error.to_string(),
        })?;
    downloader
        .download(model_hub::DownloadOptions {
            repo_id: repo.to_string(),
            revision: None,
            save_dir: cache_dir.to_path_buf(),
            files: files.map(|files| files.iter().map(|file| (*file).to_string()).collect()),
        })
        .await
        .map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo: repo.to_string(),
            message: error.to_string(),
        })
}

async fn prepare_cached_model<M: ModelSpec>(
    model: &M,
    target: &Path,
    source_name: &'static str,
) -> Result<()> {
    match model.category() {
        ModelCategory::Asr => {
            ensure_asr_tokenizer_json(target, source_name, model.huggingface_repo()).await
        }
        ModelCategory::Tts | ModelCategory::Ocr | ModelCategory::OcrVl => Ok(()),
    }
}

async fn ensure_asr_tokenizer_json(
    target: &Path,
    source_name: &'static str,
    repo: &str,
) -> Result<()> {
    if tokio::fs::try_exists(target.join("tokenizer.json"))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: error.to_string(),
        })?
    {
        return Ok(());
    }

    let tokenizer_config =
        read_cache_file(target, "tokenizer_config.json", source_name, repo).await?;
    let vocab = read_cache_file(target, "vocab.json", source_name, repo).await?;
    let merges = read_cache_file(target, "merges.txt", source_name, repo).await?;
    let tokenizer_json = build_qwen3_asr_tokenizer_json(&vocab, &merges, &tokenizer_config)
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: format!("failed to build tokenizer.json: {error}"),
        })?;

    tokio::fs::write(target.join("tokenizer.json"), tokenizer_json)
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: error.to_string(),
        })?;
    tracing::info!(path = %target.join("tokenizer.json").display(), "rebuilt ASR tokenizer.json");
    Ok(())
}

async fn read_cache_file(
    target: &Path,
    file_name: &'static str,
    source_name: &'static str,
    repo: &str,
) -> Result<String> {
    tokio::fs::read_to_string(target.join(file_name))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: format!("missing required ASR cache file `{file_name}`: {error}"),
        })
}

fn build_qwen3_asr_tokenizer_json(
    vocab: &str,
    merges: &str,
    tokenizer_config: &str,
) -> serde_json::Result<Vec<u8>> {
    let vocab_value: serde_json::Value = serde_json::from_str(vocab)?;
    let merges: Vec<&str> = merges
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .collect();

    let tokenizer_config: serde_json::Value = serde_json::from_str(tokenizer_config)?;
    let mut added_tokens = Vec::new();
    if let Some(decoder_map) = tokenizer_config["added_tokens_decoder"].as_object() {
        let mut entries: Vec<(u64, &serde_json::Value)> = decoder_map
            .iter()
            .filter_map(|(id, value)| id.parse::<u64>().ok().map(|id| (id, value)))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        for (id, value) in entries {
            added_tokens.push(serde_json::json!({
                "id": id,
                "content": value["content"],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": value["special"]
            }));
        }
    }

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"},
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab_value,
            "merges": merges
        }
    });

    serde_json::to_vec(&tokenizer_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_overrides_to_huggingface_only() {
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };
        assert_eq!(
            resolve_source(DownloadSource::Auto, &env).unwrap(),
            vec![ResolvedSource::HuggingFace]
        );
    }

    #[test]
    fn env_overrides_to_modelscope_only() {
        let env = DownloadEnv {
            orchion_model_source: Some("modelscope".to_string()),
            hf_endpoint: None,
        };
        assert_eq!(
            resolve_source(DownloadSource::Auto, &env).unwrap(),
            vec![ResolvedSource::ModelScope]
        );
    }

    #[test]
    fn auto_tries_huggingface_then_modelscope() {
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        assert_eq!(
            resolve_source(DownloadSource::Auto, &env).unwrap(),
            vec![ResolvedSource::HuggingFace, ResolvedSource::ModelScope]
        );
    }

    #[test]
    fn invalid_env_value_is_rejected() {
        let env = DownloadEnv {
            orchion_model_source: Some("mirror".to_string()),
            hf_endpoint: None,
        };
        assert!(matches!(
            resolve_source(DownloadSource::Auto, &env),
            Err(OrchionError::InvalidModelSource { value }) if value == "mirror"
        ));
    }

    #[test]
    fn paddle_ocr_dictionary_parser_preserves_full_width_space_entry() {
        let yaml = "PostProcess:\n  character_dict:\n    - 　\n    - 一\n    - A\n";

        assert_eq!(
            parse_paddle_ocr_character_dict(yaml).unwrap(),
            vec!["　", "一", "A"]
        );
    }
}

#[cfg(test)]
mod downloader_tests {
    use super::*;
    use orchion_core::{AsrModel, KnownOcrModel, TtsModel};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct UnsafeModel;

    impl ModelSpec for UnsafeModel {
        fn category(&self) -> ModelCategory {
            ModelCategory::Asr
        }

        fn huggingface_repo(&self) -> &str {
            "../victim"
        }

        fn modelscope_repo(&self) -> &str {
            "../victim"
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct UnsafeCachePathModel;

    impl ModelSpec for UnsafeCachePathModel {
        fn category(&self) -> ModelCategory {
            ModelCategory::Asr
        }

        fn huggingface_repo(&self) -> &str {
            "Safe/Model"
        }

        fn modelscope_repo(&self) -> &str {
            "Safe/Model"
        }

        fn cache_path(&self, cache_dir: impl AsRef<Path>) -> PathBuf {
            cache_dir.as_ref().join("..").join("victim")
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ReservedCacheNamespaceModel;

    impl ModelSpec for ReservedCacheNamespaceModel {
        fn category(&self) -> ModelCategory {
            ModelCategory::Asr
        }

        fn huggingface_repo(&self) -> &str {
            ".ORCHION-publish-transaction/Model"
        }

        fn modelscope_repo(&self) -> &str {
            ".ORCHION-publish-transaction/Model"
        }
    }

    #[derive(Default)]
    struct FakeDownloadClient {
        fail_huggingface: bool,
        omit_asr_tokenizer_sources: bool,
        write_ocr_vl_weights: bool,
        delay: Duration,
        calls: Arc<Mutex<Vec<&'static str>>>,
        repos: Arc<Mutex<Vec<String>>>,
        file_filters: Arc<Mutex<Vec<Option<Vec<&'static str>>>>>,
    }

    struct FakeProbe {
        huggingface_available: bool,
        calls: Arc<Mutex<usize>>,
    }

    impl SourceProbe for FakeProbe {
        fn huggingface_available<'a>(&'a self, _env: &'a DownloadEnv) -> BoxFuture<'a, bool> {
            Box::pin(async move {
                *self.calls.lock().unwrap() += 1;
                self.huggingface_available
            })
        }
    }

    #[tokio::test]
    async fn huggingface_probe_times_out_quickly() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            if let Ok((_stream, _addr)) = listener.accept() {
                std::thread::sleep(Duration::from_secs(10));
            }
        });
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: Some(endpoint),
        };

        let available = tokio::time::timeout(
            HUGGINGFACE_PROBE_TIMEOUT + Duration::from_secs(2),
            HttpSourceProbe.huggingface_available(&env),
        )
        .await
        .expect("probe should return before the outer timeout");

        assert!(!available);
    }

    impl DownloadClient for FakeDownloadClient {
        fn download<'a>(
            &'a self,
            source: ResolvedSource,
            repo: &'a str,
            _cache_dir: &'a Path,
            target: &'a Path,
            files: Option<&'a [&'static str]>,
            _env: &'a DownloadEnv,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                tokio::time::sleep(self.delay).await;
                self.calls.lock().unwrap().push(source.label());
                self.repos.lock().unwrap().push(repo.to_string());
                self.file_filters
                    .lock()
                    .unwrap()
                    .push(files.map(|files| files.to_vec()));
                if self.fail_huggingface && source == ResolvedSource::HuggingFace {
                    tokio::fs::create_dir_all(target).await.map_err(|error| {
                        OrchionError::Download {
                            source_name: source.label(),
                            repo: repo.to_string(),
                            message: error.to_string(),
                        }
                    })?;
                    tokio::fs::write(target.join("partial.bin"), "partial")
                        .await
                        .map_err(|error| OrchionError::Download {
                            source_name: source.label(),
                            repo: repo.to_string(),
                            message: error.to_string(),
                        })?;
                    return Err(OrchionError::Download {
                        source_name: source.label(),
                        repo: repo.to_string(),
                        message: "simulated failure".to_string(),
                    });
                }
                tokio::fs::create_dir_all(target).await.map_err(|error| {
                    OrchionError::Download {
                        source_name: source.label(),
                        repo: repo.to_string(),
                        message: error.to_string(),
                    }
                })?;
                tokio::fs::write(target.join("config.json"), "{}")
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name: source.label(),
                        repo: repo.to_string(),
                        message: error.to_string(),
                    })?;
                if let Some(files) = files {
                    for file_name in files {
                        tokio::fs::write(target.join(file_name), b"asset")
                            .await
                            .map_err(|error| OrchionError::Download {
                                source_name: source.label(),
                                repo: repo.to_string(),
                                message: error.to_string(),
                            })?;
                    }
                }
                if !self.omit_asr_tokenizer_sources {
                    write_asr_tokenizer_sources(target).await;
                }
                if self.write_ocr_vl_weights {
                    write_complete_ocr_vl_cache(target).await;
                }
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn auto_falls_back_to_modelscope_when_huggingface_fails() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            fail_huggingface: true,
            omit_asr_tokenizer_sources: false,
            calls: Arc::new(Mutex::new(Vec::new())),
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::Auto);

        let path = downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert!(path.join("config.json").exists());
        assert!(path.join("tokenizer.json").exists());
        assert!(!path.join("partial.bin").exists());
        assert!(!path.join(".orchion-complete").exists());
        assert_eq!(&*calls.lock().unwrap(), &["huggingface", "modelscope"]);
    }

    #[tokio::test]
    async fn failed_download_preserves_existing_cache() {
        let dir = tempfile::tempdir().unwrap();
        let model = qwen_asr_06b();
        let target = model.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("existing.bin"), "existing")
            .await
            .unwrap();
        let client = FakeDownloadClient {
            fail_huggingface: true,
            ..Default::default()
        };
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        ModelDownloader::default()
            .download_with_client(model, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert_eq!(
            tokio::fs::read_to_string(target.join("existing.bin"))
                .await
                .unwrap(),
            "existing"
        );
        assert!(!target.join("partial.bin").exists());
    }

    #[tokio::test]
    async fn downloader_rejects_unsafe_custom_model_repository() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("models");
        let victim = dir.path().join("victim");
        tokio::fs::create_dir_all(&victim).await.unwrap();
        tokio::fs::write(victim.join("keep.bin"), "keep")
            .await
            .unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let error = ModelDownloader::default()
            .download_with_client(UnsafeModel, &cache_dir, &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("invalid model id"));
        assert_eq!(
            tokio::fs::read_to_string(victim.join("keep.bin"))
                .await
                .unwrap(),
            "keep"
        );
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn downloader_rejects_unsafe_custom_model_cache_path() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("models");
        let victim = dir.path().join("victim");
        tokio::fs::create_dir_all(&victim).await.unwrap();
        tokio::fs::write(victim.join("keep.bin"), "keep")
            .await
            .unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        ModelDownloader::default()
            .download_with_client(UnsafeCachePathModel, &cache_dir, &client, &env)
            .await
            .unwrap_err();

        assert_eq!(
            tokio::fs::read_to_string(victim.join("keep.bin"))
                .await
                .unwrap(),
            "keep"
        );
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn downloader_rejects_reserved_cache_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let error = ModelDownloader::default()
            .download_with_client(ReservedCacheNamespaceModel, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("reserved `.orchion-`"));
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn downloader_rejects_symlinked_cache_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("models");
        let victim = dir.path().join("victim");
        tokio::fs::create_dir_all(&cache_dir).await.unwrap();
        tokio::fs::create_dir_all(&victim).await.unwrap();
        tokio::fs::write(victim.join("keep.bin"), "keep")
            .await
            .unwrap();
        std::os::unix::fs::symlink(&victim, cache_dir.join("Qwen")).unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let error = ModelDownloader::default()
            .download_with_client(qwen_asr_06b(), &cache_dir, &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("ancestor is a symlink"));
        assert_eq!(
            tokio::fs::read_to_string(victim.join("keep.bin"))
                .await
                .unwrap(),
            "keep"
        );
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn concurrent_downloads_publish_one_complete_cache() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            delay: Duration::from_millis(50),
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };
        let first = ModelDownloader::default();
        let second = ModelDownloader::default();

        let (first_result, second_result) = tokio::join!(
            first.download_with_client(qwen_asr_06b(), dir.path(), &client, &env),
            second.download_with_client(qwen_asr_06b(), dir.path(), &client, &env),
        );

        let target = qwen_asr_06b().cache_path(dir.path());
        assert_eq!(first_result.unwrap(), target);
        assert_eq!(second_result.unwrap(), target);
        assert_eq!(&*calls.lock().unwrap(), &["huggingface"]);
        assert!(target.join(READY_MANIFEST_FILE).exists());
    }

    #[tokio::test]
    async fn auto_skips_huggingface_when_probe_reports_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::Auto);

        let path = downloader
            .download_with_client_and_probe(
                qwen_asr_06b(),
                dir.path(),
                &client,
                &FakeProbe {
                    huggingface_available: false,
                    calls: Arc::new(Mutex::new(0)),
                },
                &env,
            )
            .await
            .unwrap();

        assert!(path.join("config.json").exists());
        assert!(!path.join(".orchion-complete").exists());
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
    }

    #[tokio::test]
    async fn auto_probe_runs_once_for_downloader_instance() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let probe_calls = Arc::new(Mutex::new(0));
        let probe = FakeProbe {
            huggingface_available: false,
            calls: Arc::clone(&probe_calls),
        };
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::Auto);

        downloader
            .download_with_client_and_probe(qwen_asr_06b(), dir.path(), &client, &probe, &env)
            .await
            .unwrap();
        downloader
            .download_with_client_and_probe(qwen_tts_base(), dir.path(), &client, &probe, &env)
            .await
            .unwrap();

        assert_eq!(*probe_calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn ready_manifest_skips_download_when_required_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        let model = qwen_asr_06b();
        let target = model.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("config.json"), "{}")
            .await
            .unwrap();
        write_asr_tokenizer_json(&target).await;
        write_ready_manifest(&target, model.huggingface_repo()).await;

        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();

        let path = downloader
            .download_with_client(model, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn downloader_rolls_back_interrupted_cache_publication() {
        let dir = tempfile::tempdir().unwrap();
        let model = qwen_asr_06b();
        let target = model.cache_path(dir.path());
        let transaction_dir = dir.path().join(PUBLISH_TRANSACTION_DIR);
        let backup = repo_cache_path(&transaction_dir, model.huggingface_repo());
        tokio::fs::create_dir_all(&backup).await.unwrap();
        tokio::fs::write(backup.join("config.json"), "{}")
            .await
            .unwrap();
        write_asr_tokenizer_json(&backup).await;
        write_ready_manifest(&backup, model.huggingface_repo()).await;
        tokio::fs::write(backup.join("old.bin"), "old")
            .await
            .unwrap();
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("new.bin"), "new")
            .await
            .unwrap();
        tokio::fs::write(
            transaction_dir.join(PUBLISH_TRANSACTION_MANIFEST),
            serde_json::to_vec(&serde_json::json!({
                "repos": [{"repo": model.huggingface_repo(), "had_target": true}]
            }))
            .unwrap(),
        )
        .await
        .unwrap();
        tokio::fs::write(
            transaction_dir.join(PUBLISH_TRANSACTION_COMMITTED),
            b"commit",
        )
        .await
        .unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let path = ModelDownloader::default()
            .download_with_client(model, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert_eq!(
            tokio::fs::read_to_string(target.join("old.bin"))
                .await
                .unwrap(),
            "old"
        );
        assert!(!target.join("new.bin").exists());
        assert!(!transaction_dir.exists());
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ready_manifest_redownloads_when_required_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let model = qwen_asr_06b();
        let target = model.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("config.json"), "{}")
            .await
            .unwrap();
        write_ready_manifest(&target, model.huggingface_repo()).await;

        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("modelscope".to_string()),
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();

        let path = downloader
            .download_with_client(model, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert!(path.join("tokenizer.json").exists());
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
    }

    #[tokio::test]
    async fn ocr_vl_incomplete_sharded_cache_is_redownloaded() {
        let temp = tempfile::tempdir().unwrap();
        let model = KnownOcrModel::PaddleOcrVl16;
        let target = model.cache_path(temp.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        for file_name in model.required_files() {
            tokio::fs::write(target.join(file_name), "{}")
                .await
                .unwrap();
        }
        super::write_ready_manifest(&model, &target, "test")
            .await
            .unwrap();
        write_ocr_vl_weight_index(&target).await;
        tokio::fs::write(target.join("model-00001-of-00002.safetensors"), b"weights")
            .await
            .unwrap();
        let client = FakeDownloadClient {
            write_ocr_vl_weights: true,
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let path = ModelDownloader::default()
            .download_with_client(model, temp.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert_eq!(&*calls.lock().unwrap(), &["huggingface"]);
        assert!(target.join("model-00002-of-00002.safetensors").exists());
    }

    #[tokio::test]
    async fn ocr_vl_indexed_cache_validates_shards_even_with_monolithic_weights() {
        let temp = tempfile::tempdir().unwrap();
        let model = KnownOcrModel::PaddleOcrVl16;
        let target = model.cache_path(temp.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        for file_name in model.required_files() {
            tokio::fs::write(target.join(file_name), "{}")
                .await
                .unwrap();
        }
        super::write_ready_manifest(&model, &target, "test")
            .await
            .unwrap();
        write_ocr_vl_weight_index(&target).await;
        tokio::fs::write(target.join("model.safetensors"), b"weights")
            .await
            .unwrap();
        tokio::fs::write(target.join("model-00001-of-00002.safetensors"), b"weights")
            .await
            .unwrap();
        let client = FakeDownloadClient {
            write_ocr_vl_weights: true,
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        ModelDownloader::default()
            .download_with_client(model, temp.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["huggingface"]);
        assert!(target.join("model-00002-of-00002.safetensors").exists());
    }

    #[tokio::test]
    async fn download_rejects_unrepairable_asr_cache_after_model_hub_success() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            fail_huggingface: false,
            omit_asr_tokenizer_sources: true,
            calls: Arc::new(Mutex::new(Vec::new())),
            ..Default::default()
        };
        let env = DownloadEnv {
            orchion_model_source: Some("modelscope".to_string()),
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let error = downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("tokenizer_config.json"));
    }

    #[tokio::test]
    async fn pp_ocrv5_mobile_downloads_modelscope_oar_registry_files() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let repos = Arc::clone(&client.repos);
        let file_filters = Arc::clone(&client.file_filters);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::Auto);

        let path = downloader
            .download_with_client(KnownOcrModel::PpOcrV5Mobile, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, KnownOcrModel::PpOcrV5Mobile.cache_path(dir.path()));
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
        assert_eq!(&*repos.lock().unwrap(), &["greatv/oar-ocr".to_string()]);
        assert_eq!(
            &*file_filters.lock().unwrap(),
            &[Some(vec![
                "pp-ocrv5_mobile_det.onnx",
                "pp-ocrv5_mobile_rec.onnx",
                "ppocrv5_dict.txt"
            ])]
        );
        assert!(path.join(".orchion-ready.json").exists());
        assert!(!path.join("pp-ocrv5_mobile_det.onnx").exists());
        assert!(!path.join("pp-ocrv5_mobile_rec.onnx").exists());
        assert!(!path.join("ppocrv5_dict.txt").exists());

        let registry_dir = dir.path().join("greatv/oar-ocr");
        assert!(registry_dir.join("pp-ocrv5_mobile_det.onnx").exists());
        assert!(registry_dir.join("pp-ocrv5_mobile_rec.onnx").exists());
        assert!(registry_dir.join("ppocrv5_dict.txt").exists());
    }

    async fn write_ready_manifest(target: &Path, repo: &str) {
        let manifest = serde_json::json!({
            "schema_version": 1,
            "repo_id": repo,
            "layout": "model-hub-native",
        });
        tokio::fs::write(target.join(".orchion-ready.json"), manifest.to_string())
            .await
            .unwrap();
    }

    fn qwen_asr_06b() -> AsrModel {
        AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap()
    }

    fn qwen_tts_base() -> TtsModel {
        TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-Base").unwrap()
    }

    async fn write_asr_tokenizer_json(target: &Path) {
        tokio::fs::write(
            target.join("tokenizer.json"),
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#,
        )
        .await
        .unwrap();
    }

    async fn write_asr_tokenizer_sources(target: &Path) {
        tokio::fs::write(
            target.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"151645":{"content":"<|im_end|>","special":true}}}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(target.join("vocab.json"), r#"{"hello":0,"world":1}"#)
            .await
            .unwrap();
        tokio::fs::write(target.join("merges.txt"), "#version: 0.2\nhello world\n")
            .await
            .unwrap();
    }

    async fn write_ocr_vl_weight_index(target: &Path) {
        tokio::fs::write(
            target.join("model.safetensors.index.json"),
            r#"{"weight_map":{"first":"model-00001-of-00002.safetensors","second":"model-00002-of-00002.safetensors"}}"#,
        )
        .await
        .unwrap();
    }

    async fn write_complete_ocr_vl_cache(target: &Path) {
        for file_name in [
            "preprocessor_config.json",
            "tokenizer.json",
            "chat_template.jinja",
        ] {
            tokio::fs::write(target.join(file_name), "{}")
                .await
                .unwrap();
        }
        write_ocr_vl_weight_index(target).await;
        for file_name in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            tokio::fs::write(target.join(file_name), b"weights")
                .await
                .unwrap();
        }
    }
}
