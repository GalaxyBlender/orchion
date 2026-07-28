mod assets;
mod provider;
mod transaction;

pub use provider::{
    DownloadFuture, DownloadProvider, DownloadProviderRegistry, DownloadSource, HubProviderOptions,
    HuggingFaceProvider, ModelScopeProvider, ProviderDownloadRequest, ProviderDownloadResult,
    ProviderModel, ResolvedDownloadFuture,
};

use assets::{ModelHubAsset, ModelHubAssetKind, uses_modelscope_file_assets};
use orchion_core::{DownloadFailure, ModelCategory, ModelId, ModelSpec, OrchionError, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const READY_MANIFEST_FILE: &str = ".orchion-ready.json";
const READY_MANIFEST_SCHEMA_VERSION: u64 = 3;
const READY_MANIFEST_LAYOUT: &str = "model-hub-native";
const CACHE_STATE_DIR: &str = ".orchion";
const DOWNLOAD_STAGING_DIR: &str = "staging";
const MODEL_LOCK_DIR: &str = "locks";
const PUBLICATION_LOCK_FILE: &str = "publish.lock";
const PUBLISH_TRANSACTION_DIR: &str = "publish-transaction";
const PUBLISH_TRANSACTION_MANIFEST: &str = "manifest.json";
const PUBLISH_TRANSACTION_COMMITTED: &str = "committed";

fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
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

#[derive(Debug)]
struct RequiredCacheFile {
    repo: String,
    path: String,
    absolute_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RepositoryRequest {
    identity: String,
    repository: String,
    requested_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RepositoryDownload {
    request: RepositoryRequest,
    resolved_revision: Option<String>,
}

impl ResolvedSource {
    const fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }

    #[cfg(test)]
    const fn default_revision(self) -> &'static str {
        match self {
            Self::HuggingFace => "main",
            Self::ModelScope => "master",
        }
    }

    #[cfg(test)]
    fn repo<M: ModelSpec>(self, model: &M) -> &str {
        match self {
            Self::HuggingFace => model.huggingface_repo(),
            Self::ModelScope => model.modelscope_repo(),
        }
    }
}

#[cfg(test)]
impl DownloadProvider for ResolvedSource {
    fn label(&self) -> &'static str {
        (*self).label()
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn default_revision(&self) -> &str {
        (*self).default_revision()
    }

    fn repository(&self, model: ProviderModel<'_>) -> String {
        match self {
            Self::HuggingFace => model.huggingface_repo(),
            Self::ModelScope => model.modelscope_repo(),
        }
        .to_string()
    }

    fn download<'a>(&'a self, _request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
        Box::pin(async { unreachable!("test manifest provider does not download") })
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
    if matches!(source, DownloadSource::Auto)
        && let Some(value) = env.orchion_model_source.as_deref()
    {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderSelection {
    BuiltIn(DownloadSource),
    Registry,
}

#[derive(Clone)]
pub struct ModelDownloader {
    selection: ProviderSelection,
    providers: DownloadProviderRegistry,
    revision: Option<String>,
    repository_revisions: HashMap<String, String>,
    huggingface_available: Arc<tokio::sync::OnceCell<bool>>,
}

impl std::fmt::Debug for ModelDownloader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelDownloader")
            .field("selection", &self.selection)
            .field("providers", &self.providers)
            .field("revision", &self.revision)
            .field("repository_revisions", &self.repository_revisions)
            .field("huggingface_available", &self.huggingface_available)
            .finish()
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new(DownloadSource::Auto)
    }
}

impl ModelDownloader {
    #[must_use]
    pub fn new(source: DownloadSource) -> Self {
        Self {
            selection: ProviderSelection::BuiltIn(source),
            providers: DownloadProviderRegistry::new()
                .with_provider(HuggingFaceProvider::new())
                .with_provider(ModelScopeProvider::new()),
            revision: None,
            repository_revisions: HashMap::new(),
            huggingface_available: Arc::new(tokio::sync::OnceCell::const_new()),
        }
    }

    #[must_use]
    pub fn from_provider<P>(provider: P) -> Self
    where
        P: DownloadProvider,
    {
        Self::from_registry(DownloadProviderRegistry::new().with_provider(provider))
    }

    #[must_use]
    pub fn from_registry(providers: DownloadProviderRegistry) -> Self {
        Self {
            selection: ProviderSelection::Registry,
            providers,
            revision: None,
            repository_revisions: HashMap::new(),
            huggingface_available: Arc::new(tokio::sync::OnceCell::const_new()),
        }
    }

    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Selects a revision for one provider-neutral repository identity.
    #[must_use]
    pub fn with_repository_revision(
        mut self,
        repository: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.repository_revisions
            .insert(repository.into(), revision.into());
        self
    }

    /// Downloads and transactionally publishes a model into the cache.
    ///
    /// # Errors
    ///
    /// Returns an error when provider selection, download, validation, or cache publication fails.
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

    #[allow(clippy::too_many_lines)]
    async fn download_with_client_and_probe<M: ModelSpec, C: DownloadClient, P: SourceProbe>(
        &self,
        model: M,
        cache_dir: impl AsRef<Path>,
        client: &C,
        probe: &P,
        env: &DownloadEnv,
    ) -> Result<PathBuf> {
        let cache_dir = cache_dir.as_ref();
        if let Some(revision) = self.revision.as_deref() {
            validate_revision(model.huggingface_repo(), revision)?;
        }
        for (repo, revision) in &self.repository_revisions {
            validate_repo_id(repo)?;
            validate_revision(repo, revision)?;
        }
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
        ensure_cache_state_dir(cache_dir)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: error.to_string(),
            })?;
        let model_lock =
            transaction::acquire_model_lock(cache_dir, model.huggingface_repo()).await?;
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
        let (mut model_lock, publication_clean) = recover_with_owned_locks(
            cache_dir.to_path_buf(),
            model.huggingface_repo().to_string(),
            model_lock,
        )
        .await?;
        let staging_prefix = transaction::model_staging_prefix(model.huggingface_repo());
        let staging_dir = cache_state_path(cache_dir, DOWNLOAD_STAGING_DIR);
        if let Err(error) = cleanup_stale_model_downloads(&staging_dir, &staging_prefix).await {
            tracing::warn!(
                model = ?model,
                path = %cache_dir.display(),
                %error,
                "failed to clean stale model download staging directories"
            );
        }

        let assets = model_hub_assets(&model);
        let cache_sources = self.configured_providers(
            env,
            uses_modelscope_file_assets(assets),
            model.huggingface_repo(),
        )?;
        if is_ready_cache(&model, &target, &cache_sources, self).await? {
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

        let candidates = if uses_modelscope_file_assets(assets)
            && matches!(self.selection, ProviderSelection::BuiltIn(_))
        {
            self.configured_providers(env, true, model.huggingface_repo())?
        } else {
            self.resolve_candidates(env, probe).await?
        };
        tracing::info!(
            model = ?model,
            path = %target.display(),
            source_count = candidates.len(),
            "ensuring model cache is available"
        );
        ensure_download_staging_dir(&staging_dir)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo().to_string(),
                message: error.to_string(),
            })?;
        let mut failures = Vec::new();
        for candidate in candidates {
            let source_name = candidate.label();
            let repo = candidate.repository(provider_model(&model));
            let repository_requests =
                self.repository_requests(&model, candidate.as_ref(), assets)?;
            let staging = tempfile::Builder::new()
                .prefix(&staging_prefix)
                .tempdir_in(&staging_dir)
                .map_err(|error| OrchionError::Download {
                    source_name,
                    repo: model.huggingface_repo().to_string(),
                    message: error.to_string(),
                })?;
            let staging_root = staging.path();
            let staging_target = validated_model_cache_path(&model, staging_root)?;
            let preparation_result = async {
                let downloads = if assets.is_empty() {
                    let request = &repository_requests[0];
                    tracing::info!(
                        source = source_name,
                        repo = request.repository,
                        path = %target.display(),
                        "downloading model"
                    );
                    let result = client
                        .download(
                            candidate.as_ref(),
                            &request.repository,
                            staging_root,
                            &staging_target,
                            None,
                            &request.requested_revision,
                            env,
                        )
                        .await?;
                    vec![RepositoryDownload {
                        request: request.clone(),
                        resolved_revision: result.resolved_revision().map(str::to_string),
                    }]
                } else {
                    download_hub_assets(
                        &model,
                        candidate.as_ref(),
                        assets,
                        &repository_requests,
                        staging_root,
                        &staging_target,
                        client,
                        env,
                    )
                    .await?
                };
                prepare_cached_model(&model, &staging_target, source_name).await?;
                ensure_ready_cache_files(&model, &staging_target, source_name).await?;
                write_ready_manifest(
                    &model,
                    &staging_target,
                    candidate.as_ref(),
                    &repo,
                    self.repository_revision(
                        model.huggingface_repo(),
                        model.huggingface_repo(),
                        candidate.as_ref(),
                    ),
                    &downloads,
                )
                .await?;
                Ok::<_, OrchionError>(())
            }
            .await;
            let transaction_result = match preparation_result {
                Ok(()) => {
                    let repos = publication_repositories(&model, assets);
                    let (returned_lock, result) = publish_staged_cache(
                        model.huggingface_repo().to_string(),
                        repos,
                        staging,
                        cache_dir.to_path_buf(),
                        source_name,
                        model_lock,
                    )
                    .await?;
                    model_lock = returned_lock;
                    result
                }
                Err(error) => Err(error),
            };
            match transaction_result {
                Ok(()) => {
                    tracing::info!(
                        source = source_name,
                        repo,
                        path = %target.display(),
                        "model candidate transaction completed"
                    );
                    return Ok(target);
                }
                Err(error) => {
                    tracing::warn!(
                        source = source_name,
                        repo,
                        path = %target.display(),
                        error = %error,
                        "model candidate transaction failed"
                    );
                    failures.push(DownloadFailure {
                        source_name,
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

    fn repository_requests<M: ModelSpec>(
        &self,
        model: &M,
        provider: &dyn DownloadProvider,
        assets: &[ModelHubAsset],
    ) -> Result<Vec<RepositoryRequest>> {
        let mut identities = Vec::new();
        if assets.is_empty() {
            identities.push(model.huggingface_repo());
        } else {
            for asset in assets {
                if !identities.contains(&asset.repo) {
                    identities.push(asset.repo);
                }
            }
        }

        identities
            .into_iter()
            .map(|identity| {
                let repository = if identity == model.huggingface_repo() {
                    provider.repository(provider_model(model))
                } else {
                    provider.repository(ProviderModel::for_repository(identity))
                };
                let requested_revision = self
                    .repository_revision(identity, model.huggingface_repo(), provider)
                    .to_string();
                validate_revision(identity, &requested_revision)?;
                Ok(RepositoryRequest {
                    identity: identity.to_string(),
                    repository,
                    requested_revision,
                })
            })
            .collect()
    }

    fn repository_revision<'a>(
        &'a self,
        identity: &str,
        canonical_identity: &str,
        provider: &'a dyn DownloadProvider,
    ) -> &'a str {
        if identity == canonical_identity
            && let Some(revision) = self.revision.as_deref()
        {
            return revision;
        }
        self.repository_revisions
            .get(identity)
            .map_or_else(|| provider.default_revision(), String::as_str)
    }

    fn configured_providers(
        &self,
        env: &DownloadEnv,
        require_modelscope: bool,
        model_repo: &str,
    ) -> Result<Vec<Arc<dyn DownloadProvider>>> {
        if matches!(self.selection, ProviderSelection::Registry) {
            return Ok(self.providers.providers().iter().map(Arc::clone).collect());
        }
        let ProviderSelection::BuiltIn(source) = self.selection else {
            unreachable!("registry selection returned above");
        };
        let mut sources = resolve_source(source, env)?;
        if require_modelscope {
            if !sources.contains(&ResolvedSource::ModelScope) {
                return Err(OrchionError::Download {
                    source_name: "huggingface",
                    repo: model_repo.to_string(),
                    message: "model has assets that are only available from ModelScope".to_string(),
                });
            }
            sources.retain(|source| *source == ResolvedSource::ModelScope);
        }
        Ok(sources
            .into_iter()
            .filter_map(|source| self.providers.provider(source.label()))
            .collect())
    }

    async fn resolve_candidates<P: SourceProbe>(
        &self,
        env: &DownloadEnv,
        probe: &P,
    ) -> Result<Vec<Arc<dyn DownloadProvider>>> {
        let candidates = self.configured_providers(env, false, "provider-selection")?;
        if !matches!(
            self.selection,
            ProviderSelection::BuiltIn(DownloadSource::Auto)
        ) || env.orchion_model_source.is_some()
        {
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
                .filter(|provider| provider.label() != "huggingface")
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
        .is_some_and(is_reserved_cache_namespace)
    {
        return Err(OrchionError::Download {
            source_name: "cache",
            repo: repo.to_string(),
            message: "repository uses the reserved `.orchion` cache namespace".to_string(),
        });
    }
    Ok(())
}

fn is_reserved_cache_namespace(segment: &str) -> bool {
    let segment = segment.to_ascii_lowercase();
    segment == CACHE_STATE_DIR || segment.starts_with(".orchion-")
}

fn validate_revision(repo: &str, revision: &str) -> Result<()> {
    if revision.trim().is_empty()
        || revision.chars().any(char::is_control)
        || revision.contains(['\\', '?', '#'])
        || revision
            .split('/')
            .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(OrchionError::Download {
            source_name: "cache",
            repo: repo.to_string(),
            message: format!("invalid model revision `{revision}`"),
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

fn publication_repositories<M: ModelSpec>(model: &M, assets: &[ModelHubAsset]) -> Vec<String> {
    let mut repos = Vec::new();
    for asset in assets {
        if asset.repo != model.huggingface_repo() && !repos.iter().any(|repo| repo == asset.repo) {
            repos.push(asset.repo.to_string());
        }
    }
    repos.push(model.huggingface_repo().to_string());
    repos
}

async fn recover_with_owned_locks(
    cache_dir: PathBuf,
    model_key: String,
    model_lock: transaction::CacheLock,
) -> Result<(transaction::CacheLock, bool)> {
    let task = tokio::spawn(async move {
        let recovery = async {
            let _publication_lock =
                transaction::acquire_publication_lock(&cache_dir, &model_key).await?;
            recover_interrupted_publication(&cache_dir)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: "cache",
                    repo: model_key,
                    message: format!("failed to recover interrupted cache publication: {error}"),
                })
        }
        .await;
        (model_lock, recovery)
    });
    let (model_lock, recovery) = task.await.map_err(|error| OrchionError::BlockingTask {
        message: format!("cache recovery task failed: {error}"),
    })?;
    Ok((model_lock, recovery?))
}

async fn publish_staged_cache(
    model_key: String,
    repos: Vec<String>,
    staging: tempfile::TempDir,
    cache_dir: PathBuf,
    source_name: &'static str,
    model_lock: transaction::CacheLock,
) -> Result<(transaction::CacheLock, Result<()>)> {
    let task = tokio::spawn(async move {
        let publication = async {
            let _publication_lock =
                transaction::acquire_publication_lock(&cache_dir, &model_key).await?;
            publish_staged_repositories(staging.path(), &cache_dir, &repos)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name,
                    repo: model_key,
                    message: error.to_string(),
                })
        }
        .await;
        (model_lock, publication)
    });
    task.await.map_err(|error| OrchionError::BlockingTask {
        message: format!("cache publication task failed: {error}"),
    })
}

#[allow(clippy::too_many_lines)]
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
    let transaction_dir = cache_state_path(cache_dir, PUBLISH_TRANSACTION_DIR);
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
    sync_cache_state(cache_dir).await?;

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
        sync_cache_state(cache_dir).await?;
    }
    Ok(())
}

async fn recover_interrupted_publication(cache_dir: &Path) -> std::io::Result<bool> {
    let transaction_dir = cache_state_path(cache_dir, PUBLISH_TRANSACTION_DIR);
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
        sync_cache_state(cache_dir).await?;
        return Ok(true);
    }

    let manifest_path = transaction_dir.join(PUBLISH_TRANSACTION_MANIFEST);
    let manifest = match tokio::fs::read(&manifest_path).await {
        Ok(manifest) => manifest,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tokio::fs::remove_dir_all(transaction_dir).await?;
            sync_cache_state(cache_dir).await?;
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
                .is_some_and(is_reserved_cache_namespace)
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
    sync_cache_state(cache_dir).await?;
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

async fn cleanup_stale_model_downloads(
    staging_dir: &Path,
    staging_prefix: &str,
) -> std::io::Result<()> {
    if validate_optional_download_staging_dir(staging_dir).await? {
        let mut entries = tokio::fs::read_dir(staging_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name();
            if file_name
                .to_str()
                .is_some_and(|name| name.starts_with(staging_prefix))
            {
                remove_stale_download_staging(entry.path()).await?;
            }
        }
    }
    Ok(())
}

async fn remove_stale_download_staging(path: PathBuf) -> std::io::Result<()> {
    remove_cache_entry(&path).await?;
    tracing::info!(path = %path.display(), "removed stale model download staging directory");
    Ok(())
}

async fn ensure_download_staging_dir(staging_dir: &Path) -> std::io::Result<()> {
    tokio::fs::create_dir_all(staging_dir).await?;
    if validate_optional_download_staging_dir(staging_dir).await? {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "model download staging directory was not created: {}",
            staging_dir.display()
        ),
    ))
}

async fn ensure_cache_state_dir(cache_dir: &Path) -> std::io::Result<()> {
    let state_dir = cache_dir.join(CACHE_STATE_DIR);
    tokio::fs::create_dir_all(&state_dir).await?;
    validate_regular_directory(&state_dir, "model cache state").await
}

async fn validate_optional_download_staging_dir(staging_dir: &Path) -> std::io::Result<bool> {
    let metadata = match tokio::fs::symlink_metadata(staging_dir).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        return Ok(true);
    }
    Err(not_regular_directory_error(
        staging_dir,
        "model download staging",
    ))
}

async fn validate_regular_directory(path: &Path, label: &str) -> std::io::Result<()> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        return Ok(());
    }
    Err(not_regular_directory_error(path, label))
}

fn not_regular_directory_error(path: &Path, label: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "{label} path is not a regular directory: {}",
            path.display()
        ),
    )
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

async fn sync_cache_state(cache_dir: &Path) -> std::io::Result<()> {
    sync_directory(&cache_dir.join(CACHE_STATE_DIR)).await?;
    sync_directory(cache_dir).await
}

async fn is_ready_cache<M: ModelSpec>(
    model: &M,
    target: &Path,
    allowed_providers: &[Arc<dyn DownloadProvider>],
    downloader: &ModelDownloader,
) -> Result<bool> {
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
        || manifest["layout"].as_str() != Some(READY_MANIFEST_LAYOUT)
    {
        return Ok(false);
    }
    let Some(source_name) = manifest["source"].as_str() else {
        return Ok(false);
    };
    let Some(provider) = allowed_providers
        .iter()
        .find(|provider| provider.label() == source_name)
    else {
        return Ok(false);
    };
    let expected_revision = downloader.repository_revision(
        model.huggingface_repo(),
        model.huggingface_repo(),
        provider.as_ref(),
    );
    let expected_repo = provider.repository(provider_model(model));
    if manifest["repo_id"].as_str() != Some(expected_repo.as_str())
        || manifest["revision"].as_str() != Some(expected_revision)
    {
        return Ok(false);
    }
    let expected_requests =
        downloader.repository_requests(model, provider.as_ref(), model_hub_assets(model))?;
    let Some(manifest_repos) = manifest["downloaded_repos"].as_array() else {
        return Ok(false);
    };
    if manifest_repos.len() != expected_requests.len()
        || manifest_repos
            .iter()
            .zip(&expected_requests)
            .any(|(actual, expected)| actual.as_str() != Some(expected.repository.as_str()))
    {
        return Ok(false);
    }
    let Some(repositories) = manifest["repositories"].as_array() else {
        return Ok(false);
    };
    if repositories.len() != expected_requests.len() {
        return Ok(false);
    }
    for expected in expected_requests {
        let Some(actual) = repositories
            .iter()
            .find(|actual| actual["identity"].as_str() == Some(expected.identity.as_str()))
        else {
            return Ok(false);
        };
        if actual["repo_id"].as_str() != Some(expected.repository.as_str())
            || actual["requested_revision"].as_str() != Some(expected.requested_revision.as_str())
        {
            return Ok(false);
        }
        if provider::is_immutable_revision(&expected.requested_revision)
            && actual["resolved_revision"].as_str() != Some(expected.requested_revision.as_str())
        {
            return Ok(false);
        }
        if actual["resolved_revision"]
            .as_str()
            .is_some_and(str::is_empty)
        {
            return Ok(false);
        }
    }

    manifest_files_match(model, target, &manifest).await
}

async fn manifest_files_match<M: ModelSpec>(
    model: &M,
    target: &Path,
    manifest: &serde_json::Value,
) -> Result<bool> {
    let Some(required_files) = required_cache_files(model, target).await? else {
        return Ok(false);
    };
    let Some(manifest_files) = manifest["files"].as_array() else {
        return Ok(false);
    };
    if manifest_files.len() != required_files.len() {
        return Ok(false);
    }
    for required in required_files {
        let Some(entry) = manifest_files.iter().find(|entry| {
            entry["repo"].as_str() == Some(required.repo.as_str())
                && entry["path"].as_str() == Some(required.path.as_str())
        }) else {
            return Ok(false);
        };
        if entry["file_type"].as_str() != Some("file") {
            return Ok(false);
        }
        let Some(expected_size) = entry["size"].as_u64().filter(|size| *size > 0) else {
            return Ok(false);
        };
        let Some(expected_sha256) = entry["sha256"].as_str().filter(|hash| hash.len() == 64) else {
            return Ok(false);
        };
        let (actual_size, actual_sha256) = match cache_file_integrity(required.absolute_path).await
        {
            Ok(integrity) => integrity,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidData
                ) =>
            {
                return Ok(false);
            }
            Err(error) => {
                return Err(OrchionError::Download {
                    source_name: "cache",
                    repo: required.repo,
                    message: error.to_string(),
                });
            }
        };
        if actual_size != expected_size || actual_sha256 != expected_sha256 {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn required_cache_files<M: ModelSpec>(
    model: &M,
    target: &Path,
) -> Result<Option<Vec<RequiredCacheFile>>> {
    let mut files = Vec::new();
    for file_name in model.required_files() {
        push_required_cache_file(
            &mut files,
            model.huggingface_repo(),
            file_name,
            target.join(file_name),
        );
    }
    if model.category() == ModelCategory::OcrVl {
        let Some(weight_files) = ocr_vl_weight_file_names(model, target).await? else {
            return Ok(None);
        };
        for file_name in weight_files {
            push_required_cache_file(
                &mut files,
                model.huggingface_repo(),
                &file_name,
                target.join(&file_name),
            );
        }
    }
    let Some(cache_dir) = cache_root_from_target(target) else {
        return Ok(None);
    };
    for asset in model_hub_assets(model) {
        push_required_cache_file(
            &mut files,
            asset.repo,
            asset.file,
            repo_cache_path(cache_dir, asset.repo).join(asset.file),
        );
        if let ModelHubAssetKind::PaddleOcrDictionary { output_file } = asset.kind {
            push_required_cache_file(
                &mut files,
                model.huggingface_repo(),
                output_file,
                target.join(output_file),
            );
        }
    }
    Ok(Some(files))
}

fn push_required_cache_file(
    files: &mut Vec<RequiredCacheFile>,
    repo: &str,
    path: &str,
    absolute_path: PathBuf,
) {
    if files
        .iter()
        .any(|file| file.repo == repo && file.path == path)
    {
        return;
    }
    files.push(RequiredCacheFile {
        repo: repo.to_string(),
        path: path.to_string(),
        absolute_path,
    });
}

async fn required_cache_files_exist<M: ModelSpec>(model: &M, target: &Path) -> Result<bool> {
    let Some(files) = required_cache_files(model, target).await? else {
        return Ok(false);
    };
    for file in files {
        match cache_file_size(&file.absolute_path).await {
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidData
                ) =>
            {
                return Ok(false);
            }
            Err(error) => {
                return Err(OrchionError::Download {
                    source_name: "cache",
                    repo: file.repo,
                    message: error.to_string(),
                });
            }
        }
    }
    Ok(true)
}

async fn cache_file_size(path: &Path) -> std::io::Result<u64> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "expected a non-empty regular file",
        ));
    }
    Ok(metadata.len())
}

fn cache_root_from_target(target: &Path) -> Option<&Path> {
    target.parent().and_then(Path::parent)
}

fn model_hub_assets<M: ModelSpec>(model: &M) -> &'static [ModelHubAsset] {
    assets::for_model(model.huggingface_repo())
}

fn repo_cache_path(cache_dir: &Path, repo: &str) -> PathBuf {
    repo.split('/')
        .fold(cache_dir.to_path_buf(), |path, segment| path.join(segment))
}

fn cache_state_path(cache_dir: &Path, name: &str) -> PathBuf {
    cache_dir.join(CACHE_STATE_DIR).join(name)
}

#[allow(clippy::too_many_arguments)]
async fn download_hub_assets<M: ModelSpec, C: DownloadClient>(
    model: &M,
    provider: &dyn DownloadProvider,
    assets: &[ModelHubAsset],
    repository_requests: &[RepositoryRequest],
    cache_dir: &Path,
    target: &Path,
    client: &C,
    env: &DownloadEnv,
) -> Result<Vec<RepositoryDownload>> {
    tokio::fs::create_dir_all(target)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: provider.label(),
            repo: model.huggingface_repo().to_string(),
            message: error.to_string(),
        })?;

    let mut downloads = Vec::with_capacity(repository_requests.len());
    for request in repository_requests {
        let repo_target = repo_cache_path(cache_dir, &request.identity);
        let repo_files = assets::files_for_repo(assets, &request.identity);
        tracing::info!(
            source = provider.label(),
            repo = request.repository,
            path = %repo_target.display(),
            "downloading model asset repo"
        );
        let result = client
            .download(
                provider,
                &request.repository,
                cache_dir,
                &repo_target,
                Some(&repo_files),
                &request.requested_revision,
                env,
            )
            .await?;
        downloads.push(RepositoryDownload {
            request: request.clone(),
            resolved_revision: result.resolved_revision().map(str::to_string),
        });
    }

    for asset in assets {
        let source_path = repo_cache_path(cache_dir, asset.repo).join(asset.file);
        match asset.kind {
            ModelHubAssetKind::RequiredFile => {
                ensure_asset_file_exists(provider.label(), asset.repo, &source_path).await?;
            }
            ModelHubAssetKind::PaddleOcrDictionary { output_file } => {
                let dictionary =
                    build_paddle_ocr_dictionary(provider.label(), asset.repo, &source_path).await?;
                tokio::fs::write(target.join(output_file), dictionary)
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name: provider.label(),
                        repo: asset.repo.to_string(),
                        message: error.to_string(),
                    })?;
            }
            ModelHubAssetKind::ModelScopeFile { output_file } => {
                ensure_asset_file_exists(provider.label(), asset.repo, &source_path).await?;
                let _ = output_file;
            }
        }
    }
    Ok(downloads)
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
    source_name: &'static str,
    repo: &'static str,
    path: &Path,
) -> Result<()> {
    if tokio::fs::try_exists(path)
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: error.to_string(),
        })?
    {
        return Ok(());
    }
    Err(OrchionError::Download {
        source_name,
        repo: repo.to_string(),
        message: format!("missing required model asset `{}`", path.display()),
    })
}

async fn build_paddle_ocr_dictionary(
    source_name: &'static str,
    repo: &'static str,
    path: &Path,
) -> Result<String> {
    let yaml = tokio::fs::read_to_string(path)
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: repo.to_string(),
            message: error.to_string(),
        })?;
    let characters =
        parse_paddle_ocr_character_dict(&yaml).ok_or_else(|| OrchionError::Download {
            source_name,
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

async fn ocr_vl_weight_file_names<M: ModelSpec>(
    model: &M,
    target: &Path,
) -> Result<Option<Vec<String>>> {
    let index = match tokio::fs::read_to_string(target.join("model.safetensors.index.json")).await {
        Ok(index) => index,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return match cache_file_size(&target.join("model.safetensors")).await {
                Ok(_) => Ok(Some(vec!["model.safetensors".to_string()])),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidData
                    ) =>
                {
                    Ok(None)
                }
                Err(error) => Err(OrchionError::Download {
                    source_name: "cache",
                    repo: model.huggingface_repo().to_string(),
                    message: error.to_string(),
                }),
            };
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
        return Ok(None);
    };
    let Some(weight_map) = index["weight_map"].as_object() else {
        return Ok(None);
    };
    let mut weight_files = vec!["model.safetensors.index.json".to_string()];
    for file_name in weight_map.values() {
        let Some(file_name) = file_name.as_str() else {
            return Ok(None);
        };
        let path = Path::new(file_name);
        if path.is_absolute()
            || !path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
        {
            return Ok(None);
        }
        if !weight_files.iter().any(|current| current == file_name) {
            weight_files.push(file_name.to_string());
        }
    }
    if weight_files.len() == 1 {
        return Ok(None);
    }
    Ok(Some(weight_files))
}

async fn write_ready_manifest<M: ModelSpec>(
    model: &M,
    target: &Path,
    provider: &dyn DownloadProvider,
    repo: &str,
    requested_revision: &str,
    downloads: &[RepositoryDownload],
) -> Result<()> {
    let source_name = provider.label();
    let downloaded_repos = downloads
        .iter()
        .map(|download| download.request.repository.as_str())
        .collect::<Vec<_>>();
    let repositories = downloads
        .iter()
        .map(|download| {
            serde_json::json!({
                "identity": download.request.identity,
                "repo_id": download.request.repository,
                "requested_revision": download.request.requested_revision,
                "resolved_revision": download.resolved_revision,
            })
        })
        .collect::<Vec<_>>();
    let resolved_revision = downloads
        .iter()
        .find(|download| download.request.identity == model.huggingface_repo())
        .and_then(|download| download.resolved_revision.as_deref());
    let required_files =
        required_cache_files(model, target)
            .await?
            .ok_or_else(|| OrchionError::Download {
                source_name,
                repo: repo.to_string(),
                message: "download completed without a complete required file set".to_string(),
            })?;
    let mut files = Vec::with_capacity(required_files.len());
    for file in required_files {
        let (size, sha256) = cache_file_integrity(file.absolute_path)
            .await
            .map_err(|error| OrchionError::Download {
                source_name,
                repo: repo.to_string(),
                message: format!("invalid required cache file `{}`: {error}", file.path),
            })?;
        files.push(serde_json::json!({
            "repo": file.repo,
            "path": file.path,
            "file_type": "file",
            "size": size,
            "sha256": sha256,
        }));
    }
    let manifest = serde_json::json!({
        "schema_version": READY_MANIFEST_SCHEMA_VERSION,
        "source": source_name,
        "repo_id": repo,
        "downloaded_repos": downloaded_repos,
        "revision": requested_revision,
        "resolved_revision": resolved_revision,
        "repositories": repositories,
        "layout": READY_MANIFEST_LAYOUT,
        "files": files,
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

fn provider_model<M: ModelSpec>(model: &M) -> ProviderModel<'_> {
    ProviderModel::new(model.huggingface_repo(), model.modelscope_repo())
}

async fn cache_file_integrity(path: PathBuf) -> std::io::Result<(u64, String)> {
    tokio::task::spawn_blocking(move || {
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "expected a non-empty regular file",
            ));
        }
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024].into_boxed_slice();
        let mut bytes_read = 0_u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            bytes_read = bytes_read
                .checked_add(u64::try_from(read).map_err(std::io::Error::other)?)
                .ok_or_else(|| std::io::Error::other("required file size overflow"))?;
            hasher.update(&buffer[..read]);
        }
        if bytes_read != metadata.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "required file changed while its integrity was calculated",
            ));
        }
        Ok((metadata.len(), encode_hex(&hasher.finalize())))
    })
    .await
    .map_err(std::io::Error::other)?
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
    const fn timeout() -> Duration {
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
            let client = match reqwest::Client::builder().timeout(Self::timeout()).build() {
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

#[allow(clippy::too_many_arguments)]
trait DownloadClient {
    fn download<'a>(
        &'a self,
        provider: &'a dyn DownloadProvider,
        repo: &'a str,
        cache_dir: &'a Path,
        target: &'a Path,
        files: Option<&'a [&'a str]>,
        revision: &'a str,
        env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<ProviderDownloadResult>>;
}

struct LibraryDownloadClient;

impl DownloadClient for LibraryDownloadClient {
    fn download<'a>(
        &'a self,
        provider: &'a dyn DownloadProvider,
        repo: &'a str,
        cache_dir: &'a Path,
        target: &'a Path,
        files: Option<&'a [&'a str]>,
        revision: &'a str,
        _env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<ProviderDownloadResult>> {
        provider.download_with_result(ProviderDownloadRequest::new(
            repo, revision, cache_dir, target, files,
        ))
    }
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
    #![allow(clippy::struct_excessive_bools, clippy::unnecessary_literal_bound)]

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
            ".ORCHION/Model"
        }

        fn modelscope_repo(&self) -> &str {
            ".ORCHION/Model"
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DifferentProviderReposModel;

    impl ModelSpec for DifferentProviderReposModel {
        fn category(&self) -> ModelCategory {
            ModelCategory::Tts
        }

        fn huggingface_repo(&self) -> &str {
            "Acme/HuggingFaceModel"
        }

        fn modelscope_repo(&self) -> &str {
            "Acme/ModelScopeModel"
        }
    }

    #[derive(Default)]
    struct FakeDownloadClient {
        fail_huggingface: bool,
        omit_asr_tokenizer_sources: bool,
        omit_huggingface_asr_tokenizer_sources: bool,
        write_empty_config: bool,
        write_ocr_vl_weights: bool,
        delay: Duration,
        calls: Arc<Mutex<Vec<&'static str>>>,
        repos: Arc<Mutex<Vec<String>>>,
        file_filters: Arc<Mutex<Vec<Option<Vec<String>>>>>,
        revisions: Arc<Mutex<Vec<String>>>,
        omit_resolved_revision: bool,
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
            provider: &'a dyn DownloadProvider,
            repo: &'a str,
            _cache_dir: &'a Path,
            target: &'a Path,
            files: Option<&'a [&'a str]>,
            revision: &'a str,
            _env: &'a DownloadEnv,
        ) -> BoxFuture<'a, Result<ProviderDownloadResult>> {
            Box::pin(async move {
                let source_name = provider.label();
                self.calls.lock().unwrap().push(source_name);
                self.repos.lock().unwrap().push(repo.to_string());
                self.revisions.lock().unwrap().push(revision.to_string());
                self.file_filters.lock().unwrap().push(
                    files.map(|files| files.iter().map(|file| (*file).to_string()).collect()),
                );
                tokio::time::sleep(self.delay).await;
                if self.fail_huggingface && source_name == "huggingface" {
                    tokio::fs::create_dir_all(target).await.map_err(|error| {
                        OrchionError::Download {
                            source_name,
                            repo: repo.to_string(),
                            message: error.to_string(),
                        }
                    })?;
                    tokio::fs::write(target.join("partial.bin"), "partial")
                        .await
                        .map_err(|error| OrchionError::Download {
                            source_name,
                            repo: repo.to_string(),
                            message: error.to_string(),
                        })?;
                    return Err(OrchionError::Download {
                        source_name,
                        repo: repo.to_string(),
                        message: "simulated failure".to_string(),
                    });
                }
                tokio::fs::create_dir_all(target).await.map_err(|error| {
                    OrchionError::Download {
                        source_name,
                        repo: repo.to_string(),
                        message: error.to_string(),
                    }
                })?;
                let config = if self.write_empty_config { "" } else { "{}" };
                tokio::fs::write(target.join("config.json"), config)
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name,
                        repo: repo.to_string(),
                        message: error.to_string(),
                    })?;
                if let Some(files) = files {
                    for file_name in files {
                        tokio::fs::write(target.join(file_name), b"asset")
                            .await
                            .map_err(|error| OrchionError::Download {
                                source_name,
                                repo: repo.to_string(),
                                message: error.to_string(),
                            })?;
                    }
                }
                let omit_tokenizer_sources = self.omit_asr_tokenizer_sources
                    || (self.omit_huggingface_asr_tokenizer_sources
                        && source_name == "huggingface");
                if !omit_tokenizer_sources {
                    write_asr_tokenizer_sources(target).await;
                }
                if self.write_ocr_vl_weights {
                    write_complete_ocr_vl_cache(target).await;
                }
                Ok(if self.omit_resolved_revision {
                    ProviderDownloadResult::unresolved()
                } else {
                    ProviderDownloadResult::with_resolved_revision(
                        "1111111111111111111111111111111111111111",
                    )
                })
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
    async fn auto_falls_back_when_huggingface_prepare_fails() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            omit_huggingface_asr_tokenizer_sources: true,
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        let path = ModelDownloader::new(DownloadSource::Auto)
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["huggingface", "modelscope"]);
        assert!(path.join("tokenizer.json").exists());
        let manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(path.join(READY_MANIFEST_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["source"], "modelscope");
    }

    #[tokio::test]
    async fn ready_manifest_records_download_identity_and_required_file_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        let path = ModelDownloader::new(DownloadSource::HuggingFace)
            .with_revision("0123456789abcdef0123456789abcdef01234567")
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        let manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(path.join(READY_MANIFEST_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["source"], "huggingface");
        assert_eq!(manifest["repo_id"], "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(
            manifest["revision"],
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(
            manifest["repositories"],
            serde_json::json!([{
                "identity": "Qwen/Qwen3-ASR-0.6B",
                "repo_id": "Qwen/Qwen3-ASR-0.6B",
                "requested_revision": "0123456789abcdef0123456789abcdef01234567",
                "resolved_revision": "1111111111111111111111111111111111111111"
            }])
        );
        assert_eq!(
            manifest["downloaded_repos"],
            serde_json::json!(["Qwen/Qwen3-ASR-0.6B"])
        );
        let config = manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|file| file["path"] == "config.json")
            .unwrap();
        assert_eq!(config["file_type"], "file");
        assert_eq!(config["size"], 2);
        assert_eq!(
            config["sha256"],
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }

    #[tokio::test]
    async fn ready_manifest_records_the_selected_providers_actual_repo() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let repos = Arc::clone(&client.repos);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        let path = ModelDownloader::new(DownloadSource::ModelScope)
            .download_with_client(DifferentProviderReposModel, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*repos.lock().unwrap(), &["Acme/ModelScopeModel"]);
        let manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(path.join(READY_MANIFEST_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["source"], "modelscope");
        assert_eq!(manifest["repo_id"], "Acme/ModelScopeModel");
        assert_eq!(
            manifest["downloaded_repos"],
            serde_json::json!(["Acme/ModelScopeModel"])
        );
        assert_eq!(manifest["revision"], "master");
    }

    #[tokio::test]
    async fn explicit_source_does_not_reuse_cache_from_another_source() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        ModelDownloader::new(DownloadSource::ModelScope)
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();
        ModelDownloader::new(DownloadSource::HuggingFace)
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["modelscope", "huggingface"]);
    }

    #[tokio::test]
    async fn ready_cache_with_same_size_modified_required_file_is_redownloaded() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::HuggingFace);

        let path = downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();
        tokio::fs::write(path.join("config.json"), "[]")
            .await
            .unwrap();
        downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["huggingface", "huggingface"]);
        assert_eq!(
            tokio::fs::read_to_string(path.join("config.json"))
                .await
                .unwrap(),
            "{}"
        );
    }

    #[tokio::test]
    async fn configured_revision_does_not_reuse_cache_from_another_revision() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        ModelDownloader::new(DownloadSource::HuggingFace)
            .with_revision("revision-a")
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();
        let path = ModelDownloader::new(DownloadSource::HuggingFace)
            .with_revision("revision-b")
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["huggingface", "huggingface"]);
        let manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(path.join(READY_MANIFEST_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["revision"], "revision-b");
    }

    #[tokio::test]
    async fn zero_size_required_file_is_not_published_as_ready() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            write_empty_config: true,
            ..Default::default()
        };
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        let error = ModelDownloader::new(DownloadSource::HuggingFace)
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            OrchionError::DownloadFallbackExhausted { failures, .. }
                if failures.len() == 1
                    && failures[0].message.contains("required cache files")
        ));
        assert!(!qwen_asr_06b().cache_path(dir.path()).exists());
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

        assert!(error.to_string().contains("reserved `.orchion`"));
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
    #[cfg(unix)]
    async fn downloader_rejects_symlinked_staging_directory() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("models");
        let external = dir.path().join("external");
        tokio::fs::create_dir_all(&cache_dir).await.unwrap();
        tokio::fs::create_dir_all(&external).await.unwrap();
        tokio::fs::create_dir_all(cache_dir.join(CACHE_STATE_DIR))
            .await
            .unwrap();
        std::os::unix::fs::symlink(
            &external,
            cache_state_path(&cache_dir, DOWNLOAD_STAGING_DIR),
        )
        .unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let error = ModelDownloader::default()
            .download_with_client(qwen_asr_06b(), &cache_dir, &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("not a regular directory"));
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
    async fn internal_cache_state_is_contained_in_orchion_directory() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        ModelDownloader::default()
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        let state_dir = dir.path().join(CACHE_STATE_DIR);
        assert!(state_dir.join(DOWNLOAD_STAGING_DIR).is_dir());
        assert!(state_dir.join(MODEL_LOCK_DIR).is_dir());
        assert!(state_dir.join(PUBLICATION_LOCK_FILE).is_file());
        assert!(!state_dir.join(PUBLISH_TRANSACTION_DIR).exists());

        let mut entries = tokio::fs::read_dir(dir.path()).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            assert!(name == CACHE_STATE_DIR || !name.starts_with(".orchion"));
        }
    }

    #[tokio::test]
    async fn stale_model_staging_is_removed_without_touching_other_models() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let model = qwen_asr_06b();

        downloader
            .download_with_client(model.clone(), dir.path(), &client, &env)
            .await
            .unwrap();

        let staging_dir = cache_state_path(dir.path(), DOWNLOAD_STAGING_DIR);
        let stale = staging_dir.join(format!(
            "{}stale",
            transaction::model_staging_prefix(model.huggingface_repo())
        ));
        let other_model_staging = staging_dir.join(format!(
            "{}active",
            transaction::model_staging_prefix(qwen_tts_base().huggingface_repo())
        ));
        tokio::fs::create_dir_all(&stale).await.unwrap();
        tokio::fs::create_dir_all(&other_model_staging)
            .await
            .unwrap();

        downloader
            .download_with_client(model, dir.path(), &client, &env)
            .await
            .unwrap();

        assert!(!stale.exists());
        assert!(other_model_staging.exists());
    }

    #[tokio::test]
    async fn cancelling_download_does_not_release_publication_resources_early() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().to_path_buf();
        let model = qwen_asr_06b();
        let model_key = model.huggingface_repo().to_string();
        let staging_prefix = transaction::model_staging_prefix(&model_key);
        let target = model.cache_path(&cache_dir);
        let client = FakeDownloadClient {
            delay: Duration::from_millis(200),
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };
        let download_cache = cache_dir.clone();
        let download = tokio::spawn(async move {
            ModelDownloader::default()
                .download_with_client(model, download_cache, &client, &env)
                .await
        });

        tokio::time::timeout(Duration::from_secs(2), async {
            while calls.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let publication_lock =
            transaction::acquire_publication_lock(&cache_dir, "test-publication")
                .await
                .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let mut entries =
                    tokio::fs::read_dir(cache_state_path(&cache_dir, DOWNLOAD_STAGING_DIR))
                        .await
                        .unwrap();
                let mut manifest_is_staged = false;
                while let Some(entry) = entries.next_entry().await.unwrap() {
                    let is_staging = entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(&staging_prefix);
                    if is_staging
                        && tokio::fs::try_exists(
                            entry.path().join(&model_key).join(READY_MANIFEST_FILE),
                        )
                        .await
                        .unwrap()
                    {
                        manifest_is_staged = true;
                        break;
                    }
                }
                if manifest_is_staged {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        download.abort();
        assert!(download.await.unwrap_err().is_cancelled());

        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                transaction::acquire_model_lock(&cache_dir, &model_key),
            )
            .await
            .is_err()
        );
        drop(publication_lock);
        let completed_lock = tokio::time::timeout(
            Duration::from_secs(2),
            transaction::acquire_model_lock(&cache_dir, &model_key),
        )
        .await
        .unwrap()
        .unwrap();
        drop(completed_lock);

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
        write_ready_manifest(&model, &target, ResolvedSource::HuggingFace).await;

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
        let transaction_dir = cache_state_path(dir.path(), PUBLISH_TRANSACTION_DIR);
        let backup = repo_cache_path(&transaction_dir, model.huggingface_repo());
        tokio::fs::create_dir_all(&backup).await.unwrap();
        tokio::fs::write(backup.join("config.json"), "{}")
            .await
            .unwrap();
        write_asr_tokenizer_json(&backup).await;
        write_ready_manifest(&model, &backup, ResolvedSource::HuggingFace).await;
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
        write_asr_tokenizer_json(&target).await;
        write_ready_manifest(&model, &target, ResolvedSource::ModelScope).await;
        tokio::fs::remove_file(target.join("tokenizer.json"))
            .await
            .unwrap();

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
        write_ocr_vl_weight_index(&target).await;
        for file_name in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            tokio::fs::write(target.join(file_name), b"weights")
                .await
                .unwrap();
        }
        write_ready_manifest(&model, &target, ResolvedSource::HuggingFace).await;
        tokio::fs::remove_file(target.join("model-00002-of-00002.safetensors"))
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
        write_ocr_vl_weight_index(&target).await;
        tokio::fs::write(target.join("model.safetensors"), b"weights")
            .await
            .unwrap();
        for file_name in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            tokio::fs::write(target.join(file_name), b"weights")
                .await
                .unwrap();
        }
        write_ready_manifest(&model, &target, ResolvedSource::HuggingFace).await;
        tokio::fs::remove_file(target.join("model-00002-of-00002.safetensors"))
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
        let revisions = Arc::clone(&client.revisions);
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
        assert_eq!(&*revisions.lock().unwrap(), &["master"]);
        assert_eq!(
            &*file_filters.lock().unwrap(),
            &[Some(vec![
                "pp-ocrv5_mobile_det.onnx".to_string(),
                "pp-ocrv5_mobile_rec.onnx".to_string(),
                "ppocrv5_dict.txt".to_string()
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
        let manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(path.join(READY_MANIFEST_FILE))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest["downloaded_repos"],
            serde_json::json!(["greatv/oar-ocr"])
        );
        assert_eq!(manifest["repositories"][0]["requested_revision"], "master");
        assert_eq!(
            manifest["repositories"][0]["resolved_revision"],
            "1111111111111111111111111111111111111111"
        );
    }

    #[tokio::test]
    async fn canonical_revision_is_not_applied_to_auxiliary_repositories() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let revisions = Arc::clone(&client.revisions);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        ModelDownloader::new(DownloadSource::Auto)
            .with_revision("canonical-revision")
            .download_with_client(KnownOcrModel::PpOcrV5Mobile, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*revisions.lock().unwrap(), &["master"]);
    }

    #[tokio::test]
    async fn repository_revision_is_applied_only_to_the_selected_auxiliary_repo() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let revisions = Arc::clone(&client.revisions);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        ModelDownloader::new(DownloadSource::Auto)
            .with_repository_revision("greatv/oar-ocr", "asset-revision")
            .download_with_client(KnownOcrModel::PpOcrV5Mobile, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*revisions.lock().unwrap(), &["asset-revision"]);
    }

    #[tokio::test]
    async fn explicit_huggingface_rejects_modelscope_only_assets() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };

        let error = ModelDownloader::new(DownloadSource::HuggingFace)
            .download_with_client(KnownOcrModel::PpOcrV5Mobile, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("only available from ModelScope"));
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn auto_with_explicit_huggingface_env_rejects_modelscope_only_assets() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: Some("huggingface".to_string()),
            hf_endpoint: None,
        };

        let error = ModelDownloader::new(DownloadSource::Auto)
            .download_with_client(KnownOcrModel::PpOcrV5Mobile, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("only available from ModelScope"));
        assert!(client.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unresolved_mutable_revision_reuses_verified_local_cache() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            omit_resolved_revision: true,
            ..Default::default()
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::HuggingFace);

        downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();
        downloader
            .download_with_client(qwen_asr_06b(), dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(&*calls.lock().unwrap(), &["huggingface"]);
    }

    async fn write_ready_manifest<M: ModelSpec>(model: &M, target: &Path, source: ResolvedSource) {
        let downloader = ModelDownloader::new(match source {
            ResolvedSource::HuggingFace => DownloadSource::HuggingFace,
            ResolvedSource::ModelScope => DownloadSource::ModelScope,
        });
        let downloads = downloader
            .repository_requests(model, &source, model_hub_assets(model))
            .unwrap()
            .into_iter()
            .map(|request| RepositoryDownload {
                request,
                resolved_revision: Some("1111111111111111111111111111111111111111".to_string()),
            })
            .collect::<Vec<_>>();
        super::write_ready_manifest(
            model,
            target,
            &source,
            source.repo(model),
            source.default_revision(),
            &downloads,
        )
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
