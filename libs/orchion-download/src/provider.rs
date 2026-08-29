use orchion_core::{DownloadRetryability, ModelId, OrchionError, Result};
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

pub type DownloadFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
pub type ResolvedDownloadFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ProviderDownloadResult>> + Send + 'a>>;
pub type PreflightFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ProviderPreflightResult>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderPreflightResult {
    files: Vec<String>,
    resolved_revision: Option<String>,
}

impl ProviderPreflightResult {
    pub(crate) fn new(files: Vec<String>) -> Self {
        Self {
            files,
            resolved_revision: None,
        }
    }

    pub(crate) fn with_resolved_revision(
        files: Vec<String>,
        resolved_revision: impl Into<String>,
    ) -> Self {
        Self {
            files,
            resolved_revision: Some(resolved_revision.into()),
        }
    }

    #[must_use]
    pub fn files(&self) -> &[String] {
        &self.files
    }

    #[must_use]
    pub fn resolved_revision(&self) -> Option<&str> {
        self.resolved_revision.as_deref()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderDownloadResult {
    resolved_revision: Option<String>,
}

impl ProviderDownloadResult {
    #[must_use]
    pub const fn unresolved() -> Self {
        Self {
            resolved_revision: None,
        }
    }

    #[must_use]
    pub fn with_resolved_revision(revision: impl Into<String>) -> Self {
        Self {
            resolved_revision: Some(revision.into()),
        }
    }

    #[must_use]
    pub fn resolved_revision(&self) -> Option<&str> {
        self.resolved_revision.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadSource {
    Auto,
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderModel<'a> {
    huggingface_repo: &'a str,
    modelscope_repo: &'a str,
    repository_identity: Option<&'a str>,
}

impl<'a> ProviderModel<'a> {
    pub(crate) const fn new(huggingface_repo: &'a str, modelscope_repo: &'a str) -> Self {
        Self {
            huggingface_repo,
            modelscope_repo,
            repository_identity: None,
        }
    }

    pub(crate) const fn for_repository(repository: &'a str) -> Self {
        Self {
            huggingface_repo: repository,
            modelscope_repo: repository,
            repository_identity: Some(repository),
        }
    }

    #[must_use]
    pub const fn huggingface_repo(self) -> &'a str {
        self.huggingface_repo
    }

    #[must_use]
    pub const fn modelscope_repo(self) -> &'a str {
        self.modelscope_repo
    }

    /// Returns the provider-neutral identity when locating an auxiliary repository.
    #[must_use]
    pub const fn repository_identity(self) -> Option<&'a str> {
        self.repository_identity
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderDownloadRequest<'a> {
    repository: &'a str,
    revision: &'a str,
    cache_dir: &'a Path,
    target: &'a Path,
    files: Option<&'a [&'a str]>,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderPreflightRequest<'a> {
    repository: &'a str,
    revision: &'a str,
    files: Option<&'a [&'a str]>,
}

impl<'a> ProviderPreflightRequest<'a> {
    pub(crate) const fn new(
        repository: &'a str,
        revision: &'a str,
        files: Option<&'a [&'a str]>,
    ) -> Self {
        Self {
            repository,
            revision,
            files,
        }
    }

    #[must_use]
    pub const fn repository(&self) -> &'a str {
        self.repository
    }

    #[must_use]
    pub const fn revision(&self) -> &'a str {
        self.revision
    }

    #[must_use]
    pub const fn files(&self) -> Option<&'a [&'a str]> {
        self.files
    }
}

impl<'a> ProviderDownloadRequest<'a> {
    pub(crate) const fn new(
        repository: &'a str,
        revision: &'a str,
        cache_dir: &'a Path,
        target: &'a Path,
        files: Option<&'a [&'a str]>,
    ) -> Self {
        Self {
            repository,
            revision,
            cache_dir,
            target,
            files,
        }
    }

    #[must_use]
    pub const fn repository(&self) -> &'a str {
        self.repository
    }

    #[must_use]
    pub const fn revision(&self) -> &'a str {
        self.revision
    }

    #[must_use]
    pub const fn cache_dir(&self) -> &'a Path {
        self.cache_dir
    }

    #[must_use]
    pub const fn target(&self) -> &'a Path {
        self.target
    }

    #[must_use]
    pub const fn files(&self) -> Option<&'a [&'a str]> {
        self.files
    }
}

pub trait DownloadProvider: Send + Sync + 'static {
    fn label(&self) -> &'static str;
    fn default_revision(&self) -> &str;
    fn repository(&self, model: ProviderModel<'_>) -> String;
    fn download<'a>(&'a self, request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a>;

    fn preflight<'a>(&'a self, request: ProviderPreflightRequest<'a>) -> PreflightFuture<'a> {
        Box::pin(async move {
            Err(provider_error(
                self.label(),
                request.repository(),
                "provider does not expose typed preflight metadata",
                DownloadRetryability::Terminal,
            ))
        })
    }

    fn download_with_result<'a>(
        &'a self,
        request: ProviderDownloadRequest<'a>,
    ) -> ResolvedDownloadFuture<'a> {
        Box::pin(async move {
            let revision = request.revision().to_string();
            self.download(request).await?;
            Ok(if is_immutable_revision(&revision) {
                ProviderDownloadResult::with_resolved_revision(revision)
            } else {
                ProviderDownloadResult::unresolved()
            })
        })
    }
}

#[derive(Clone, Default)]
pub struct DownloadProviderRegistry {
    providers: Vec<Arc<dyn DownloadProvider>>,
}

impl DownloadProviderRegistry {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_provider<P>(mut self, provider: P) -> Self
    where
        P: DownloadProvider,
    {
        self.register(provider);
        self
    }

    #[must_use]
    pub fn with_shared_provider(mut self, provider: Arc<dyn DownloadProvider>) -> Self {
        self.register_shared(provider);
        self
    }

    pub fn register<P>(&mut self, provider: P)
    where
        P: DownloadProvider,
    {
        self.register_shared(Arc::new(provider));
    }

    pub fn register_shared(&mut self, provider: Arc<dyn DownloadProvider>) {
        if let Some(existing) = self
            .providers
            .iter_mut()
            .find(|existing| existing.label() == provider.label())
        {
            *existing = provider;
        } else {
            self.providers.push(provider);
        }
    }

    #[must_use]
    pub fn provider(&self, label: &str) -> Option<Arc<dyn DownloadProvider>> {
        self.providers
            .iter()
            .find(|provider| provider.label() == label)
            .map(Arc::clone)
    }

    pub(crate) fn providers(&self) -> &[Arc<dyn DownloadProvider>] {
        &self.providers
    }
}

impl fmt::Debug for DownloadProviderRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels = self
            .providers
            .iter()
            .map(|provider| provider.label())
            .collect::<Vec<_>>();
        formatter
            .debug_struct("DownloadProviderRegistry")
            .field("providers", &labels)
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct HubProviderOptions {
    token: Option<String>,
    concurrency: Option<usize>,
    max_retries: Option<u32>,
    metadata_endpoint: Option<String>,
}

impl HubProviderOptions {
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    #[must_use]
    pub const fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    #[must_use]
    pub fn with_metadata_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.metadata_endpoint = Some(endpoint.into());
        self
    }

    fn with_env_token(mut self, names: &[&str]) -> Self {
        if self.token.is_none() {
            self.token = names
                .iter()
                .find_map(|name| std::env::var(name).ok().filter(|token| !token.is_empty()));
        }
        self
    }

    fn configure(&self, mut downloader: model_hub::ModelDownloader) -> model_hub::ModelDownloader {
        if let Some(concurrency) = self.concurrency {
            downloader = downloader.with_concurrency(concurrency);
        }
        if let Some(max_retries) = self.max_retries {
            downloader = downloader.with_max_retries(max_retries);
        }
        downloader
    }
}

impl fmt::Debug for HubProviderOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HubProviderOptions")
            .field("token", &self.token.as_ref().map(|_| Redacted))
            .field("concurrency", &self.concurrency)
            .field("max_retries", &self.max_retries)
            .field("metadata_endpoint", &self.metadata_endpoint)
            .finish()
    }
}

struct Redacted;

impl fmt::Debug for Redacted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("REDACTED")
    }
}

#[derive(Clone)]
pub struct HuggingFaceProvider {
    options: HubProviderOptions,
}

impl HuggingFaceProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(HubProviderOptions::default())
    }

    #[must_use]
    pub fn with_options(options: HubProviderOptions) -> Self {
        Self {
            options: options.with_env_token(&["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]),
        }
    }

    fn hub_provider(&self) -> model_hub::HubProvider {
        model_hub::HubProvider::HuggingFace {
            token: self.options.token.clone(),
        }
    }
}

impl Default for HuggingFaceProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HuggingFaceProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HuggingFaceProvider")
            .field("options", &self.options)
            .finish()
    }
}

impl DownloadProvider for HuggingFaceProvider {
    fn label(&self) -> &'static str {
        "huggingface"
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn default_revision(&self) -> &str {
        "main"
    }

    fn repository(&self, model: ProviderModel<'_>) -> String {
        model.huggingface_repo().to_string()
    }

    fn download<'a>(&'a self, request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
        Box::pin(async move {
            self.download_with_result(request).await?;
            Ok(())
        })
    }

    fn preflight<'a>(&'a self, request: ProviderPreflightRequest<'a>) -> PreflightFuture<'a> {
        preflight_hugging_face(self, request)
    }

    fn download_with_result<'a>(
        &'a self,
        request: ProviderDownloadRequest<'a>,
    ) -> ResolvedDownloadFuture<'a> {
        download_from_model_hub(self, &self.options, self.hub_provider(), request)
    }
}

#[derive(Clone)]
pub struct ModelScopeProvider {
    options: HubProviderOptions,
}

impl ModelScopeProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(HubProviderOptions::default())
    }

    #[must_use]
    pub fn with_options(options: HubProviderOptions) -> Self {
        Self {
            options: options.with_env_token(&["MODELSCOPE_API_TOKEN"]),
        }
    }

    fn hub_provider(&self) -> model_hub::HubProvider {
        model_hub::HubProvider::ModelScope {
            token: self.options.token.clone(),
        }
    }
}

impl Default for ModelScopeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ModelScopeProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelScopeProvider")
            .field("options", &self.options)
            .finish()
    }
}

impl DownloadProvider for ModelScopeProvider {
    fn label(&self) -> &'static str {
        "modelscope"
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn default_revision(&self) -> &str {
        "master"
    }

    fn repository(&self, model: ProviderModel<'_>) -> String {
        model.modelscope_repo().to_string()
    }

    fn download<'a>(&'a self, request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
        Box::pin(async move {
            self.download_with_result(request).await?;
            Ok(())
        })
    }

    fn preflight<'a>(&'a self, request: ProviderPreflightRequest<'a>) -> PreflightFuture<'a> {
        preflight_model_scope(self, request)
    }

    fn download_with_result<'a>(
        &'a self,
        request: ProviderDownloadRequest<'a>,
    ) -> ResolvedDownloadFuture<'a> {
        download_from_model_hub(self, &self.options, self.hub_provider(), request)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "resolves revision metadata before paginating the immutable tree"
)]
fn preflight_hugging_face<'a>(
    provider: &'a HuggingFaceProvider,
    request: ProviderPreflightRequest<'a>,
) -> PreflightFuture<'a> {
    Box::pin(async move {
        let endpoint = provider
            .options
            .metadata_endpoint
            .clone()
            .unwrap_or_else(|| {
                std::env::var("HF_ENDPOINT")
                    .unwrap_or_else(|_| "https://huggingface.co".to_string())
            });
        let client = preflight_client(provider.label(), request.repository(), &provider.options)?;
        let revision_url = hugging_face_api_url(
            provider.label(),
            request.repository(),
            request.revision(),
            &endpoint,
            "revision",
            request.revision(),
            false,
        )?;
        let revision_response = client.get(revision_url).send().await.map_err(|error| {
            classify_reqwest_error(provider.label(), request.repository(), &error)
        })?;
        let revision_response =
            require_preflight_success(provider.label(), request.repository(), revision_response)?;
        let revision_metadata = revision_response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| {
                classify_reqwest_error(provider.label(), request.repository(), &error)
            })?;
        let resolved_revision = revision_metadata["sha"].as_str().ok_or_else(|| {
            provider_error(
                provider.label(),
                request.repository(),
                "Hugging Face revision metadata omitted immutable sha",
                DownloadRetryability::Terminal,
            )
        })?;
        if !is_immutable_revision(resolved_revision) {
            return Err(provider_error(
                provider.label(),
                request.repository(),
                "Hugging Face revision metadata returned malformed sha",
                DownloadRetryability::Terminal,
            ));
        }
        let resolved_revision = resolved_revision.to_string();
        let tree_url = hugging_face_api_url(
            provider.label(),
            request.repository(),
            request.revision(),
            &endpoint,
            "tree",
            &resolved_revision,
            true,
        )?;
        let mut next = Some(tree_url.to_string());
        let mut available = Vec::new();
        while let Some(url) = next.take() {
            let response = client.get(url).send().await.map_err(|error| {
                classify_reqwest_error(provider.label(), request.repository(), &error)
            })?;
            let response =
                require_preflight_success(provider.label(), request.repository(), response)?;
            if let Some(revision) = response
                .headers()
                .get("x-repo-commit")
                .and_then(|value| value.to_str().ok())
            {
                let mut expected = Some(resolved_revision.clone());
                require_consistent_immutable_revision(
                    provider.label(),
                    request.repository(),
                    &mut expected,
                    revision,
                )?;
            }
            next = response
                .headers()
                .get(reqwest::header::LINK)
                .and_then(|value| value.to_str().ok())
                .and_then(preflight_link_next);
            let metadata = response
                .json::<serde_json::Value>()
                .await
                .map_err(|error| {
                    classify_reqwest_error(provider.label(), request.repository(), &error)
                })?;
            available.extend(
                metadata
                    .as_array()
                    .ok_or_else(|| {
                        provider_error(
                            provider.label(),
                            request.repository(),
                            "invalid Hugging Face file metadata",
                            DownloadRetryability::Terminal,
                        )
                    })?
                    .iter()
                    .filter(|entry| entry["type"].as_str().is_none_or(|kind| kind == "file"))
                    .filter_map(|entry| entry["path"].as_str().map(str::to_string)),
            );
        }
        let result = resolve_preflight_files(provider.label(), request, available)?;
        Ok(ProviderPreflightResult::with_resolved_revision(
            result.files,
            resolved_revision,
        ))
    })
}

fn hugging_face_api_url(
    source_name: &'static str,
    repository: &str,
    requested_revision: &str,
    endpoint: &str,
    operation: &str,
    operation_revision: &str,
    recursive: bool,
) -> Result<reqwest::Url> {
    let repository_id = ModelId::parse(repository).map_err(|error| {
        provider_error(
            source_name,
            repository,
            format!("invalid repository path: {error}"),
            DownloadRetryability::Terminal,
        )
    })?;
    if !valid_requested_revision(requested_revision)
        || !valid_requested_revision(operation_revision)
    {
        return Err(provider_error(
            source_name,
            repository,
            "invalid revision path",
            DownloadRetryability::Terminal,
        ));
    }
    let mut url = reqwest::Url::parse(endpoint).map_err(|error| {
        provider_error(
            source_name,
            repository,
            format!("invalid metadata endpoint: {error}"),
            DownloadRetryability::Terminal,
        )
    })?;
    {
        let mut segments = url.path_segments_mut().map_err(|()| {
            provider_error(
                source_name,
                repository,
                "metadata endpoint cannot contain path segments",
                DownloadRetryability::Terminal,
            )
        })?;
        segments.pop_if_empty();
        segments.extend([
            "api",
            "models",
            repository_id.vendor(),
            repository_id.name(),
        ]);
        segments.push(operation);
        segments.push(operation_revision);
    }
    if recursive {
        url.query_pairs_mut().append_pair("recursive", "1");
    }
    Ok(url)
}

fn valid_requested_revision(revision: &str) -> bool {
    !revision.trim().is_empty()
        && !revision.contains(['%', '\\', '?', '#'])
        && !revision.chars().any(char::is_control)
        && revision
            .split('/')
            .all(|segment| !segment.is_empty() && !matches!(segment, "." | ".."))
}

fn preflight_model_scope<'a>(
    provider: &'a ModelScopeProvider,
    request: ProviderPreflightRequest<'a>,
) -> PreflightFuture<'a> {
    Box::pin(async move {
        let endpoint = provider
            .options
            .metadata_endpoint
            .as_deref()
            .unwrap_or("https://modelscope.cn");
        let url = format!(
            "{}/api/v1/models/{}/repo/files?Recursive=true&Revision={}",
            endpoint.trim_end_matches('/'),
            request.repository(),
            request.revision()
        );
        let response = preflight_client(provider.label(), request.repository(), &provider.options)?
            .get(url)
            .send()
            .await
            .map_err(|error| {
                classify_reqwest_error(provider.label(), request.repository(), &error)
            })?;
        let response = require_preflight_success(provider.label(), request.repository(), response)?;
        let metadata = response
            .json::<serde_json::Value>()
            .await
            .map_err(|error| {
                classify_reqwest_error(provider.label(), request.repository(), &error)
            })?;
        if metadata["Success"].as_bool() != Some(true) {
            return Err(provider_error(
                provider.label(),
                request.repository(),
                "ModelScope metadata reported failure",
                DownloadRetryability::Terminal,
            ));
        }
        let available = metadata["Data"]["Files"]
            .as_array()
            .ok_or_else(|| {
                provider_error(
                    provider.label(),
                    request.repository(),
                    "invalid ModelScope file metadata",
                    DownloadRetryability::Terminal,
                )
            })?
            .iter()
            .filter(|entry| entry["Type"].as_str().is_none_or(|kind| kind == "blob"))
            .filter_map(|entry| entry["Path"].as_str().map(str::to_string))
            .collect::<Vec<_>>();
        let resolved_revision = modelscope_resolved_revision(&metadata).ok_or_else(|| {
            provider_error(
                provider.label(),
                request.repository(),
                "ModelScope metadata did not resolve the revision to an immutable commit",
                DownloadRetryability::Terminal,
            )
        })?;
        let result = resolve_preflight_files(provider.label(), request, available)?;
        Ok(ProviderPreflightResult::with_resolved_revision(
            result.files,
            resolved_revision,
        ))
    })
}

fn require_consistent_immutable_revision(
    source_name: &'static str,
    repo: &str,
    current: &mut Option<String>,
    revision: &str,
) -> Result<()> {
    if !is_immutable_revision(revision) {
        return Err(provider_error(
            source_name,
            repo,
            "provider metadata returned a mutable or invalid revision",
            DownloadRetryability::Terminal,
        ));
    }
    if current
        .as_deref()
        .is_some_and(|current| current != revision)
    {
        return Err(provider_error(
            source_name,
            repo,
            "provider metadata pages returned inconsistent revisions",
            DownloadRetryability::Terminal,
        ));
    }
    *current = Some(revision.to_string());
    Ok(())
}

fn modelscope_resolved_revision(metadata: &serde_json::Value) -> Option<String> {
    for pointer in [
        "/Data/Revision",
        "/Data/CommitId",
        "/Data/Commit/Id",
        "/Data/LatestCommitter/Id",
    ] {
        if let Some(revision) = metadata
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            && is_immutable_revision(revision)
        {
            return Some(revision.to_string());
        }
    }
    let short = metadata["Data"]["LatestCommitter"]["ShortId"].as_str()?;
    if short.is_empty() || !short.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut revisions = metadata["Data"]["Files"]
        .as_array()?
        .iter()
        .filter_map(|file| file["Revision"].as_str())
        .filter(|revision| revision.starts_with(short) && is_immutable_revision(revision))
        .map(str::to_string)
        .collect::<Vec<_>>();
    revisions.sort();
    revisions.dedup();
    (revisions.len() == 1).then(|| revisions.remove(0))
}

fn preflight_client(
    source_name: &'static str,
    repo: &str,
    options: &HubProviderOptions,
) -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(token) = &options.token {
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).map_err(
            |error| {
                provider_error(
                    source_name,
                    repo,
                    error.to_string(),
                    DownloadRetryability::Terminal,
                )
            },
        )?;
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|error| classify_reqwest_error(source_name, repo, &error))
}

fn require_preflight_success(
    source_name: &'static str,
    repo: &str,
    response: reqwest::Response,
) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    Err(provider_error(
        source_name,
        repo,
        format!("HTTP {status}"),
        classify_status(status),
    ))
}

fn resolve_preflight_files(
    source_name: &'static str,
    request: ProviderPreflightRequest<'_>,
    mut available: Vec<String>,
) -> Result<ProviderPreflightResult> {
    available.sort();
    available.dedup();
    if available.is_empty() || available.iter().any(|path| !valid_artifact_path(path)) {
        return Err(provider_error(
            source_name,
            request.repository(),
            "provider metadata did not produce a trustworthy nonempty file plan",
            DownloadRetryability::Terminal,
        ));
    }
    if let Some(requested) = request.files() {
        if requested.is_empty() {
            return Err(provider_error(
                source_name,
                request.repository(),
                "artifact request contained an empty file plan",
                DownloadRetryability::Terminal,
            ));
        }
        let available = available
            .iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>();
        if let Some(missing) = requested.iter().find(|file| !available.contains(**file)) {
            return Err(provider_error(
                source_name,
                request.repository(),
                format!("file not found: {missing}"),
                DownloadRetryability::RetryableNotFound,
            ));
        }
        return Ok(ProviderPreflightResult::new(
            requested.iter().map(|file| (*file).to_string()).collect(),
        ));
    }
    Ok(ProviderPreflightResult::new(available))
}

fn valid_artifact_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with(['/', '\\'])
        && !path.contains(['\\', '?', '#'])
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && !matches!(segment, "." | ".."))
}

fn preflight_link_next(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, relation) = part.trim().split_once(';')?;
        (relation.trim() == r#"rel="next""#).then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

fn classify_reqwest_error(
    source_name: &'static str,
    repo: &str,
    error: &reqwest::Error,
) -> OrchionError {
    let retryability = error.status().map_or_else(
        || {
            if error.is_timeout() || error.is_connect() {
                DownloadRetryability::RetryableNetwork
            } else {
                DownloadRetryability::Terminal
            }
        },
        classify_status,
    );
    provider_error(source_name, repo, error.to_string(), retryability)
}

fn classify_reqwest_error_ref(error: &reqwest::Error) -> DownloadRetryability {
    error.status().map_or_else(
        || {
            if error.is_timeout() || error.is_connect() {
                DownloadRetryability::RetryableNetwork
            } else {
                DownloadRetryability::Terminal
            }
        },
        classify_status,
    )
}

fn classify_status(status: reqwest::StatusCode) -> DownloadRetryability {
    if status == reqwest::StatusCode::NOT_FOUND {
        DownloadRetryability::RetryableNotFound
    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        DownloadRetryability::RetryableRateLimit
    } else if status.is_server_error() {
        DownloadRetryability::RetryableServer
    } else {
        DownloadRetryability::Terminal
    }
}

fn provider_error(
    source_name: &'static str,
    repo: &str,
    message: impl Into<String>,
    retryability: DownloadRetryability,
) -> OrchionError {
    OrchionError::ProviderDownload {
        source_name,
        repo: repo.to_string(),
        message: message.into(),
        retryability,
    }
}

fn download_from_model_hub<'a>(
    provider: &'a dyn DownloadProvider,
    options: &'a HubProviderOptions,
    hub_provider: model_hub::HubProvider,
    request: ProviderDownloadRequest<'a>,
) -> ResolvedDownloadFuture<'a> {
    Box::pin(async move {
        let downloader = model_hub::ModelDownloader::new(hub_provider).map_err(|error| {
            provider_error(
                provider.label(),
                request.repository(),
                error.to_string(),
                DownloadRetryability::Terminal,
            )
        })?;
        let downloader = options.configure(downloader);
        downloader
            .download(model_hub::DownloadOptions {
                repo_id: request.repository().to_string(),
                revision: Some(request.revision().to_string()),
                save_dir: request.cache_dir().to_path_buf(),
                files: request
                    .files()
                    .map(|files| files.iter().map(|file| (*file).to_string()).collect()),
            })
            .await
            .map_err(|error| {
                let retryability = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
                    .map_or(DownloadRetryability::Terminal, classify_reqwest_error_ref);
                provider_error(
                    provider.label(),
                    request.repository(),
                    error.to_string(),
                    retryability,
                )
            })?;

        let downloaded_target = repo_cache_path(request.cache_dir(), request.repository());
        if downloaded_target != request.target() {
            let parent = request
                .target()
                .parent()
                .ok_or_else(|| OrchionError::Download {
                    source_name: provider.label(),
                    repo: request.repository().to_string(),
                    message: "model cache target has no parent".to_string(),
                })?;
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: provider.label(),
                    repo: request.repository().to_string(),
                    message: error.to_string(),
                })?;
            tokio::fs::rename(downloaded_target, request.target())
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: provider.label(),
                    repo: request.repository().to_string(),
                    message: error.to_string(),
                })?;
        }
        Ok(if is_immutable_revision(request.revision()) {
            ProviderDownloadResult::with_resolved_revision(request.revision())
        } else {
            ProviderDownloadResult::unresolved()
        })
    })
}

pub(crate) fn is_immutable_revision(revision: &str) -> bool {
    matches!(revision.len(), 40 | 64) && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn repo_cache_path(cache_dir: &Path, repo: &str) -> PathBuf {
    repo.split('/')
        .fold(cache_dir.to_path_buf(), |path, segment| path.join(segment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};

    fn serve_metadata_responses(
        build: impl FnOnce(&str) -> Vec<String>,
    ) -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let responses = build(&endpoint);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let task = std::thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                recorded
                    .lock()
                    .unwrap()
                    .push(request.lines().next().unwrap_or_default().to_string());
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (endpoint, requests, task)
    }

    struct DownloadOnlyProvider;

    impl DownloadProvider for DownloadOnlyProvider {
        fn label(&self) -> &'static str {
            "download-only"
        }

        fn default_revision(&self) -> &'static str {
            "main"
        }

        fn repository(&self, model: ProviderModel<'_>) -> String {
            model.huggingface_repo().to_string()
        }

        fn download<'a>(&'a self, _request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
            Box::pin(async { Ok(()) })
        }
    }

    #[test]
    fn configured_token_reaches_model_hub_provider_and_is_redacted() {
        let secret = "hf_private_test_token";
        let provider =
            HuggingFaceProvider::with_options(HubProviderOptions::default().with_token(secret));

        assert!(matches!(
            provider.hub_provider(),
            model_hub::HubProvider::HuggingFace { token: Some(token) } if token == secret
        ));
        assert!(!format!("{provider:?}").contains(secret));
    }

    #[test]
    fn full_git_object_ids_are_immutable_revisions() {
        assert!(is_immutable_revision(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(is_immutable_revision(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
    }

    #[test]
    fn mutable_or_abbreviated_refs_are_not_immutable_revisions() {
        assert!(!is_immutable_revision("main"));
        assert!(!is_immutable_revision("0123456789abcdef"));
    }

    #[tokio::test]
    async fn hugging_face_preflight_resolves_main_then_lists_tree_without_commit_header() {
        let revision = "1111111111111111111111111111111111111111";
        let (endpoint, requests, task) = serve_metadata_responses(|_| {
            let metadata = serde_json::json!({"sha": revision}).to_string();
            let tree = r#"[{"type":"file","path":"model.onnx"}]"#;
            vec![
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
                    metadata.len()
                ),
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{tree}",
                    tree.len()
                ),
            ]
        });
        let provider = HuggingFaceProvider::with_options(
            HubProviderOptions::default().with_metadata_endpoint(endpoint.clone()),
        );

        let result = provider
            .preflight(ProviderPreflightRequest::new(
                "Owner/Repo",
                "main",
                Some(&["model.onnx"]),
            ))
            .await
            .unwrap();
        task.join().unwrap();

        assert_eq!(result.files(), ["model.onnx"]);
        assert_eq!(result.resolved_revision(), Some(revision));
        assert_eq!(
            *requests.lock().unwrap(),
            vec![
                "GET /api/models/Owner/Repo/revision/main HTTP/1.1".to_string(),
                format!("GET /api/models/Owner/Repo/tree/{revision}?recursive=1 HTTP/1.1")
            ]
        );
    }

    #[tokio::test]
    async fn hugging_face_tree_commit_header_must_match_revision_metadata() {
        let revision = "1111111111111111111111111111111111111111";
        let conflict = "2222222222222222222222222222222222222222";
        let (endpoint, _, task) = serve_metadata_responses(|_| {
            let metadata = serde_json::json!({"sha": revision}).to_string();
            let tree = r#"[{"type":"file","path":"model.onnx"}]"#;
            vec![
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
                    metadata.len()
                ),
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Repo-Commit: {conflict}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{tree}",
                    tree.len()
                ),
            ]
        });
        let provider = HuggingFaceProvider::with_options(
            HubProviderOptions::default().with_metadata_endpoint(endpoint),
        );
        let error = provider
            .preflight(ProviderPreflightRequest::new(
                "Owner/Repo",
                "main",
                Some(&["model.onnx"]),
            ))
            .await
            .unwrap_err();
        task.join().unwrap();
        assert!(error.to_string().contains("inconsistent revisions"));
    }

    #[tokio::test]
    async fn hugging_face_paginated_tree_does_not_require_commit_headers() {
        let revision = "1111111111111111111111111111111111111111";
        let (endpoint, _, task) = serve_metadata_responses(|endpoint| {
            let metadata = serde_json::json!({"sha": revision}).to_string();
            let first = r#"[{"type":"file","path":"first.onnx"}]"#;
            let second = r#"[{"type":"file","path":"second.txt"}]"#;
            vec![
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
                    metadata.len()
                ),
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nLink: <{endpoint}/page-2>; rel=\"next\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first}",
                    first.len()
                ),
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second}",
                    second.len()
                ),
            ]
        });
        let provider = HuggingFaceProvider::with_options(
            HubProviderOptions::default().with_metadata_endpoint(endpoint),
        );
        let requested = ["first.onnx", "second.txt"];
        let result = provider
            .preflight(ProviderPreflightRequest::new(
                "Owner/Repo",
                "main",
                Some(&requested),
            ))
            .await
            .unwrap();
        task.join().unwrap();
        assert_eq!(result.files(), requested);
        assert_eq!(result.resolved_revision(), Some(revision));
    }

    #[tokio::test]
    async fn modelscope_preflight_resolves_master_from_file_metadata() {
        let revision = "2222222222222222222222222222222222222222";
        let body = serde_json::json!({
            "Code": 200,
            "Success": true,
            "Data": {
                "LatestCommitter": {"ShortId": "22222222", "Id": ""},
                "Files": [{
                    "Type": "blob",
                    "Path": "model.onnx",
                    "Revision": revision
                }]
            }
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let (endpoint, _, task) = serve_metadata_responses(|_| vec![response]);
        let provider = ModelScopeProvider::with_options(
            HubProviderOptions::default().with_metadata_endpoint(endpoint),
        );

        let result = provider
            .preflight(ProviderPreflightRequest::new(
                "Owner/Repo",
                "master",
                Some(&["model.onnx"]),
            ))
            .await
            .unwrap();
        task.join().unwrap();

        assert_eq!(result.files(), ["model.onnx"]);
        assert_eq!(result.resolved_revision(), Some(revision));
    }

    #[test]
    fn http_status_retryability_is_allowlisted() {
        assert_eq!(
            classify_status(reqwest::StatusCode::NOT_FOUND),
            DownloadRetryability::RetryableNotFound
        );
        assert_eq!(
            classify_status(reqwest::StatusCode::TOO_MANY_REQUESTS),
            DownloadRetryability::RetryableRateLimit
        );
        assert_eq!(
            classify_status(reqwest::StatusCode::SERVICE_UNAVAILABLE),
            DownloadRetryability::RetryableServer
        );
        assert_eq!(
            classify_status(reqwest::StatusCode::UNAUTHORIZED),
            DownloadRetryability::Terminal
        );
        assert_eq!(
            classify_status(reqwest::StatusCode::FORBIDDEN),
            DownloadRetryability::Terminal
        );
        assert_eq!(
            classify_status(reqwest::StatusCode::BAD_REQUEST),
            DownloadRetryability::Terminal
        );
    }

    #[test]
    fn metadata_must_resolve_to_a_trustworthy_nonempty_plan() {
        let empty = resolve_preflight_files(
            "huggingface",
            ProviderPreflightRequest::new("Owner/Repo", "main", None),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            empty,
            OrchionError::ProviderDownload {
                retryability: DownloadRetryability::Terminal,
                ..
            }
        ));

        let unsafe_path = resolve_preflight_files(
            "huggingface",
            ProviderPreflightRequest::new("Owner/Repo", "main", None),
            vec!["../model.bin".to_string()],
        )
        .unwrap_err();
        assert!(matches!(
            unsafe_path,
            OrchionError::ProviderDownload {
                retryability: DownloadRetryability::Terminal,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn default_download_result_preserves_immutable_revision() {
        let cache = tempfile::tempdir().unwrap();
        let target = cache.path().join("target");
        let revision = "1111111111111111111111111111111111111111";

        let result = DownloadOnlyProvider
            .download_with_result(ProviderDownloadRequest::new(
                "Canonical/Model",
                revision,
                cache.path(),
                &target,
                None,
            ))
            .await
            .unwrap();

        assert_eq!(result.resolved_revision(), Some(revision));
    }

    #[tokio::test]
    async fn configured_token_is_not_exposed_by_provider_errors() {
        let secret = "hf_private_test_token\ninvalid";
        let provider =
            HuggingFaceProvider::with_options(HubProviderOptions::default().with_token(secret));
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");

        let error = provider
            .download(ProviderDownloadRequest::new(
                "owner/model",
                "main",
                temp.path(),
                &target,
                None,
            ))
            .await
            .unwrap_err();

        assert!(!error.to_string().contains("hf_private_test_token"));
    }
}
