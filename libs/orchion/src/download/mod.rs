use crate::error::{OrchionError, Result};
use crate::model::{ModelCategory, ModelSpec};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

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
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new(DownloadSource::Auto)
    }
}

impl ModelDownloader {
    pub const fn new(source: DownloadSource) -> Self {
        Self { source }
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
        let target = model.cache_path(cache_dir.as_ref());
        let marker = target.join(".orchion-complete");
        if tokio::fs::try_exists(&marker)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo(),
                message: error.to_string(),
            })?
        {
            prepare_cached_model(model, &target, "cache").await?;
            tracing::debug!(model = ?model, path = %target.display(), "model cache hit");
            return Ok(target);
        }
        if tokio::fs::try_exists(&target)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo(),
                message: error.to_string(),
            })?
        {
            tracing::warn!(model = ?model, path = %target.display(), "model cache is incomplete; removing before download");
            tokio::fs::remove_dir_all(&target)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: "cache",
                    repo: model.huggingface_repo(),
                    message: error.to_string(),
                })?;
        }

        let candidates = self.resolve_candidates(env, probe).await?;
        tracing::info!(
            model = ?model,
            path = %target.display(),
            source_count = candidates.len(),
            "model cache missing; starting download"
        );
        let mut failures = Vec::new();
        for candidate in candidates {
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
            match client.download(candidate, repo, &target, env).await {
                Ok(()) => {
                    prepare_cached_model(model, &target, candidate.label()).await?;
                    tokio::fs::write(
                        target.join(".orchion-complete"),
                        format!("{}\n", candidate.label()),
                    )
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name: candidate.label(),
                        repo,
                        message: error.to_string(),
                    })?;
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
                    let _ = tokio::fs::remove_dir_all(&target).await;
                    failures.push(error.to_string());
                }
            }
        }

        Err(OrchionError::DownloadFallbackExhausted {
            repo: model.huggingface_repo(),
            messages: failures.join("; "),
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
        if probe.huggingface_available(env).await {
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

struct AlwaysAvailableProbe;

impl SourceProbe for AlwaysAvailableProbe {
    fn huggingface_available<'a>(&'a self, _env: &'a DownloadEnv) -> BoxFuture<'a, bool> {
        Box::pin(async { true })
    }
}

trait DownloadClient {
    fn download<'a>(
        &'a self,
        source: ResolvedSource,
        repo: &'static str,
        target: &'a Path,
        env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<()>>;
}

struct LibraryDownloadClient;

impl DownloadClient for LibraryDownloadClient {
    fn download<'a>(
        &'a self,
        source: ResolvedSource,
        repo: &'static str,
        target: &'a Path,
        env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            match source {
                ResolvedSource::HuggingFace => download_huggingface(repo, target, env).await,
                ResolvedSource::ModelScope => download_modelscope(repo, target).await,
            }
        })
    }
}

async fn download_huggingface(repo: &'static str, target: &Path, env: &DownloadEnv) -> Result<()> {
    let staging = staging_dir(target, "hf-cache", repo, ResolvedSource::HuggingFace)?;
    let _ = tokio::fs::remove_dir_all(&staging).await;
    tokio::fs::create_dir_all(target)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: ResolvedSource::HuggingFace.label(),
            repo,
            message: error.to_string(),
        })?;

    let mut builder = hf_hub::api::tokio::ApiBuilder::from_env().with_cache_dir(staging.clone());
    if let Some(endpoint) = env.hf_endpoint.as_deref() {
        builder = builder.with_endpoint(endpoint.trim_end_matches('/').to_string());
    }
    let api = builder.build().map_err(|error| OrchionError::Download {
        source_name: ResolvedSource::HuggingFace.label(),
        repo,
        message: error.to_string(),
    })?;
    let repository = api.model(repo.to_string());
    let info = repository
        .info()
        .await
        .map_err(|error| OrchionError::Download {
            source_name: ResolvedSource::HuggingFace.label(),
            repo,
            message: error.to_string(),
        })?;
    for sibling in info.siblings {
        let filename = sibling.rfilename;
        let source_path = repository
            .download(filename.as_str())
            .await
            .map_err(|error| OrchionError::Download {
                source_name: ResolvedSource::HuggingFace.label(),
                repo,
                message: error.to_string(),
            })?;
        let destination = target.join(&filename);
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| OrchionError::Download {
                    source_name: ResolvedSource::HuggingFace.label(),
                    repo,
                    message: error.to_string(),
                })?;
        }
        tokio::fs::copy(source_path, destination)
            .await
            .map_err(|error| OrchionError::Download {
                source_name: ResolvedSource::HuggingFace.label(),
                repo,
                message: error.to_string(),
            })?;
    }
    let _ = tokio::fs::remove_dir_all(staging).await;
    Ok(())
}

async fn download_modelscope(repo: &'static str, target: &Path) -> Result<()> {
    let parent = target.parent().ok_or_else(|| OrchionError::Download {
        source_name: ResolvedSource::ModelScope.label(),
        repo,
        message: "target cache path has no parent directory".to_string(),
    })?;
    let staging = staging_dir(target, "modelscope-cache", repo, ResolvedSource::ModelScope)?;
    let _ = tokio::fs::remove_dir_all(&staging).await;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: ResolvedSource::ModelScope.label(),
            repo,
            message: error.to_string(),
        })?;
    modelscope::ModelScope::download(repo, &staging)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: ResolvedSource::ModelScope.label(),
            repo,
            message: error.to_string(),
        })?;
    let downloaded = staging.join(repo);
    tokio::fs::rename(&downloaded, target)
        .await
        .map_err(|error| OrchionError::Download {
            source_name: ResolvedSource::ModelScope.label(),
            repo,
            message: error.to_string(),
        })?;
    let _ = tokio::fs::remove_dir_all(staging).await;
    Ok(())
}

fn staging_dir(
    target: &Path,
    suffix: &str,
    repo: &'static str,
    source: ResolvedSource,
) -> Result<PathBuf> {
    let parent = target.parent().ok_or_else(|| OrchionError::Download {
        source_name: source.label(),
        repo,
        message: "target cache path has no parent directory".to_string(),
    })?;
    let name = target.file_name().ok_or_else(|| OrchionError::Download {
        source_name: source.label(),
        repo,
        message: "target cache path has no final directory name".to_string(),
    })?;
    Ok(parent.join(format!(".{}.{}", name.to_string_lossy(), suffix)))
}

async fn prepare_cached_model<M: ModelSpec>(
    model: M,
    target: &Path,
    source_name: &'static str,
) -> Result<()> {
    match model.category() {
        ModelCategory::Asr => {
            ensure_asr_tokenizer_json(target, source_name, model.huggingface_repo()).await
        }
        ModelCategory::Tts => Ok(()),
    }
}

async fn ensure_asr_tokenizer_json(
    target: &Path,
    source_name: &'static str,
    repo: &'static str,
) -> Result<()> {
    if tokio::fs::try_exists(target.join("tokenizer.json"))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo,
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
            repo,
            message: format!("failed to build tokenizer.json: {error}"),
        })?;

    tokio::fs::write(target.join("tokenizer.json"), tokenizer_json)
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo,
            message: error.to_string(),
        })?;
    tracing::info!(path = %target.join("tokenizer.json").display(), "rebuilt ASR tokenizer.json");
    Ok(())
}

async fn read_cache_file(
    target: &Path,
    file_name: &'static str,
    source_name: &'static str,
    repo: &'static str,
) -> Result<String> {
    tokio::fs::read_to_string(target.join(file_name))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo,
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
}

#[cfg(test)]
mod downloader_tests {
    use super::*;
    use crate::model::AsrModel;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct FakeDownloadClient {
        fail_huggingface: bool,
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    struct FakeProbe {
        huggingface_available: bool,
    }

    impl SourceProbe for FakeProbe {
        fn huggingface_available<'a>(&'a self, _env: &'a DownloadEnv) -> BoxFuture<'a, bool> {
            Box::pin(async move { self.huggingface_available })
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
            repo: &'static str,
            target: &'a Path,
            _env: &'a DownloadEnv,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(source.label());
                if self.fail_huggingface && source == ResolvedSource::HuggingFace {
                    return Err(OrchionError::Download {
                        source_name: source.label(),
                        repo,
                        message: "simulated failure".to_string(),
                    });
                }
                tokio::fs::create_dir_all(target).await.map_err(|error| {
                    OrchionError::Download {
                        source_name: source.label(),
                        repo,
                        message: error.to_string(),
                    }
                })?;
                tokio::fs::write(target.join("config.json"), "{}")
                    .await
                    .map_err(|error| OrchionError::Download {
                        source_name: source.label(),
                        repo,
                        message: error.to_string(),
                    })?;
                write_asr_tokenizer_sources(target).await;
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn auto_falls_back_to_modelscope_when_huggingface_fails() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            fail_huggingface: true,
            calls: Arc::new(Mutex::new(Vec::new())),
        };
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::new(DownloadSource::Auto);

        let path = downloader
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap();

        assert!(path.join("config.json").exists());
        assert!(path.join(".orchion-complete").exists());
        assert_eq!(&*calls.lock().unwrap(), &["huggingface", "modelscope"]);
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
                AsrModel::Qwen3Asr06B,
                dir.path(),
                &client,
                &FakeProbe {
                    huggingface_available: false,
                },
                &env,
            )
            .await
            .unwrap();

        assert!(path.join("config.json").exists());
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
    }

    #[tokio::test]
    async fn complete_marker_skips_download() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        write_asr_tokenizer_json(&target).await;
        tokio::fs::write(target.join(".orchion-complete"), "huggingface\n")
            .await
            .unwrap();

        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let path = downloader
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn incomplete_cache_is_removed_and_downloaded_again() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("partial.bin"), "partial")
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
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert!(!path.join("partial.bin").exists());
        assert!(path.join(".orchion-complete").exists());
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
    }

    #[tokio::test]
    async fn complete_marker_repairs_missing_asr_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        write_asr_tokenizer_sources(&target).await;
        tokio::fs::write(target.join(".orchion-complete"), "modelscope\n")
            .await
            .unwrap();

        let client = FakeDownloadClient::default();
        let calls = Arc::clone(&client.calls);
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let path = downloader
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap();

        assert_eq!(path, target);
        assert!(path.join("tokenizer.json").exists());
        assert!(
            std::fs::read_to_string(path.join("tokenizer.json"))
                .unwrap()
                .contains("\"type\":\"BPE\"")
        );
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn complete_marker_rejects_unrepairable_asr_cache() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join(".orchion-complete"), "modelscope\n")
            .await
            .unwrap();

        let client = FakeDownloadClient::default();
        let env = DownloadEnv {
            orchion_model_source: None,
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let error = downloader
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("tokenizer_config.json"));
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
}
