use orchion_core::{ModelCategory, ModelSpec, OrchionError, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const READY_MANIFEST_FILE: &str = ".orchion-ready.json";
const READY_MANIFEST_SCHEMA_VERSION: u64 = 1;
const READY_MANIFEST_LAYOUT: &str = "model-hub-native";

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
        let target = model.cache_path(cache_dir);

        if is_ready_cache(model, &target).await? {
            tracing::debug!(model = ?model, path = %target.display(), "model cache ready");
            return Ok(target);
        }

        let candidates = self.resolve_candidates(env, probe).await?;
        tracing::info!(
            model = ?model,
            path = %target.display(),
            source_count = candidates.len(),
            "ensuring model cache is available"
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
            match client
                .download(candidate, repo, cache_dir, &target, env)
                .await
            {
                Ok(()) => {
                    prepare_cached_model(model, &target, candidate.label()).await?;
                    write_ready_manifest(model, &target, candidate.label()).await?;
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

async fn is_ready_cache<M: ModelSpec>(model: M, target: &Path) -> Result<bool> {
    let manifest = match tokio::fs::read_to_string(target.join(READY_MANIFEST_FILE)).await {
        Ok(manifest) => manifest,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(OrchionError::Download {
                source_name: "cache",
                repo: model.huggingface_repo(),
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

async fn required_cache_files_exist<M: ModelSpec>(model: M, target: &Path) -> Result<bool> {
    if !cache_file_exists(model, target, "config.json").await? {
        return Ok(false);
    }
    match model.category() {
        ModelCategory::Asr => cache_file_exists(model, target, "tokenizer.json").await,
        ModelCategory::Tts => Ok(true),
    }
}

async fn cache_file_exists<M: ModelSpec>(model: M, target: &Path, file_name: &str) -> Result<bool> {
    tokio::fs::try_exists(target.join(file_name))
        .await
        .map_err(|error| OrchionError::Download {
            source_name: "cache",
            repo: model.huggingface_repo(),
            message: error.to_string(),
        })
}

async fn write_ready_manifest<M: ModelSpec>(
    model: M,
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
            repo: model.huggingface_repo(),
            message: error.to_string(),
        })?;
    tokio::fs::rename(&tmp, target.join(READY_MANIFEST_FILE))
        .await
        .map_err(|error| OrchionError::Download {
            source_name,
            repo: model.huggingface_repo(),
            message: error.to_string(),
        })
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
        repo: &'static str,
        cache_dir: &'a Path,
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
        cache_dir: &'a Path,
        _target: &'a Path,
        _env: &'a DownloadEnv,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { download_model_hub(source, repo, cache_dir).await })
    }
}

async fn download_model_hub(
    source: ResolvedSource,
    repo: &'static str,
    cache_dir: &Path,
) -> Result<()> {
    let provider = match source {
        ResolvedSource::HuggingFace => model_hub::HubProvider::HuggingFace { token: None },
        ResolvedSource::ModelScope => model_hub::HubProvider::ModelScope { token: None },
    };
    let downloader =
        model_hub::ModelDownloader::new(provider).map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo,
            message: error.to_string(),
        })?;
    downloader
        .download(model_hub::DownloadOptions {
            repo_id: repo.to_string(),
            revision: None,
            save_dir: cache_dir.to_path_buf(),
            files: None,
        })
        .await
        .map_err(|error| OrchionError::Download {
            source_name: source.label(),
            repo,
            message: error.to_string(),
        })
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
    use orchion_core::{AsrModel, TtsModel};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct FakeDownloadClient {
        fail_huggingface: bool,
        omit_asr_tokenizer_sources: bool,
        calls: Arc<Mutex<Vec<&'static str>>>,
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
            repo: &'static str,
            _cache_dir: &'a Path,
            target: &'a Path,
            _env: &'a DownloadEnv,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(source.label());
                if self.fail_huggingface && source == ResolvedSource::HuggingFace {
                    tokio::fs::create_dir_all(target).await.map_err(|error| {
                        OrchionError::Download {
                            source_name: source.label(),
                            repo,
                            message: error.to_string(),
                        }
                    })?;
                    tokio::fs::write(target.join("partial.bin"), "partial")
                        .await
                        .map_err(|error| OrchionError::Download {
                            source_name: source.label(),
                            repo,
                            message: error.to_string(),
                        })?;
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
                if !self.omit_asr_tokenizer_sources {
                    write_asr_tokenizer_sources(target).await;
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
        assert!(path.join("tokenizer.json").exists());
        assert!(!path.join("partial.bin").exists());
        assert!(!path.join(".orchion-complete").exists());
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
            .download_with_client_and_probe(
                AsrModel::Qwen3Asr06B,
                dir.path(),
                &client,
                &probe,
                &env,
            )
            .await
            .unwrap();
        downloader
            .download_with_client_and_probe(
                TtsModel::Qwen3Tts06BBase,
                dir.path(),
                &client,
                &probe,
                &env,
            )
            .await
            .unwrap();

        assert_eq!(*probe_calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn ready_manifest_skips_download_when_required_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("config.json"), "{}")
            .await
            .unwrap();
        write_asr_tokenizer_json(&target).await;
        write_ready_manifest(&target, AsrModel::Qwen3Asr06B.huggingface_repo()).await;

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
    async fn ready_manifest_redownloads_when_required_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
        tokio::fs::write(target.join("config.json"), "{}")
            .await
            .unwrap();
        write_ready_manifest(&target, AsrModel::Qwen3Asr06B.huggingface_repo()).await;

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
        assert!(path.join("tokenizer.json").exists());
        assert_eq!(&*calls.lock().unwrap(), &["modelscope"]);
    }

    #[tokio::test]
    async fn download_rejects_unrepairable_asr_cache_after_model_hub_success() {
        let dir = tempfile::tempdir().unwrap();
        let client = FakeDownloadClient {
            fail_huggingface: false,
            omit_asr_tokenizer_sources: true,
            calls: Arc::new(Mutex::new(Vec::new())),
        };
        let env = DownloadEnv {
            orchion_model_source: Some("modelscope".to_string()),
            hf_endpoint: None,
        };
        let downloader = ModelDownloader::default();
        let error = downloader
            .download_with_client(AsrModel::Qwen3Asr06B, dir.path(), &client, &env)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("tokenizer_config.json"));
    }

    async fn write_ready_manifest(target: &Path, repo: &'static str) {
        let manifest = serde_json::json!({
            "schema_version": 1,
            "repo_id": repo,
            "layout": "model-hub-native",
        });
        tokio::fs::write(target.join(".orchion-ready.json"), manifest.to_string())
            .await
            .unwrap();
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
