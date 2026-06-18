use crate::error::{OrchionError, Result};
use crate::model::ModelSpec;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

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
        self.download_with_client(model, cache_dir, &LibraryDownloadClient, &env)
            .await
    }

    async fn download_with_client<M: ModelSpec, C: DownloadClient>(
        &self,
        model: M,
        cache_dir: impl AsRef<Path>,
        client: &C,
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
            return Err(OrchionError::IncompleteCache { path: target });
        }

        let candidates = resolve_source(self.source, env)?;
        let mut failures = Vec::new();
        for candidate in candidates {
            let repo = match candidate {
                ResolvedSource::HuggingFace => model.huggingface_repo(),
                ResolvedSource::ModelScope => model.modelscope_repo(),
            };
            match client.download(candidate, repo, &target, env).await {
                Ok(()) => {
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
                    return Ok(target);
                }
                Err(error) => {
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
                    })
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
    async fn complete_marker_skips_download() {
        let dir = tempfile::tempdir().unwrap();
        let target = AsrModel::Qwen3Asr06B.cache_path(dir.path());
        tokio::fs::create_dir_all(&target).await.unwrap();
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
}
