#![allow(clippy::unnecessary_literal_bound)]

use orchion_core::{KnownOcrModel, ModelCategory, ModelSpec, OrchionError, Result};
use orchion_download::{
    DownloadFuture, DownloadProvider, DownloadProviderRegistry, HubProviderOptions,
    HuggingFaceProvider, ModelDownloader, ProviderDownloadRequest, ProviderDownloadResult,
    ProviderModel, ResolvedDownloadFuture,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FirstModel;

impl ModelSpec for FirstModel {
    fn category(&self) -> ModelCategory {
        ModelCategory::Tts
    }

    fn huggingface_repo(&self) -> &str {
        "Canonical/First"
    }

    fn modelscope_repo(&self) -> &str {
        "Mirror/First"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SecondModel;

impl ModelSpec for SecondModel {
    fn category(&self) -> ModelCategory {
        ModelCategory::Tts
    }

    fn huggingface_repo(&self) -> &str {
        "Canonical/Second"
    }

    fn modelscope_repo(&self) -> &str {
        "Mirror/Second"
    }
}

#[derive(Default)]
struct ProviderState {
    calls: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    repositories: Mutex<Vec<String>>,
    revisions: Mutex<Vec<String>>,
}

struct EnterpriseProvider {
    state: Arc<ProviderState>,
    delay: Duration,
}

impl EnterpriseProvider {
    fn new(delay: Duration) -> (Self, Arc<ProviderState>) {
        let state = Arc::new(ProviderState::default());
        (
            Self {
                state: Arc::clone(&state),
                delay,
            },
            state,
        )
    }
}

impl DownloadProvider for EnterpriseProvider {
    fn label(&self) -> &'static str {
        "enterprise"
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn default_revision(&self) -> &str {
        "production"
    }

    fn repository(&self, model: ProviderModel<'_>) -> String {
        model.repository_identity().map_or_else(
            || format!("enterprise/{}", model.huggingface_repo()),
            |identity| format!("enterprise-assets/{identity}"),
        )
    }

    fn download<'a>(&'a self, request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
        Box::pin(async move {
            self.state.calls.fetch_add(1, Ordering::SeqCst);
            self.state
                .repositories
                .lock()
                .unwrap()
                .push(request.repository().to_string());
            self.state
                .revisions
                .lock()
                .unwrap()
                .push(request.revision().to_string());
            let current = self.state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.state
                .max_in_flight
                .fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            tokio::fs::create_dir_all(request.target())
                .await
                .map_err(|error| download_error(request.repository(), &error))?;
            tokio::fs::write(request.target().join("config.json"), b"{}")
                .await
                .map_err(|error| download_error(request.repository(), &error))?;
            if let Some(files) = request.files() {
                for file in files {
                    tokio::fs::write(request.target().join(file), b"asset")
                        .await
                        .map_err(|error| download_error(request.repository(), &error))?;
                }
            }
            self.state.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
    }

    fn download_with_result<'a>(
        &'a self,
        request: ProviderDownloadRequest<'a>,
    ) -> ResolvedDownloadFuture<'a> {
        Box::pin(async move {
            self.download(request).await?;
            Ok(ProviderDownloadResult::with_resolved_revision(
                "2222222222222222222222222222222222222222",
            ))
        })
    }
}

fn download_error(repo: &str, error: &std::io::Error) -> OrchionError {
    OrchionError::Download {
        source_name: "enterprise",
        repo: repo.to_string(),
        message: error.to_string(),
    }
}

#[tokio::test]
async fn custom_provider_controls_identity_locator_revision_and_download() -> Result<()> {
    let cache = tempfile::tempdir().unwrap();
    let (provider, state) = EnterpriseProvider::new(Duration::ZERO);
    let registry = DownloadProviderRegistry::new().with_provider(provider);

    let path = ModelDownloader::from_registry(registry)
        .download(FirstModel, cache.path())
        .await?;

    assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    assert_eq!(path, cache.path().join("Canonical/First"));
    let manifest: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(path.join(".orchion-ready.json"))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["source"], "enterprise");
    assert_eq!(manifest["repo_id"], "enterprise/Canonical/First");
    assert_eq!(manifest["revision"], "production");
    assert_eq!(
        manifest["resolved_revision"],
        "2222222222222222222222222222222222222222"
    );
    Ok(())
}

#[tokio::test]
async fn custom_provider_maps_auxiliary_asset_identity_and_revision() -> Result<()> {
    let cache = tempfile::tempdir().unwrap();
    let (provider, state) = EnterpriseProvider::new(Duration::ZERO);

    let path = ModelDownloader::from_provider(provider)
        .with_revision("canonical-only")
        .with_repository_revision("greatv/oar-ocr", "asset-release")
        .download(KnownOcrModel::PpOcrV5Mobile, cache.path())
        .await?;

    assert_eq!(
        &*state.repositories.lock().unwrap(),
        &["enterprise-assets/greatv/oar-ocr"]
    );
    assert_eq!(&*state.revisions.lock().unwrap(), &["asset-release"]);
    let manifest: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(path.join(".orchion-ready.json"))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["repositories"],
        serde_json::json!([{
            "identity": "greatv/oar-ocr",
            "repo_id": "enterprise-assets/greatv/oar-ocr",
            "requested_revision": "asset-release",
            "resolved_revision": "2222222222222222222222222222222222222222"
        }])
    );
    Ok(())
}

#[tokio::test]
async fn different_canonical_models_download_in_parallel() -> Result<()> {
    let cache = tempfile::tempdir().unwrap();
    let (provider, state) = EnterpriseProvider::new(Duration::from_millis(100));
    let downloader = ModelDownloader::from_provider(provider);

    let (first, second) = tokio::join!(
        downloader.download(FirstModel, cache.path()),
        downloader.download(SecondModel, cache.path()),
    );
    first?;
    second?;

    assert_eq!(state.calls.load(Ordering::SeqCst), 2);
    assert_eq!(state.max_in_flight.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn same_canonical_model_is_single_flight() -> Result<()> {
    let cache = tempfile::tempdir().unwrap();
    let (provider, state) = EnterpriseProvider::new(Duration::from_millis(100));
    let downloader = ModelDownloader::from_provider(provider);

    let (first, second) = tokio::join!(
        downloader.download(FirstModel, cache.path()),
        downloader.download(FirstModel, cache.path()),
    );
    first?;
    second?;

    assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.max_in_flight.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn configured_provider_token_is_redacted_from_debug_output() {
    let secret = "hf_private_test_token";
    let provider =
        HuggingFaceProvider::with_options(HubProviderOptions::default().with_token(secret));
    let provider_debug = format!("{provider:?}");
    let downloader = ModelDownloader::from_provider(provider);

    let debug = format!("{downloader:?}");
    assert!(!debug.contains(secret));
    assert!(!provider_debug.contains(secret));
    assert!(provider_debug.contains("REDACTED"));
}
