#![allow(clippy::unnecessary_literal_bound)]

use orchion_core::{KnownOcrModel, ModelCategory, ModelUrl, OcrModel, OrchionError};
use orchion_download::{
    ArtifactRole, DeploymentArtifactPlan, DeploymentArtifactRequest, DeploymentArtifactSource,
    DownloadFuture, DownloadProvider, DownloadSource, ModelDownloader, PreflightFuture,
    ProviderDownloadRequest, ProviderDownloadResult, ProviderModel, ProviderPreflightRequest,
    ProviderPreflightResult, ResolvedDownloadFuture,
};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

// Use a process-lifetime subscriber like the server, isolated from unit tests that
// exercise the same callsites without a subscriber. Cases serialize buffer access.
static LOG_TEST: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

impl Write for CapturedLogs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl CapturedLogs {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }

    fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

fn capture_logs() -> &'static CapturedLogs {
    static LOGS: OnceLock<CapturedLogs> = OnceLock::new();
    let logs = LOGS.get_or_init(|| {
        let logs = CapturedLogs::default();
        let writer = logs.clone();
        tracing_subscriber::fmt()
            .with_env_filter("orchion_server=info,orchion=info,tower_http=warn")
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .try_init()
            .unwrap();
        logs
    });
    logs.clear();
    logs
}

#[derive(Default)]
struct Gate {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

impl Gate {
    async fn wait(&self) {
        self.started.notify_one();
        self.release.notified().await;
    }
}

#[derive(Default)]
struct ProviderState {
    preflights: AtomicUsize,
    downloads: AtomicUsize,
    preflight_gate: Option<Arc<Gate>>,
    download_gate: Option<Arc<Gate>>,
    failure_repo: Option<&'static str>,
}

struct FixtureProvider(Arc<ProviderState>);

impl DownloadProvider for FixtureProvider {
    fn label(&self) -> &'static str {
        "modelscope"
    }

    fn default_revision(&self) -> &str {
        "master"
    }

    fn repository(&self, model: ProviderModel<'_>) -> String {
        model
            .repository_identity()
            .unwrap_or_else(|| model.modelscope_repo())
            .to_string()
    }

    fn preflight<'a>(&'a self, request: ProviderPreflightRequest<'a>) -> PreflightFuture<'a> {
        Box::pin(async move {
            self.0.preflights.fetch_add(1, Ordering::SeqCst);
            if let Some(gate) = &self.0.preflight_gate {
                gate.wait().await;
            }
            Ok(ProviderPreflightResult::with_resolved_revision(
                request
                    .files()
                    .unwrap()
                    .iter()
                    .map(|file| (*file).to_string())
                    .collect(),
                "1111111111111111111111111111111111111111",
            ))
        })
    }

    fn download<'a>(&'a self, request: ProviderDownloadRequest<'a>) -> DownloadFuture<'a> {
        Box::pin(async move { self.download_with_result(request).await.map(|_| ()) })
    }

    fn download_with_result<'a>(
        &'a self,
        request: ProviderDownloadRequest<'a>,
    ) -> ResolvedDownloadFuture<'a> {
        Box::pin(async move {
            self.0.downloads.fetch_add(1, Ordering::SeqCst);
            if let Some(gate) = &self.0.download_gate {
                gate.wait().await;
            }
            let error = |message: String| OrchionError::Download {
                source_name: "modelscope",
                repo: request.repository().to_string(),
                message,
            };
            if self.0.failure_repo == Some(request.repository()) {
                return Err(error("injected download failure".to_string()));
            }
            tokio::fs::create_dir_all(request.target())
                .await
                .map_err(|e| error(e.to_string()))?;
            for file in request.files().unwrap() {
                tokio::fs::write(request.target().join(file), b"asset")
                    .await
                    .map_err(|e| error(e.to_string()))?;
            }
            Ok(ProviderDownloadResult::with_resolved_revision(
                request.revision(),
            ))
        })
    }
}

fn fixture_downloader(state: &Arc<ProviderState>) -> ModelDownloader {
    ModelDownloader::from_provider(FixtureProvider(Arc::clone(state)))
}

fn deployment_plan() -> (OcrModel, DeploymentArtifactPlan) {
    let primary = KnownOcrModel::PpOcrV5Mobile.into_model();
    let locator = ModelUrl::parse("//PaddlePaddle/PP-OCRv5_mobile").unwrap();
    let mut artifacts = ModelDownloader::model_artifact_plan(&primary, &locator)
        .unwrap()
        .into_iter()
        .map(|artifact| DeploymentArtifactRequest {
            role: artifact.role,
            source: DeploymentArtifactSource::Neutral,
            repository: Some(artifact.repository),
            files: artifact.files.unwrap(),
            required_source: artifact.required_source,
        })
        .collect::<Vec<_>>();
    artifacts.extend(
        [
            (ArtifactRole::OcrLayout, "Acme/Layout", "layout.onnx"),
            (
                ArtifactRole::OcrTableStructureModel,
                "Acme/Table",
                "table.onnx",
            ),
            (
                ArtifactRole::OcrTableStructureDictionary,
                "Acme/Table",
                "table_dict.txt",
            ),
        ]
        .into_iter()
        .map(|(role, repo, file)| DeploymentArtifactRequest {
            role,
            source: DeploymentArtifactSource::Neutral,
            repository: Some(repo.to_string()),
            files: vec![file.to_string()],
            required_source: None,
        }),
    );
    let plan = DeploymentArtifactPlan {
        deployment_id: primary.id().clone(),
        category: ModelCategory::Ocr,
        source_intent: "ocr-logging-test".to_string(),
        artifacts,
        neutral_candidates: vec![DownloadSource::HuggingFace, DownloadSource::ModelScope],
    };
    (primary, plan)
}

#[tokio::test]
async fn info_logs_are_visible_before_remote_preflight_finishes() {
    let _guard = LOG_TEST.lock().await;
    let logs = capture_logs();
    let cache = tempfile::tempdir().unwrap();
    let (primary, plan) = deployment_plan();
    let gate = Arc::new(Gate::default());
    let state = Arc::new(ProviderState {
        preflight_gate: Some(Arc::clone(&gate)),
        ..Default::default()
    });
    let downloader = fixture_downloader(&state);
    let preparation = downloader.provision_deployment(primary, &plan, cache.path());
    tokio::pin!(preparation);
    tokio::select! {
        () = gate.started.notified() => {},
        result = &mut preparation => panic!("preflight should remain blocked: {result:?}"),
        () = tokio::time::sleep(Duration::from_secs(2)) => panic!("preflight did not start"),
    }
    let output = logs.text();
    assert!(output.contains("checking deployment cache"), "{output}");
    assert!(output.contains("preparing deployment cache"), "{output}");
    assert!(output.contains("checking deployment source"), "{output}");
    assert!(!output.contains("deployment cache ready"), "{output}");
}

#[tokio::test]
async fn info_logs_are_visible_before_download_finishes() {
    let _guard = LOG_TEST.lock().await;
    let logs = capture_logs();
    let cache = tempfile::tempdir().unwrap();
    let (primary, plan) = deployment_plan();
    let gate = Arc::new(Gate::default());
    let state = Arc::new(ProviderState {
        download_gate: Some(Arc::clone(&gate)),
        ..Default::default()
    });
    let downloader = fixture_downloader(&state);
    let preparation = downloader.provision_deployment(primary, &plan, cache.path());
    tokio::pin!(preparation);
    tokio::select! {
        () = gate.started.notified() => {},
        result = &mut preparation => panic!("download should remain blocked: {result:?}"),
        () = tokio::time::sleep(Duration::from_secs(2)) => panic!("download did not start"),
    }
    let output = logs.text();
    assert!(
        output.contains("downloading deployment artifacts"),
        "{output}"
    );
    assert!(
        !output.contains("deployment artifact download completed"),
        "{output}"
    );
    assert!(!output.contains("deployment cache ready"), "{output}");
}

#[tokio::test]
async fn info_logs_distinguish_downloads_from_cached_restart() {
    let _guard = LOG_TEST.lock().await;
    let logs = capture_logs();
    let cache = tempfile::tempdir().unwrap();
    let (primary, plan) = deployment_plan();
    let state = Arc::new(ProviderState::default());
    fixture_downloader(&state)
        .provision_deployment(primary.clone(), &plan, cache.path())
        .await
        .unwrap();
    let output = logs.text();
    for message in [
        "preparing deployment cache",
        "deployment source selected",
        "downloading deployment artifacts",
        "deployment artifact download completed",
        "deployment cache ready",
    ] {
        assert!(output.contains(message), "missing {message}: {output}");
    }
    for message in [
        "downloading deployment artifacts",
        "deployment artifact download completed",
    ] {
        assert_eq!(
            output
                .lines()
                .filter(|line| line.contains("Acme/Table") && line.contains(message))
                .count(),
            1
        );
    }
    let downloads = state.downloads.load(Ordering::SeqCst);
    let preflights = state.preflights.load(Ordering::SeqCst);
    logs.clear();
    fixture_downloader(&state)
        .provision_deployment(primary, &plan, cache.path())
        .await
        .unwrap();
    let output = logs.text();
    assert!(output.contains("reusing cached deployment"), "{output}");
    assert!(
        !output.contains("downloading deployment artifacts"),
        "{output}"
    );
    assert!(!output.contains("checking deployment source"), "{output}");
    assert_eq!(state.downloads.load(Ordering::SeqCst), downloads);
    assert_eq!(state.preflights.load(Ordering::SeqCst), preflights);
}

#[tokio::test]
async fn info_logs_do_not_report_failed_download_as_ready() {
    let _guard = LOG_TEST.lock().await;
    let logs = capture_logs();
    let cache = tempfile::tempdir().unwrap();
    let (primary, plan) = deployment_plan();
    let state = Arc::new(ProviderState {
        failure_repo: Some("Acme/Table"),
        ..Default::default()
    });
    fixture_downloader(&state)
        .provision_deployment(primary, &plan, cache.path())
        .await
        .unwrap_err();
    let output = logs.text();
    assert!(
        output.contains("deployment artifact download failed"),
        "{output}"
    );
    assert!(!output.contains("deployment cache ready"), "{output}");
    assert!(
        !output.lines().any(|line| line.contains("Acme/Table")
            && line.contains("deployment artifact download completed")),
        "{output}"
    );
}
