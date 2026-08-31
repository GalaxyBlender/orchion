use orchion::{
    ArtifactRole, AsrModel, DeploymentArtifactPlan, DeploymentArtifactSource,
    DeploymentPublication, DevicePreference, ModelCapabilities, ModelId, ModelSpec, Ocr, OcrEngine,
    OcrEngineFuture, OcrLimits, OcrModel, OcrOptions, OcrResult, PublishedDeploymentArtifact,
    TableStructureAssets, TtsModel,
};
use orchion_server::application::ServerApplication;
use orchion_server::application::model_lifecycle::{
    ModelLifecycleRuntime, ModelSelector, ModelService,
};
use orchion_server::config::ServerConfig;
use orchion_server::model_cache::{
    ModelCache, ModelProvisionFuture, ModelProvisioner, ModelProvisioning,
};
use orchion_server::routes::router;
use orchion_server::state::{
    AppState, AsrRuntimeFuture, ModelRuntimeFactory, OcrDeploymentFuture, OcrDeploymentProvisioner,
    OcrRuntimeFuture, TtsRuntimeFuture,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

#[derive(Default)]
struct FakeProvisioner {
    calls: Mutex<Vec<String>>,
    locators: Mutex<Vec<Option<String>>>,
    sources: Mutex<Vec<Vec<orchion::DownloadSource>>>,
    deployment_plans: Mutex<Vec<DeploymentArtifactPlan>>,
    failures: Mutex<HashSet<String>>,
    locator_failures: Mutex<HashSet<String>>,
    active: AtomicUsize,
    max_active: AtomicUsize,
    delay: Mutex<Option<Duration>>,
}

struct TestOcrEngine {
    model: ModelId,
}

impl OcrEngine for TestOcrEngine {
    fn model(&self) -> &ModelId {
        &self.model
    }

    fn recognize_file_with_limits(
        &self,
        _path: PathBuf,
        _options: OcrOptions,
        _limits: OcrLimits,
    ) -> OcrEngineFuture<'_, OcrResult> {
        Box::pin(async {
            Err(orchion::OrchionError::Inference {
                message: "inference is not used by this test".to_string(),
            })
        })
    }
}

struct SuccessfulOcrRuntimeFactory {
    fail: bool,
    paths: Arc<Mutex<Vec<ObservedOcrRuntimePaths>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedOcrRuntimePaths {
    model_dir: PathBuf,
    cache_root: PathBuf,
    layout: PathBuf,
    table_model: PathBuf,
    table_dictionary: PathBuf,
}

impl ModelRuntimeFactory for SuccessfulOcrRuntimeFactory {
    fn load_asr(
        &self,
        _model: AsrModel,
        _path: PathBuf,
        _device: DevicePreference,
    ) -> AsrRuntimeFuture<'_> {
        Box::pin(async { anyhow::bail!("ASR is not used by this test") })
    }

    fn load_tts(
        &self,
        _model: TtsModel,
        _path: PathBuf,
        _device: DevicePreference,
    ) -> TtsRuntimeFuture<'_> {
        Box::pin(async { anyhow::bail!("TTS is not used by this test") })
    }

    fn load_ocr(
        &self,
        model: OcrModel,
        model_dir: PathBuf,
        cache_root: PathBuf,
        layout: Option<(OcrModel, PathBuf)>,
        table: Option<TableStructureAssets>,
        _device: DevicePreference,
    ) -> OcrRuntimeFuture<'_> {
        let fail = self.fail;
        let paths = Arc::clone(&self.paths);
        Box::pin(async move {
            anyhow::ensure!(!fail, "injected OCR runtime probe failure");
            let (_, layout_path) = layout.context("layout was not assembled")?;
            anyhow::ensure!(layout_path.is_file(), "layout path was not published");
            let table = table.context("table structure was not assembled")?;
            anyhow::ensure!(table.model.is_file(), "table model path was not published");
            anyhow::ensure!(
                table.dictionary.is_file(),
                "table dictionary path was not published"
            );
            paths.lock().unwrap().push(ObservedOcrRuntimePaths {
                model_dir,
                cache_root,
                layout: layout_path,
                table_model: table.model,
                table_dictionary: table.dictionary,
            });
            Ok(Ocr::from_engine(Arc::new(TestOcrEngine {
                model: model.id().clone(),
            })))
        })
    }
}

impl FakeProvisioner {
    fn failing(model: impl Into<String>) -> Self {
        Self {
            failures: Mutex::new(HashSet::from([model.into()])),
            ..Self::default()
        }
    }

    fn delayed(delay: Duration) -> Self {
        Self {
            delay: Mutex::new(Some(delay)),
            ..Self::default()
        }
    }

    fn failing_locator(locator: impl Into<String>) -> Self {
        Self {
            locator_failures: Mutex::new(HashSet::from([locator.into()])),
            ..Self::default()
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn locators(&self) -> Vec<Option<String>> {
        self.locators.lock().unwrap().clone()
    }

    fn sources(&self) -> Vec<Vec<orchion::DownloadSource>> {
        self.sources.lock().unwrap().clone()
    }

    fn deployment_plans(&self) -> Vec<DeploymentArtifactPlan> {
        self.deployment_plans.lock().unwrap().clone()
    }

    fn provision<M: ModelSpec>(
        &self,
        model: M,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        let id = model.huggingface_repo().to_string();
        let locator = provisioning
            .as_ref()
            .map(|provisioning| provisioning.model_url.to_string());
        self.calls.lock().unwrap().push(id.clone());
        self.locators.lock().unwrap().push(locator.clone());
        self.sources.lock().unwrap().push(
            provisioning
                .and_then(|provisioning| provisioning.source_plan)
                .map_or_else(Vec::new, |plan| plan.candidates),
        );
        let should_fail = self.failures.lock().unwrap().contains(&id)
            || locator
                .as_ref()
                .is_some_and(|locator| self.locator_failures.lock().unwrap().contains(locator));
        let delay = *self.delay.lock().unwrap();
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        let path = model.cache_path(models_dir);
        drop(model);

        Box::pin(async move {
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            if should_fail {
                anyhow::bail!("injected download failure for {id}");
            }
            Ok(path)
        })
    }
}

#[tokio::test]
async fn table_structure_uses_exact_model_and_dictionary_deployment_plan() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[models]
source = "modelscope"
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = { model = "//Acme/Table/table.onnx", dictionary = "//Acme/Table/table_dict.txt", table_type = "wireless" }
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let plans = provisioner.deployment_plans();
    let plan = &plans[0];
    assert!(plan.artifacts.iter().any(|artifact| {
        artifact.role == ArtifactRole::OcrTableStructureModel
            && artifact.repository.as_deref() == Some("Acme/Table")
            && artifact.files == ["table.onnx".to_string()]
    }));
    assert!(plan.artifacts.iter().any(|artifact| {
        artifact.role == ArtifactRole::OcrTableStructureDictionary
            && artifact.repository.as_deref() == Some("Acme/Table")
            && artifact.files == ["table_dict.txt".to_string()]
    }));
    assert!(
        plan.artifacts
            .iter()
            .all(|artifact| !artifact.files.is_empty())
    );
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "covers capability absence, failed probe, successful load, HTTP publication, and unload"
)]
async fn table_capability_appears_only_after_successful_runtime_load() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = false
[services.tts]
enabled = false
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = { model = "//Acme/Table/table.onnx", dictionary = "//Acme/Table/table_dict.txt", table_type = "wired" }
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let models_dir = config.models.dir.clone();
    let primary = OcrModel::new(
        ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap(),
        orchion::OcrModelKind::TraditionalOcr,
    );
    let layout = OcrModel::new(
        ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap(),
        orchion::OcrModelKind::Layout,
    );
    let failed_state = AppState::load_with_components(
        config.clone(),
        Arc::new(FakeProvisioner::default()),
        Arc::new(SuccessfulOcrRuntimeFactory {
            fail: true,
            paths: Arc::default(),
        }),
    )
    .await
    .unwrap();
    assert!(
        failed_state
            .ocr(primary.clone(), Some(layout.clone()))
            .await
            .is_err()
    );
    assert!(
        !failed_state.model_catalog().await[0]
            .capabilities
            .contains(ModelCapabilities::OCR_TABLE_STRUCTURE)
    );
    let observed_paths = Arc::new(Mutex::new(Vec::new()));
    let state = AppState::load_with_components(
        config,
        Arc::new(FakeProvisioner::default()),
        Arc::new(SuccessfulOcrRuntimeFactory {
            fail: false,
            paths: Arc::clone(&observed_paths),
        }),
    )
    .await
    .unwrap();

    let before = state.model_catalog().await;
    assert_eq!(before.len(), 1);
    assert!(
        !before[0]
            .capabilities
            .contains(ModelCapabilities::OCR_TABLE_STRUCTURE)
    );

    let lease = state.ocr(primary, Some(layout)).await.unwrap().unwrap();
    assert_eq!(
        observed_paths.lock().unwrap().as_slice(),
        [ObservedOcrRuntimePaths {
            model_dir: models_dir.join("PaddlePaddle/PP-OCRv6_tiny"),
            cache_root: models_dir.clone(),
            layout: models_dir.join("PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"),
            table_model: models_dir.join("Acme/Table/table.onnx"),
            table_dictionary: models_dir.join("Acme/Table/table_dict.txt"),
        }]
    );

    let after = state.model_catalog().await;
    assert_eq!(after.len(), 1);
    assert!(
        after[0]
            .capabilities
            .contains(ModelCapabilities::OCR_TABLE_STRUCTURE)
    );
    let response = router(Arc::clone(&state))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert!(
        body["data"][0]["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|capability| capability == "ocr_table_structure")
    );
    assert!(body["data"][0].get("artifacts").is_none());
    drop(lease);
    state
        .unload_model(ModelSelector {
            model: "paddlepaddle/pp-ocrv6-tiny".to_string(),
            service: ModelService::Ocr,
        })
        .await
        .unwrap()
        .unwrap();
    assert!(
        !state.model_catalog().await[0]
            .capabilities
            .contains(ModelCapabilities::OCR_TABLE_STRUCTURE)
    );
}

#[tokio::test]
async fn explicit_table_artifact_failure_blocks_the_default_deployment() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dictionary = "ms://Acme/Table/table_dict.txt";
    let config = ServerConfig::from_toml_str(
        &format!(
            r#"
[services.asr]
enabled = false
[services.tts]
enabled = false
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
layout_model = "hf://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = {{ model = "hf://Acme/Table/table.onnx", dictionary = "{dictionary}", table_type = "wired" }}
"#
        ),
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing_locator(dictionary));

    let Err(error) = AppState::load_with_provisioner(config, Arc::clone(&provisioner)).await else {
        panic!("table dictionary failure should block the deployment");
    };
    assert!(format!("{error:#}").contains("injected download failure"));
    assert_eq!(provisioner.deployment_plans().len(), 1);
}

#[tokio::test]
async fn explicit_table_artifacts_preserve_intentional_mixed_providers() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = { model = "hf://Acme/Table/table.onnx", dictionary = "ms://Acme/Table/table_dict.txt", table_type = "wireless" }
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());
    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();
    let plans = provisioner.deployment_plans();
    let plan = &plans[0];
    assert!(plan.artifacts.iter().any(|artifact| {
        artifact.role == ArtifactRole::OcrTableStructureModel
            && artifact.source == DeploymentArtifactSource::HuggingFace
    }));
    assert!(plan.artifacts.iter().any(|artifact| {
        artifact.role == ArtifactRole::OcrTableStructureDictionary
            && artifact.source == DeploymentArtifactSource::ModelScope
    }));
}

#[tokio::test]
async fn prepared_load_rejects_missing_deployment_publication() {
    let temp_dir = tempfile::tempdir().unwrap();
    let layout = temp_dir.path().join("layout.onnx");
    let dictionary = temp_dir.path().join("table_dict.txt");
    std::fs::write(&layout, b"layout").unwrap();
    std::fs::write(&dictionary, b"dictionary").unwrap();
    let config = ServerConfig::from_toml_str(
        &format!(
            r#"
[services.asr]
enabled = false
[services.tts]
enabled = false
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "file://{}"
table_structure = {{ model = "file://{}", dictionary = "file://{}", table_type = "wired" }}
"#,
            layout.display(),
            temp_dir.path().join("missing-table.onnx").display(),
            dictionary.display(),
        ),
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();

    let state = AppState::from_prepared_config(config).unwrap();
    let model = OcrModel::new(
        orchion::ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap(),
        orchion::OcrModelKind::TraditionalOcr,
    );
    let layout = OcrModel::new(
        orchion::ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap(),
        orchion::OcrModelKind::Layout,
    );
    let Err(error) = state.ocr(model, Some(layout)).await else {
        panic!("incomplete prepared deployment should fail to load");
    };
    assert!(format!("{error:#}").contains("orchion-deployment.json"));
}

#[tokio::test]
async fn table_structure_locator_but_not_runtime_profile_changes_source_intent() {
    async fn intent_for(table_model: &str, extra: &str) -> String {
        let temp_dir = tempfile::tempdir().unwrap();
        let document = format!(
            r#"
[models]
source = "modelscope"
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = {{ model = "//Acme/Table/{table_model}", dictionary = "//Acme/Table/table_dict.txt", table_type = "wired"{extra} }}
"#
        );
        let config =
            ServerConfig::from_toml_str(&document, &temp_dir.path().join("orchion-server"))
                .unwrap();
        let provisioner = Arc::new(FakeProvisioner::default());
        AppState::load_with_provisioner(config, Arc::clone(&provisioner))
            .await
            .unwrap();
        provisioner.deployment_plans()[0].source_intent.clone()
    }

    assert_eq!(
        intent_for("table.onnx", "").await,
        intent_for("table.onnx", ", score_threshold = 0.8").await
    );
    assert_ne!(
        intent_for("table.onnx", "").await,
        intent_for("table-v2.onnx", "").await
    );
}

impl ModelProvisioner<AsrModel> for FakeProvisioner {
    fn provision(
        &self,
        model: AsrModel,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        self.provision(model, provisioning, models_dir)
    }
}

impl ModelProvisioner<TtsModel> for FakeProvisioner {
    fn provision(
        &self,
        model: TtsModel,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        self.provision(model, provisioning, models_dir)
    }
}

impl ModelProvisioner<OcrModel> for FakeProvisioner {
    fn provision(
        &self,
        model: OcrModel,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        self.provision(model, provisioning, models_dir)
    }
}

impl OcrDeploymentProvisioner for FakeProvisioner {
    fn provision_deployment(
        &self,
        primary: OcrModel,
        plan: DeploymentArtifactPlan,
        models_dir: PathBuf,
    ) -> OcrDeploymentFuture<'_> {
        self.deployment_plans.lock().unwrap().push(plan.clone());
        let failed = plan.artifacts.iter().find_map(|artifact| {
            let repository = artifact.repository.as_deref()?;
            let source = match artifact.source {
                DeploymentArtifactSource::Neutral => "//",
                DeploymentArtifactSource::HuggingFace => "hf://",
                DeploymentArtifactSource::ModelScope => "ms://",
                DeploymentArtifactSource::File(_) => return None,
            };
            artifact.files.iter().find_map(|file| {
                let locator = format!("{source}{repository}/{file}");
                self.locator_failures
                    .lock()
                    .unwrap()
                    .contains(&locator)
                    .then_some(locator)
            })
        });
        Box::pin(async move {
            if let Some(locator) = failed {
                anyhow::bail!("injected download failure for {locator}");
            }
            tokio::fs::create_dir_all(ModelSpec::cache_path(&primary, &models_dir)).await?;
            let mut published = Vec::new();
            for artifact in plan.artifacts {
                let mut files = Vec::new();
                match &artifact.source {
                    DeploymentArtifactSource::File(path) => files.push(path.clone()),
                    _ => {
                        let repository = artifact
                            .repository
                            .as_deref()
                            .context("remote fake artifact has no repository")?;
                        for file in &artifact.files {
                            let path = models_dir.join(repository).join(file);
                            if let Some(parent) = path.parent() {
                                tokio::fs::create_dir_all(parent).await?;
                            }
                            tokio::fs::write(&path, b"artifact").await?;
                            files.push(path);
                        }
                    }
                }
                published.push(PublishedDeploymentArtifact {
                    role: artifact.role,
                    source: match artifact.source {
                        DeploymentArtifactSource::Neutral => DeploymentArtifactSource::ModelScope,
                        source => source,
                    },
                    repository: artifact.repository,
                    files,
                    source_files: artifact.files,
                    requested_revision: None,
                    resolved_revision: None,
                });
            }
            Ok(DeploymentPublication::from_artifacts(models_dir, published))
        })
    }
}

#[test]
fn prepared_state_rejects_programmatic_duplicate_ids() {
    let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
    config
        .services
        .asr
        .models
        .push(config.services.asr.models[0].clone());

    let Err(error) = AppState::from_prepared_config(config) else {
        panic!("duplicate programmatic model ids should fail validation");
    };

    assert!(error.to_string().contains("configured more than once"));
}

#[tokio::test]
async fn load_with_provisioner_rejects_programmatic_default_mismatch() {
    let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
    config.services.asr.default_model = AsrModel::parse("alibaba/qwen3-asr-1.7b").unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    let Err(error) = AppState::load_with_provisioner(config, provisioner).await else {
        panic!("programmatic default mismatch should fail validation");
    };

    assert!(error.to_string().contains("must match exactly one entry"));
}

#[tokio::test]
async fn non_default_model_download_failure_does_not_block_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "alibaba/qwen3-asr-0.6b"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
[[services.asr.models]]
id = "alibaba/qwen3-asr-1.7b"
model = "//Qwen/Qwen3-ASR-1.7B"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("Qwen/Qwen3-ASR-1.7B"));

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let calls = provisioner.calls();
    assert!(calls.contains(&"Qwen/Qwen3-ASR-0.6B".to_string()));
    assert!(!calls.contains(&"Qwen/Qwen3-ASR-1.7B".to_string()));
}

#[tokio::test]
async fn only_default_asr_model_is_provisioned_at_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "alibaba/qwen3-asr-0.6b"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
[[services.asr.models]]
id = "alibaba/qwen3-asr-1.7b"
model = "//Qwen/Qwen3-ASR-1.7B"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let calls = provisioner.calls();
    assert!(calls.contains(&"Qwen/Qwen3-ASR-0.6B".to_string()));
    assert!(!calls.contains(&"Qwen/Qwen3-ASR-1.7B".to_string()));
}

#[tokio::test]
async fn configured_model_locator_reaches_startup_provisioner() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "hf://Mirror/Qwen3-ASR-Package"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    assert_eq!(
        provisioner.locators(),
        [Some("hf://Mirror/Qwen3-ASR-Package".to_string())]
    );
}

#[tokio::test]
async fn only_default_ocr_model_is_provisioned_at_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[models]
source = "modelscope"

[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-small"
model = "//PaddlePaddle/PP-OCRv6_small"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let calls = provisioner.calls();
    assert!(calls.contains(&"PaddlePaddle/PP-OCRv6_tiny".to_string()));
    assert!(!calls.contains(&"PaddlePaddle/PP-OCRv6_small".to_string()));
}

#[tokio::test]
async fn default_model_download_failure_still_blocks_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "alibaba/qwen3-asr-0.6b"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
[[services.asr.models]]
id = "alibaba/qwen3-asr-1.7b"
model = "//Qwen/Qwen3-ASR-1.7B"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("Qwen/Qwen3-ASR-0.6B"));

    let Err(error) = AppState::load_with_provisioner(config, Arc::clone(&provisioner)).await else {
        panic!("default model failure should block startup");
    };

    assert!(
        format!("{error:#}").contains("injected download failure for Qwen/Qwen3-ASR-0.6B"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn required_default_layout_download_failure_blocks_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[models]
source = "modelscope"

[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("PaddlePaddle/PP-DocLayoutV3"));

    let Err(error) = AppState::load_with_provisioner(config, Arc::clone(&provisioner)).await else {
        panic!("required default layout failure should block startup");
    };

    assert!(
        format!("{error:#}").contains("injected download failure for PaddlePaddle/PP-DocLayoutV3"),
        "unexpected error: {error:#}"
    );
    assert!(provisioner.locators().contains(&Some(
        "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx".to_string()
    )));
    assert_eq!(
        provisioner.sources(),
        [
            vec![orchion::DownloadSource::ModelScope],
            vec![orchion::DownloadSource::ModelScope],
        ]
    );
}

#[tokio::test]
async fn explicit_ocr_artifacts_may_intentionally_mix_providers() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[models]
source = "modelscope"

[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    assert!(provisioner.sources().iter().all(Vec::is_empty));
    assert!(
        provisioner
            .locators()
            .contains(&Some("hf://PaddlePaddle/PP-OCRv6_tiny".to_string()))
    );
    assert!(provisioner.locators().contains(&Some(
        "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx".to_string()
    )));
}

#[tokio::test]
async fn non_default_model_is_provisioned_lazily_with_a_single_flight() {
    let temp_dir = tempfile::tempdir().unwrap();
    let default = AsrModel::parse("alibaba/qwen3-asr-0.6b").unwrap();
    let optional = AsrModel::parse("alibaba/qwen3-asr-1.7b").unwrap();
    let provisioner = Arc::new(FakeProvisioner::delayed(Duration::from_millis(20)));
    let cache = ModelCache::new_with_provisioner(
        "asr",
        vec![default, optional.clone()],
        Duration::from_mins(1),
        1,
        temp_dir.path().to_path_buf(),
        Arc::clone(&provisioner),
    );
    let engine_loads = Arc::new(AtomicUsize::new(0));

    let first = tokio::spawn({
        let cache = cache.clone();
        let optional = optional.clone();
        let engine_loads = Arc::clone(&engine_loads);
        async move {
            cache
                .get_or_load(optional, move |_, path| async move {
                    assert!(path.ends_with("Qwen/Qwen3-ASR-1.7B"));
                    Ok(engine_loads.fetch_add(1, Ordering::SeqCst) + 1)
                })
                .await
        }
    });
    let second = tokio::spawn({
        let cache = cache.clone();
        let engine_loads = Arc::clone(&engine_loads);
        async move {
            cache
                .get_or_load(optional, move |_, _| async move {
                    Ok(engine_loads.fetch_add(1, Ordering::SeqCst) + 1)
                })
                .await
        }
    });

    assert_eq!(*first.await.unwrap().unwrap().unwrap(), 1);
    assert_eq!(*second.await.unwrap().unwrap().unwrap(), 1);
    assert_eq!(provisioner.calls(), ["Qwen/Qwen3-ASR-1.7B"]);
    assert_eq!(engine_loads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn startup_model_provisioning_has_bounded_concurrency() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"

[services.tts]
enabled = true
[[services.tts.models]]
id = "alibaba/qwen3-tts-12hz-0.6b-customvoice"
model = "//Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"

[services.ocr-vl]
enabled = true
default_model = "paddlepaddle/paddleocr-vl-1.6"
[[services.ocr-vl.models]]
id = "paddlepaddle/paddleocr-vl-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::delayed(Duration::from_millis(20)));

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let max_active = provisioner.max_active.load(Ordering::SeqCst);
    assert!(max_active > 1, "startup provisioning remained serial");
    assert!(
        max_active <= 2,
        "startup provisioning was not bounded: {max_active}"
    );
    assert_eq!(
        provisioner
            .calls()
            .iter()
            .filter(|model| model.as_str() == "PaddlePaddle/PP-DocLayoutV3")
            .count(),
        2
    );
}
use anyhow::Context;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
