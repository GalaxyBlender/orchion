use crate::application::llm::{LlmCommand, LlmGenerationFuture, LlmRuntime, ManagedGeneration};
use crate::application::model_cache::{
    AsrModelCache, CacheTracker, GlobalModelCacheLimiter, ModelCache, ModelCacheKey, ModelLease,
    ModelProvisionFuture, ModelProvisioner, ModelProvisioning, ModelResidencyStatus,
    ResidencyDomain, TtsModelCache,
};
use crate::application::model_lifecycle::{
    ModelControlFuture, ModelLifecycleRuntime, ModelResidency, ModelSelector, ModelService,
    ModelStatus, ModelStatusesFuture,
};
use crate::application::ocr::{
    OcrFuture, OcrPolicy, OcrRuntime, OcrServiceChoice, OcrServicePolicy, OcrVlServicePolicy,
};
use crate::application::resource_policy::ResourcePolicy;
use crate::application::speech::{SpeechPolicy, SpeechRuntime, SpeechRuntimeFuture};
use crate::application::streaming_transcription::{
    LeasedAsrModel, StreamingModelFuture, StreamingTranscriptionRuntime,
};
use crate::application::transcription::{
    TranscriptionFuture, TranscriptionPolicy, TranscriptionRuntime,
};
use crate::application::{
    ActivityPolicy, ApiModel, ApiPolicy, AsrApiPolicy, RuntimeError, ServerApplication,
};
use crate::settings::{
    LlmContextSize, LlmGpuLayers, LlmModelDeployment, ServerConfig, TableStructureConfig,
};
use anyhow::Context;
use orchion::{
    ArtifactRequest, ArtifactRole, Asr, AsrModel, DeploymentArtifactPlan,
    DeploymentArtifactRequest, DeploymentArtifactSource, DeploymentPublication,
    DeploymentSourcePlan, DevicePreference, DownloadSource, GenerationOptions, GenerationRequest,
    LlmBackendGuard, LlmEngine, LlmEngineConfig, LlmModel, LlmTemplateEngine, ModelCapabilities,
    ModelCategory, ModelDownloader, ModelId, ModelSpec, ModelUrlSource, Ocr, OcrAssets, OcrModel,
    OcrModelAssetKind, OcrModelAssetRole, OcrModelKind, RuntimeProvider, TableStructureAssets, Tts,
    TtsModel, initialize_llm_backend, model_descriptor,
};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Semaphore, watch};
use tokio::task::{JoinHandle, JoinSet};

const MAX_CONCURRENT_MODEL_PROVISIONS: usize = 2;
type LayoutModelCache = ModelCache<DeploymentLayoutModel, ()>;
type OcrAssetCache = ModelCache<OcrModel, ()>;
type OcrRuntimeCache = ModelCache<OcrRuntimeKey, Ocr>;
type LlmRuntimeCache = ModelCache<LlmRuntimeKey, LlmEngine>;
pub type AsrRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Asr>> + Send + 'a>>;
pub type TtsRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Tts>> + Send + 'a>>;
pub type OcrRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Ocr>> + Send + 'a>>;
pub type OcrDeploymentFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<DeploymentPublication>> + Send + 'a>>;

pub type LlmRuntimeFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<LlmEngine>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LlmRuntimeKey(LlmModel);

impl LlmRuntimeKey {
    fn new(id: ModelId) -> Self {
        Self(LlmModel::new(id))
    }

    fn id(&self) -> &ModelId {
        self.0.id()
    }

    fn public_model(&self) -> LlmModel {
        self.0.clone()
    }
}

impl std::fmt::Display for LlmRuntimeKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.id().fmt(formatter)
    }
}

impl ModelCacheKey for LlmRuntimeKey {
    fn cache_path(&self, cache_dir: &std::path::Path) -> PathBuf {
        self.id()
            .as_str()
            .split('/')
            .fold(cache_dir.join(".llm-logical"), |path, segment| {
                path.join(segment)
            })
    }
}

pub trait OcrDeploymentProvisioner: Send + Sync {
    fn provision_deployment(
        &self,
        primary: OcrModel,
        plan: DeploymentArtifactPlan,
        models_dir: PathBuf,
    ) -> OcrDeploymentFuture<'_>;

    fn provision_llm_deployment(
        &self,
        _primary: ModelId,
        _plan: DeploymentArtifactPlan,
        _models_dir: PathBuf,
    ) -> OcrDeploymentFuture<'_> {
        Box::pin(async {
            anyhow::bail!("LLM deployment provisioning is not supported by this provisioner")
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OcrRuntimeKey {
    primary: OcrModel,
    layout: Option<DeploymentLayoutModel>,
    table_structure: Option<TableStructureConfig>,
    table_source_intent: Option<String>,
}

impl OcrRuntimeKey {
    const fn new(primary: OcrModel, layout: Option<DeploymentLayoutModel>) -> Self {
        Self {
            primary,
            layout,
            table_structure: None,
            table_source_intent: None,
        }
    }

    fn with_table_structure(mut self, table_structure: Option<TableStructureConfig>) -> Self {
        self.table_structure = table_structure;
        self
    }

    fn with_table_source_intent(mut self, source_intent: Option<String>) -> Self {
        self.table_source_intent = source_intent;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DeploymentLayoutModel {
    deployment_id: ModelId,
    model: OcrModel,
}

impl DeploymentLayoutModel {
    const fn new(deployment_id: ModelId, model: OcrModel) -> Self {
        Self {
            deployment_id,
            model,
        }
    }
}

impl std::fmt::Display for DeploymentLayoutModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.model.fmt(formatter)
    }
}

impl ModelSpec for DeploymentLayoutModel {
    fn category(&self) -> ModelCategory {
        self.model.category()
    }

    fn huggingface_repo(&self) -> &str {
        self.model.huggingface_repo()
    }

    fn modelscope_repo(&self) -> &str {
        self.model.modelscope_repo()
    }

    fn required_files(&self) -> &'static [&'static str] {
        self.model.required_files()
    }
}

impl ModelSpec for OcrRuntimeKey {
    fn category(&self) -> ModelCategory {
        self.primary.category()
    }

    fn huggingface_repo(&self) -> &str {
        self.primary.huggingface_repo()
    }

    fn modelscope_repo(&self) -> &str {
        self.primary.modelscope_repo()
    }

    fn required_files(&self) -> &'static [&'static str] {
        self.primary.required_files()
    }
}

pub trait ModelRuntimeFactory: Send + Sync + 'static {
    fn supports_asr(&self, _model: &AsrModel) -> bool {
        true
    }

    fn supports_tts(&self, _model: &TtsModel) -> bool {
        true
    }

    fn supports_ocr(&self, _model: &OcrModel) -> bool {
        true
    }

    fn load_asr(
        &self,
        model: AsrModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> AsrRuntimeFuture<'_>;

    fn load_tts(
        &self,
        model: TtsModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> TtsRuntimeFuture<'_>;

    fn load_ocr(
        &self,
        model: OcrModel,
        model_dir: PathBuf,
        cache_root: PathBuf,
        layout_model: Option<(OcrModel, PathBuf)>,
        table_structure: Option<TableStructureAssets>,
        device: DevicePreference,
    ) -> OcrRuntimeFuture<'_>;

    fn load_llm(
        &self,
        _model: LlmModel,
        _path: PathBuf,
        _deployment: LlmModelDeployment,
    ) -> LlmRuntimeFuture<'_> {
        Box::pin(async { anyhow::bail!("LLM is not supported by this runtime factory") })
    }
}

#[derive(Debug, Default)]
pub struct BuiltinModelRuntimeFactory;

#[cfg(test)]
struct TestLlmRuntimeFactory {
    llm: LlmEngine,
    builtin: BuiltinModelRuntimeFactory,
}

#[cfg(test)]
impl ModelRuntimeFactory for TestLlmRuntimeFactory {
    fn load_asr(
        &self,
        model: AsrModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> AsrRuntimeFuture<'_> {
        self.builtin.load_asr(model, path, device)
    }

    fn load_tts(
        &self,
        model: TtsModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> TtsRuntimeFuture<'_> {
        self.builtin.load_tts(model, path, device)
    }

    fn load_ocr(
        &self,
        model: OcrModel,
        model_dir: PathBuf,
        cache_root: PathBuf,
        layout_model: Option<(OcrModel, PathBuf)>,
        table_structure: Option<TableStructureAssets>,
        device: DevicePreference,
    ) -> OcrRuntimeFuture<'_> {
        self.builtin.load_ocr(
            model,
            model_dir,
            cache_root,
            layout_model,
            table_structure,
            device,
        )
    }

    fn load_llm(
        &self,
        _model: LlmModel,
        _path: PathBuf,
        _deployment: LlmModelDeployment,
    ) -> LlmRuntimeFuture<'_> {
        let llm = self.llm.clone();
        Box::pin(async move { Ok(llm) })
    }
}

impl ModelRuntimeFactory for BuiltinModelRuntimeFactory {
    fn supports_asr(&self, model: &AsrModel) -> bool {
        model_descriptor(model.as_str()).is_none_or(|descriptor| {
            descriptor.category == orchion::ModelCategory::Asr
                && descriptor.runtime_provider == RuntimeProvider::Qwen3
        })
    }

    fn supports_tts(&self, model: &TtsModel) -> bool {
        model_descriptor(model.as_str()).is_none_or(|descriptor| {
            descriptor.category == orchion::ModelCategory::Tts
                && descriptor.runtime_provider == RuntimeProvider::Qwen3
        })
    }

    fn supports_ocr(&self, model: &OcrModel) -> bool {
        model.known().is_some()
    }

    fn load_asr(
        &self,
        model: AsrModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> AsrRuntimeFuture<'_> {
        Box::pin(async move {
            Asr::load_with_device(model, path, device)
                .await
                .map_err(Into::into)
        })
    }

    fn load_tts(
        &self,
        model: TtsModel,
        path: PathBuf,
        device: DevicePreference,
    ) -> TtsRuntimeFuture<'_> {
        Box::pin(async move {
            Tts::load_with_device(model, path, device)
                .await
                .map_err(Into::into)
        })
    }

    fn load_ocr(
        &self,
        model: OcrModel,
        model_dir: PathBuf,
        cache_root: PathBuf,
        layout_model: Option<(OcrModel, PathBuf)>,
        table_structure: Option<TableStructureAssets>,
        device: DevicePreference,
    ) -> OcrRuntimeFuture<'_> {
        Box::pin(async move {
            let known = model
                .known()
                .with_context(|| format!("unsupported built-in OCR model `{model}`"))?;
            let layout = layout_model
                .map(|(layout_model, path)| {
                    if path.is_file() {
                        return Ok(path);
                    }
                    let layout_cache_root = path
                        .parent()
                        .and_then(std::path::Path::parent)
                        .unwrap_or(&cache_root);
                    let known_layout = layout_model.known().with_context(|| {
                        format!("unsupported built-in OCR layout model `{layout_model}`")
                    })?;
                    match OcrAssets::from_cache_layout(known_layout, &path, layout_cache_root) {
                        OcrAssets::Layout { model } => Ok(model),
                        OcrAssets::Traditional { .. } | OcrAssets::VisionLanguage { .. } => {
                            anyhow::bail!("model `{layout_model}` did not resolve to layout assets")
                        }
                    }
                })
                .transpose()?;
            let assets = OcrAssets::from_cache_layout(known, model_dir, cache_root)
                .with_layout(layout)
                .with_table_structure(table_structure);
            Ocr::load_with_assets_and_device(known.id(), assets, device)
                .await
                .map_err(Into::into)
        })
    }

    fn load_llm(
        &self,
        _model: LlmModel,
        path: PathBuf,
        deployment: LlmModelDeployment,
    ) -> LlmRuntimeFuture<'_> {
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let runtime = deployment.runtime;
                LlmEngine::load(
                    path,
                    LlmEngineConfig {
                        context_size: match runtime.context_size {
                            LlmContextSize::Model => None,
                            LlmContextSize::Tokens(value) => std::num::NonZeroU32::new(value),
                        },
                        batch_size: runtime.batch_size,
                        micro_batch_size: runtime.micro_batch_size,
                        threads: runtime.threads,
                        gpu_layers: match runtime.gpu_layers {
                            LlmGpuLayers::All => u32::MAX,
                            LlmGpuLayers::Count(value) => value,
                        },
                        parallel_sequences: runtime.parallel_sequences,
                        request_queue_capacity: runtime.request_queue_capacity,
                        event_queue_capacity: runtime.event_queue_capacity,
                        chat_template: deployment.chat_template.template,
                        template_engine: match deployment.chat_template.engine {
                            crate::settings::ChatTemplateEngine::LlamaCpp => {
                                LlmTemplateEngine::LlamaCpp
                            }
                            crate::settings::ChatTemplateEngine::Jinja => LlmTemplateEngine::Jinja,
                        },
                        enable_thinking: deployment.chat_template.enable_thinking,
                    },
                )
                .map_err(anyhow::Error::from)
            })
            .await
            .map_err(|error| anyhow::anyhow!("LLM load task failed: {error}"))?
        })
    }
}

impl<M: ModelSpec> ModelProvisioner<M> for ModelDownloader {
    fn provision(
        &self,
        model: M,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        Box::pin(async move {
            match provisioning {
                Some(provisioning) => {
                    self.download_model_url_with_plan(
                        model,
                        &provisioning.model_url,
                        &provisioning.source_intent,
                        None,
                        provisioning.source_plan.as_ref(),
                        models_dir,
                    )
                    .await
                }
                None => self.download(model, models_dir).await,
            }
            .map_err(anyhow::Error::from)
        })
    }
}

impl OcrDeploymentProvisioner for ModelDownloader {
    fn provision_deployment(
        &self,
        primary: OcrModel,
        plan: DeploymentArtifactPlan,
        models_dir: PathBuf,
    ) -> OcrDeploymentFuture<'_> {
        Box::pin(async move {
            ModelDownloader::provision_deployment(self, primary, &plan, models_dir)
                .await
                .map_err(anyhow::Error::from)
        })
    }

    fn provision_llm_deployment(
        &self,
        primary: ModelId,
        plan: DeploymentArtifactPlan,
        models_dir: PathBuf,
    ) -> OcrDeploymentFuture<'_> {
        Box::pin(async move {
            ModelDownloader::provision_logical_deployment(
                self,
                &primary,
                ModelCategory::Llm,
                &plan,
                models_dir,
            )
            .await
            .map_err(anyhow::Error::from)
        })
    }
}

#[derive(Clone)]
pub struct AppState {
    config: ServerConfig,
    source_candidates: Vec<DownloadSource>,
    api_policy: ApiPolicy,
    asr_models: AsrModelCache,
    tts_models: TtsModelCache,
    ocr_models: OcrRuntimeCache,
    ocr_vl_models: OcrRuntimeCache,
    ocr_assets: OcrAssetCache,
    layout_models: LayoutModelCache,
    llm_models: LlmRuntimeCache,
    llm_deployment_plans: HashMap<LlmRuntimeKey, DeploymentArtifactPlan>,
    llm_deployment_provisioner: Option<Arc<dyn OcrDeploymentProvisioner>>,
    llm_backend: Arc<StdMutex<Option<LlmBackendGuard>>>,
    ocr_deployment_plans: HashMap<OcrRuntimeKey, DeploymentArtifactPlan>,
    ocr_deployment_provisioner: Option<Arc<dyn OcrDeploymentProvisioner>>,
    global_models: GlobalModelCacheLimiter,
    model_residency: ResidencyDomain,
    resources: ResourcePolicy,
    runtime_factory: Arc<dyn ModelRuntimeFactory>,
    cleanup_shutdown: watch::Sender<bool>,
    cleanup_task: Arc<StdMutex<Option<JoinHandle<()>>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{
        ChatTemplateConfig, LlmGenerationConfig, LlmModelDeployment, LlmRuntimeConfig,
        ModelDeployment, OcrModelDeployment,
    };
    use orchion::{
        AsrEngine, AsrEngineFuture, AsrOptions, AsrStreamSession, AsrStreamingOptions,
        AsrTranscript, GenerationEvent, GenerationFinishReason, KnownOcrModel, LlmMessage, LlmRole,
        LlmScriptedControl, LlmUsage, OcrEngine, OcrEngineFuture, OcrLimits, OcrOptions, OcrResult,
        scripted_llm_engine, scripted_panicking_llm_engine,
        scripted_preparation_panicking_llm_engine, scripted_slow_preparation_llm_engine,
    };

    #[tokio::test]
    async fn disabled_ocr_services_skip_empty_caches() {
        let temp_dir = tempfile::tempdir().unwrap();
        let exe_path = temp_dir.path().join("orchion-server");
        let mut config = ServerConfig::default_for_exe(&exe_path);
        config.models.dir = temp_dir.path().join("models");
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let models_dir = config.models.dir.clone();

        let state = AppState::load(config).await.unwrap();

        assert!(
            state
                .ocr(KnownOcrModel::PpOcrV6Tiny.into_model(), None)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ocr_vl(KnownOcrModel::PaddleOcrVl16.into_model(), None)
                .await
                .unwrap()
                .is_none()
        );
        assert!(!models_dir.exists());
    }

    #[tokio::test]
    async fn inactive_ocr_services_still_validate_unknown_deployments() {
        let mut config = test_config();
        let unknown_model = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.enabled = false;
        config.services.ocr.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            unknown_model.clone(),
            OcrModelKind::TraditionalOcr,
        ))];
        config.services.ocr_vl.enabled = false;

        let Err(error) = AppState::load(config).await else {
            panic!("unknown disabled OCR deployment should fail validation");
        };

        assert!(
            error
                .to_string()
                .contains("not a supported traditional OCR")
        );
    }

    #[tokio::test]
    async fn known_ocr_model_reaches_injected_runtime_factory() {
        let mut config = test_config();
        config.services.ocr.enabled = true;
        let model_id = ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap();
        config.services.ocr.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            model_id.clone(),
            OcrModelKind::TraditionalOcr,
        ))];

        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(FailingRuntimeFactory),
        )
        .unwrap();
        let Err(error) = state
            .ocr(OcrModel::new(model_id, OcrModelKind::TraditionalOcr), None)
            .await
        else {
            panic!("injected OCR runtime factory should be called");
        };

        assert!(format!("{error:#}").contains("injected OCR runtime factory"));
    }

    #[tokio::test]
    async fn known_ocr_vl_model_reaches_injected_runtime_factory() {
        let mut config = test_config();
        config.services.ocr_vl.enabled = true;
        let model_id = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        config.services.ocr_vl.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            model_id.clone(),
            OcrModelKind::OcrVl,
        ))];

        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(FailingRuntimeFactory),
        )
        .unwrap();
        let Err(error) = state
            .ocr_vl(OcrModel::new(model_id, OcrModelKind::OcrVl), None)
            .await
        else {
            panic!("injected OCR runtime factory should be called");
        };

        assert!(format!("{error:#}").contains("injected OCR runtime factory"));
    }

    struct FailingRuntimeFactory;

    struct ScriptedLlmRuntimeFactory {
        engines: Vec<LlmEngine>,
        loads: std::sync::atomic::AtomicUsize,
    }

    impl ModelRuntimeFactory for ScriptedLlmRuntimeFactory {
        fn load_asr(
            &self,
            _model: AsrModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> AsrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("ASR is not used by scripted LLM tests") })
        }

        fn load_tts(
            &self,
            _model: TtsModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> TtsRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("TTS is not used by scripted LLM tests") })
        }

        fn load_ocr(
            &self,
            _model: OcrModel,
            _model_dir: PathBuf,
            _cache_root: PathBuf,
            _layout_model: Option<(OcrModel, PathBuf)>,
            _table_structure: Option<TableStructureAssets>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("OCR is not used by scripted LLM tests") })
        }

        fn load_llm(
            &self,
            _model: LlmModel,
            _path: PathBuf,
            _deployment: LlmModelDeployment,
        ) -> LlmRuntimeFuture<'_> {
            let index = self
                .loads
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                .min(self.engines.len() - 1);
            let engine = self.engines[index].clone();
            Box::pin(async move { Ok(engine) })
        }
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
            Box::pin(async { panic!("recognition is not used by this test engine") })
        }
    }

    #[derive(Default)]
    struct RecordingOcrRuntimeFactory {
        layouts: StdMutex<Vec<Option<ModelId>>>,
    }

    #[derive(Default)]
    struct RecordingModelProvisioner {
        models: StdMutex<Vec<String>>,
    }

    impl RecordingModelProvisioner {
        fn provision<M: ModelSpec>(
            &self,
            model: &M,
            models_dir: PathBuf,
        ) -> ModelProvisionFuture<'_> {
            self.models
                .lock()
                .unwrap()
                .push(model.huggingface_repo().to_string());
            let path = model.cache_path(models_dir);
            Box::pin(async move { Ok(path) })
        }
    }

    impl ModelProvisioner<AsrModel> for RecordingModelProvisioner {
        fn provision(
            &self,
            model: AsrModel,
            _provisioning: Option<ModelProvisioning>,
            models_dir: PathBuf,
        ) -> ModelProvisionFuture<'_> {
            self.provision(&model, models_dir)
        }
    }

    impl ModelProvisioner<TtsModel> for RecordingModelProvisioner {
        fn provision(
            &self,
            model: TtsModel,
            _provisioning: Option<ModelProvisioning>,
            models_dir: PathBuf,
        ) -> ModelProvisionFuture<'_> {
            self.provision(&model, models_dir)
        }
    }

    impl ModelProvisioner<OcrModel> for RecordingModelProvisioner {
        fn provision(
            &self,
            model: OcrModel,
            _provisioning: Option<ModelProvisioning>,
            models_dir: PathBuf,
        ) -> ModelProvisionFuture<'_> {
            self.provision(&model, models_dir)
        }
    }

    impl OcrDeploymentProvisioner for RecordingModelProvisioner {
        fn provision_deployment(
            &self,
            _primary: OcrModel,
            _plan: DeploymentArtifactPlan,
            _models_dir: PathBuf,
        ) -> OcrDeploymentFuture<'_> {
            Box::pin(async {
                anyhow::bail!("OCR deployment provisioning is not used by this test")
            })
        }
    }

    impl ModelRuntimeFactory for RecordingOcrRuntimeFactory {
        fn load_asr(
            &self,
            _model: AsrModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> AsrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("ASR is not used by this test factory") })
        }

        fn load_tts(
            &self,
            _model: TtsModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> TtsRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("TTS is not used by this test factory") })
        }

        fn load_ocr(
            &self,
            model: OcrModel,
            _model_dir: PathBuf,
            _cache_root: PathBuf,
            layout_model: Option<(OcrModel, PathBuf)>,
            _table_structure: Option<TableStructureAssets>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            self.layouts
                .lock()
                .unwrap()
                .push(layout_model.map(|(model, _)| model.id().clone()));
            let model = model.id().clone();
            Box::pin(async move { Ok(Ocr::from_engine(Arc::new(TestOcrEngine { model }))) })
        }
    }

    #[tokio::test]
    async fn ocr_runtime_cache_keys_include_the_deployment_layout_model() {
        let mut config = test_config();
        config.models.max_loaded = 3;
        config.services.ocr.enabled = true;
        config.services.ocr.max_loaded = 3;
        let primary_id = ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap();
        let layout_id = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let primary = OcrModel::new(primary_id.clone(), OcrModelKind::TraditionalOcr);
        config.services.ocr.models =
            vec![OcrModelDeployment::from_runtime(primary.clone()).with_supported_layout()];
        let factory = Arc::new(RecordingOcrRuntimeFactory::default());
        let provisioner = Arc::new(RecordingModelProvisioner::default());
        let state = AppState::load_with_components(config, provisioner.clone(), factory.clone())
            .await
            .unwrap();
        let primary = OcrModel::new(primary_id, OcrModelKind::TraditionalOcr);
        let layout = OcrModel::new(layout_id.clone(), OcrModelKind::Layout);

        drop(state.ocr(primary.clone(), None).await.unwrap().unwrap());
        drop(
            state
                .ocr(primary.clone(), Some(layout.clone()))
                .await
                .unwrap()
                .unwrap(),
        );
        drop(state.ocr(primary, Some(layout)).await.unwrap().unwrap());

        assert_eq!(
            *factory.layouts.lock().unwrap(),
            vec![None, Some(layout_id)]
        );
        assert_eq!(
            *provisioner.models.lock().unwrap(),
            vec!["PaddlePaddle/PP-OCRv6_tiny", "PaddlePaddle/PP-DocLayoutV3"]
        );
        state.shutdown().await;
    }

    struct TestAsrEngine {
        model: AsrModel,
    }

    impl AsrEngine for TestAsrEngine {
        fn model(&self) -> AsrModel {
            self.model.clone()
        }

        fn transcribe_file_with(
            &self,
            _path: PathBuf,
            _options: AsrOptions,
        ) -> AsrEngineFuture<'_, AsrTranscript> {
            Box::pin(async { Ok(test_transcript()) })
        }

        fn transcribe_samples_with(
            &self,
            _samples: Vec<f32>,
            _sample_rate: u32,
            _options: AsrOptions,
        ) -> AsrEngineFuture<'_, AsrTranscript> {
            Box::pin(async { Ok(test_transcript()) })
        }

        fn start_streaming_with(
            &self,
            _options: AsrStreamingOptions,
        ) -> AsrEngineFuture<'_, Box<dyn AsrStreamSession>> {
            Box::pin(async {
                Err(orchion::OrchionError::InvalidAudio {
                    reason: "streaming is not used by this test engine".to_string(),
                })
            })
        }
    }

    fn test_transcript() -> AsrTranscript {
        AsrTranscript {
            text: "test".to_string(),
            language: "en".to_string(),
            raw_output: "test".to_string(),
            segments: Vec::new(),
        }
    }

    struct SuccessfulAsrRuntimeFactory;

    #[derive(Default)]
    struct RecordingAsrPathRuntimeFactory {
        paths: StdMutex<Vec<PathBuf>>,
    }

    impl ModelRuntimeFactory for RecordingAsrPathRuntimeFactory {
        fn load_asr(
            &self,
            model: AsrModel,
            path: PathBuf,
            _device: DevicePreference,
        ) -> AsrRuntimeFuture<'_> {
            self.paths.lock().unwrap().push(path);
            Box::pin(async move { Ok(Asr::from_engine(Arc::new(TestAsrEngine { model }))) })
        }

        fn load_tts(
            &self,
            _model: TtsModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> TtsRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("TTS is not used by this test factory") })
        }

        fn load_ocr(
            &self,
            _model: OcrModel,
            _model_dir: PathBuf,
            _cache_root: PathBuf,
            _layout_model: Option<(OcrModel, PathBuf)>,
            _table_structure: Option<TableStructureAssets>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("OCR is not used by this test factory") })
        }
    }

    impl ModelRuntimeFactory for SuccessfulAsrRuntimeFactory {
        fn load_asr(
            &self,
            model: AsrModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> AsrRuntimeFuture<'_> {
            Box::pin(async move { Ok(Asr::from_engine(Arc::new(TestAsrEngine { model }))) })
        }

        fn load_tts(
            &self,
            _model: TtsModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> TtsRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("TTS is not used by this test factory") })
        }

        fn load_ocr(
            &self,
            _model: OcrModel,
            _model_dir: PathBuf,
            _cache_root: PathBuf,
            _layout_model: Option<(OcrModel, PathBuf)>,
            _table_structure: Option<TableStructureAssets>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("OCR is not used by this test factory") })
        }
    }

    impl ModelRuntimeFactory for FailingRuntimeFactory {
        fn load_asr(
            &self,
            _model: AsrModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> AsrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("injected ASR runtime factory") })
        }

        fn load_tts(
            &self,
            _model: TtsModel,
            _path: PathBuf,
            _device: DevicePreference,
        ) -> TtsRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("injected TTS runtime factory") })
        }

        fn load_ocr(
            &self,
            _model: OcrModel,
            _model_dir: PathBuf,
            _cache_root: PathBuf,
            _layout_model: Option<(OcrModel, PathBuf)>,
            _table_structure: Option<TableStructureAssets>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("injected OCR runtime factory") })
        }
    }

    #[tokio::test]
    async fn prepared_remote_model_uses_deterministic_deployment_path() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        config.models.source = crate::settings::ModelSource::HuggingFace;
        config.services.asr.models[0].model =
            orchion::ModelUrl::parse("//Mirror/Shared-ASR-Package").unwrap();
        let model = config.services.asr.default_model.clone();
        let provisioning =
            deployment_provisioning(&config.services.asr.models, &[DownloadSource::HuggingFace])
                .remove(&model)
                .unwrap();
        let expected =
            resolve_prepared_provisioning_path(&model, &provisioning, &config.models.dir).unwrap();
        let factory = Arc::new(RecordingAsrPathRuntimeFactory::default());
        let state =
            AppState::from_prepared_config_with_runtime_factory(config, factory.clone()).unwrap();

        drop(state.asr(model).await.unwrap().unwrap());

        assert_eq!(*factory.paths.lock().unwrap(), [expected]);
    }

    #[tokio::test]
    async fn prepared_local_model_uses_validated_local_package_path() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        let package = config.models.dir.parent().unwrap().join("local-asr");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("config.json"), "{}").unwrap();
        std::fs::write(package.join("tokenizer.json"), "{}").unwrap();
        config.services.asr.models[0].model =
            orchion::ModelUrl::parse(&format!("file://{}", package.display())).unwrap();
        let model = config.services.asr.default_model.clone();
        let factory = Arc::new(RecordingAsrPathRuntimeFactory::default());
        let state =
            AppState::from_prepared_config_with_runtime_factory(config, factory.clone()).unwrap();

        drop(state.asr(model).await.unwrap().unwrap());

        assert_eq!(*factory.paths.lock().unwrap(), [package]);
    }

    #[test]
    fn prepared_constructor_rejects_missing_local_package() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        let missing = config
            .models
            .dir
            .parent()
            .unwrap()
            .join("missing-local-asr");
        config.services.asr.models[0].model =
            orchion::ModelUrl::parse(&format!("file://{}", missing.display())).unwrap();

        let Err(error) = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(RecordingAsrPathRuntimeFactory::default()),
        ) else {
            panic!("missing prepared local package should fail construction");
        };

        assert!(error.to_string().contains("invalid local model path"));
    }

    #[test]
    fn explicit_active_locators_ignore_invalid_source_environment_and_disabled_defaults() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        config.services.asr.models[0].model =
            orchion::ModelUrl::parse("hf://Qwen/Qwen3-ASR-0.6B").unwrap();
        assert!(
            config
                .services
                .tts
                .models
                .iter()
                .any(|deployment| deployment.model.source() == ModelUrlSource::Neutral)
        );

        let candidates =
            resolve_config_source_candidates_with_env(&config, Some("invalid-provider")).unwrap();

        assert!(candidates.is_empty());
        let hugging_face =
            deployment_provisioning(&config.services.asr.models, &[DownloadSource::HuggingFace]);
        let model_scope =
            deployment_provisioning(&config.services.asr.models, &[DownloadSource::ModelScope]);
        let model = &config.services.asr.default_model;
        assert_eq!(
            hugging_face[model].source_intent,
            model_scope[model].source_intent
        );
        assert!(hugging_face[model].source_plan.is_none());
        assert!(model_scope[model].source_plan.is_none());
    }

    #[test]
    fn active_neutral_auto_uses_documented_candidate_order() {
        let mut config = test_config();
        config.services.asr.enabled = true;

        assert_eq!(
            resolve_config_source_candidates_with_env(&config, None).unwrap(),
            [DownloadSource::HuggingFace, DownloadSource::ModelScope]
        );
        assert!(
            resolve_config_source_candidates_with_env(&config, Some("invalid-provider")).is_err()
        );
    }

    #[test]
    fn distinct_layout_locators_have_deployment_scoped_runtime_keys() {
        let config = ServerConfig::from_toml_str(
            r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "hf://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_small"
model = "//PaddlePaddle/PP-OCRv6_small"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
            std::path::Path::new("/tmp/orchion-server"),
        )
        .unwrap();

        let resolved = resolve_configured_ocr_models(
            &config,
            &[DownloadSource::HuggingFace, DownloadSource::ModelScope],
        );

        assert_eq!(resolved.layout.len(), 2);
        assert_ne!(
            resolved.layout[0].deployment_id,
            resolved.layout[1].deployment_id
        );
        let urls = resolved
            .layout
            .iter()
            .map(|key| {
                resolved
                    .layout_locators
                    .get(key)
                    .unwrap()
                    .model_url
                    .source()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            urls,
            [ModelUrlSource::HuggingFace, ModelUrlSource::ModelScope]
        );
    }

    #[tokio::test]
    async fn custom_runtime_factory_controls_model_loading() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        let model = config.services.asr.default_model.clone();
        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(FailingRuntimeFactory),
        )
        .unwrap();

        let Err(error) = state.asr(model).await else {
            panic!("custom runtime factory should control ASR loading");
        };

        assert!(format!("{error:#}").contains("injected ASR runtime factory"));
    }

    #[tokio::test]
    async fn model_lifecycle_loads_reports_and_unloads_runtime() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        let model = config.services.asr.default_model.clone();
        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(SuccessfulAsrRuntimeFactory),
        )
        .unwrap();
        let selector = ModelSelector {
            model: model.as_str().to_string(),
            service: ModelService::Asr,
        };

        let loaded = state.load_model(selector.clone()).await.unwrap().unwrap();
        assert_eq!(loaded.status, ModelResidency::Loaded);
        assert!(state.model_statuses().await.iter().any(|status| {
            status.id == model.as_str() && status.status == ModelResidency::Loaded
        }));

        let unloaded = state.unload_model(selector).await.unwrap().unwrap();
        assert_eq!(unloaded.status, ModelResidency::Unloaded);
        assert!(state.model_statuses().await.iter().any(|status| {
            status.id == model.as_str() && status.status == ModelResidency::Unloaded
        }));

        state
            .load_model(ModelSelector {
                model: model.as_str().to_string(),
                service: ModelService::Asr,
            })
            .await
            .unwrap()
            .unwrap();
        state.shutdown().await;
        assert_eq!(
            state.asr_models.status(&model).await,
            Some(ModelResidencyStatus::Unloaded)
        );
    }

    #[tokio::test]
    async fn idle_cleanup_uses_model_deadline_and_joins_on_shutdown() {
        let mut config = test_config();
        config.services.asr.enabled = true;
        config.services.asr.idle_timeout = std::time::Duration::from_millis(20);
        let model = config.services.asr.default_model.clone();
        let state = Arc::new(
            AppState::from_prepared_config_with_runtime_factory(
                config,
                Arc::new(SuccessfulAsrRuntimeFactory),
            )
            .unwrap(),
        );
        state.spawn_idle_cleanup();

        let lease = state.asr(model.clone()).await.unwrap().unwrap();
        drop(lease);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if state.asr_models.status(&model).await == Some(ModelResidencyStatus::Unloaded) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(1), state.shutdown())
            .await
            .unwrap();
        assert!(
            state
                .cleanup_task
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none()
        );
    }

    #[test]
    fn builtin_runtime_factory_rejects_foreign_registered_speech_models() {
        let mut config = test_config();
        let model = AsrModel::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        config.services.asr.enabled = true;
        config.services.asr.default_model = model.clone();
        config.services.asr.models = vec![ModelDeployment::from_asr_runtime(model)];

        let Err(error) = AppState::from_prepared_config(config) else {
            panic!("builtin runtime factory should reject a foreign ASR descriptor");
        };

        assert!(
            error
                .to_string()
                .contains("not a supported ASR runtime model")
        );
    }

    #[test]
    fn builtin_runtime_factory_rejects_custom_ocr_models_before_serving() {
        let mut config = test_config();
        let model = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.enabled = true;
        config.services.ocr.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            model,
            OcrModelKind::TraditionalOcr,
        ))];

        let Err(error) = AppState::from_prepared_config(config) else {
            panic!("builtin runtime factory should reject a custom OCR model");
        };

        assert!(
            error
                .to_string()
                .contains("not a supported traditional OCR")
        );
    }

    #[test]
    fn llm_plan_uses_exact_main_and_mmproj_roles_and_profiles_source_intent() {
        let config = ServerConfig::from_toml_str(
            r#"
            [services.llm]
            enabled = true
            default_model = "qwen/test"
            [[services.llm.models]]
            id = "qwen/test"
            model = "//owner/repo/main.gguf"
            mmproj_model = "//owner/repo/mmproj.gguf"
            runtime = { parallel_sequences = 1 }
            chat_template = { engine = "jinja", template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}" }
            generation = { max_tokens = 128 }
        "#,
            std::path::Path::new("/tmp/orchion-server"),
        )
        .unwrap();
        let plan = llm_deployment_artifact_plan(
            &config.services.llm.models[0],
            &[DownloadSource::HuggingFace, DownloadSource::ModelScope],
        );
        assert_eq!(plan.category, ModelCategory::Llm);
        assert_eq!(plan.artifacts.len(), 2);
        assert_eq!(plan.artifacts[0].role, ArtifactRole::LlmModel);
        assert_eq!(plan.artifacts[0].files, ["main.gguf"]);
        assert_eq!(plan.artifacts[1].role, ArtifactRole::LlmMmproj);
        assert_eq!(plan.artifacts[1].files, ["mmproj.gguf"]);
        assert!(plan.source_intent.contains("parallel_sequences: 1"));
        assert!(plan.source_intent.contains("generation="));
        assert!(plan.source_intent.contains("engine: Jinja"));
        assert!(plan.source_intent.contains("enable_thinking: false"));
        assert!(plan.source_intent.contains("add_generation_prompt"));
    }

    #[tokio::test]
    async fn managed_generation_keeps_llm_lease_and_global_permit_until_terminal_ack() {
        let (_root, state, control) = scripted_llm_state(
            vec![
                GenerationEvent::ContentDelta("hello".to_string()),
                GenerationEvent::Finished {
                    reason: GenerationFinishReason::Stop,
                    usage: LlmUsage {
                        prompt_tokens: 2,
                        completion_tokens: 1,
                        total_tokens: 3,
                        queue_time_ms: None,
                        eval_time_ms: None,
                    },
                },
            ],
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        assert!(
            !start.is_finished(),
            "readiness must precede handler return"
        );
        let unload = tokio::spawn({
            let state = Arc::clone(&state);
            async move {
                state
                    .unload_model(ModelSelector {
                        model: "qwen/test".to_string(),
                        service: ModelService::Llm,
                    })
                    .await
                    .unwrap()
            }
        });
        control.release_ready();
        let mut generation = start.await.unwrap().unwrap().unwrap();
        assert!(!unload.is_finished());
        assert!(matches!(
            generation.next().await.unwrap().unwrap(),
            GenerationEvent::ContentDelta(_)
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), generation.next())
                .await
                .is_err(),
            "wire terminal must wait for native cleanup ack"
        );
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(20),
                state.resources.acquire_inference(),
            )
            .await
            .is_err()
        );
        control.release_cleanup();
        let GenerationEvent::Finished { usage, .. } = generation.next().await.unwrap().unwrap()
        else {
            panic!("expected terminal event");
        };
        assert!(usage.queue_time_ms.is_some());
        assert!(usage.eval_time_ms.is_some());
        assert_eq!(
            unload.await.unwrap().unwrap().status,
            ModelResidency::Unloaded
        );
        drop(generation);
        state.shutdown().await;
    }

    #[tokio::test]
    async fn llm_queue_and_generation_deadlines_are_structured_and_preempt_work() {
        let (_root, state, control) = scripted_llm_state(
            Vec::new(),
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(20),
        )
        .await;
        let queue = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        assert!(matches!(
            queue.await.unwrap(),
            Err(RuntimeError::Timeout(_))
        ));

        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        control.release_ready();
        let mut generation = start.await.unwrap().unwrap().unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        control.release_cleanup();
        assert!(matches!(
            generation.next().await.unwrap(),
            Err(RuntimeError::Timeout(_))
        ));
        state.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_remains_first_cause_when_worker_acknowledges_cancelled() {
        let (_root, state, control) = scripted_llm_state(
            Vec::new(),
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(10),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let mut generation = start.await.unwrap().unwrap().unwrap();
        state.begin_shutdown();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        control.release_cleanup();
        assert!(matches!(
            generation.next().await.unwrap(),
            Err(RuntimeError::ShuttingDown)
        ));
        state.shutdown().await;
    }

    #[tokio::test]
    async fn ready_reservation_does_not_execute_or_take_global_permit_before_commit() {
        let (_root, state, control) = scripted_llm_state(
            vec![GenerationEvent::ContentDelta("must not run".to_string())],
            std::time::Duration::from_millis(50),
            std::time::Duration::from_secs(1),
        )
        .await;
        let global = state.resources.acquire_inference().await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(!start.is_finished());
        assert!(!control.has_executed());
        assert!(matches!(
            start.await.unwrap(),
            Err(RuntimeError::Timeout(_))
        ));
        assert!(!control.has_executed());
        drop(global);
        state.shutdown().await;
    }

    #[tokio::test]
    async fn begin_shutdown_fences_new_llm_admission_before_worker_reservation() {
        let (_root, state, control) = scripted_llm_state(
            Vec::new(),
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .await;
        state.begin_shutdown();
        assert!(matches!(
            state.start_generation(test_llm_command()).await,
            Err(RuntimeError::ShuttingDown)
        ));
        assert!(!control.has_started());
        assert!(!control.has_executed());
        state.shutdown().await;
    }

    #[tokio::test]
    async fn llm_shutdown_cancels_a_backpressured_stream_and_does_not_deadlock() {
        let (_root, state, control) = scripted_llm_state(
            vec![
                GenerationEvent::ContentDelta("one".to_string()),
                GenerationEvent::ContentDelta("two".to_string()),
            ],
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(10),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let _generation = start.await.unwrap().unwrap().unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        state.begin_shutdown();
        let shutdown = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.shutdown().await }
        });
        control.release_cleanup();
        tokio::time::timeout(std::time::Duration::from_secs(1), shutdown)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn full_http_queue_timeout_releases_lease_before_terminal_delivery() {
        let (_root, state, control) = scripted_llm_state(
            vec![
                GenerationEvent::ContentDelta("one".to_string()),
                GenerationEvent::ContentDelta("two".to_string()),
            ],
            std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(20),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let mut generation = start.await.unwrap().unwrap().unwrap();
        let unload = tokio::spawn({
            let state = Arc::clone(&state);
            async move {
                state
                    .unload_model(ModelSelector {
                        model: "qwen/test".to_string(),
                        service: ModelService::Llm,
                    })
                    .await
                    .unwrap()
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        control.release_cleanup();
        let unloaded = tokio::time::timeout(std::time::Duration::from_secs(1), unload)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(unloaded.status, ModelResidency::Unloaded);
        assert!(matches!(
            generation.next().await.unwrap().unwrap(),
            GenerationEvent::ContentDelta(_)
        ));
        assert!(matches!(
            generation.next().await.unwrap(),
            Err(RuntimeError::Timeout(_))
        ));
        assert!(generation.next().await.is_none());
        state.shutdown().await;
    }

    #[tokio::test]
    async fn panicked_llm_worker_is_retired_and_status_is_not_loaded() {
        let (engine, control) = scripted_panicking_llm_engine();
        let (healthy_engine, healthy_control) =
            scripted_llm_engine(vec![GenerationEvent::Finished {
                reason: GenerationFinishReason::Stop,
                usage: LlmUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    queue_time_ms: None,
                    eval_time_ms: None,
                },
            }]);
        let (_root, state, control) = scripted_llm_state_with_engines(
            vec![engine, healthy_engine],
            control,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let mut generation = start.await.unwrap().unwrap().unwrap();
        control.release_cleanup();
        assert!(matches!(
            generation.next().await.unwrap(),
            Err(RuntimeError::Core(
                orchion::OrchionError::LlmWorkerFailed { .. }
            ))
        ));
        assert_eq!(
            state
                .llm_models
                .status(&LlmRuntimeKey::new(ModelId::parse("qwen/test").unwrap()))
                .await,
            Some(ModelResidencyStatus::Unloaded)
        );
        let reload = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let healthy_started = healthy_control.clone();
        tokio::task::spawn_blocking(move || healthy_started.wait_started())
            .await
            .unwrap();
        healthy_control.release_ready();
        let mut healthy_generation = reload.await.unwrap().unwrap().unwrap();
        healthy_control.release_cleanup();
        assert!(matches!(
            healthy_generation.next().await.unwrap().unwrap(),
            GenerationEvent::Finished { .. }
        ));
        state.shutdown().await;
    }

    #[tokio::test]
    async fn dropped_handler_readiness_retires_worker_that_panics_after_commit() {
        let (engine, control) = scripted_panicking_llm_engine();
        let (_root, state, control) = scripted_llm_state_with_engine(
            engine,
            control,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .await;
        let defaults = &state.config.services.llm.models[0].generation;
        let request = GenerationRequest {
            messages: test_llm_command().messages,
            options: GenerationOptions {
                max_tokens: defaults.max_tokens,
                temperature: defaults.temperature,
                top_p: defaults.top_p,
                top_k: defaults.top_k,
                min_p: defaults.min_p,
                presence_penalty: defaults.presence_penalty,
                frequency_penalty: defaults.frequency_penalty,
                repeat_penalty: defaults.repeat_penalty,
                seed: u32::MAX,
                stop: Vec::new(),
            },
        };
        let (ready, readiness) = tokio::sync::oneshot::channel();
        drop(readiness);
        let (events, event_receiver) = tokio::sync::mpsc::channel(1);
        drop(event_receiver);
        let (terminal, terminal_receiver) = tokio::sync::oneshot::channel();
        drop(terminal_receiver);
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancellation = Arc::new(tokio::sync::Notify::new());
        let owner = tokio::spawn(own_llm_generation(
            state.as_ref().clone(),
            LlmRuntimeKey::new(ModelId::parse("qwen/test").unwrap()),
            request,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
            ready,
            events,
            terminal,
            cancelled,
            cancellation,
        ));
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !control.has_executed() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        control.release_cleanup();
        owner.await.unwrap();
        assert_eq!(
            state
                .llm_models
                .status(&LlmRuntimeKey::new(ModelId::parse("qwen/test").unwrap()))
                .await,
            Some(ModelResidencyStatus::Unloaded)
        );
        state.shutdown().await;
    }

    #[tokio::test]
    async fn preparation_panic_is_retired_before_error_and_cold_reloads() {
        let (panicking, control) = scripted_preparation_panicking_llm_engine();
        let (healthy, healthy_control) = scripted_llm_engine(vec![GenerationEvent::Finished {
            reason: GenerationFinishReason::Stop,
            usage: LlmUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                queue_time_ms: None,
                eval_time_ms: None,
            },
        }]);
        let (_root, state, control) = scripted_llm_state_with_engines(
            vec![panicking, healthy],
            control,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        assert!(matches!(
            start.await.unwrap(),
            Err(RuntimeError::Core(
                orchion::OrchionError::LlmWorkerFailed { .. }
            ))
        ));
        assert_eq!(
            state
                .llm_models
                .status(&LlmRuntimeKey::new(ModelId::parse("qwen/test").unwrap()))
                .await,
            Some(ModelResidencyStatus::Unloaded)
        );

        let reload = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = healthy_control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        healthy_control.release_ready();
        let mut generation = reload.await.unwrap().unwrap().unwrap();
        healthy_control.release_cleanup();
        assert!(matches!(
            generation.next().await.unwrap().unwrap(),
            GenerationEvent::Finished { .. }
        ));
        state.shutdown().await;
    }

    #[tokio::test]
    async fn slow_preparation_times_out_before_readiness_and_releases_resources() {
        let (engine, control) = scripted_slow_preparation_llm_engine();
        let (_root, state, control) = scripted_llm_state_with_engine(
            engine,
            control,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(20),
        )
        .await;
        let start = tokio::spawn({
            let state = Arc::clone(&state);
            async move { state.start_generation(test_llm_command()).await }
        });
        let started = control.clone();
        tokio::task::spawn_blocking(move || started.wait_started())
            .await
            .unwrap();
        control.release_ready();
        let preparation = control.clone();
        tokio::task::spawn_blocking(move || preparation.wait_preparation_started())
            .await
            .unwrap();
        assert!(matches!(
            start.await.unwrap(),
            Err(RuntimeError::Timeout(message)) if message == "LLM preparation timed out"
        ));
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            state.resources.acquire_inference(),
        )
        .await
        .unwrap();
        assert!(!control.has_executed());
        state.shutdown().await;
    }

    async fn scripted_llm_state(
        script: Vec<GenerationEvent>,
        queue_timeout: std::time::Duration,
        generation_timeout: std::time::Duration,
    ) -> (tempfile::TempDir, Arc<AppState>, LlmScriptedControl) {
        let (engine, control) = scripted_llm_engine(script);
        scripted_llm_state_with_engine(engine, control, queue_timeout, generation_timeout).await
    }

    async fn scripted_llm_state_with_engine(
        engine: LlmEngine,
        control: LlmScriptedControl,
        queue_timeout: std::time::Duration,
        generation_timeout: std::time::Duration,
    ) -> (tempfile::TempDir, Arc<AppState>, LlmScriptedControl) {
        scripted_llm_state_with_engines(vec![engine], control, queue_timeout, generation_timeout)
            .await
    }

    async fn scripted_llm_state_with_engines(
        engines: Vec<LlmEngine>,
        control: LlmScriptedControl,
        queue_timeout: std::time::Duration,
        generation_timeout: std::time::Duration,
    ) -> (tempfile::TempDir, Arc<AppState>, LlmScriptedControl) {
        let root = tempfile::tempdir().unwrap();
        let model_file = root.path().join("model.gguf");
        std::fs::write(&model_file, b"scripted model").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        config.models.dir = root.path().join("models");
        config.server.max_concurrent_inference = 1;
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config.services.ocr.enabled = false;
        config.services.ocr_vl.enabled = false;
        let id = ModelId::parse("qwen/test").unwrap();
        let runtime = LlmRuntimeConfig {
            event_queue_capacity: 1,
            queue_timeout,
            generation_timeout,
            ..LlmRuntimeConfig::default()
        };
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: None,
            model: orchion::ModelUrl::parse(&format!("file://{}", model_file.display())).unwrap(),
            mmproj_model: None,
            runtime,
            chat_template: ChatTemplateConfig::default(),
            generation: LlmGenerationConfig::default(),
        }];
        let state = AppState::load_with_components(
            config,
            Arc::new(ModelDownloader::new(DownloadSource::Auto)),
            Arc::new(ScriptedLlmRuntimeFactory {
                engines,
                loads: std::sync::atomic::AtomicUsize::new(0),
            }),
        )
        .await
        .unwrap();
        (root, state, control)
    }

    fn test_llm_command() -> LlmCommand {
        LlmCommand {
            model: "qwen/test".to_string(),
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: "hello".to_string(),
            }],
            options: crate::application::llm::LlmGenerationOverrides::default(),
            max_tokens_param: "max_completion_tokens",
            queue_timeout: None,
            generation_timeout: None,
        }
    }

    fn test_config() -> ServerConfig {
        let temp_dir = tempfile::tempdir().unwrap().keep();
        let exe_path = temp_dir.join("orchion-server");
        let mut config = ServerConfig::default_for_exe(&exe_path);
        config.models.dir = temp_dir.join("models");
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config
    }
}

impl AppState {
    #[cfg(test)]
    pub(crate) async fn load_with_test_llm_engine(
        config: ServerConfig,
        llm: LlmEngine,
    ) -> anyhow::Result<Arc<Self>> {
        let provisioner = Arc::new(
            ModelDownloader::new(config.models.source.into())
                .with_file_integrity_verification(config.models.verify_file_integrity),
        );
        Self::load_with_components(
            config,
            provisioner,
            Arc::new(TestLlmRuntimeFactory {
                llm,
                builtin: BuiltinModelRuntimeFactory,
            }),
        )
        .await
    }

    /// # Errors
    ///
    /// Returns an error when configuration, provisioning, or startup model loading fails.
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        config.validate()?;
        let provisioner = Arc::new(
            ModelDownloader::new(config.models.source.into())
                .with_file_integrity_verification(config.models.verify_file_integrity),
        );
        Self::load_with_provisioner(config, provisioner).await
    }

    /// # Errors
    ///
    /// Returns an error when configuration, provisioning, or startup model loading fails.
    pub async fn load_with_provisioner<P>(
        config: ServerConfig,
        provisioner: Arc<P>,
    ) -> anyhow::Result<Arc<Self>>
    where
        P: ModelProvisioner<AsrModel>
            + ModelProvisioner<TtsModel>
            + ModelProvisioner<OcrModel>
            + OcrDeploymentProvisioner
            + 'static,
    {
        config.validate()?;
        Self::load_with_components(config, provisioner, Arc::new(BuiltinModelRuntimeFactory)).await
    }

    /// Loads server state with injected provisioning and runtime adapters.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration, provisioning, or startup model loading fails.
    pub async fn load_with_components<P>(
        config: ServerConfig,
        provisioner: Arc<P>,
        runtime_factory: Arc<dyn ModelRuntimeFactory>,
    ) -> anyhow::Result<Arc<Self>>
    where
        P: ModelProvisioner<AsrModel>
            + ModelProvisioner<TtsModel>
            + ModelProvisioner<OcrModel>
            + OcrDeploymentProvisioner
            + 'static,
    {
        config.validate()?;
        validate_runtime_factory(&config, runtime_factory.as_ref())?;
        let source_candidates = resolve_config_source_candidates(&config)?;
        let llm_backend = initialize_configured_llm_backend(&config)?;
        let resolved_ocr_models = resolve_configured_ocr_models(&config, &source_candidates);
        let provisioners = ModelProvisioners::new(provisioner);
        let state = Arc::new(Self::build(
            config,
            resolved_ocr_models,
            Some(&provisioners),
            runtime_factory,
            &source_candidates,
            llm_backend,
        ));
        let counts = state.ensure_startup_models().await?;
        state.spawn_idle_cleanup();
        tracing::info!(
            asr = counts.asr,
            tts = counts.tts,
            ocr = counts.ocr,
            ocr_vl = counts.ocr_vl,
            llm = counts.llm,
            layout = counts.layout,
            "model cache ready"
        );
        Ok(state)
    }

    /// # Errors
    ///
    /// Returns an error when configured OCR model identifiers cannot be resolved.
    pub fn from_prepared_config(config: ServerConfig) -> anyhow::Result<Self> {
        config.validate()?;
        Self::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(BuiltinModelRuntimeFactory),
        )
    }

    /// Builds server state around already provisioned models and an injected runtime factory.
    ///
    /// # Errors
    ///
    /// Returns an error when configured OCR model identifiers cannot be resolved.
    pub fn from_prepared_config_with_runtime_factory(
        config: ServerConfig,
        runtime_factory: Arc<dyn ModelRuntimeFactory>,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        validate_runtime_factory(&config, runtime_factory.as_ref())?;
        let source_candidates = resolve_config_source_candidates(&config)?;
        let llm_backend = initialize_configured_llm_backend(&config)?;
        let resolved_ocr_models = resolve_configured_ocr_models(&config, &source_candidates);
        validate_prepared_model_paths(&config, &source_candidates, &resolved_ocr_models)?;
        Ok(Self::build(
            config,
            resolved_ocr_models,
            None,
            runtime_factory,
            &source_candidates,
            llm_backend,
        ))
    }

    #[must_use]
    pub const fn config(&self) -> &ServerConfig {
        &self.config
    }

    #[allow(
        clippy::too_many_lines,
        reason = "constructs the complete set of model caches from one validated configuration"
    )]
    fn build(
        config: ServerConfig,
        resolved_ocr_models: ResolvedOcrModels,
        provisioners: Option<&ModelProvisioners>,
        runtime_factory: Arc<dyn ModelRuntimeFactory>,
        source_candidates: &[DownloadSource],
        llm_backend: Option<LlmBackendGuard>,
    ) -> Self {
        let ResolvedOcrModels {
            assets: resolved_ocr_assets,
            asset_locators: resolved_ocr_asset_locators,
            ocr: resolved_ocr,
            ocr_vl: resolved_ocr_vl,
            layout: resolved_layout,
            layout_locators: resolved_layout_locators,
            deployment_plans: ocr_deployment_plans,
        } = resolved_ocr_models;
        let api_policy = api_policy(&config);
        let ocr_deployment_provisioner =
            provisioners.map(|provisioners| Arc::clone(&provisioners.ocr_deployments));
        let llm_deployment_provisioner =
            provisioners.map(|provisioners| Arc::clone(&provisioners.ocr_deployments));
        let llm_deployment_plans = config
            .services
            .llm
            .models
            .iter()
            .filter(|_| config.services.llm.active())
            .map(|deployment| {
                (
                    LlmRuntimeKey::new(deployment.id.clone()),
                    llm_deployment_artifact_plan(deployment, source_candidates),
                )
            })
            .collect::<HashMap<_, _>>();
        let model_residency = ResidencyDomain::new();
        let asr_models = build_model_cache(
            "asr",
            config.services.asr.runtime_models(),
            deployment_provisioning(&config.services.asr.models, source_candidates),
            config.services.asr.idle_timeout,
            config.services.asr.max_loaded,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.asr)),
            model_residency.clone(),
        );
        let tts_models = build_model_cache(
            "tts",
            config.services.tts.runtime_models(),
            deployment_provisioning(&config.services.tts.models, source_candidates),
            config.services.tts.idle_timeout,
            config.services.tts.max_loaded,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.tts)),
            model_residency.clone(),
        );
        let ocr_assets = build_model_cache(
            "ocr-assets",
            resolved_ocr_assets,
            resolved_ocr_asset_locators,
            config
                .services
                .ocr
                .idle_timeout
                .min(config.services.ocr_vl.idle_timeout),
            1,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.ocr)),
            model_residency.clone(),
        );
        let ocr_models = build_model_cache(
            "ocr",
            resolved_ocr,
            HashMap::new(),
            config.services.ocr.idle_timeout,
            config.services.ocr.max_loaded,
            config.models.dir.clone(),
            None,
            model_residency.clone(),
        );
        let ocr_vl_models = build_model_cache(
            "ocr-vl",
            resolved_ocr_vl,
            HashMap::new(),
            config.services.ocr_vl.idle_timeout,
            config.services.ocr_vl.max_loaded,
            config.models.dir.clone(),
            None,
            model_residency.clone(),
        );
        let layout_models = build_model_cache(
            "ocr-layout",
            resolved_layout,
            resolved_layout_locators,
            config
                .services
                .ocr
                .idle_timeout
                .min(config.services.ocr_vl.idle_timeout),
            1,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.layout)),
            model_residency.clone(),
        );
        let llm_models = build_model_cache(
            "llm",
            config
                .services
                .llm
                .models
                .iter()
                .filter(|_| config.services.llm.active())
                .map(|deployment| LlmRuntimeKey::new(deployment.id.clone()))
                .collect(),
            HashMap::new(),
            config.services.llm.idle_timeout,
            config.services.llm.max_loaded,
            config.models.dir.clone(),
            None,
            model_residency.clone(),
        );
        let global_models = GlobalModelCacheLimiter::new_in_domain(
            config.models.max_loaded,
            model_residency.clone(),
        );
        let (cleanup_shutdown, _) = watch::channel(false);
        let resources = ResourcePolicy::new(
            config.server.max_concurrent_inference,
            config.server.max_websocket_connections,
            config.server.max_pending_websocket_connections,
        );
        Self {
            config,
            source_candidates: source_candidates.to_vec(),
            api_policy,
            asr_models,
            tts_models,
            ocr_models,
            ocr_vl_models,
            ocr_assets,
            layout_models,
            llm_models,
            llm_deployment_plans,
            llm_deployment_provisioner,
            llm_backend: Arc::new(StdMutex::new(llm_backend)),
            ocr_deployment_plans,
            ocr_deployment_provisioner,
            global_models,
            model_residency,
            resources,
            runtime_factory,
            cleanup_shutdown,
            cleanup_task: Arc::new(StdMutex::new(None)),
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "provisions each service default and its deployment-owned OCR auxiliaries"
    )]
    async fn ensure_startup_models(self: &Arc<Self>) -> anyhow::Result<ProvisionedModelCounts> {
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_MODEL_PROVISIONS));
        let mut tasks = JoinSet::new();

        if self.config.services.asr.enabled {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.asr_models.clone(),
                self.config.services.asr.default_model.clone(),
                ProvisionedModelKind::Asr,
                "provision default ASR model",
            );
        }
        if self.config.services.tts.enabled {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.tts_models.clone(),
                self.config.services.tts.default_model.clone(),
                ProvisionedModelKind::Tts,
                "provision default TTS model",
            );
        }
        if self.config.services.ocr.active()
            && let Some(model) = &self.config.services.ocr.default_model
            && !self
                .ocr_deployment_plans
                .keys()
                .any(|key| key.primary.id() == model)
        {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.ocr_assets.clone(),
                OcrModel::new(model.clone(), OcrModelKind::TraditionalOcr),
                ProvisionedModelKind::Ocr,
                "provision default OCR model",
            );
        }
        if self.config.services.ocr_vl.active()
            && let Some(model) = &self.config.services.ocr_vl.default_model
        {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.ocr_assets.clone(),
                OcrModel::new(model.clone(), OcrModelKind::OcrVl),
                ProvisionedModelKind::OcrVl,
                "provision default OCR-VL model",
            );
        }

        let mut required_layouts = HashSet::new();
        if self.config.services.ocr.active()
            && let Some(deployment_id) = &self.config.services.ocr.default_model
            && !self
                .ocr_deployment_plans
                .keys()
                .any(|key| key.primary.id() == deployment_id)
            && let Some(model) = self.config.services.ocr.default_layout_runtime()
        {
            required_layouts.insert(DeploymentLayoutModel::new(
                deployment_id.clone(),
                model.clone(),
            ));
        }
        if self.config.services.ocr_vl.active()
            && let Some(deployment_id) = &self.config.services.ocr_vl.default_model
            && let Some(model) = self.config.services.ocr_vl.default_layout_runtime()
        {
            required_layouts.insert(DeploymentLayoutModel::new(
                deployment_id.clone(),
                model.clone(),
            ));
        }
        for layout in required_layouts {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.layout_models.clone(),
                layout,
                ProvisionedModelKind::Layout,
                "provision default OCR layout model",
            );
        }

        let mut counts = ProvisionedModelCounts::default();
        while let Some(result) = tasks.join_next().await {
            counts
                .record(result.map_err(|error| {
                    anyhow::anyhow!("model provision task failed: {error:#}")
                })??);
        }
        if let (Some(default), Some(provisioner)) = (
            self.config.services.ocr.default_model.as_ref(),
            self.ocr_deployment_provisioner.as_ref(),
        ) && let Some((key, plan)) = self
            .ocr_deployment_plans
            .iter()
            .find(|(key, _)| key.primary.id() == default)
        {
            provisioner
                .provision_deployment(
                    key.primary.clone(),
                    plan.clone(),
                    self.config.models.dir.clone(),
                )
                .await
                .context("provision default OCR deployment")?;
            counts.ocr += 1;
        }
        if let (Some(default), Some(provisioner)) = (
            self.config.services.llm.default_model.as_ref(),
            self.llm_deployment_provisioner.as_ref(),
        ) && self.config.services.llm.active()
            && let Some((model, plan)) = self
                .llm_deployment_plans
                .iter()
                .find(|(model, _)| model.id() == default)
        {
            provisioner
                .provision_llm_deployment(
                    model.id().clone(),
                    plan.clone(),
                    self.config.models.dir.clone(),
                )
                .await
                .context("provision default LLM deployment")?;
            counts.llm += 1;
        }
        Ok(counts)
    }

    #[must_use]
    pub const fn resources(&self) -> &ResourcePolicy {
        &self.resources
    }

    /// # Errors
    ///
    /// Returns an error when the ASR model cannot be provisioned or loaded.
    pub async fn asr(&self, model: orchion::AsrModel) -> anyhow::Result<Option<ModelLease<Asr>>> {
        if !self.config.services.asr.enabled {
            return Ok(None);
        }
        let device = self.config.services.asr.device;
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.asr_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading ASR model");
                    runtime_factory
                        .load_asr(model, path, device)
                        .await
                        .context("load ASR model")
                },
            )
            .await
    }

    /// # Errors
    ///
    /// Returns an error when the TTS model cannot be provisioned or loaded.
    pub async fn tts(&self, model: orchion::TtsModel) -> anyhow::Result<Option<ModelLease<Tts>>> {
        if !self.config.services.tts.enabled {
            return Ok(None);
        }
        let device = self.config.services.tts.device;
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.tts_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading TTS model");
                    runtime_factory
                        .load_tts(model, path, device)
                        .await
                        .context("load TTS model")
                },
            )
            .await
    }

    /// # Errors
    ///
    /// Returns an error when OCR assets cannot be provisioned or loaded.
    pub async fn ocr(
        &self,
        model: OcrModel,
        layout: Option<OcrModel>,
    ) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr.active() {
            return Ok(None);
        }
        let deployment = self
            .config
            .services
            .ocr
            .models
            .iter()
            .find(|deployment| deployment.id == *model.id());
        let layout = layout.map(|layout| DeploymentLayoutModel::new(model.id().clone(), layout));
        let table_structure = deployment.and_then(|deployment| deployment.table_structure.clone());
        let table_source_intent = if table_structure.is_some() {
            deployment
                .map(|deployment| ocr_deployment_source_intent(deployment, &self.source_candidates))
        } else {
            None
        };
        let key = OcrRuntimeKey::new(model, layout)
            .with_table_structure(table_structure)
            .with_table_source_intent(table_source_intent);
        let device = self.config.services.ocr.device;
        let models_dir = self.config.models.dir.clone();
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let ocr_assets = self.ocr_assets.clone();
        let layout_models = self.layout_models.clone();
        let deployment_plan = self.ocr_deployment_plans.get(&key).cloned();
        let deployment_provisioner = self.ocr_deployment_provisioner.clone();
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_models,
                all_caches.as_slice(),
                key,
                move |key, _| async move {
                    load_ocr_runtime(
                        key,
                        models_dir,
                        ocr_assets,
                        layout_models,
                        deployment_plan,
                        deployment_provisioner,
                        runtime_factory,
                        device,
                    )
                    .await
                    .context("load OCR model")
                },
            )
            .await
    }

    /// # Errors
    ///
    /// Returns an error when OCR-VL assets cannot be provisioned or loaded.
    pub async fn ocr_vl(
        &self,
        model: OcrModel,
        layout: Option<OcrModel>,
    ) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr_vl.active() {
            return Ok(None);
        }
        let layout = layout.map(|layout| DeploymentLayoutModel::new(model.id().clone(), layout));
        let key = OcrRuntimeKey::new(model, layout);
        let device = self.config.services.ocr_vl.device;
        let models_dir = self.config.models.dir.clone();
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let ocr_assets = self.ocr_assets.clone();
        let layout_models = self.layout_models.clone();
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_vl_models,
                all_caches.as_slice(),
                key,
                move |key, _| async move {
                    load_ocr_runtime(
                        key,
                        models_dir,
                        ocr_assets,
                        layout_models,
                        None,
                        None,
                        runtime_factory,
                        device,
                    )
                    .await
                    .context("load OCR-VL model")
                },
            )
            .await
    }

    async fn llm(&self, model: LlmRuntimeKey) -> anyhow::Result<Option<ModelLease<LlmEngine>>> {
        if !self.config.services.llm.active() {
            return Ok(None);
        }
        let Some(deployment) = self
            .config
            .services
            .llm
            .models
            .iter()
            .find(|deployment| deployment.id == *model.id())
            .cloned()
        else {
            return Ok(None);
        };
        let plan = self
            .llm_deployment_plans
            .get(&model)
            .cloned()
            .context("configured LLM deployment has no artifact plan")?;
        for attempt in 0..2 {
            let models_dir = self.config.models.dir.clone();
            let provisioner = self.llm_deployment_provisioner.clone();
            let runtime_factory = Arc::clone(&self.runtime_factory);
            let all_caches = self.active_model_caches();
            let deployment = deployment.clone();
            let plan = plan.clone();
            let lease = self
                .global_models
                .get_or_load(
                    &self.llm_models,
                    all_caches.as_slice(),
                    model.clone(),
                    move |model, _| async move {
                        let publication = if let Some(provisioner) = provisioner {
                            provisioner
                                .provision_llm_deployment(model.id().clone(), plan, models_dir)
                                .await?
                        } else {
                            let prepared_id = model.id().clone();
                            tokio::task::spawn_blocking(move || {
                                ModelDownloader::resolve_prepared_logical_deployment(
                                    &prepared_id,
                                    ModelCategory::Llm,
                                    &plan,
                                    &models_dir,
                                )
                            })
                            .await
                            .map_err(|error| {
                                anyhow::anyhow!("prepared LLM resolver failed: {error}")
                            })??
                        };
                        let path = publication
                            .artifact_file(ArtifactRole::LlmModel)
                            .context("published LLM deployment is missing its main GGUF")?
                            .to_path_buf();
                        runtime_factory
                            .load_llm(model.public_model(), path, deployment)
                            .await
                    },
                )
                .await?;
            let Some(lease) = lease else {
                return Ok(None);
            };
            if lease.is_healthy() {
                return Ok(Some(lease));
            }
            drop(lease);
            self.llm_models.unload(model.clone()).await?;
            if attempt == 1 {
                anyhow::bail!("LLM worker remained unhealthy after cold reload");
            }
        }
        unreachable!("bounded LLM reload loop returns")
    }

    fn active_model_caches(&self) -> Vec<&dyn CacheTracker> {
        let mut caches: Vec<&dyn CacheTracker> = Vec::with_capacity(5);
        if self.config.services.asr.enabled {
            caches.push(&self.asr_models);
        }
        if self.config.services.tts.enabled {
            caches.push(&self.tts_models);
        }
        if self.config.services.ocr.active() {
            caches.push(&self.ocr_models);
        }
        if self.config.services.ocr_vl.active() {
            caches.push(&self.ocr_vl_models);
        }
        if self.config.services.llm.active() {
            caches.push(&self.llm_models);
        }
        caches
    }

    fn spawn_idle_cleanup(self: &Arc<Self>) {
        let mut task = self
            .cleanup_task
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if task.is_some() {
            return;
        }

        let state = Arc::downgrade(self);
        let mut shutdown = self.cleanup_shutdown.subscribe();
        let mut residency = self.model_residency.subscribe();
        *task = Some(tokio::spawn(async move {
            loop {
                let Some(current) = state.upgrade() else {
                    break;
                };
                let deadline = current.next_idle_deadline().await;
                drop(current);

                let due = if let Some(deadline) = deadline {
                    tokio::select! {
                        () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => true,
                        changed = residency.changed() => {
                            if changed.is_err() { break; }
                            false
                        }
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() { break; }
                            false
                        }
                    }
                } else {
                    tokio::select! {
                        changed = residency.changed() => {
                            if changed.is_err() { break; }
                            false
                        }
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() { break; }
                            false
                        }
                    }
                };
                if due {
                    let Some(current) = state.upgrade() else {
                        break;
                    };
                    current.cleanup_idle_models().await;
                }
            }
        }));
    }

    async fn next_idle_deadline(&self) -> Option<std::time::Instant> {
        let mut deadline = None;
        if self.config.services.asr.enabled {
            deadline = earlier_deadline(deadline, self.asr_models.next_idle_deadline().await);
        }
        if self.config.services.tts.enabled {
            deadline = earlier_deadline(deadline, self.tts_models.next_idle_deadline().await);
        }
        if self.config.services.ocr.active() {
            deadline = earlier_deadline(deadline, self.ocr_models.next_idle_deadline().await);
        }
        if self.config.services.ocr_vl.active() {
            deadline = earlier_deadline(deadline, self.ocr_vl_models.next_idle_deadline().await);
        }
        if self.config.services.llm.active() {
            deadline = earlier_deadline(deadline, self.llm_models.next_idle_deadline().await);
        }
        deadline
    }

    async fn cleanup_idle_models(&self) {
        if self.config.services.asr.enabled {
            self.asr_models.cleanup_idle().await;
        }
        if self.config.services.tts.enabled {
            self.tts_models.cleanup_idle().await;
        }
        if self.config.services.ocr.active() {
            self.ocr_models.cleanup_idle().await;
        }
        if self.config.services.ocr_vl.active() {
            self.ocr_vl_models.cleanup_idle().await;
        }
        if self.config.services.llm.active() {
            self.llm_models.cleanup_idle().await;
        }
    }

    pub async fn shutdown(&self) {
        self.begin_shutdown();
        self.global_models.close_and_drain().await;
        let task = self
            .cleanup_task
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(task) = task
            && let Err(error) = task.await
        {
            tracing::error!(%error, "model cleanup task failed during shutdown");
        }
        self.unload_all_models().await;
        self.llm_backend
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }

    pub fn begin_shutdown(&self) {
        self.cleanup_shutdown.send_replace(true);
    }

    async fn unload_all_models(&self) {
        let source_candidates = &self.source_candidates;
        for deployment in &self.config.services.asr.models {
            let model = &deployment.runtime;
            if let Err(error) = self.asr_models.unload(model.clone()).await {
                tracing::error!(model = %model, %error, "failed to unload ASR model during shutdown");
            }
        }
        for deployment in &self.config.services.tts.models {
            let model = &deployment.runtime;
            if let Err(error) = self.tts_models.unload(model.clone()).await {
                tracing::error!(model = %model, %error, "failed to unload TTS model during shutdown");
            }
        }
        for deployment in &self.config.services.ocr.models {
            let id = &deployment.id;
            for key in ocr_runtime_keys_for_deployment(deployment, source_candidates) {
                if let Err(error) = self.ocr_models.unload(key).await {
                    tracing::error!(model = %id, %error, "failed to unload OCR model during shutdown");
                }
            }
        }
        for deployment in &self.config.services.ocr_vl.models {
            let id = &deployment.id;
            for key in ocr_runtime_keys_for_deployment(deployment, source_candidates) {
                if let Err(error) = self.ocr_vl_models.unload(key).await {
                    tracing::error!(model = %id, %error, "failed to unload OCR-VL model during shutdown");
                }
            }
        }
        for deployment in &self.config.services.llm.models {
            let model = LlmRuntimeKey::new(deployment.id.clone());
            if let Err(error) = self.llm_models.unload(model).await {
                tracing::error!(model = %deployment.id, %error, "failed to unload LLM during shutdown");
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "keeps legacy caches and atomic deployment publication dependencies explicit"
)]
async fn load_ocr_runtime(
    key: OcrRuntimeKey,
    models_dir: PathBuf,
    ocr_assets: OcrAssetCache,
    layout_models: LayoutModelCache,
    deployment_plan: Option<DeploymentArtifactPlan>,
    deployment_provisioner: Option<Arc<dyn OcrDeploymentProvisioner>>,
    runtime_factory: Arc<dyn ModelRuntimeFactory>,
    device: DevicePreference,
) -> anyhow::Result<Ocr> {
    let primary = key.primary;
    if let Some(plan) = deployment_plan {
        let publication = if let Some(provisioner) = deployment_provisioner {
            provisioner
                .provision_deployment(primary.clone(), plan, models_dir.clone())
                .await?
        } else {
            let prepared_primary = primary.clone();
            tokio::task::spawn_blocking(move || {
                ModelDownloader::resolve_prepared_deployment(&prepared_primary, &plan, &models_dir)
            })
            .await
            .map_err(|error| anyhow::anyhow!("prepared deployment resolver failed: {error}"))??
        };
        return load_published_ocr_runtime(
            primary,
            key.layout,
            key.table_structure,
            publication,
            runtime_factory,
            device,
        )
        .await;
    }
    let primary_path = ocr_assets
        .ensure_provisioned(primary.clone())
        .await?
        .with_context(|| format!("configured OCR model `{primary}` is unavailable"))?;
    let layout = if let Some(layout) = key.layout {
        let path = layout_models
            .ensure_provisioned(layout.clone())
            .await?
            .with_context(|| format!("configured OCR layout model `{layout}` is unavailable"))?;
        Some((layout.model, path))
    } else {
        None
    };
    let primary_cache_root =
        if primary_path.starts_with(models_dir.join(".orchion").join("deployments")) {
            primary_path
                .parent()
                .and_then(std::path::Path::parent)
                .map_or_else(|| models_dir.clone(), std::path::Path::to_path_buf)
        } else {
            models_dir.clone()
        };
    anyhow::ensure!(
        key.table_structure.is_none(),
        "OCR table structure deployment has no atomic artifact plan"
    );
    tracing::info!(model = ?primary, layout = ?layout.as_ref().map(|(model, _)| model), device = %device, "loading OCR model");
    runtime_factory
        .load_ocr(
            primary,
            primary_path,
            primary_cache_root,
            layout,
            None,
            device,
        )
        .await
}

async fn load_published_ocr_runtime(
    primary: OcrModel,
    layout: Option<DeploymentLayoutModel>,
    table: Option<TableStructureConfig>,
    publication: DeploymentPublication,
    runtime_factory: Arc<dyn ModelRuntimeFactory>,
    device: DevicePreference,
) -> anyhow::Result<Ocr> {
    let primary_cache_root = publication.role_root(ArtifactRole::Model);
    let primary_path = primary
        .id()
        .as_str()
        .split('/')
        .fold(primary_cache_root.clone(), |path, segment| {
            path.join(segment)
        });
    let layout = layout
        .map(|layout| {
            publication
                .artifact_file(ArtifactRole::OcrLayout)
                .map(|path| (layout.model, path.to_path_buf()))
                .with_context(|| "published OCR deployment is missing its layout artifact")
        })
        .transpose()?;
    let table_structure = table
        .map(|table| {
            Ok::<_, anyhow::Error>(TableStructureAssets {
                model: publication
                    .artifact_file(ArtifactRole::OcrTableStructureModel)
                    .context("published OCR deployment is missing its table model")?
                    .to_path_buf(),
                dictionary: publication
                    .artifact_file(ArtifactRole::OcrTableStructureDictionary)
                    .context("published OCR deployment is missing its table dictionary")?
                    .to_path_buf(),
                table_type: table.table_type.as_str().to_string(),
                score_threshold: table.score_threshold,
                max_structure_length: table.max_structure_length,
            })
        })
        .transpose()?;
    runtime_factory
        .load_ocr(
            primary,
            primary_path,
            primary_cache_root,
            layout,
            table_structure,
            device,
        )
        .await
}

fn earlier_deadline(
    current: Option<std::time::Instant>,
    candidate: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    match (current, candidate) {
        (Some(current), Some(candidate)) => Some(current.min(candidate)),
        (Some(current), None) => Some(current),
        (None, candidate) => candidate,
    }
}

async fn ocr_runtime_status(
    cache: &OcrRuntimeCache,
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> Option<ModelResidencyStatus> {
    let mut selected = None;
    for key in ocr_runtime_keys_for_deployment(deployment, source_candidates) {
        let Some(status) = cache.status(&key).await else {
            continue;
        };
        if selected
            .is_none_or(|current| residency_status_rank(status) > residency_status_rank(current))
        {
            selected = Some(status);
        }
    }
    selected
}

const fn residency_status_rank(status: ModelResidencyStatus) -> u8 {
    match status {
        ModelResidencyStatus::Unloaded => 0,
        ModelResidencyStatus::Unloading => 1,
        ModelResidencyStatus::Loading => 2,
        ModelResidencyStatus::Loaded => 3,
    }
}

async fn unload_ocr_runtimes(
    cache: &OcrRuntimeCache,
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> anyhow::Result<Option<bool>> {
    cache
        .unload_many(ocr_runtime_keys_for_deployment(
            deployment,
            source_candidates,
        ))
        .await
}

impl ServerApplication for AppState {
    fn api_policy(&self) -> &ApiPolicy {
        &self.api_policy
    }

    fn model_catalog(&self) -> crate::application::ModelCatalogFuture<'_> {
        Box::pin(async move {
            let mut models = self.api_policy.models.clone();
            for deployment in self
                .config
                .services
                .ocr
                .models
                .iter()
                .filter(|deployment| deployment.table_structure.is_some())
            {
                let loaded = ocr_runtime_keys_for_deployment(deployment, &self.source_candidates)
                    .into_iter();
                let mut table_loaded = false;
                for key in loaded {
                    if self.ocr_models.status(&key).await == Some(ModelResidencyStatus::Loaded) {
                        table_loaded = true;
                        break;
                    }
                }
                if table_loaded
                    && let Some(model) = models.iter_mut().find(|model| {
                        model.service == ModelService::Ocr && model.id == deployment.id
                    })
                {
                    model.capabilities = model
                        .capabilities
                        .union(ModelCapabilities::OCR_TABLE_STRUCTURE);
                }
            }
            models
        })
    }

    fn acquire_inference(&self) -> crate::application::InferenceGuardFuture<'_> {
        Box::pin(self.resources.acquire_inference())
    }

    fn try_acquire_websocket(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.resources.try_acquire_websocket()
    }

    fn try_acquire_pending_websocket(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.resources.try_acquire_pending_websocket()
    }
}

impl LlmRuntime for AppState {
    #[allow(
        clippy::too_many_lines,
        reason = "owns admission, cancellation, forwarding, and terminal resource release as one transaction"
    )]
    fn start_generation(&self, command: LlmCommand) -> LlmGenerationFuture<'_> {
        Box::pin(async move {
            if *self.cleanup_shutdown.borrow() {
                return Err(RuntimeError::ShuttingDown);
            }
            let id = ModelId::parse(&command.model).ok();
            let Some(deployment) = id.as_ref().and_then(|id| {
                self.config
                    .services
                    .llm
                    .models
                    .iter()
                    .find(|deployment| deployment.id == *id)
            }) else {
                return Ok(None);
            };
            if !self.config.services.llm.active() {
                return Ok(None);
            }
            let model = LlmRuntimeKey::new(deployment.id.clone());
            let capacity = deployment.runtime.event_queue_capacity;
            let queue_timeout = command
                .queue_timeout
                .unwrap_or(deployment.runtime.queue_timeout);
            let generation_timeout = command
                .generation_timeout
                .unwrap_or(deployment.runtime.generation_timeout);
            if command
                .options
                .max_tokens
                .is_some_and(|value| value == 0 || value > deployment.generation.max_tokens)
            {
                return Err(RuntimeError::InvalidRequest {
                    message: format!(
                        "max output tokens must be in 1..={}",
                        deployment.generation.max_tokens
                    ),
                    param: command.max_tokens_param,
                    code: "invalid_parameter",
                });
            }
            let defaults = &deployment.generation;
            let options = GenerationOptions {
                max_tokens: command.options.max_tokens.unwrap_or(defaults.max_tokens),
                temperature: command.options.temperature.unwrap_or(defaults.temperature),
                top_p: command.options.top_p.unwrap_or(defaults.top_p),
                top_k: defaults.top_k,
                min_p: defaults.min_p,
                presence_penalty: command
                    .options
                    .presence_penalty
                    .unwrap_or(defaults.presence_penalty),
                frequency_penalty: command
                    .options
                    .frequency_penalty
                    .unwrap_or(defaults.frequency_penalty),
                repeat_penalty: defaults.repeat_penalty,
                seed: command.options.seed.unwrap_or(u32::MAX),
                stop: command.options.stop,
            };
            let (events, receiver) = tokio::sync::mpsc::channel(capacity);
            let (terminal, terminal_receiver) = tokio::sync::oneshot::channel();
            let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let cancellation = Arc::new(tokio::sync::Notify::new());
            let (ready, readiness) = tokio::sync::oneshot::channel();
            let state = self.clone();
            let task_cancelled = Arc::clone(&cancelled);
            let task_cancellation = Arc::clone(&cancellation);
            tokio::spawn(async move {
                own_llm_generation(
                    state,
                    model,
                    GenerationRequest {
                        messages: command.messages,
                        options,
                    },
                    queue_timeout,
                    generation_timeout,
                    ready,
                    events,
                    terminal,
                    task_cancelled,
                    task_cancellation,
                )
                .await;
            });
            let mut pending = PendingManagedReadiness {
                cancelled: Arc::clone(&cancelled),
                cancellation: Arc::clone(&cancellation),
                committed: false,
            };
            readiness.await.unwrap_or(Err(RuntimeError::Internal(
                "LLM admission task stopped before readiness".to_string(),
            )))?;
            pending.committed = true;
            Ok(Some(ManagedGeneration::new(
                receiver,
                terminal_receiver,
                cancelled,
                cancellation,
            )))
        })
    }
}

struct PendingManagedReadiness {
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    cancellation: Arc<tokio::sync::Notify>,
    committed: bool,
}

impl Drop for PendingManagedReadiness {
    fn drop(&mut self) {
        if !self.committed
            && !self
                .cancelled
                .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            self.cancellation.notify_one();
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "owns the complete admitted generation and its readiness/cleanup protocol"
)]
async fn own_llm_generation(
    state: AppState,
    model: LlmRuntimeKey,
    request: GenerationRequest,
    queue_timeout: std::time::Duration,
    generation_timeout: std::time::Duration,
    ready: tokio::sync::oneshot::Sender<Result<(), RuntimeError>>,
    events: tokio::sync::mpsc::Sender<Result<orchion::GenerationEvent, RuntimeError>>,
    terminal_sender: tokio::sync::oneshot::Sender<Result<orchion::GenerationEvent, RuntimeError>>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    cancellation: Arc<tokio::sync::Notify>,
) {
    let admission_started = std::time::Instant::now();
    let mut shutdown = state.cleanup_shutdown.subscribe();
    if *shutdown.borrow() {
        let _ = ready.send(Err(RuntimeError::ShuttingDown));
        return;
    }
    let admission_deadline = tokio::time::Instant::now() + queue_timeout;
    let reserved = {
        let reservation = async {
            let lease = state
                .llm(model.clone())
                .await
                .map_err(|error| RuntimeError::Internal(format!("{error:#}")))?
                .ok_or_else(|| RuntimeError::Internal("configured LLM disappeared".to_string()))?;
            let reservation = lease.reserve(request).await.map_err(RuntimeError::Core)?;
            Ok::<_, RuntimeError>((lease, reservation))
        };
        tokio::pin!(reservation);
        tokio::select! {
            result = &mut reservation => result,
            () = tokio::time::sleep_until(admission_deadline) => Err(RuntimeError::Timeout("LLM admission timed out".to_string())),
            () = cancellation.notified() => Err(RuntimeError::Timeout("LLM request was cancelled while waiting for admission".to_string())),
            _ = shutdown.changed() => Err(RuntimeError::ShuttingDown),
        }
    };
    let (lease, reservation) = match reserved {
        Ok(reserved) => reserved,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let inference = tokio::select! {
        inference = state.resources.acquire_inference() => Ok(inference),
        () = tokio::time::sleep_until(admission_deadline) => Err(RuntimeError::Timeout("LLM admission timed out waiting for global inference capacity".to_string())),
        () = cancellation.notified() => Err(RuntimeError::Timeout("LLM request was cancelled before commit".to_string())),
        _ = shutdown.changed() => Err(RuntimeError::ShuttingDown),
    };
    let inference = match inference {
        Ok(inference) => inference,
        Err(error) => {
            reservation.abort();
            drop(lease);
            let _ = ready.send(Err(error));
            return;
        }
    };
    if cancelled.load(std::sync::atomic::Ordering::Acquire) || *shutdown.borrow() {
        reservation.abort();
        drop(inference);
        drop(lease);
        let error = if *shutdown.borrow() {
            RuntimeError::ShuttingDown
        } else {
            RuntimeError::Timeout("LLM request was cancelled before commit".to_string())
        };
        let _ = ready.send(Err(error));
        return;
    }
    let generation_deadline = tokio::time::Instant::now() + generation_timeout;
    let mut reservation = reservation;
    let committed = {
        let commit = reservation.commit_reserved();
        tokio::pin!(commit);
        tokio::select! {
            result = &mut commit => result.map_err(RuntimeError::Core),
            () = tokio::time::sleep_until(generation_deadline) => Err(RuntimeError::Timeout("LLM preparation timed out".to_string())),
            () = cancellation.notified() => Err(RuntimeError::Timeout("LLM request was cancelled during preparation".to_string())),
            _ = shutdown.changed() => Err(RuntimeError::ShuttingDown),
        }
    };
    let mut generation = match committed {
        Ok(generation) => generation,
        Err(first_error) => {
            reservation.cancel();
            let _ = reservation.wait_for_ack().await;
            let worker_healthy = lease.is_healthy();
            drop(inference);
            drop(lease);
            retire_unhealthy_llm(&state, model, worker_healthy).await;
            let _ = ready.send(Err(first_error));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        generation.cancel();
        let _ = generation.wait_for_ack().await;
        let worker_healthy = lease.is_healthy();
        drop(inference);
        drop(lease);
        retire_unhealthy_llm(&state, model, worker_healthy).await;
        return;
    }
    let queue_time_ms = u64::try_from(admission_started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let evaluation_started = std::time::Instant::now();

    let deadline = tokio::time::sleep_until(generation_deadline);
    tokio::pin!(deadline);
    let mut client_connected = true;
    let mut forward_content = true;
    let mut terminal = None;
    let mut terminal_error = None;
    loop {
        let event = tokio::select! {
            event = generation.next() => Some(event),
            () = cancellation.notified(), if forward_content => {
                generation.cancel();
                client_connected = false;
                forward_content = false;
                None
            }
            () = &mut deadline, if terminal_error.is_none() => {
                generation.cancel();
                forward_content = false;
                terminal_error = Some(RuntimeError::Timeout("LLM generation timed out".to_string()));
                None
            }
            _ = shutdown.changed(), if terminal_error.is_none() => {
                generation.cancel();
                forward_content = false;
                terminal_error = Some(RuntimeError::ShuttingDown);
                None
            }
        };
        let Some(event) = event else { continue };
        match event {
            Ok(Some(orchion::GenerationEvent::Finished { reason, mut usage })) => {
                usage.queue_time_ms = Some(queue_time_ms);
                usage.eval_time_ms = Some(
                    u64::try_from(evaluation_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                );
                terminal = Some(orchion::GenerationEvent::Finished { reason, usage });
                break;
            }
            Ok(Some(event @ orchion::GenerationEvent::ContentDelta(_))) if forward_content => {
                tokio::select! {
                    result = events.send(Ok(event)) => {
                        if result.is_err() {
                            generation.cancel();
                            client_connected = false;
                            forward_content = false;
                        }
                    }
                    () = cancellation.notified() => {
                        generation.cancel();
                        client_connected = false;
                        forward_content = false;
                    }
                    () = &mut deadline, if terminal_error.is_none() => {
                        generation.cancel();
                        forward_content = false;
                        terminal_error = Some(RuntimeError::Timeout("LLM generation timed out".to_string()));
                    }
                    _ = shutdown.changed(), if terminal_error.is_none() => {
                        generation.cancel();
                        forward_content = false;
                        terminal_error = Some(RuntimeError::ShuttingDown);
                    }
                }
            }
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(error) => {
                if terminal_error.is_none() {
                    terminal_error = Some(RuntimeError::Core(error));
                }
                break;
            }
        }
    }
    if let Err(error) = generation.wait_for_ack().await
        && terminal_error.is_none()
    {
        terminal_error = Some(RuntimeError::Core(error));
    }
    let worker_healthy = lease.is_healthy();
    drop(inference);
    drop(lease);
    retire_unhealthy_llm(&state, model, worker_healthy).await;
    drop(events);
    if client_connected {
        let delivery = terminal_error.map_or_else(|| terminal.map(Ok), |error| Some(Err(error)));
        if let Some(delivery) = delivery {
            let _ = terminal_sender.send(delivery);
        }
    }
}

async fn retire_unhealthy_llm(state: &AppState, model: LlmRuntimeKey, worker_healthy: bool) {
    if !worker_healthy && let Err(error) = state.llm_models.unload(model).await {
        tracing::error!(%error, "failed to retire unhealthy LLM worker");
    }
}

impl ModelLifecycleRuntime for AppState {
    fn model_statuses(&self) -> ModelStatusesFuture<'_> {
        Box::pin(async move {
            let mut statuses = Vec::new();
            let source_candidates = &self.source_candidates;
            if self.config.services.asr.enabled {
                for deployment in &self.config.services.asr.models {
                    let model = &deployment.runtime;
                    if let Some(status) = self.asr_models.status(model).await {
                        statuses.push(model_status(model.as_str(), ModelService::Asr, status));
                    }
                }
            }
            if self.config.services.tts.enabled {
                for deployment in &self.config.services.tts.models {
                    let model = &deployment.runtime;
                    if let Some(status) = self.tts_models.status(model).await {
                        statuses.push(model_status(model.as_str(), ModelService::Tts, status));
                    }
                }
            }
            if self.config.services.ocr.active() {
                for deployment in &self.config.services.ocr.models {
                    let id = &deployment.id;
                    if let Some(status) =
                        ocr_runtime_status(&self.ocr_models, deployment, source_candidates).await
                    {
                        statuses.push(model_status(id.as_str(), ModelService::Ocr, status));
                    }
                }
            }
            if self.config.services.ocr_vl.active() {
                for deployment in &self.config.services.ocr_vl.models {
                    let id = &deployment.id;
                    if let Some(status) =
                        ocr_runtime_status(&self.ocr_vl_models, deployment, source_candidates).await
                    {
                        statuses.push(model_status(id.as_str(), ModelService::OcrVl, status));
                    }
                }
            }
            if self.config.services.llm.active() {
                for deployment in &self.config.services.llm.models {
                    let model = LlmRuntimeKey::new(deployment.id.clone());
                    if let Some(status) = self.llm_models.status(&model).await {
                        statuses.push(model_status(
                            deployment.id.as_str(),
                            ModelService::Llm,
                            status,
                        ));
                    }
                }
            }
            statuses
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "dispatches lifecycle control across the configured model services"
    )]
    fn load_model(&self, selector: ModelSelector) -> ModelControlFuture<'_> {
        Box::pin(async move {
            let status = match selector.service {
                ModelService::Asr => {
                    let Ok(model) = AsrModel::parse(&selector.model) else {
                        return Ok(None);
                    };
                    let Some(lease) = self
                        .asr(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    self.asr_models.status(&model).await
                }
                ModelService::Tts => {
                    let Ok(model) = TtsModel::parse(&selector.model) else {
                        return Ok(None);
                    };
                    let Some(lease) = self
                        .tts(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    self.tts_models.status(&model).await
                }
                ModelService::Ocr => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr.model_ids(),
                        OcrModelKind::TraditionalOcr,
                    ) else {
                        return Ok(None);
                    };
                    let layout = self
                        .config
                        .services
                        .ocr
                        .models
                        .iter()
                        .find(|deployment| deployment.id == *model.id())
                        .and_then(|deployment| deployment.layout_runtime.clone());
                    let Some(lease) = self
                        .ocr(model.clone(), layout)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    let deployment = self
                        .config
                        .services
                        .ocr
                        .models
                        .iter()
                        .find(|deployment| deployment.id == *model.id())
                        .expect("selected OCR model has a deployment");
                    ocr_runtime_status(&self.ocr_models, deployment, &self.source_candidates).await
                }
                ModelService::OcrVl => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr_vl.model_ids(),
                        OcrModelKind::OcrVl,
                    ) else {
                        return Ok(None);
                    };
                    let Some(lease) = self
                        .ocr_vl(model.clone(), None)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    let deployment = self
                        .config
                        .services
                        .ocr_vl
                        .models
                        .iter()
                        .find(|deployment| deployment.id == *model.id())
                        .expect("selected OCR-VL model has a deployment");
                    ocr_runtime_status(&self.ocr_vl_models, deployment, &self.source_candidates)
                        .await
                }
                ModelService::Llm => {
                    let Some(deployment) = self
                        .config
                        .services
                        .llm
                        .models
                        .iter()
                        .find(|deployment| deployment.id.as_str() == selector.model)
                    else {
                        return Ok(None);
                    };
                    let model = LlmRuntimeKey::new(deployment.id.clone());
                    let Some(lease) = self
                        .llm(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    self.llm_models.status(&model).await
                }
            };
            Ok(status.map(|status| model_status(&selector.model, selector.service, status)))
        })
    }

    fn unload_model(&self, selector: ModelSelector) -> ModelControlFuture<'_> {
        Box::pin(async move {
            let status = match selector.service {
                ModelService::Asr if self.config.services.asr.enabled => {
                    let Ok(model) = AsrModel::parse(&selector.model) else {
                        return Ok(None);
                    };
                    if self
                        .asr_models
                        .unload(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                        .is_none()
                    {
                        return Ok(None);
                    }
                    Some(ModelResidencyStatus::Unloaded)
                }
                ModelService::Tts if self.config.services.tts.enabled => {
                    let Ok(model) = TtsModel::parse(&selector.model) else {
                        return Ok(None);
                    };
                    if self
                        .tts_models
                        .unload(model)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                        .is_none()
                    {
                        return Ok(None);
                    }
                    Some(ModelResidencyStatus::Unloaded)
                }
                ModelService::Ocr if self.config.services.ocr.active() => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr.model_ids(),
                        OcrModelKind::TraditionalOcr,
                    ) else {
                        return Ok(None);
                    };
                    let deployment = self
                        .config
                        .services
                        .ocr
                        .models
                        .iter()
                        .find(|deployment| deployment.id == *model.id())
                        .expect("selected OCR model has a deployment");
                    unload_ocr_runtimes(&self.ocr_models, deployment, &self.source_candidates)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?;
                    Some(ModelResidencyStatus::Unloaded)
                }
                ModelService::OcrVl if self.config.services.ocr_vl.active() => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr_vl.model_ids(),
                        OcrModelKind::OcrVl,
                    ) else {
                        return Ok(None);
                    };
                    let deployment = self
                        .config
                        .services
                        .ocr_vl
                        .models
                        .iter()
                        .find(|deployment| deployment.id == *model.id())
                        .expect("selected OCR-VL model has a deployment");
                    unload_ocr_runtimes(&self.ocr_vl_models, deployment, &self.source_candidates)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?;
                    Some(ModelResidencyStatus::Unloaded)
                }
                ModelService::Llm if self.config.services.llm.active() => {
                    let Some(deployment) = self
                        .config
                        .services
                        .llm
                        .models
                        .iter()
                        .find(|deployment| deployment.id.as_str() == selector.model)
                    else {
                        return Ok(None);
                    };
                    if self
                        .llm_models
                        .unload(LlmRuntimeKey::new(deployment.id.clone()))
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                        .is_none()
                    {
                        return Ok(None);
                    }
                    Some(ModelResidencyStatus::Unloaded)
                }
                _ => return Ok(None),
            };
            Ok(status.map(|status| model_status(&selector.model, selector.service, status)))
        })
    }
}

fn selected_ocr_model(value: &str, available: &[ModelId], kind: OcrModelKind) -> Option<OcrModel> {
    let id = ModelId::parse(value).ok()?;
    available.contains(&id).then(|| OcrModel::new(id, kind))
}

fn model_status(id: &str, service: ModelService, status: ModelResidencyStatus) -> ModelStatus {
    let status = match status {
        ModelResidencyStatus::Unloaded => ModelResidency::Unloaded,
        ModelResidencyStatus::Loading => ModelResidency::Loading,
        ModelResidencyStatus::Loaded => ModelResidency::Loaded,
        ModelResidencyStatus::Unloading => ModelResidency::Unloading,
    };
    ModelStatus::new(id, service, status)
}

fn model_lifecycle_error(error: &anyhow::Error) -> RuntimeError {
    RuntimeError::Internal(format!("{error:#}"))
}

impl TranscriptionRuntime for AppState {
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy {
            models: self.config.services.asr.runtime_models(),
            max_audio_duration: self.config.services.asr.max_audio_duration,
        }
    }

    fn transcribe(
        &self,
        model: AsrModel,
        samples: Vec<f32>,
        sample_rate: u32,
        options: orchion::AsrOptions,
        with_segments: bool,
    ) -> TranscriptionFuture<'_> {
        Box::pin(async move {
            let Some(asr) = AppState::asr(self, model)
                .await
                .map_err(|error| RuntimeError::Internal(format!("{error:#}")))?
            else {
                return Ok(None);
            };
            let inference = self.resources.inference_limiter();
            let transcript = asr
                .run_with_inference(inference, move |asr| async move {
                    if with_segments {
                        asr.transcribe_samples_with_segments(&samples, sample_rate, options)
                            .await
                    } else {
                        asr.transcribe_samples_with(&samples, sample_rate, options)
                            .await
                    }
                })
                .await
                .map_err(|error| {
                    RuntimeError::Internal(format!("ASR operation task failed: {error:#}"))
                })?
                .map_err(RuntimeError::Core)?;
            Ok(Some(transcript))
        })
    }
}

impl StreamingTranscriptionRuntime for AppState {
    fn lease_streaming_model(&self, model: AsrModel) -> StreamingModelFuture<'_> {
        Box::pin(async move {
            let Some(asr) = AppState::asr(self, model)
                .await
                .map_err(|error| RuntimeError::Internal(format!("{error:#}")))?
            else {
                return Ok(None);
            };
            Ok(Some(LeasedAsrModel::new(
                asr,
                self.resources.inference_limiter(),
            )))
        })
    }
}

impl SpeechRuntime for AppState {
    fn speech_policy(&self) -> SpeechPolicy {
        SpeechPolicy {
            default_format: self
                .config
                .services
                .tts
                .format
                .parse()
                .expect("validated TTS response format"),
            max_length: self.config.services.tts.max_length,
            max_reference_audio_duration: self.config.services.tts.max_reference_audio_duration,
        }
    }

    fn synthesize_speech(
        &self,
        model: TtsModel,
        input: String,
        voice: orchion::TtsVoice,
        options: orchion::TtsOptions,
    ) -> SpeechRuntimeFuture<'_> {
        Box::pin(async move {
            let Some(tts) = AppState::tts(self, model)
                .await
                .map_err(|error| RuntimeError::Internal(format!("{error:#}")))?
            else {
                return Ok(None);
            };
            let inference = self.resources.inference_limiter();
            let audio = tts
                .run_with_inference(inference, move |tts| async move {
                    tts.synthesize_with(input, voice, options).await
                })
                .await
                .map_err(|error| {
                    RuntimeError::Internal(format!("TTS operation task failed: {error:#}"))
                })?
                .map_err(RuntimeError::Core)?;
            Ok(Some(audio))
        })
    }
}

impl OcrRuntime for AppState {
    fn ocr_policy(&self) -> OcrPolicy {
        let ocr = &self.config.services.ocr;
        let ocr_vl = &self.config.services.ocr_vl;
        OcrPolicy {
            ocr: OcrServicePolicy {
                active: ocr.active(),
                models: ocr.model_ids(),
                model_layouts: ocr.model_layouts(),
                format: ocr.format,
                max_pixels: ocr.max_pixels,
            },
            ocr_vl: OcrVlServicePolicy {
                active: ocr_vl.active(),
                models: ocr_vl.model_ids(),
                model_layouts: ocr_vl.model_layouts(),
                format: ocr_vl.format,
                max_tokens: ocr_vl.max_tokens,
                max_pixels: ocr_vl.max_pixels,
            },
        }
    }

    fn recognize(
        &self,
        choice: OcrServiceChoice,
        image_path: PathBuf,
        options: orchion::OcrOptions,
        limits: orchion::OcrLimits,
    ) -> OcrFuture<'_> {
        Box::pin(async move {
            let layout = options
                .layout_model
                .clone()
                .map(|model| OcrModel::new(model, OcrModelKind::Layout));
            let ocr = match choice {
                OcrServiceChoice::Ocr { model } => {
                    AppState::ocr(
                        self,
                        OcrModel::new(model, OcrModelKind::TraditionalOcr),
                        layout,
                    )
                    .await
                }
                OcrServiceChoice::OcrVl { model } => {
                    AppState::ocr_vl(self, OcrModel::new(model, OcrModelKind::OcrVl), layout).await
                }
            }
            .map_err(|error| RuntimeError::Internal(format!("{error:#}")))?;
            let Some(ocr) = ocr else {
                return Ok(None);
            };
            let inference = self.resources.inference_limiter();
            let result = ocr
                .run_with_inference(inference, move |ocr| async move {
                    ocr.recognize_file_with_limits(image_path, options, limits)
                        .await
                })
                .await
                .map_err(|error| {
                    RuntimeError::Internal(format!("OCR operation task failed: {error:#}"))
                })?
                .map_err(RuntimeError::Core)?;
            Ok(Some(result))
        })
    }
}

struct ModelProvisioners {
    asr: Arc<dyn ModelProvisioner<AsrModel>>,
    tts: Arc<dyn ModelProvisioner<TtsModel>>,
    ocr: Arc<dyn ModelProvisioner<OcrModel>>,
    layout: Arc<dyn ModelProvisioner<DeploymentLayoutModel>>,
    ocr_deployments: Arc<dyn OcrDeploymentProvisioner>,
}

struct LayoutModelProvisioner {
    inner: Arc<dyn ModelProvisioner<OcrModel>>,
}

impl ModelProvisioner<DeploymentLayoutModel> for LayoutModelProvisioner {
    fn provision(
        &self,
        model: DeploymentLayoutModel,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        self.inner.provision(model.model, provisioning, models_dir)
    }
}

impl ModelProvisioners {
    fn new<P>(provisioner: Arc<P>) -> Self
    where
        P: ModelProvisioner<AsrModel>
            + ModelProvisioner<TtsModel>
            + ModelProvisioner<OcrModel>
            + OcrDeploymentProvisioner
            + 'static,
    {
        let asr: Arc<dyn ModelProvisioner<AsrModel>> = provisioner.clone();
        let tts: Arc<dyn ModelProvisioner<TtsModel>> = provisioner.clone();
        let ocr: Arc<dyn ModelProvisioner<OcrModel>> = provisioner.clone();
        let layout: Arc<dyn ModelProvisioner<DeploymentLayoutModel>> =
            Arc::new(LayoutModelProvisioner {
                inner: Arc::clone(&ocr),
            });
        let ocr_deployments: Arc<dyn OcrDeploymentProvisioner> = provisioner;
        Self {
            asr,
            tts,
            ocr,
            layout,
            ocr_deployments,
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "selects between cache constructors while preserving their independent inputs"
)]
fn build_model_cache<M, E>(
    cache_id: &'static str,
    models: Vec<M>,
    provisioning: HashMap<M, ModelProvisioning>,
    idle_timeout: std::time::Duration,
    max_loaded: usize,
    dir: PathBuf,
    provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
    residency: ResidencyDomain,
) -> ModelCache<M, E>
where
    M: ModelCacheKey + std::hash::Hash,
    E: Clone,
{
    if provisioning.is_empty() {
        ModelCache::new_in_domain(
            cache_id,
            models,
            idle_timeout,
            max_loaded,
            dir,
            provisioner,
            residency,
        )
    } else {
        ModelCache::new_in_domain_with_provisioning(
            cache_id,
            models,
            provisioning,
            idle_timeout,
            max_loaded,
            dir,
            provisioner,
            residency,
        )
    }
}

fn spawn_model_provision<M, E>(
    tasks: &mut JoinSet<anyhow::Result<ProvisionedModelKind>>,
    semaphore: Arc<Semaphore>,
    cache: ModelCache<M, E>,
    model: M,
    kind: ProvisionedModelKind,
    context: &'static str,
) where
    M: ModelCacheKey + std::hash::Hash,
    E: Clone + Send + 'static,
{
    tasks.spawn(async move {
        let _permit = semaphore
            .acquire_owned()
            .await
            .expect("model provision semaphore must remain open");
        if cache
            .ensure_provisioned(model)
            .await
            .with_context(|| context)?
            .is_none()
        {
            anyhow::bail!("{context}: model is not in the service allowlist");
        }
        Ok(kind)
    });
}

#[derive(Clone, Copy)]
enum ProvisionedModelKind {
    Asr,
    Tts,
    Ocr,
    OcrVl,
    Layout,
}

#[derive(Default)]
struct ProvisionedModelCounts {
    asr: usize,
    tts: usize,
    ocr: usize,
    ocr_vl: usize,
    layout: usize,
    llm: usize,
}

impl ProvisionedModelCounts {
    fn record(&mut self, kind: ProvisionedModelKind) {
        match kind {
            ProvisionedModelKind::Asr => self.asr += 1,
            ProvisionedModelKind::Tts => self.tts += 1,
            ProvisionedModelKind::Ocr => self.ocr += 1,
            ProvisionedModelKind::OcrVl => self.ocr_vl += 1,
            ProvisionedModelKind::Layout => self.layout += 1,
        }
    }
}

struct ResolvedOcrModels {
    assets: Vec<OcrModel>,
    asset_locators: HashMap<OcrModel, ModelProvisioning>,
    ocr: Vec<OcrRuntimeKey>,
    ocr_vl: Vec<OcrRuntimeKey>,
    layout: Vec<DeploymentLayoutModel>,
    layout_locators: HashMap<DeploymentLayoutModel, ModelProvisioning>,
    deployment_plans: HashMap<OcrRuntimeKey, DeploymentArtifactPlan>,
}

#[allow(
    clippy::too_many_lines,
    reason = "projects one validated OCR deployment set into asset, runtime, and layout caches"
)]
fn resolve_configured_ocr_models(
    config: &ServerConfig,
    source_candidates: &[DownloadSource],
) -> ResolvedOcrModels {
    let ocr_models = if config.services.ocr.active() {
        config
            .services
            .ocr
            .models
            .iter()
            .filter(|deployment| deployment.table_structure.is_none())
            .map(|model| model.runtime.clone())
            .collect()
    } else {
        Vec::new()
    };
    let ocr_vl_models = if config.services.ocr_vl.active() {
        config
            .services
            .ocr_vl
            .models
            .iter()
            .map(|model| model.runtime.clone())
            .collect()
    } else {
        Vec::new()
    };
    let mut layout = Vec::new();
    for deployment in config
        .services
        .ocr
        .models
        .iter()
        .filter(|_| config.services.ocr.active())
        .filter(|deployment| deployment.table_structure.is_none())
        .chain(
            config
                .services
                .ocr_vl
                .models
                .iter()
                .filter(|_| config.services.ocr_vl.active()),
        )
    {
        if let Some(model) = &deployment.layout_runtime {
            let model = DeploymentLayoutModel::new(deployment.id.clone(), model.clone());
            if !layout.contains(&model) {
                layout.push(model);
            }
        }
    }
    let assets = ocr_models.iter().chain(&ocr_vl_models).cloned().collect();
    let asset_locators = config
        .services
        .ocr
        .models
        .iter()
        .chain(&config.services.ocr_vl.models)
        .filter(|deployment| deployment.table_structure.is_none())
        .filter(|deployment| match deployment.runtime.kind() {
            OcrModelKind::TraditionalOcr => config.services.ocr.active(),
            OcrModelKind::OcrVl => config.services.ocr_vl.active(),
            OcrModelKind::Layout => false,
        })
        .map(|deployment| {
            let source_intent = ocr_deployment_source_intent(deployment, source_candidates);
            let source_plan = ocr_deployment_source_plan(deployment, source_candidates);
            (
                deployment.runtime.clone(),
                provisioning_for_url(&deployment.model, source_intent, source_plan),
            )
        })
        .collect();
    let layout_locators = config
        .services
        .ocr
        .models
        .iter()
        .filter(|_| config.services.ocr.active())
        .filter(|deployment| deployment.table_structure.is_none())
        .chain(
            config
                .services
                .ocr_vl
                .models
                .iter()
                .filter(|_| config.services.ocr_vl.active()),
        )
        .filter_map(|deployment| {
            let runtime = DeploymentLayoutModel::new(
                deployment.id.clone(),
                deployment.layout_runtime.clone()?,
            );
            let url = deployment.layout_model.as_ref()?;
            let source_intent = format!(
                "deployment={}|layout={url}{}",
                deployment.id,
                neutral_policy_suffix(
                    (url.source() == ModelUrlSource::Neutral).then_some(source_candidates)
                )
            );
            let source_plan = ocr_deployment_source_plan(deployment, source_candidates);
            Some((
                runtime,
                provisioning_for_url(url, source_intent, source_plan),
            ))
        })
        .collect();
    let ocr = if config.services.ocr.active() {
        resolve_deployment_runtime_keys(&config.services.ocr.models, source_candidates)
    } else {
        Vec::new()
    };
    let ocr_vl = if config.services.ocr_vl.active() {
        resolve_deployment_runtime_keys(&config.services.ocr_vl.models, source_candidates)
    } else {
        Vec::new()
    };
    let deployment_plans = config
        .services
        .ocr
        .models
        .iter()
        .filter(|_| config.services.ocr.active())
        .filter(|deployment| deployment.table_structure.is_some())
        .flat_map(|deployment| {
            let plan = ocr_deployment_artifact_plan(deployment, source_candidates);
            ocr_runtime_keys_for_deployment(deployment, source_candidates)
                .into_iter()
                .map(move |key| (key, plan.clone()))
        })
        .collect();
    ResolvedOcrModels {
        assets,
        asset_locators,
        ocr,
        ocr_vl,
        layout,
        layout_locators,
        deployment_plans,
    }
}

fn ocr_deployment_artifact_plan(
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> DeploymentArtifactPlan {
    let primary = deployment
        .runtime
        .known()
        .expect("validated OCR deployment has a known runtime recipe");
    let mut artifacts = primary
        .download_assets()
        .iter()
        .map(|artifact| DeploymentArtifactRequest {
            role: match artifact.role {
                OcrModelAssetRole::Detector => ArtifactRole::OcrDetector,
                OcrModelAssetRole::Recognizer => ArtifactRole::OcrRecognizer,
                OcrModelAssetRole::Dictionary => ArtifactRole::OcrDictionary,
                OcrModelAssetRole::Layout => ArtifactRole::OcrLayout,
            },
            source: deployment_artifact_source(&deployment.model),
            repository: Some(artifact.repo.to_string()),
            files: vec![artifact.file.to_string()],
            required_source: matches!(artifact.kind, OcrModelAssetKind::ModelScopeFile { .. })
                .then_some(DownloadSource::ModelScope),
        })
        .collect::<Vec<_>>();
    if let (Some(layout), Some(layout_runtime)) = (
        deployment.layout_model.as_ref(),
        deployment.layout_runtime.as_ref(),
    ) {
        artifacts.extend(deployment_artifact_requests(
            ArtifactRole::OcrLayout,
            layout_runtime,
            layout,
        ));
    }
    if let Some(table) = &deployment.table_structure {
        artifacts.push(exact_deployment_artifact(
            ArtifactRole::OcrTableStructureModel,
            &table.model,
        ));
        artifacts.push(exact_deployment_artifact(
            ArtifactRole::OcrTableStructureDictionary,
            &table.dictionary,
        ));
    }
    DeploymentArtifactPlan {
        deployment_id: deployment.id.clone(),
        category: deployment.runtime.category(),
        source_intent: ocr_deployment_source_intent(deployment, source_candidates),
        artifacts,
        neutral_candidates: source_candidates.to_vec(),
    }
}

fn llm_deployment_artifact_plan(
    deployment: &LlmModelDeployment,
    source_candidates: &[DownloadSource],
) -> DeploymentArtifactPlan {
    let mut artifacts = vec![exact_deployment_artifact(
        ArtifactRole::LlmModel,
        &deployment.model,
    )];
    if let Some(mmproj) = &deployment.mmproj_model {
        artifacts.push(exact_deployment_artifact(ArtifactRole::LlmMmproj, mmproj));
    }
    DeploymentArtifactPlan {
        deployment_id: deployment.id.clone(),
        category: ModelCategory::Llm,
        source_intent: llm_deployment_source_intent(deployment, source_candidates),
        artifacts,
        neutral_candidates: source_candidates.to_vec(),
    }
}

fn llm_deployment_source_intent(
    deployment: &LlmModelDeployment,
    source_candidates: &[DownloadSource],
) -> String {
    let neutral = deployment.model.source() == ModelUrlSource::Neutral
        || deployment
            .mmproj_model
            .as_ref()
            .is_some_and(|model| model.source() == ModelUrlSource::Neutral);
    format!(
        "id={}|model={}|mmproj={}|runtime={:?}|template={:?}|generation={:?}{}",
        deployment.id,
        deployment.model,
        deployment
            .mmproj_model
            .as_ref()
            .map_or("none", orchion::ModelUrl::as_str),
        deployment.runtime,
        deployment.chat_template,
        deployment.generation,
        neutral_policy_suffix(neutral.then_some(source_candidates)),
    )
}

fn deployment_artifact_requests(
    role: ArtifactRole,
    runtime: &OcrModel,
    url: &orchion::ModelUrl,
) -> Vec<DeploymentArtifactRequest> {
    if url.source() == ModelUrlSource::File {
        return vec![exact_deployment_artifact(role, url)];
    }
    ModelDownloader::model_artifact_plan(runtime, url)
        .expect("validated OCR auxiliary has an artifact recipe")
        .into_iter()
        .map(|artifact| DeploymentArtifactRequest {
            role,
            source: deployment_artifact_source(url),
            repository: Some(artifact.repository),
            files: artifact
                .files
                .expect("validated OCR auxiliary recipe has exact files"),
            required_source: artifact.required_source,
        })
        .collect()
}

fn exact_deployment_artifact(
    role: ArtifactRole,
    url: &orchion::ModelUrl,
) -> DeploymentArtifactRequest {
    if url.source() == ModelUrlSource::File {
        let path = PathBuf::from(url.path().expect("validated file URL has a path"));
        return DeploymentArtifactRequest {
            role,
            source: DeploymentArtifactSource::File(path.clone()),
            repository: None,
            files: vec![
                path.file_name()
                    .expect("validated file URL has a file name")
                    .to_string_lossy()
                    .to_string(),
            ],
            required_source: None,
        };
    }
    DeploymentArtifactRequest {
        role,
        source: deployment_artifact_source(url),
        repository: Some(format!(
            "{}/{}",
            url.owner().expect("validated hub URL has an owner"),
            url.repository()
                .expect("validated hub URL has a repository")
        )),
        files: vec![
            url.path()
                .expect("validated auxiliary URL has an exact path")
                .to_string(),
        ],
        required_source: None,
    }
}

fn deployment_artifact_source(url: &orchion::ModelUrl) -> DeploymentArtifactSource {
    match url.source() {
        ModelUrlSource::Neutral => DeploymentArtifactSource::Neutral,
        ModelUrlSource::HuggingFace => DeploymentArtifactSource::HuggingFace,
        ModelUrlSource::ModelScope => DeploymentArtifactSource::ModelScope,
        ModelUrlSource::File => DeploymentArtifactSource::File(PathBuf::from(
            url.path().expect("validated file URL has a path"),
        )),
    }
}

fn deployment_provisioning<M>(
    deployments: &[crate::settings::ModelDeployment<M>],
    source_candidates: &[DownloadSource],
) -> HashMap<M, ModelProvisioning>
where
    M: ModelSpec + std::hash::Hash,
{
    deployments
        .iter()
        .map(|deployment| {
            let source_intent = format!(
                "id={}|model={}{}",
                deployment.id,
                deployment.model,
                neutral_policy_suffix(
                    (deployment.model.source() == ModelUrlSource::Neutral)
                        .then_some(source_candidates)
                )
            );
            let artifacts = if deployment.model.source() == ModelUrlSource::Neutral {
                ModelDownloader::model_artifact_plan(&deployment.runtime, &deployment.model)
                    .expect("validated speech deployment has a compatible artifact plan")
            } else {
                Vec::new()
            };
            let source_plan =
                deployment_source_plan(source_intent.clone(), artifacts, source_candidates);
            (
                deployment.runtime.clone(),
                provisioning_for_url(&deployment.model, source_intent, source_plan),
            )
        })
        .collect()
}

fn validate_prepared_model_paths(
    config: &ServerConfig,
    source_candidates: &[DownloadSource],
    ocr: &ResolvedOcrModels,
) -> anyhow::Result<()> {
    for (model, provisioning) in
        deployment_provisioning(&config.services.asr.models, source_candidates)
    {
        resolve_prepared_provisioning_path(&model, &provisioning, &config.models.dir)?;
    }
    for (model, provisioning) in
        deployment_provisioning(&config.services.tts.models, source_candidates)
    {
        resolve_prepared_provisioning_path(&model, &provisioning, &config.models.dir)?;
    }
    for (model, provisioning) in &ocr.asset_locators {
        resolve_prepared_provisioning_path(model, provisioning, &config.models.dir)?;
    }
    for (model, provisioning) in &ocr.layout_locators {
        resolve_prepared_provisioning_path(&model.model, provisioning, &config.models.dir)?;
    }
    if config.services.llm.active() {
        for deployment in &config.services.llm.models {
            let plan = llm_deployment_artifact_plan(deployment, source_candidates);
            ModelDownloader::resolve_prepared_logical_deployment(
                &deployment.id,
                ModelCategory::Llm,
                &plan,
                &config.models.dir,
            )?;
        }
    }
    Ok(())
}

fn resolve_prepared_provisioning_path<M: ModelSpec>(
    model: &M,
    provisioning: &ModelProvisioning,
    models_dir: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    match &provisioning.source_plan {
        Some(plan) => Ok(ModelDownloader::resolve_prepared_model_url_path_with_plan(
            model,
            &provisioning.model_url,
            &provisioning.source_intent,
            plan,
            models_dir,
        )?),
        None => Ok(ModelDownloader::resolve_prepared_model_url_path(
            model,
            &provisioning.model_url,
            &provisioning.source_intent,
            models_dir,
        )?),
    }
}

fn provisioning_for_url(
    url: &orchion::ModelUrl,
    source_intent: String,
    source_plan: Option<DeploymentSourcePlan>,
) -> ModelProvisioning {
    ModelProvisioning {
        model_url: url.clone(),
        source_intent,
        source_plan: (url.source() == ModelUrlSource::Neutral)
            .then_some(source_plan)
            .flatten(),
    }
}

fn ocr_deployment_source_intent(
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> String {
    let has_neutral = deployment.model.source() == ModelUrlSource::Neutral
        || deployment
            .layout_model
            .as_ref()
            .is_some_and(|url| url.source() == ModelUrlSource::Neutral)
        || deployment.table_structure.as_ref().is_some_and(|table| {
            table.model.source() == ModelUrlSource::Neutral
                || table.dictionary.source() == ModelUrlSource::Neutral
        });
    let table = deployment.table_structure.as_ref().map_or_else(
        || "none".to_string(),
        |table| {
            format!(
                "model={},dictionary={},type={},score={:08x},max_length={}",
                table.model,
                table.dictionary,
                table.table_type.as_str(),
                table.score_threshold.to_bits(),
                table.max_structure_length
            )
        },
    );
    format!(
        "id={}|model={}|layout={}|table={table}{}",
        deployment.id,
        deployment.model,
        deployment
            .layout_model
            .as_ref()
            .map_or("none", orchion::ModelUrl::as_str),
        neutral_policy_suffix(has_neutral.then_some(source_candidates))
    )
}

fn neutral_policy_suffix(candidates: Option<&[DownloadSource]>) -> String {
    candidates.map_or_else(String::new, |candidates| {
        let policy = candidates
            .iter()
            .map(|source| match source {
                DownloadSource::HuggingFace => "huggingface",
                DownloadSource::ModelScope => "modelscope",
                DownloadSource::Auto => "auto",
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("|neutral-policy={policy}")
    })
}

fn deployment_source_plan(
    key: String,
    artifacts: Vec<ArtifactRequest>,
    source_candidates: &[DownloadSource],
) -> Option<DeploymentSourcePlan> {
    (!artifacts.is_empty() && !source_candidates.is_empty()).then(|| DeploymentSourcePlan {
        key,
        artifacts,
        candidates: source_candidates.to_vec(),
    })
}

fn ocr_deployment_source_plan(
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> Option<DeploymentSourcePlan> {
    let mut artifacts = Vec::new();
    if deployment.model.source() == ModelUrlSource::Neutral {
        artifacts.extend(
            ModelDownloader::model_artifact_plan(&deployment.runtime, &deployment.model)
                .expect("validated OCR deployment has a compatible primary artifact plan"),
        );
    }
    if let (Some(runtime), Some(url)) = (
        deployment.layout_runtime.as_ref(),
        deployment.layout_model.as_ref(),
    ) && url.source() == ModelUrlSource::Neutral
    {
        artifacts.extend(
            ModelDownloader::model_artifact_plan(runtime, url)
                .expect("validated OCR deployment has a compatible layout artifact plan"),
        );
    }
    deployment_source_plan(
        format!("ocr-deployment={}", deployment.id),
        artifacts,
        source_candidates,
    )
}

fn resolve_config_source_candidates(config: &ServerConfig) -> anyhow::Result<Vec<DownloadSource>> {
    if !active_neutral_locators(config) {
        return Ok(Vec::new());
    }
    let environment = match std::env::var("ORCHION_MODEL_SOURCE") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ORCHION_MODEL_SOURCE contains non-Unicode data")
        }
    };
    resolve_config_source_candidates_with_env(config, environment.as_deref())
}

fn resolve_config_source_candidates_with_env(
    config: &ServerConfig,
    environment: Option<&str>,
) -> anyhow::Result<Vec<DownloadSource>> {
    if !active_neutral_locators(config) {
        return Ok(Vec::new());
    }
    match config.models.source {
        crate::settings::ModelSource::HuggingFace => Ok(vec![DownloadSource::HuggingFace]),
        crate::settings::ModelSource::ModelScope => Ok(vec![DownloadSource::ModelScope]),
        crate::settings::ModelSource::Auto => match environment {
            Some(value) => match value.trim().to_ascii_lowercase().as_str() {
                "hf" | "huggingface" => Ok(vec![DownloadSource::HuggingFace]),
                "ms" | "modelscope" => Ok(vec![DownloadSource::ModelScope]),
                "auto" => Ok(vec![
                    DownloadSource::HuggingFace,
                    DownloadSource::ModelScope,
                ]),
                _ => anyhow::bail!("invalid ORCHION_MODEL_SOURCE `{value}`"),
            },
            None => Ok(vec![
                DownloadSource::HuggingFace,
                DownloadSource::ModelScope,
            ]),
        },
    }
}

fn active_neutral_locators(config: &ServerConfig) -> bool {
    config.services.asr.enabled
        && config
            .services
            .asr
            .models
            .iter()
            .any(|deployment| deployment.model.source() == ModelUrlSource::Neutral)
        || config.services.tts.enabled
            && config
                .services
                .tts
                .models
                .iter()
                .any(|deployment| deployment.model.source() == ModelUrlSource::Neutral)
        || config.services.ocr.active()
            && config.services.ocr.models.iter().any(|deployment| {
                deployment.model.source() == ModelUrlSource::Neutral
                    || deployment
                        .layout_model
                        .as_ref()
                        .is_some_and(|url| url.source() == ModelUrlSource::Neutral)
                    || deployment.table_structure.as_ref().is_some_and(|table| {
                        table.model.source() == ModelUrlSource::Neutral
                            || table.dictionary.source() == ModelUrlSource::Neutral
                    })
            })
        || config.services.ocr_vl.active()
            && config.services.ocr_vl.models.iter().any(|deployment| {
                deployment.model.source() == ModelUrlSource::Neutral
                    || deployment
                        .layout_model
                        .as_ref()
                        .is_some_and(|url| url.source() == ModelUrlSource::Neutral)
            })
        || config.services.llm.active()
            && config.services.llm.models.iter().any(|deployment| {
                deployment.model.source() == ModelUrlSource::Neutral
                    || deployment
                        .mmproj_model
                        .as_ref()
                        .is_some_and(|url| url.source() == ModelUrlSource::Neutral)
            })
}

fn validate_runtime_factory(
    config: &ServerConfig,
    runtime_factory: &dyn ModelRuntimeFactory,
) -> anyhow::Result<()> {
    if config.services.asr.enabled {
        for deployment in &config.services.asr.models {
            let model = &deployment.runtime;
            anyhow::ensure!(
                runtime_factory.supports_asr(model),
                "runtime factory does not support configured ASR model `{model}`"
            );
        }
    }
    if config.services.tts.enabled {
        for deployment in &config.services.tts.models {
            let model = &deployment.runtime;
            anyhow::ensure!(
                runtime_factory.supports_tts(model),
                "runtime factory does not support configured TTS model `{model}`"
            );
        }
    }

    let resolved = resolve_configured_ocr_models(config, &[DownloadSource::HuggingFace]);
    for model in resolved.assets {
        anyhow::ensure!(
            runtime_factory.supports_ocr(&model),
            "runtime factory does not support configured {} model `{model}`",
            match model.kind() {
                OcrModelKind::TraditionalOcr => "OCR",
                OcrModelKind::OcrVl => "OCR-VL",
                OcrModelKind::Layout => "OCR layout",
            }
        );
    }
    for layout in resolved.layout {
        let model = layout.model;
        anyhow::ensure!(
            runtime_factory.supports_ocr(&model),
            "runtime factory does not support configured OCR layout model `{model}`"
        );
    }
    Ok(())
}

fn initialize_configured_llm_backend(
    config: &ServerConfig,
) -> anyhow::Result<Option<LlmBackendGuard>> {
    config
        .services
        .llm
        .active()
        .then(initialize_llm_backend)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn api_policy(config: &ServerConfig) -> ApiPolicy {
    let mut models = Vec::new();
    if config.services.asr.enabled {
        models.extend(config.services.asr.models.iter().map(|deployment| {
            ApiModel {
                id: deployment.id.clone(),
                name: deployment.name.clone(),
                service: ModelService::Asr,
                capabilities: deployment
                    .runtime
                    .descriptor()
                    .map_or(ModelCapabilities::ASR_TRANSCRIPTION, |descriptor| {
                        descriptor.capabilities
                    }),
            }
        }));
    }
    if config.services.tts.enabled {
        models.extend(config.services.tts.models.iter().map(|deployment| {
            ApiModel {
                id: deployment.id.clone(),
                name: deployment.name.clone(),
                service: ModelService::Tts,
                capabilities: deployment
                    .runtime
                    .descriptor()
                    .map_or(ModelCapabilities::NONE, |descriptor| {
                        descriptor.capabilities
                    }),
            }
        }));
    }
    if config.services.ocr.active() {
        models.extend(
            config
                .services
                .ocr
                .models
                .iter()
                .map(|deployment| ocr_api_model(deployment, ModelService::Ocr)),
        );
    }
    if config.services.ocr_vl.active() {
        models.extend(
            config
                .services
                .ocr_vl
                .models
                .iter()
                .map(|deployment| ocr_api_model(deployment, ModelService::OcrVl)),
        );
    }
    if config.services.llm.active() {
        models.extend(config.services.llm.models.iter().map(|deployment| {
            ApiModel {
                id: deployment.id.clone(),
                name: deployment.name.clone(),
                service: ModelService::Llm,
                capabilities: ModelCapabilities::LLM_CHAT
                    .union(ModelCapabilities::LLM_RESPONSES)
                    .union(ModelCapabilities::LLM_STREAMING),
            }
        }));
    }
    ApiPolicy {
        api_key: config.auth.api_key.clone(),
        cors_allowed_origins: config.server.cors_allowed_origins.clone(),
        max_upload_size: config.server.max_upload_size,
        max_pdf_pages: config.server.max_pdf_pages,
        max_pdf_pixels: config.server.max_pdf_pixels,
        max_pdf_output_size: config.server.max_pdf_output_size,
        max_websocket_message_size: config.server.max_websocket_message_size,
        activity: ActivityPolicy {
            enabled: config.activity.enabled,
            history_capacity: config.activity.history_capacity,
        },
        models,
        asr: config.services.asr.enabled.then(|| AsrApiPolicy {
            models: config.services.asr.runtime_models(),
            stream_target_segment: config.services.asr.stream_target_segment,
            stream_max_segment: config.services.asr.stream_max_segment,
            stream_idle_timeout: config.services.asr.stream_idle_timeout,
            stream_max_duration: config.services.asr.stream_max_duration,
            stream_chunk_size: config.services.asr.stream_chunk_size,
        }),
        tts_models: config
            .services
            .tts
            .enabled
            .then(|| config.services.tts.runtime_models()),
        ocr_enabled: config.services.ocr.active() || config.services.ocr_vl.active(),
        llm_enabled: config.services.llm.active(),
    }
}

fn ocr_api_model(
    deployment: &crate::settings::OcrModelDeployment,
    service: ModelService,
) -> ApiModel {
    let available = deployment
        .layout_runtime
        .as_ref()
        .map_or(ModelCapabilities::NONE, |_| ModelCapabilities::OCR_LAYOUT);
    let capabilities = deployment
        .runtime
        .known()
        .expect("validated OCR deployment has a registered runtime")
        .descriptor()
        .effective_capabilities(available)
        .union(available);
    ApiModel {
        id: deployment.id.clone(),
        name: deployment.name.clone(),
        service,
        capabilities,
    }
}

fn resolve_deployment_runtime_keys(
    deployments: &[crate::settings::OcrModelDeployment],
    source_candidates: &[DownloadSource],
) -> Vec<OcrRuntimeKey> {
    deployments
        .iter()
        .flat_map(|deployment| ocr_runtime_keys_for_deployment(deployment, source_candidates))
        .collect()
}

fn ocr_runtime_keys_for_deployment(
    deployment: &crate::settings::OcrModelDeployment,
    source_candidates: &[DownloadSource],
) -> Vec<OcrRuntimeKey> {
    let layout = deployment
        .layout_runtime
        .clone()
        .map(|layout| DeploymentLayoutModel::new(deployment.id.clone(), layout));
    if let Some(table) = deployment.table_structure.clone() {
        return vec![
            OcrRuntimeKey::new(deployment.runtime.clone(), layout)
                .with_table_structure(Some(table))
                .with_table_source_intent(Some(ocr_deployment_source_intent(
                    deployment,
                    source_candidates,
                ))),
        ];
    }
    std::iter::once(OcrRuntimeKey::new(deployment.runtime.clone(), None))
        .chain(layout.map(|layout| OcrRuntimeKey::new(deployment.runtime.clone(), Some(layout))))
        .collect()
}
