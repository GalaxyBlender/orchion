use crate::application::model_cache::{
    AsrModelCache, CacheTracker, GlobalModelCacheLimiter, ModelCache, ModelLease,
    ModelProvisionFuture, ModelProvisioner, ModelResidencyStatus, OcrModelCache, OcrVlModelCache,
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
    ActivityPolicy, ApiPolicy, AsrApiPolicy, OcrApiModels, RuntimeError, ServerApplication,
};
use crate::settings::ServerConfig;
use anyhow::Context;
use orchion::{
    Asr, AsrModel, DevicePreference, ModelDownloader, ModelId, ModelSpec, Ocr, OcrAssets, OcrModel,
    OcrModelKind, RuntimeProvider, Tts, TtsModel, model_descriptor,
};
use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Semaphore, watch};
use tokio::task::{JoinHandle, JoinSet};

const MAX_CONCURRENT_MODEL_PROVISIONS: usize = 2;
type LayoutModelCache = ModelCache<OcrModel, ()>;
pub type AsrRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Asr>> + Send + 'a>>;
pub type TtsRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Tts>> + Send + 'a>>;
pub type OcrRuntimeFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Ocr>> + Send + 'a>>;

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
        layout_models: Vec<(OcrModel, PathBuf)>,
        device: DevicePreference,
    ) -> OcrRuntimeFuture<'_>;
}

#[derive(Debug, Default)]
pub struct BuiltinModelRuntimeFactory;

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
        layout_models: Vec<(OcrModel, PathBuf)>,
        device: DevicePreference,
    ) -> OcrRuntimeFuture<'_> {
        Box::pin(async move {
            let known = model
                .known()
                .with_context(|| format!("unsupported built-in OCR model `{model}`"))?;
            let layout = layout_models.into_iter().find_map(|(layout_model, path)| {
                let known_layout = layout_model.known()?;
                match OcrAssets::from_cache_layout(known_layout, path, &cache_root) {
                    OcrAssets::Layout { model } => Some(model),
                    OcrAssets::Traditional { .. } | OcrAssets::VisionLanguage { .. } => None,
                }
            });
            let assets =
                OcrAssets::from_cache_layout(known, model_dir, cache_root).with_layout(layout);
            Ocr::load_with_assets_and_device(known.id(), assets, device)
                .await
                .map_err(Into::into)
        })
    }
}

impl<M: ModelSpec> ModelProvisioner<M> for ModelDownloader {
    fn provision(&self, model: M, models_dir: PathBuf) -> ModelProvisionFuture<'_> {
        Box::pin(async move {
            self.download(model, models_dir)
                .await
                .map_err(anyhow::Error::from)
        })
    }
}

#[derive(Clone)]
pub struct AppState {
    config: ServerConfig,
    api_policy: ApiPolicy,
    asr_models: AsrModelCache,
    tts_models: TtsModelCache,
    ocr_models: OcrModelCache,
    ocr_vl_models: OcrVlModelCache,
    layout_models: LayoutModelCache,
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
    use orchion::{
        AsrEngine, AsrEngineFuture, AsrOptions, AsrStreamSession, AsrStreamingOptions,
        AsrTranscript, KnownOcrModel,
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
                .ocr(KnownOcrModel::PpOcrV6Tiny.into_model())
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ocr_vl(KnownOcrModel::PaddleOcrVl16.into_model())
                .await
                .unwrap()
                .is_none()
        );
        assert!(!models_dir.exists());
    }

    #[tokio::test]
    async fn inactive_ocr_services_ignore_unknown_available_models() {
        let mut config = test_config();
        let unknown_model = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.enabled = false;
        config.services.ocr.available_models = vec![unknown_model.clone()];
        config.services.ocr_vl.enabled = false;
        config.services.ocr_vl.available_models = vec![unknown_model];

        let state = AppState::load(config).await.unwrap();

        assert!(
            state
                .ocr(KnownOcrModel::PpOcrV6Tiny.into_model())
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ocr_vl(KnownOcrModel::PaddleOcrVl16.into_model())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn custom_ocr_model_reaches_injected_runtime_factory() {
        let mut config = test_config();
        config.services.ocr.enabled = true;
        let model_id = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.available_models = vec![model_id.clone()];

        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(FailingRuntimeFactory),
        )
        .unwrap();
        let Err(error) = state
            .ocr(OcrModel::new(model_id, OcrModelKind::TraditionalOcr))
            .await
        else {
            panic!("injected OCR runtime factory should be called");
        };

        assert!(format!("{error:#}").contains("injected OCR runtime factory"));
    }

    #[tokio::test]
    async fn custom_ocr_vl_model_reaches_injected_runtime_factory() {
        let mut config = test_config();
        config.services.ocr_vl.enabled = true;
        let model_id = ModelId::parse("Acme/Experimental-OCR-VL").unwrap();
        config.services.ocr_vl.available_models = vec![model_id.clone()];

        let state = AppState::from_prepared_config_with_runtime_factory(
            config,
            Arc::new(FailingRuntimeFactory),
        )
        .unwrap();
        let Err(error) = state
            .ocr_vl(OcrModel::new(model_id, OcrModelKind::OcrVl))
            .await
        else {
            panic!("injected OCR runtime factory should be called");
        };

        assert!(format!("{error:#}").contains("injected OCR runtime factory"));
    }

    struct FailingRuntimeFactory;

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
            _layout_models: Vec<(OcrModel, PathBuf)>,
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
            _layout_models: Vec<(OcrModel, PathBuf)>,
            _device: DevicePreference,
        ) -> OcrRuntimeFuture<'_> {
            Box::pin(async { anyhow::bail!("injected OCR runtime factory") })
        }
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
        config.services.asr.available_models = vec![model];

        let Err(error) = AppState::from_prepared_config(config) else {
            panic!("builtin runtime factory should reject a foreign ASR descriptor");
        };

        assert!(
            error
                .to_string()
                .contains("does not support configured ASR model")
        );
    }

    #[test]
    fn builtin_runtime_factory_rejects_custom_ocr_models_before_serving() {
        let mut config = test_config();
        let model = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.enabled = true;
        config.services.ocr.available_models = vec![model];

        let Err(error) = AppState::from_prepared_config(config) else {
            panic!("builtin runtime factory should reject a custom OCR model");
        };

        assert!(
            error
                .to_string()
                .contains("does not support configured OCR model")
        );
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
    /// # Errors
    ///
    /// Returns an error when configuration, provisioning, or startup model loading fails.
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let provisioner = Arc::new(ModelDownloader::new(config.models.source.into()));
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
            + 'static,
    {
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
            + 'static,
    {
        validate_runtime_factory(&config, runtime_factory.as_ref())?;
        let resolved_ocr_models = resolve_configured_ocr_models(&config);
        let provisioners = ModelProvisioners::new(provisioner);
        let state = Arc::new(Self::build(
            config,
            resolved_ocr_models,
            Some(&provisioners),
            runtime_factory,
        ));
        let counts = state.ensure_startup_models().await?;
        state.spawn_idle_cleanup();
        tracing::info!(
            asr = counts.asr,
            tts = counts.tts,
            ocr = counts.ocr,
            ocr_vl = counts.ocr_vl,
            layout = counts.layout,
            "model cache ready"
        );
        Ok(state)
    }

    /// # Errors
    ///
    /// Returns an error when configured OCR model identifiers cannot be resolved.
    pub fn from_prepared_config(config: ServerConfig) -> anyhow::Result<Self> {
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
        validate_runtime_factory(&config, runtime_factory.as_ref())?;
        let resolved_ocr_models = resolve_configured_ocr_models(&config);
        Ok(Self::build(
            config,
            resolved_ocr_models,
            None,
            runtime_factory,
        ))
    }

    #[must_use]
    pub const fn config(&self) -> &ServerConfig {
        &self.config
    }

    fn build(
        config: ServerConfig,
        resolved_ocr_models: ResolvedOcrModels,
        provisioners: Option<&ModelProvisioners>,
        runtime_factory: Arc<dyn ModelRuntimeFactory>,
    ) -> Self {
        let api_policy = api_policy(&config);
        let model_residency = ResidencyDomain::new();
        let asr_models = build_model_cache(
            "asr",
            config.services.asr.available_models.clone(),
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
            config.services.tts.available_models.clone(),
            config.services.tts.idle_timeout,
            config.services.tts.max_loaded,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.tts)),
            model_residency.clone(),
        );
        let ocr_models = build_model_cache(
            "ocr",
            resolved_ocr_models.ocr,
            config.services.ocr.idle_timeout,
            config.services.ocr.max_loaded,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.ocr)),
            model_residency.clone(),
        );
        let ocr_vl_models = build_model_cache(
            "ocr-vl",
            resolved_ocr_models.ocr_vl,
            config.services.ocr_vl.idle_timeout,
            config.services.ocr_vl.max_loaded,
            config.models.dir.clone(),
            provisioners
                .as_ref()
                .map(|provisioners| Arc::clone(&provisioners.ocr)),
            model_residency.clone(),
        );
        let layout_models = build_model_cache(
            "ocr-layout",
            resolved_ocr_models.layout,
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
            api_policy,
            asr_models,
            tts_models,
            ocr_models,
            ocr_vl_models,
            layout_models,
            global_models,
            model_residency,
            resources,
            runtime_factory,
            cleanup_shutdown,
            cleanup_task: Arc::new(StdMutex::new(None)),
        }
    }

    async fn ensure_startup_models(self: &Arc<Self>) -> anyhow::Result<ProvisionedModelCounts> {
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_MODEL_PROVISIONS));
        let mut tasks = JoinSet::new();

        if self.config.services.asr.enabled {
            for model in &self.config.services.asr.available_models {
                spawn_model_provision(
                    &mut tasks,
                    Arc::clone(&semaphore),
                    self.asr_models.clone(),
                    model.clone(),
                    ProvisionedModelKind::Asr,
                    "provision configured ASR model",
                );
            }
        }
        if self.config.services.tts.enabled {
            for model in &self.config.services.tts.available_models {
                spawn_model_provision(
                    &mut tasks,
                    Arc::clone(&semaphore),
                    self.tts_models.clone(),
                    model.clone(),
                    ProvisionedModelKind::Tts,
                    "provision configured TTS model",
                );
            }
        }
        if self.config.services.ocr.active() {
            for model in &self.config.services.ocr.available_models {
                spawn_model_provision(
                    &mut tasks,
                    Arc::clone(&semaphore),
                    self.ocr_models.clone(),
                    OcrModel::new(model.clone(), OcrModelKind::TraditionalOcr),
                    ProvisionedModelKind::Ocr,
                    "provision configured OCR model",
                );
            }
        }
        if self.config.services.ocr_vl.active() {
            for model in &self.config.services.ocr_vl.available_models {
                spawn_model_provision(
                    &mut tasks,
                    Arc::clone(&semaphore),
                    self.ocr_vl_models.clone(),
                    OcrModel::new(model.clone(), OcrModelKind::OcrVl),
                    ProvisionedModelKind::OcrVl,
                    "provision configured OCR-VL model",
                );
            }
        }

        let mut required_layouts = HashSet::new();
        if self.config.services.ocr.active() {
            required_layouts.extend(resolve_layout_models(
                &self.config.services.ocr.layout_available_models,
            ));
        }
        if self.config.services.ocr_vl.active() {
            required_layouts.extend(resolve_layout_models(
                &self.config.services.ocr_vl.layout_available_models,
            ));
        }
        for layout in required_layouts {
            spawn_model_provision(
                &mut tasks,
                Arc::clone(&semaphore),
                self.layout_models.clone(),
                layout,
                ProvisionedModelKind::Layout,
                "provision configured OCR layout model",
            );
        }

        let mut counts = ProvisionedModelCounts::default();
        while let Some(result) = tasks.join_next().await {
            counts
                .record(result.map_err(|error| {
                    anyhow::anyhow!("model provision task failed: {error:#}")
                })??);
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
    pub async fn ocr(&self, model: OcrModel) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr.active() {
            return Ok(None);
        }
        let layout_models = self
            .ensure_layout_models(&self.config.services.ocr.layout_available_models)
            .await?;
        let device = self.config.services.ocr.device;
        let models_dir = self.config.models.dir.clone();
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading OCR model");
                    runtime_factory
                        .load_ocr(model, path, models_dir, layout_models, device)
                        .await
                        .context("load OCR model")
                },
            )
            .await
    }

    /// # Errors
    ///
    /// Returns an error when OCR-VL assets cannot be provisioned or loaded.
    pub async fn ocr_vl(&self, model: OcrModel) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr_vl.active() {
            return Ok(None);
        }
        let layout_models = self
            .ensure_layout_models(&self.config.services.ocr_vl.layout_available_models)
            .await?;
        let device = self.config.services.ocr_vl.device;
        let models_dir = self.config.models.dir.clone();
        let runtime_factory = Arc::clone(&self.runtime_factory);
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_vl_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading OCR-VL model");
                    runtime_factory
                        .load_ocr(model, path, models_dir, layout_models, device)
                        .await
                        .context("load OCR-VL model")
                },
            )
            .await
    }

    async fn ensure_layout_models(
        &self,
        configured: &[ModelId],
    ) -> anyhow::Result<Vec<(OcrModel, PathBuf)>> {
        let mut provisioned = Vec::with_capacity(configured.len());
        for model_id in configured {
            let model = OcrModel::new(model_id.clone(), OcrModelKind::Layout);
            let path = self
                .layout_models
                .ensure_provisioned(model.clone())
                .await?
                .with_context(|| format!("configured OCR layout model `{model}` is unavailable"))?;
            provisioned.push((model, path));
        }
        Ok(provisioned)
    }

    fn active_model_caches(&self) -> Vec<&dyn CacheTracker> {
        let mut caches: Vec<&dyn CacheTracker> = Vec::with_capacity(4);
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
    }

    pub async fn shutdown(&self) {
        self.global_models.close_and_drain().await;
        self.cleanup_shutdown.send_replace(true);
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
    }

    async fn unload_all_models(&self) {
        for model in &self.config.services.asr.available_models {
            if let Err(error) = self.asr_models.unload(model.clone()).await {
                tracing::error!(model = %model, %error, "failed to unload ASR model during shutdown");
            }
        }
        for model in &self.config.services.tts.available_models {
            if let Err(error) = self.tts_models.unload(model.clone()).await {
                tracing::error!(model = %model, %error, "failed to unload TTS model during shutdown");
            }
        }
        for id in &self.config.services.ocr.available_models {
            let model = OcrModel::new(id.clone(), OcrModelKind::TraditionalOcr);
            if let Err(error) = self.ocr_models.unload(model).await {
                tracing::error!(model = %id, %error, "failed to unload OCR model during shutdown");
            }
        }
        for id in &self.config.services.ocr_vl.available_models {
            let model = OcrModel::new(id.clone(), OcrModelKind::OcrVl);
            if let Err(error) = self.ocr_vl_models.unload(model).await {
                tracing::error!(model = %id, %error, "failed to unload OCR-VL model during shutdown");
            }
        }
    }
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

impl ServerApplication for AppState {
    fn api_policy(&self) -> &ApiPolicy {
        &self.api_policy
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

impl ModelLifecycleRuntime for AppState {
    fn model_statuses(&self) -> ModelStatusesFuture<'_> {
        Box::pin(async move {
            let mut statuses = Vec::new();
            if self.config.services.asr.enabled {
                for model in &self.config.services.asr.available_models {
                    if let Some(status) = self.asr_models.status(model).await {
                        statuses.push(model_status(model.as_str(), ModelService::Asr, status));
                    }
                }
            }
            if self.config.services.tts.enabled {
                for model in &self.config.services.tts.available_models {
                    if let Some(status) = self.tts_models.status(model).await {
                        statuses.push(model_status(model.as_str(), ModelService::Tts, status));
                    }
                }
            }
            if self.config.services.ocr.active() {
                for id in &self.config.services.ocr.available_models {
                    let model = OcrModel::new(id.clone(), OcrModelKind::TraditionalOcr);
                    if let Some(status) = self.ocr_models.status(&model).await {
                        statuses.push(model_status(id.as_str(), ModelService::Ocr, status));
                    }
                }
            }
            if self.config.services.ocr_vl.active() {
                for id in &self.config.services.ocr_vl.available_models {
                    let model = OcrModel::new(id.clone(), OcrModelKind::OcrVl);
                    if let Some(status) = self.ocr_vl_models.status(&model).await {
                        statuses.push(model_status(id.as_str(), ModelService::OcrVl, status));
                    }
                }
            }
            statuses
        })
    }

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
                        &self.config.services.ocr.available_models,
                        OcrModelKind::TraditionalOcr,
                    ) else {
                        return Ok(None);
                    };
                    let Some(lease) = self
                        .ocr(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    self.ocr_models.status(&model).await
                }
                ModelService::OcrVl => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr_vl.available_models,
                        OcrModelKind::OcrVl,
                    ) else {
                        return Ok(None);
                    };
                    let Some(lease) = self
                        .ocr_vl(model.clone())
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?
                    else {
                        return Ok(None);
                    };
                    drop(lease);
                    self.ocr_vl_models.status(&model).await
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
                        &self.config.services.ocr.available_models,
                        OcrModelKind::TraditionalOcr,
                    ) else {
                        return Ok(None);
                    };
                    self.ocr_models
                        .unload(model)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?;
                    Some(ModelResidencyStatus::Unloaded)
                }
                ModelService::OcrVl if self.config.services.ocr_vl.active() => {
                    let Some(model) = selected_ocr_model(
                        &selector.model,
                        &self.config.services.ocr_vl.available_models,
                        OcrModelKind::OcrVl,
                    ) else {
                        return Ok(None);
                    };
                    self.ocr_vl_models
                        .unload(model)
                        .await
                        .map_err(|error| model_lifecycle_error(&error))?;
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
            available_models: self.config.services.asr.available_models.clone(),
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
                default_model: ocr.default_model.clone(),
                available_models: ocr.available_models.clone(),
                layout_default_model: ocr.layout_default_model.clone(),
                layout_available_models: ocr.layout_available_models.clone(),
                format: ocr.format,
                max_pixels: ocr.max_pixels,
            },
            ocr_vl: OcrVlServicePolicy {
                active: ocr_vl.active(),
                default_model: ocr_vl.default_model.clone(),
                available_models: ocr_vl.available_models.clone(),
                layout_default_model: ocr_vl.layout_default_model.clone(),
                layout_available_models: ocr_vl.layout_available_models.clone(),
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
            let ocr = match choice {
                OcrServiceChoice::Ocr { model } => {
                    AppState::ocr(self, OcrModel::new(model, OcrModelKind::TraditionalOcr)).await
                }
                OcrServiceChoice::OcrVl { model } => {
                    AppState::ocr_vl(self, OcrModel::new(model, OcrModelKind::OcrVl)).await
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
}

impl ModelProvisioners {
    fn new<P>(provisioner: Arc<P>) -> Self
    where
        P: ModelProvisioner<AsrModel>
            + ModelProvisioner<TtsModel>
            + ModelProvisioner<OcrModel>
            + 'static,
    {
        let asr: Arc<dyn ModelProvisioner<AsrModel>> = provisioner.clone();
        let tts: Arc<dyn ModelProvisioner<TtsModel>> = provisioner.clone();
        let ocr: Arc<dyn ModelProvisioner<OcrModel>> = provisioner;
        Self { asr, tts, ocr }
    }
}

fn build_model_cache<M, E>(
    cache_id: &'static str,
    available_models: Vec<M>,
    idle_timeout: std::time::Duration,
    max_loaded: usize,
    dir: PathBuf,
    provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
    residency: ResidencyDomain,
) -> ModelCache<M, E>
where
    M: ModelSpec + std::hash::Hash,
    E: Clone,
{
    ModelCache::new_in_domain(
        cache_id,
        available_models,
        idle_timeout,
        max_loaded,
        dir,
        provisioner,
        residency,
    )
}

fn spawn_model_provision<M, E>(
    tasks: &mut JoinSet<anyhow::Result<ProvisionedModelKind>>,
    semaphore: Arc<Semaphore>,
    cache: ModelCache<M, E>,
    model: M,
    kind: ProvisionedModelKind,
    context: &'static str,
) where
    M: ModelSpec + std::hash::Hash,
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
    ocr: Vec<OcrModel>,
    ocr_vl: Vec<OcrModel>,
    layout: Vec<OcrModel>,
}

fn resolve_configured_ocr_models(config: &ServerConfig) -> ResolvedOcrModels {
    let ocr = if config.services.ocr.active() {
        resolve_ocr_models(
            &config.services.ocr.available_models,
            OcrModelKind::TraditionalOcr,
        )
    } else {
        Vec::new()
    };
    let ocr_vl = if config.services.ocr_vl.active() {
        resolve_ocr_models(
            &config.services.ocr_vl.available_models,
            OcrModelKind::OcrVl,
        )
    } else {
        Vec::new()
    };
    let mut layout = Vec::new();
    if config.services.ocr.active() {
        layout.extend(resolve_layout_models(
            &config.services.ocr.layout_available_models,
        ));
    }
    if config.services.ocr_vl.active() {
        for model in resolve_layout_models(&config.services.ocr_vl.layout_available_models) {
            if !layout.contains(&model) {
                layout.push(model);
            }
        }
    }
    ResolvedOcrModels {
        ocr,
        ocr_vl,
        layout,
    }
}

fn validate_runtime_factory(
    config: &ServerConfig,
    runtime_factory: &dyn ModelRuntimeFactory,
) -> anyhow::Result<()> {
    if config.services.asr.enabled {
        for model in &config.services.asr.available_models {
            anyhow::ensure!(
                runtime_factory.supports_asr(model),
                "runtime factory does not support configured ASR model `{model}`"
            );
        }
    }
    if config.services.tts.enabled {
        for model in &config.services.tts.available_models {
            anyhow::ensure!(
                runtime_factory.supports_tts(model),
                "runtime factory does not support configured TTS model `{model}`"
            );
        }
    }

    let resolved = resolve_configured_ocr_models(config);
    for model in resolved
        .ocr
        .into_iter()
        .chain(resolved.ocr_vl)
        .chain(resolved.layout)
    {
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
    Ok(())
}

fn api_policy(config: &ServerConfig) -> ApiPolicy {
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
        asr: config.services.asr.enabled.then(|| AsrApiPolicy {
            available_models: config.services.asr.available_models.clone(),
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
            .then(|| config.services.tts.available_models.clone()),
        ocr: config.services.ocr.active().then(|| OcrApiModels {
            models: config.services.ocr.available_models.clone(),
            layout_models: config.services.ocr.layout_available_models.clone(),
        }),
        ocr_vl: config.services.ocr_vl.active().then(|| OcrApiModels {
            models: config.services.ocr_vl.available_models.clone(),
            layout_models: config.services.ocr_vl.layout_available_models.clone(),
        }),
    }
}

fn resolve_ocr_models(models: &[ModelId], kind: OcrModelKind) -> Vec<OcrModel> {
    models
        .iter()
        .cloned()
        .map(|model| OcrModel::new(model, kind))
        .collect()
}

fn resolve_layout_models(models: &[ModelId]) -> Vec<OcrModel> {
    resolve_ocr_models(models, OcrModelKind::Layout)
}
