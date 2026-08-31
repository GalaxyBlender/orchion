use orchion_core::{
    DevicePreference, KnownOcrModel, ModelId, OcrLimits, OcrModel, OcrModelKind, OcrOptions,
    OcrResult, OrchionError, Result,
};
pub use orchion_ocr::OcrAssets;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

pub type OcrEngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// A validated OCR model identity paired with its complete local runtime assets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrDeployment {
    model: OcrModel,
    assets: OcrAssets,
}

impl OcrDeployment {
    pub fn from_assets(model: OcrModel, assets: OcrAssets) -> Result<Self> {
        let known = known_ocr_model(&model)?;
        let assets_kind = match &assets {
            OcrAssets::Traditional { .. } => OcrModelKind::TraditionalOcr,
            OcrAssets::Layout { .. } => OcrModelKind::Layout,
            OcrAssets::VisionLanguage { .. } => OcrModelKind::OcrVl,
        };
        if known.kind() != assets_kind {
            return Err(OrchionError::ModelLoad {
                message: format!(
                    "OCR model `{}` has kind {:?}, but the supplied assets are {:?}",
                    model.id(),
                    known.kind(),
                    assets_kind
                ),
            });
        }
        Ok(Self { model, assets })
    }

    #[must_use]
    pub const fn model(&self) -> &OcrModel {
        &self.model
    }

    #[must_use]
    pub const fn assets(&self) -> &OcrAssets {
        &self.assets
    }

    #[must_use]
    pub fn into_assets(self) -> OcrAssets {
        self.assets
    }

    #[cfg(feature = "download-all")]
    pub async fn provision(model: OcrModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        Self::provision_with_downloader(
            model,
            None,
            cache_dir,
            &orchion_download::ModelDownloader::default(),
        )
        .await
    }

    #[cfg(feature = "download-all")]
    pub async fn provision_with_downloader(
        model: OcrModel,
        layout: Option<OcrModel>,
        cache_dir: impl AsRef<Path>,
        downloader: &orchion_download::ModelDownloader,
    ) -> Result<Self> {
        let known = known_ocr_model(&model)?;
        let layout = layout
            .map(|layout| {
                let layout_known = known_ocr_model(&layout)?;
                if !layout_known.is_layout_model() {
                    return Err(OrchionError::ModelLoad {
                        message: format!(
                            "OCR layout model `{}` is not a Layout model",
                            layout.id()
                        ),
                    });
                }
                Ok(layout)
            })
            .transpose()?;
        let model_dir = downloader
            .download(model.clone(), cache_dir.as_ref())
            .await?;
        if let Some(layout) = layout {
            downloader.download(layout, cache_dir.as_ref()).await?;
        }
        let assets = OcrAssets::from_cache_layout(known, model_dir, cache_dir);
        Self::from_assets(model, assets)
    }
}

pub trait OcrEngine: Send + Sync {
    fn model(&self) -> &ModelId;

    fn recognize_file_with_limits(
        &self,
        path: PathBuf,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> OcrEngineFuture<'_, OcrResult>;
}

#[derive(Clone)]
pub struct Ocr {
    model: ModelId,
    inner: Arc<dyn OcrEngine>,
}

struct OarOcrEngine {
    model: ModelId,
    inner: orchion_ocr::OcrEngine,
}

impl Ocr {
    pub fn from_engine(engine: Arc<dyn OcrEngine>) -> Self {
        Self {
            model: engine.model().clone(),
            inner: engine,
        }
    }

    pub async fn load(model: impl AsRef<str>, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    pub async fn load_with_device(
        model: impl AsRef<str>,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        let (id, known) = parse_model(model.as_ref())?;
        let inner = orchion_ocr::OcrEngine::load_with_device(known, model_dir, device).await?;
        Ok(Self::from_engine(Arc::new(OarOcrEngine {
            model: id,
            inner,
        })))
    }

    pub async fn load_with_assets(model: impl AsRef<str>, assets: OcrAssets) -> Result<Self> {
        Self::load_with_assets_and_device(model, assets, DevicePreference::Auto).await
    }

    pub async fn load_with_assets_and_device(
        model: impl AsRef<str>,
        assets: OcrAssets,
        device: DevicePreference,
    ) -> Result<Self> {
        let (id, known) = parse_model(model.as_ref())?;
        Self::load_parsed_with_assets(id, known, assets, device).await
    }

    pub async fn load_deployment(
        deployment: OcrDeployment,
        device: DevicePreference,
    ) -> Result<Self> {
        let OcrDeployment { model, assets } = deployment;
        let known = known_ocr_model(&model)?;
        Self::load_parsed_with_assets(model.id().clone(), known, assets, device).await
    }

    #[cfg(feature = "download-all")]
    pub async fn load_or_download(model: OcrModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_or_download_with_device(model, cache_dir, DevicePreference::Auto).await
    }

    #[cfg(feature = "download-all")]
    pub async fn load_or_download_with_device(
        model: OcrModel,
        cache_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        let deployment = OcrDeployment::provision(model, cache_dir).await?;
        Self::load_deployment(deployment, device).await
    }

    async fn load_parsed_with_assets(
        id: ModelId,
        known: KnownOcrModel,
        assets: OcrAssets,
        device: DevicePreference,
    ) -> Result<Self> {
        let inner = orchion_ocr::OcrEngine::load_with_assets(known, assets, device).await?;
        Ok(Self::from_engine(Arc::new(OarOcrEngine {
            model: id,
            inner,
        })))
    }

    #[must_use]
    pub const fn model(&self) -> &ModelId {
        &self.model
    }

    pub async fn recognize_file(&self, path: impl AsRef<Path>) -> Result<OcrResult> {
        self.recognize_file_with(path, OcrOptions::default()).await
    }

    pub async fn recognize_file_with(
        &self,
        path: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<OcrResult> {
        self.recognize_file_with_limits(path, options, OcrLimits::default())
            .await
    }

    pub async fn recognize_file_with_limits(
        &self,
        path: impl AsRef<Path>,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> Result<OcrResult> {
        self.inner
            .recognize_file_with_limits(path.as_ref().to_path_buf(), options, limits)
            .await
    }
}

fn parse_model(model: &str) -> Result<(ModelId, KnownOcrModel)> {
    let id = ModelId::parse(model).map_err(|error| OrchionError::ModelLoad {
        message: error.to_string(),
    })?;
    let known = KnownOcrModel::from_model_id(&id)?;
    Ok((id, known))
}

fn known_ocr_model(model: &OcrModel) -> Result<KnownOcrModel> {
    model.known().ok_or_else(|| OrchionError::ModelLoad {
        message: format!(
            "unsupported OCR model `{}` with kind {:?}",
            model.id(),
            model.kind()
        ),
    })
}

impl OcrEngine for OarOcrEngine {
    fn model(&self) -> &ModelId {
        &self.model
    }

    fn recognize_file_with_limits(
        &self,
        path: PathBuf,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> OcrEngineFuture<'_, OcrResult> {
        Box::pin(async move {
            self.inner
                .recognize_file_with_limits(path, options, limits)
                .await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CustomEngine {
        model: ModelId,
    }

    impl OcrEngine for CustomEngine {
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
                Err(OrchionError::Inference {
                    message: "custom engine".to_string(),
                })
            })
        }
    }

    fn assets_for(kind: OcrModelKind) -> OcrAssets {
        match kind {
            OcrModelKind::TraditionalOcr => OcrAssets::Traditional {
                detector: "detector.onnx".into(),
                recognizer: "recognizer.onnx".into(),
                dictionary: "dictionary.txt".into(),
                layout: None,
                table_structure: None,
            },
            OcrModelKind::Layout => OcrAssets::Layout {
                model: "layout.onnx".into(),
            },
            OcrModelKind::OcrVl => OcrAssets::VisionLanguage {
                model_dir: "vl".into(),
                layout: None,
            },
        }
    }

    #[tokio::test]
    async fn facade_rejects_invalid_model_id_before_loading_runtime() {
        let result = Ocr::load("not-a-model", "/tmp/orchion-test-models").await;

        assert!(matches!(
            result,
            Err(orchion_core::OrchionError::ModelLoad { .. })
        ));
    }

    #[test]
    fn deployment_validates_model_kind_and_exposes_assets() {
        for known in [
            KnownOcrModel::PpOcrV6Tiny,
            KnownOcrModel::PpDocLayoutV3,
            KnownOcrModel::PaddleOcrVl16,
        ] {
            let model = known.into_model();
            let assets = assets_for(known.kind());
            let deployment = OcrDeployment::from_assets(model.clone(), assets.clone()).unwrap();
            assert_eq!(deployment.model(), &model);
            assert_eq!(deployment.assets(), &assets);
            assert_eq!(deployment.into_assets(), assets);
        }

        let result = OcrDeployment::from_assets(
            KnownOcrModel::PpOcrV6Tiny.into_model(),
            assets_for(OcrModelKind::OcrVl),
        );
        assert!(matches!(result, Err(OrchionError::ModelLoad { .. })));
    }

    #[test]
    fn deployment_rejects_unknown_or_mislabeled_models() {
        let unknown = OcrModel::new(
            ModelId::parse("acme/ocr").unwrap(),
            OcrModelKind::TraditionalOcr,
        );
        assert!(matches!(
            OcrDeployment::from_assets(unknown, assets_for(OcrModelKind::TraditionalOcr)),
            Err(OrchionError::ModelLoad { .. })
        ));

        let mislabeled = OcrModel::new(
            KnownOcrModel::PpOcrV6Tiny.into_model().id().clone(),
            OcrModelKind::OcrVl,
        );
        assert!(matches!(
            OcrDeployment::from_assets(mislabeled, assets_for(OcrModelKind::OcrVl)),
            Err(OrchionError::ModelLoad { .. })
        ));
    }

    #[tokio::test]
    async fn from_engine_preserves_the_custom_engine_seam() {
        let model = ModelId::parse("acme/custom-ocr").unwrap();
        let ocr = Ocr::from_engine(Arc::new(CustomEngine {
            model: model.clone(),
        }));
        assert_eq!(ocr.model(), &model);
        assert!(matches!(
            ocr.recognize_file("image.png").await,
            Err(OrchionError::Inference { message }) if message == "custom engine"
        ));
    }
}
