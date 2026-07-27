use orchion_core::{
    DevicePreference, KnownOcrModel, ModelId, OcrLimits, OcrOptions, OcrResult, OrchionError,
    Result,
};
pub use orchion_ocr::OcrAssets;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

pub type OcrEngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

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

    #[tokio::test]
    async fn facade_rejects_invalid_model_id_before_loading_runtime() {
        let result = Ocr::load("not-a-model", "/tmp/orchion-test-models").await;

        assert!(matches!(
            result,
            Err(orchion_core::OrchionError::ModelLoad { .. })
        ));
    }
}
