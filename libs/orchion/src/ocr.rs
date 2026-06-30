use orchion_core::{DevicePreference, KnownOcrModel, ModelId, OcrOptions, OcrResult, Result};
use std::path::Path;

#[derive(Clone)]
pub struct Ocr {
    model: ModelId,
    inner: orchion_ocr::OcrEngine,
}

impl Ocr {
    pub async fn load(model: impl AsRef<str>, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    pub async fn load_with_device(
        model: impl AsRef<str>,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        let id = ModelId::parse(model.as_ref()).map_err(|error| {
            orchion_core::OrchionError::ModelLoad {
                message: error.to_string(),
            }
        })?;
        let known = KnownOcrModel::from_model_id(&id)?;
        Ok(Self {
            model: id,
            inner: orchion_ocr::OcrEngine::load_with_device(known, model_dir, device).await?,
        })
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
        self.inner.recognize_file_with(path, options).await
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
