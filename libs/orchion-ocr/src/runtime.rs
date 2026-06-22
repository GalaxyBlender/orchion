use crate::result::{self, LoadedOcrRuntime};
use orchion_core::{DevicePreference, KnownOcrModel, OcrOptions, OcrResult, Result};
use std::path::Path;

/// OCR runtime handle for one loaded model directory and device preference.
#[derive(Clone)]
pub struct OcrEngine {
    model: KnownOcrModel,
    runtime: LoadedOcrRuntime,
}

impl OcrEngine {
    /// Creates an OCR engine handle for a known model and local model directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected model capability is not compiled in or model
    /// assets cannot be loaded on the requested device.
    pub async fn load_with_device(
        model: KnownOcrModel,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let runtime = result::load_runtime(model, model_dir, device).await?;
        Ok(Self { model, runtime })
    }

    /// Returns the model associated with this engine.
    #[must_use]
    pub const fn model(&self) -> KnownOcrModel {
        self.model
    }

    /// Runs OCR for an image file using explicit OCR options.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected model capability is not compiled in, model
    /// assets cannot be loaded, inference fails, or the blocking worker cannot join.
    pub async fn recognize_file_with(
        &self,
        path: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<OcrResult> {
        result::run_ocr(self.model, self.runtime.clone(), path.as_ref(), options).await
    }
}
