use crate::OcrAssets;
use crate::result::{self, LoadedOcrRuntime};
use orchion_core::{DevicePreference, KnownOcrModel, OcrLimits, OcrOptions, OcrResult, Result};
use std::path::Path;

/// OCR runtime handle for one loaded asset bundle and device preference.
#[derive(Clone)]
pub struct OcrEngine {
    model: KnownOcrModel,
    runtime: LoadedOcrRuntime,
}

impl OcrEngine {
    /// Creates an OCR engine using the cache layout accepted by earlier releases.
    ///
    /// New integrations should prefer [`Self::load_with_assets`].
    ///
    /// # Errors
    ///
    /// Returns an error when the legacy asset layout cannot be inferred or loaded.
    pub async fn load_with_device(
        model: KnownOcrModel,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        let assets = OcrAssets::from_legacy_cache_layout(model, model_dir.as_ref())?;
        Self::load_with_assets(model, assets, device).await
    }

    /// Creates an OCR engine handle from explicit local asset paths.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected model capability is not compiled in or model
    /// assets cannot be loaded on the requested device.
    pub async fn load_with_assets(
        model: KnownOcrModel,
        assets: OcrAssets,
        device: DevicePreference,
    ) -> Result<Self> {
        let runtime = result::load_runtime(model, assets, device).await?;
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
        self.recognize_file_with_limits(path, options, OcrLimits::default())
            .await
    }

    /// Runs OCR for an image file using explicit OCR options and resource limits.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the image exceeds the supplied limits.
    pub async fn recognize_file_with_limits(
        &self,
        path: impl AsRef<Path>,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> Result<OcrResult> {
        result::run_ocr(
            self.model,
            self.runtime.clone(),
            path.as_ref(),
            options,
            limits,
        )
        .await
    }
}
