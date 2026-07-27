#[cfg(feature = "tts-qwen3")]
use orchion_core::{DevicePreference, OrchionError, RuntimeProvider, model_descriptor};
use orchion_core::{Result, TtsAudio, TtsModel, TtsOptions, TtsVoice};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

pub type TtsEngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

pub trait TtsEngine: Send + Sync {
    fn model(&self) -> TtsModel;

    fn synthesize_with(
        &self,
        text: String,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> TtsEngineFuture<'_, TtsAudio>;

    fn synthesize_to_file(
        &self,
        text: String,
        voice: TtsVoice,
        output_path: PathBuf,
    ) -> TtsEngineFuture<'_, ()>;
}

#[derive(Clone)]
pub struct Tts {
    inner: Arc<dyn TtsEngine>,
}

#[cfg(feature = "tts-qwen3")]
struct QwenTtsEngine {
    inner: orchion_qwen3::Tts,
}

impl Tts {
    pub fn from_engine(engine: Arc<dyn TtsEngine>) -> Self {
        Self { inner: engine }
    }

    #[cfg(feature = "tts-qwen3")]
    pub async fn load(model: TtsModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    #[cfg(feature = "tts-qwen3")]
    pub async fn load_with_device(
        model: TtsModel,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        ensure_qwen_tts_model(&model)?;
        let inner = orchion_qwen3::Tts::load_with_device(model, model_dir, device).await?;
        Ok(Self::from_engine(Arc::new(QwenTtsEngine { inner })))
    }

    #[cfg(all(feature = "tts-qwen3", feature = "download-all"))]
    pub async fn load_or_download(model: TtsModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = orchion_download::ModelDownloader::default()
            .download(model.clone(), cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub fn model(&self) -> TtsModel {
        self.inner.model()
    }

    pub async fn synthesize(&self, text: impl AsRef<str>, voice: TtsVoice) -> Result<TtsAudio> {
        self.synthesize_with(text, voice, TtsOptions::default())
            .await
    }

    pub async fn synthesize_with(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> Result<TtsAudio> {
        self.inner
            .synthesize_with(text.as_ref().to_string(), voice, options)
            .await
    }

    pub async fn synthesize_to_file(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.inner
            .synthesize_to_file(
                text.as_ref().to_string(),
                voice,
                output_path.as_ref().to_path_buf(),
            )
            .await
    }
}

#[cfg(feature = "tts-qwen3")]
impl TtsEngine for QwenTtsEngine {
    fn model(&self) -> TtsModel {
        self.inner.model()
    }

    fn synthesize_with(
        &self,
        text: String,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> TtsEngineFuture<'_, TtsAudio> {
        Box::pin(async move { self.inner.synthesize_with(text, voice, options).await })
    }

    fn synthesize_to_file(
        &self,
        text: String,
        voice: TtsVoice,
        output_path: PathBuf,
    ) -> TtsEngineFuture<'_, ()> {
        Box::pin(async move {
            self.inner
                .synthesize_to_file(text, voice, output_path)
                .await
        })
    }
}

#[cfg(feature = "tts-qwen3")]
fn ensure_qwen_tts_model(model: &TtsModel) -> Result<()> {
    if model_descriptor(model.as_str())
        .is_none_or(|descriptor| descriptor.runtime_provider == RuntimeProvider::Qwen3)
    {
        Ok(())
    } else {
        Err(OrchionError::ModelLoad {
            message: format!("unsupported TTS model `{model}`"),
        })
    }
}

#[cfg(all(test, feature = "tts-qwen3"))]
mod provider_tests {
    use super::*;

    #[test]
    fn qwen_loader_preserves_fallback_for_unregistered_checkpoints() {
        assert!(ensure_qwen_tts_model(&TtsModel::parse("Acme/New-TTS").unwrap()).is_ok());
    }

    #[test]
    fn qwen_loader_rejects_models_registered_to_another_provider() {
        assert!(
            ensure_qwen_tts_model(&TtsModel::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap())
                .is_err()
        );
    }
}
