use orchion_core::{Result, TtsAudio, TtsModel, TtsOptions, TtsVoice};
use std::path::Path;

#[derive(Clone)]
pub struct Tts {
    inner: orchion_qwen3::Tts,
}

impl Tts {
    pub async fn load(model: TtsModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: orchion_qwen3::Tts::load(model, model_dir).await?,
        })
    }

    #[cfg(feature = "download-all")]
    pub async fn load_or_download(model: TtsModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = orchion_download::ModelDownloader::default()
            .download(model, cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub const fn model(&self) -> TtsModel {
        self.inner.model()
    }

    pub async fn synthesize(&self, text: impl AsRef<str>, voice: TtsVoice) -> Result<TtsAudio> {
        self.inner.synthesize(text, voice).await
    }

    pub async fn synthesize_with(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> Result<TtsAudio> {
        self.inner.synthesize_with(text, voice, options).await
    }

    pub async fn synthesize_to_file(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.inner
            .synthesize_to_file(text, voice, output_path)
            .await
    }
}
