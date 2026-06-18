use orchion_core::{AsrModel, AsrOptions, AsrTranscript, Result};
use std::path::Path;
use std::time::Instant;

#[derive(Clone)]
pub struct Asr {
    inner: orchion_qwen3::Asr,
}

impl Asr {
    pub async fn load(model: AsrModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: orchion_qwen3::Asr::load(model, model_dir).await?,
        })
    }

    #[cfg(feature = "download-all")]
    pub async fn load_or_download(model: AsrModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = orchion_download::ModelDownloader::default()
            .download(model, cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub const fn model(&self) -> AsrModel {
        self.inner.model()
    }

    pub async fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<AsrTranscript> {
        self.inner.transcribe_file(path).await
    }

    pub async fn transcribe_file_with(
        &self,
        path: impl AsRef<Path>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        self.inner.transcribe_file_with(path, options).await
    }

    pub async fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AsrTranscript> {
        self.inner.transcribe_samples(samples, sample_rate).await
    }

    pub async fn transcribe_samples_with(
        &self,
        samples: &[f32],
        sample_rate: u32,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        self.inner
            .transcribe_samples_with(samples, sample_rate, options)
            .await
    }

    #[cfg(feature = "audio-ffmpeg")]
    pub async fn transcribe_audio_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<AsrTranscript> {
        self.transcribe_audio_bytes_with(bytes, AsrOptions::default())
            .await
    }

    #[cfg(feature = "audio-ffmpeg")]
    pub async fn transcribe_audio_bytes_with(
        &self,
        bytes: impl Into<Vec<u8>>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = orchion_audio::decode_audio_bytes(bytes.into()).await?;
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_started.elapsed().as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.inner
            .transcribe_samples_with(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|_| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    "ASR inference completed"
                );
            })
    }

    pub async fn start_streaming(&self) -> Result<orchion_qwen3::AsrStream> {
        self.inner.start_streaming().await
    }

    pub async fn start_streaming_with(
        &self,
        options: orchion_core::AsrStreamingOptions,
    ) -> Result<orchion_qwen3::AsrStream> {
        self.inner.start_streaming_with(options).await
    }
}
