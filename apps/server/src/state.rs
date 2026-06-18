use crate::config::ServerConfig;
use anyhow::Context;
use orchion::{Asr, ModelDownloader, Tts};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: ServerConfig,
    pub asr: Asr,
    pub tts: Tts,
}

impl AppState {
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let downloader = ModelDownloader::new(config.models.source.into());
        let asr_dir = downloader
            .download(config.models.asr, &config.models.dir)
            .await
            .context("download ASR model")?;
        let tts_dir = downloader
            .download(config.models.tts, &config.models.dir)
            .await
            .context("download TTS model")?;
        let asr = Asr::load(config.models.asr, asr_dir)
            .await
            .context("load ASR model")?;
        let tts = Tts::load(config.models.tts, tts_dir)
            .await
            .context("load TTS model")?;
        Ok(Arc::new(Self { config, asr, tts }))
    }
}
