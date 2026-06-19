use crate::application::model_cache::{AsrModelCache, TtsModelCache, ensure_available_models};
use crate::settings::ServerConfig;
use anyhow::Context;
use orchion::{Asr, ModelDownloader, Tts};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub struct AppState {
    pub config: ServerConfig,
    pub asr_models: AsrModelCache,
    pub tts_models: TtsModelCache,
}

impl AppState {
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let downloader = ModelDownloader::new(config.models.source.into());
        let asr_count = ensure_available_models(
            "ASR",
            &downloader,
            &config.models.asr.available,
            &config.models.dir,
        )
        .await
        .context("download ASR models")?;
        let tts_count = ensure_available_models(
            "TTS",
            &downloader,
            &config.models.tts.available,
            &config.models.dir,
        )
        .await
        .context("download TTS models")?;
        let asr_models = AsrModelCache::new(config.models.asr.clone(), config.models.dir.clone());
        let tts_models = TtsModelCache::new(config.models.tts.clone(), config.models.dir.clone());
        let state = Arc::new(Self {
            config,
            asr_models,
            tts_models,
        });
        state.spawn_idle_cleanup();
        tracing::info!(asr = asr_count, tts = tts_count, "model cache ready");
        Ok(state)
    }

    pub async fn asr(&self, model: orchion::AsrModel) -> anyhow::Result<Option<Asr>> {
        let device = self.config.models.asr.device;
        self.asr_models
            .get_or_load(model, |model, path| async move {
                tracing::info!(model = ?model, device = %device, "loading ASR model");
                Asr::load_with_device(model, path, device)
                    .await
                    .context("load ASR model")
            })
            .await
    }

    pub async fn tts(&self, model: orchion::TtsModel) -> anyhow::Result<Option<Tts>> {
        let device = self.config.models.tts.device;
        self.tts_models
            .get_or_load(model, |model, path| async move {
                tracing::info!(model = ?model, device = %device, "loading TTS model");
                Tts::load_with_device(model, path, device)
                    .await
                    .context("load TTS model")
            })
            .await
    }

    fn spawn_idle_cleanup(self: &Arc<Self>) {
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let interval = state
                .asr_models
                .idle_timeout()
                .min(state.tts_models.idle_timeout())
                .min(Duration::from_secs(60));
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                state.asr_models.cleanup_idle().await;
                state.tts_models.cleanup_idle().await;
            }
        });
    }
}
