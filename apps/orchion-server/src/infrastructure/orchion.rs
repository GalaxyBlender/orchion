use crate::application::model_cache::{
    AsrModelCache, GlobalModelCacheLimiter, TtsModelCache, ensure_available_models,
};
use crate::settings::ServerConfig;
use anyhow::Context;
use orchion::{Asr, ModelDownloader, Tts};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: ServerConfig,
    pub asr_models: AsrModelCache,
    pub tts_models: TtsModelCache,
    pub global_models: GlobalModelCacheLimiter,
}

impl AppState {
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let downloader = ModelDownloader::new(config.models.source.into());
        let asr_count = if config.services.asr.enabled {
            ensure_available_models(
                "ASR",
                &downloader,
                &config.services.asr.available_models,
                &config.models.dir,
            )
            .await
            .context("download ASR models")?
        } else {
            tracing::trace!("ASR model download check skipped because service is disabled");
            0
        };
        let tts_count = if config.services.tts.enabled {
            ensure_available_models(
                "TTS",
                &downloader,
                &config.services.tts.available_models,
                &config.models.dir,
            )
            .await
            .context("download TTS models")?
        } else {
            tracing::trace!("TTS model download check skipped because service is disabled");
            0
        };
        let asr_models = AsrModelCache::new(
            config.services.asr.available_models.clone(),
            config.services.asr.idle_timeout,
            config.services.asr.max_loaded,
            config.models.dir.clone(),
        );
        let tts_models = TtsModelCache::new(
            config.services.tts.available_models.clone(),
            config.services.tts.idle_timeout,
            config.services.tts.max_loaded,
            config.models.dir.clone(),
        );
        let global_models = GlobalModelCacheLimiter::new(config.models.max_loaded);
        let state = Arc::new(Self {
            config,
            asr_models,
            tts_models,
            global_models,
        });
        state.spawn_idle_cleanup();
        tracing::info!(asr = asr_count, tts = tts_count, "model cache ready");
        Ok(state)
    }

    pub async fn asr(&self, model: orchion::AsrModel) -> anyhow::Result<Option<Asr>> {
        if !self.config.services.asr.enabled {
            return Ok(None);
        }
        let device = self.config.services.asr.device;
        self.global_models
            .get_or_load(
                &self.asr_models,
                &self.tts_models,
                model,
                |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading ASR model");
                    Asr::load_with_device(model, path, device)
                        .await
                        .context("load ASR model")
                },
            )
            .await
    }

    pub async fn tts(&self, model: orchion::TtsModel) -> anyhow::Result<Option<Tts>> {
        if !self.config.services.tts.enabled {
            return Ok(None);
        }
        let device = self.config.services.tts.device;
        self.global_models
            .get_or_load(
                &self.tts_models,
                &self.asr_models,
                model,
                |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading TTS model");
                    Tts::load_with_device(model, path, device)
                        .await
                        .context("load TTS model")
                },
            )
            .await
    }

    fn spawn_idle_cleanup(self: &Arc<Self>) {
        let asr_enabled = self.config.services.asr.enabled;
        let tts_enabled = self.config.services.tts.enabled;
        if !asr_enabled && !tts_enabled {
            return;
        }

        let cleanup_interval = match (asr_enabled, tts_enabled) {
            (true, true) => self
                .config
                .services
                .asr
                .idle_timeout
                .min(self.config.services.tts.idle_timeout),
            (true, false) => self.config.services.asr.idle_timeout,
            (false, true) => self.config.services.tts.idle_timeout,
            (false, false) => unreachable!(),
        };
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                if asr_enabled {
                    state.asr_models.cleanup_idle().await;
                }
                if tts_enabled {
                    state.tts_models.cleanup_idle().await;
                }
            }
        });
    }
}
