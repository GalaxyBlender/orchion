use crate::application::model_cache::{
    AsrModelCache, CacheTracker, GlobalModelCacheLimiter, ModelLease, OcrModelCache,
    OcrVlModelCache, TtsModelCache, ensure_available_models,
};
use crate::settings::ServerConfig;
use anyhow::Context;
use orchion::{Asr, KnownOcrModel, ModelDownloader, ModelId, Ocr, Tts};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    config: ServerConfig,
    asr_models: AsrModelCache,
    tts_models: TtsModelCache,
    ocr_models: OcrModelCache,
    ocr_vl_models: OcrVlModelCache,
    global_models: GlobalModelCacheLimiter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use orchion::KnownOcrModel;

    #[tokio::test]
    async fn disabled_ocr_services_skip_empty_caches() {
        let temp_dir = tempfile::tempdir().unwrap();
        let exe_path = temp_dir.path().join("orchion-server");
        let mut config = ServerConfig::default_for_exe(&exe_path);
        config.models.dir = temp_dir.path().join("models");
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let models_dir = config.models.dir.clone();

        let state = AppState::load(config).await.unwrap();

        assert!(
            state
                .ocr(KnownOcrModel::PpOcrV6Tiny)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ocr_vl(KnownOcrModel::PaddleOcrVl16)
                .await
                .unwrap()
                .is_none()
        );
        assert!(!models_dir.exists());
    }

    #[tokio::test]
    async fn inactive_ocr_services_ignore_unknown_available_models() {
        let mut config = test_config();
        let unknown_model = ModelId::parse("Acme/Experimental-OCR").unwrap();
        config.services.ocr.enabled = false;
        config.services.ocr.available_models = vec![unknown_model.clone()];
        config.services.ocr_vl.enabled = false;
        config.services.ocr_vl.available_models = vec![unknown_model];

        let state = AppState::load(config).await.unwrap();

        assert!(
            state
                .ocr(KnownOcrModel::PpOcrV6Tiny)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ocr_vl(KnownOcrModel::PaddleOcrVl16)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn active_ocr_services_reject_unknown_available_models() {
        let mut config = test_config();
        config.services.ocr.enabled = true;
        config.services.ocr.available_models =
            vec![ModelId::parse("Acme/Experimental-OCR").unwrap()];

        let error = match AppState::load(config).await {
            Ok(_) => panic!("active unknown OCR model should fail"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("resolve configured OCR models"),
            "unexpected error: {error:#}"
        );
        assert!(
            format!("{error:#}").contains("Acme/Experimental-OCR"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn active_ocr_vl_services_reject_unknown_available_models() {
        let mut config = test_config();
        config.services.ocr_vl.enabled = true;
        config.services.ocr_vl.available_models =
            vec![ModelId::parse("Acme/Experimental-OCR").unwrap()];

        let error = match AppState::load(config).await {
            Ok(_) => panic!("active unknown OCR-VL model should fail"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("resolve configured OCR-VL models"),
            "unexpected error: {error:#}"
        );
        assert!(
            format!("{error:#}").contains("Acme/Experimental-OCR"),
            "unexpected error: {error:#}"
        );
    }

    fn test_config() -> ServerConfig {
        let temp_dir = tempfile::tempdir().unwrap().keep();
        let exe_path = temp_dir.join("orchion-server");
        let mut config = ServerConfig::default_for_exe(&exe_path);
        config.models.dir = temp_dir.join("models");
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config
    }
}

impl AppState {
    pub async fn load(config: ServerConfig) -> anyhow::Result<Arc<Self>> {
        let downloader = ModelDownloader::new(config.models.source.into());
        let ocr_active = config.services.ocr.active();
        let ocr_vl_active = config.services.ocr_vl.active();
        let resolved_ocr_models = resolve_configured_ocr_models(&config)?;
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
        let ocr_count = if ocr_active {
            let layout_models = resolve_layout_models(
                &config.services.ocr.layout_available_models,
                "resolve configured OCR layout models",
            )?;
            let layout_count = ensure_available_models(
                "OCR layout",
                &downloader,
                &layout_models,
                &config.models.dir,
            )
            .await
            .context("download OCR layout models")?;
            ensure_available_models(
                "OCR",
                &downloader,
                &resolved_ocr_models.ocr,
                &config.models.dir,
            )
            .await
            .context("download OCR models")?
                + layout_count
        } else {
            tracing::trace!("OCR model download check skipped because service is inactive");
            0
        };
        let ocr_vl_count = if ocr_vl_active {
            let layout_models = resolve_layout_models(
                &config.services.ocr_vl.layout_available_models,
                "resolve configured OCR-VL layout models",
            )?;
            let layout_count = ensure_available_models(
                "OCR-VL layout",
                &downloader,
                &layout_models,
                &config.models.dir,
            )
            .await
            .context("download OCR-VL layout models")?;
            let ocr_vl_count = ensure_available_models(
                "OCR-VL",
                &downloader,
                &resolved_ocr_models.ocr_vl,
                &config.models.dir,
            )
            .await
            .context("download OCR-VL models")?;
            layout_count + ocr_vl_count
        } else {
            tracing::trace!("OCR-VL model download check skipped because service is inactive");
            0
        };
        let state = Arc::new(Self::build(config, resolved_ocr_models));
        state.spawn_idle_cleanup();
        tracing::info!(
            asr = asr_count,
            tts = tts_count,
            ocr = ocr_count,
            ocr_vl = ocr_vl_count,
            "model cache ready"
        );
        Ok(state)
    }

    pub fn from_prepared_config(config: ServerConfig) -> anyhow::Result<Self> {
        let resolved_ocr_models = resolve_configured_ocr_models(&config)?;
        Ok(Self::build(config, resolved_ocr_models))
    }

    pub const fn config(&self) -> &ServerConfig {
        &self.config
    }

    fn build(config: ServerConfig, resolved_ocr_models: ResolvedOcrModels) -> Self {
        let asr_models = AsrModelCache::new(
            "asr",
            config.services.asr.available_models.clone(),
            config.services.asr.idle_timeout,
            config.services.asr.max_loaded,
            config.models.dir.clone(),
        );
        let tts_models = TtsModelCache::new(
            "tts",
            config.services.tts.available_models.clone(),
            config.services.tts.idle_timeout,
            config.services.tts.max_loaded,
            config.models.dir.clone(),
        );
        let ocr_models = OcrModelCache::new(
            "ocr",
            resolved_ocr_models.ocr,
            config.services.ocr.idle_timeout,
            config.services.ocr.max_loaded,
            config.models.dir.clone(),
        );
        let ocr_vl_models = OcrVlModelCache::new(
            "ocr-vl",
            resolved_ocr_models.ocr_vl,
            config.services.ocr_vl.idle_timeout,
            config.services.ocr_vl.max_loaded,
            config.models.dir.clone(),
        );
        let global_models = GlobalModelCacheLimiter::new(config.models.max_loaded);
        Self {
            config,
            asr_models,
            tts_models,
            ocr_models,
            ocr_vl_models,
            global_models,
        }
    }

    pub async fn asr(&self, model: orchion::AsrModel) -> anyhow::Result<Option<ModelLease<Asr>>> {
        if !self.config.services.asr.enabled {
            return Ok(None);
        }
        let device = self.config.services.asr.device;
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.asr_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading ASR model");
                    Asr::load_with_device(model, path, device)
                        .await
                        .context("load ASR model")
                },
            )
            .await
    }

    pub async fn tts(&self, model: orchion::TtsModel) -> anyhow::Result<Option<ModelLease<Tts>>> {
        if !self.config.services.tts.enabled {
            return Ok(None);
        }
        let device = self.config.services.tts.device;
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.tts_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading TTS model");
                    Tts::load_with_device(model, path, device)
                        .await
                        .context("load TTS model")
                },
            )
            .await
    }

    pub async fn ocr(&self, model: KnownOcrModel) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr.active() {
            return Ok(None);
        }
        let device = self.config.services.ocr.device;
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading OCR model");
                    Ocr::load_with_device(model.id(), path, device)
                        .await
                        .context("load OCR model")
                },
            )
            .await
    }

    pub async fn ocr_vl(&self, model: KnownOcrModel) -> anyhow::Result<Option<ModelLease<Ocr>>> {
        if !self.config.services.ocr_vl.active() {
            return Ok(None);
        }
        let device = self.config.services.ocr_vl.device;
        let all_caches = self.active_model_caches();
        self.global_models
            .get_or_load(
                &self.ocr_vl_models,
                all_caches.as_slice(),
                model,
                move |model, path| async move {
                    tracing::info!(model = ?model, device = %device, "loading OCR-VL model");
                    Ocr::load_with_device(model.id(), path, device)
                        .await
                        .context("load OCR-VL model")
                },
            )
            .await
    }

    fn active_model_caches(&self) -> ActiveModelCaches<'_> {
        match (
            self.config.services.asr.enabled,
            self.config.services.tts.enabled,
            self.config.services.ocr.active(),
            self.config.services.ocr_vl.active(),
        ) {
            (false, false, false, false) => ActiveModelCaches::Empty([]),
            (true, false, false, false) => ActiveModelCaches::One([&self.asr_models]),
            (false, true, false, false) => ActiveModelCaches::One([&self.tts_models]),
            (false, false, true, false) => ActiveModelCaches::One([&self.ocr_models]),
            (false, false, false, true) => ActiveModelCaches::One([&self.ocr_vl_models]),
            (true, true, false, false) => {
                ActiveModelCaches::Two([&self.asr_models, &self.tts_models])
            }
            (true, false, true, false) => {
                ActiveModelCaches::Two([&self.asr_models, &self.ocr_models])
            }
            (true, false, false, true) => {
                ActiveModelCaches::Two([&self.asr_models, &self.ocr_vl_models])
            }
            (false, true, true, false) => {
                ActiveModelCaches::Two([&self.tts_models, &self.ocr_models])
            }
            (false, true, false, true) => {
                ActiveModelCaches::Two([&self.tts_models, &self.ocr_vl_models])
            }
            (false, false, true, true) => {
                ActiveModelCaches::Two([&self.ocr_models, &self.ocr_vl_models])
            }
            (true, true, true, false) => {
                ActiveModelCaches::Three([&self.asr_models, &self.tts_models, &self.ocr_models])
            }
            (true, true, false, true) => {
                ActiveModelCaches::Three([&self.asr_models, &self.tts_models, &self.ocr_vl_models])
            }
            (true, false, true, true) => {
                ActiveModelCaches::Three([&self.asr_models, &self.ocr_models, &self.ocr_vl_models])
            }
            (false, true, true, true) => {
                ActiveModelCaches::Three([&self.tts_models, &self.ocr_models, &self.ocr_vl_models])
            }
            (true, true, true, true) => ActiveModelCaches::Four([
                &self.asr_models,
                &self.tts_models,
                &self.ocr_models,
                &self.ocr_vl_models,
            ]),
        }
    }

    fn spawn_idle_cleanup(self: &Arc<Self>) {
        let asr_enabled = self.config.services.asr.enabled;
        let tts_enabled = self.config.services.tts.enabled;
        let ocr_active = self.config.services.ocr.active();
        let ocr_vl_active = self.config.services.ocr_vl.active();

        let cleanup_interval = [
            asr_enabled.then_some(self.config.services.asr.idle_timeout),
            tts_enabled.then_some(self.config.services.tts.idle_timeout),
            ocr_active.then_some(self.config.services.ocr.idle_timeout),
            ocr_vl_active.then_some(self.config.services.ocr_vl.idle_timeout),
        ]
        .into_iter()
        .flatten()
        .min();
        let Some(cleanup_interval) = cleanup_interval else {
            return;
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
                if ocr_active {
                    state.ocr_models.cleanup_idle().await;
                }
                if ocr_vl_active {
                    state.ocr_vl_models.cleanup_idle().await;
                }
            }
        });
    }
}

struct ResolvedOcrModels {
    ocr: Vec<KnownOcrModel>,
    ocr_vl: Vec<KnownOcrModel>,
}

fn resolve_configured_ocr_models(config: &ServerConfig) -> anyhow::Result<ResolvedOcrModels> {
    let ocr = if config.services.ocr.active() {
        resolve_ocr_models(
            &config.services.ocr.available_models,
            KnownOcrModel::from_traditional_model_id,
        )
        .context("resolve configured OCR models")?
    } else {
        Vec::new()
    };
    let ocr_vl = if config.services.ocr_vl.active() {
        resolve_ocr_models(
            &config.services.ocr_vl.available_models,
            KnownOcrModel::from_ocr_vl_model_id,
        )
        .context("resolve configured OCR-VL models")?
    } else {
        Vec::new()
    };
    Ok(ResolvedOcrModels { ocr, ocr_vl })
}

enum ActiveModelCaches<'a> {
    Empty([&'a dyn CacheTracker; 0]),
    One([&'a dyn CacheTracker; 1]),
    Two([&'a dyn CacheTracker; 2]),
    Three([&'a dyn CacheTracker; 3]),
    Four([&'a dyn CacheTracker; 4]),
}

impl<'a> ActiveModelCaches<'a> {
    fn as_slice(&self) -> &[&'a dyn CacheTracker] {
        match self {
            Self::Empty(caches) => caches,
            Self::One(caches) => caches,
            Self::Two(caches) => caches,
            Self::Three(caches) => caches,
            Self::Four(caches) => caches,
        }
    }
}

fn resolve_ocr_models(
    models: &[ModelId],
    resolve: fn(&ModelId) -> orchion::Result<KnownOcrModel>,
) -> anyhow::Result<Vec<KnownOcrModel>> {
    models
        .iter()
        .map(|model| {
            resolve(model).with_context(|| format!("resolve configured OCR model `{model}`"))
        })
        .collect()
}

fn resolve_layout_models(
    models: &[ModelId],
    context: &'static str,
) -> anyhow::Result<Vec<KnownOcrModel>> {
    models
        .iter()
        .map(|model| {
            KnownOcrModel::from_layout_model_id(model)
                .with_context(|| format!("resolve OCR layout model `{model}`"))?;
            Ok(KnownOcrModel::PpDocLayoutV3)
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .context(context)
}
