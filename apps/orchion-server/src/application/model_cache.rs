use crate::settings::ModelRegistrySection;
use orchion::{Asr, AsrModel, ModelDownloader, ModelSpec, Tts, TtsModel};
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct ModelCache<M, E> {
    inner: Arc<Mutex<ModelCacheState<M, E>>>,
    dir: PathBuf,
    idle_timeout: Duration,
    max_loaded: usize,
}

struct ModelCacheState<M, E> {
    available: Vec<M>,
    loaded: HashMap<M, LoadedModel<E>>,
    loading: HashMap<M, Arc<Mutex<()>>>,
}

struct LoadedModel<E> {
    engine: E,
    last_used: Instant,
}

impl<M, E> ModelCache<M, E>
where
    M: ModelSpec + std::hash::Hash,
    E: Clone,
{
    pub fn new(registry: ModelRegistrySection<M>, dir: PathBuf) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelCacheState {
                available: registry.available,
                loaded: HashMap::new(),
                loading: HashMap::new(),
            })),
            dir,
            idle_timeout: registry.idle_timeout,
            max_loaded: registry.max_loaded,
        }
    }

    pub async fn get_or_load<F, Fut>(&self, model: M, load: F) -> anyhow::Result<Option<E>>
    where
        F: FnOnce(M, PathBuf) -> Fut,
        Fut: Future<Output = anyhow::Result<E>>,
    {
        let loading = {
            let mut state = self.inner.lock().await;
            if !state.available.contains(&model) {
                return Ok(None);
            }
            state.evict_idle(self.idle_timeout);
            if let Some(loaded) = state.loaded.get_mut(&model) {
                loaded.last_used = Instant::now();
                return Ok(Some(loaded.engine.clone()));
            }
            Arc::clone(
                state
                    .loading
                    .entry(model)
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };

        let _load_guard = loading.lock().await;
        {
            let mut state = self.inner.lock().await;
            state.evict_idle(self.idle_timeout);
            if let Some(loaded) = state.loaded.get_mut(&model) {
                loaded.last_used = Instant::now();
                return Ok(Some(loaded.engine.clone()));
            }
        }

        let engine = load(model, model.cache_path(&self.dir)).await?;
        let mut state = self.inner.lock().await;
        state.evict_idle(self.idle_timeout);
        state.evict_lru_until_below(self.max_loaded);
        let engine = state
            .loaded
            .entry(model)
            .or_insert_with(|| LoadedModel {
                engine,
                last_used: Instant::now(),
            })
            .engine
            .clone();
        Ok(Some(engine))
    }

    pub async fn cleanup_idle(&self) {
        self.inner.lock().await.evict_idle(self.idle_timeout);
    }

    pub const fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }
}

impl<M, E> ModelCacheState<M, E>
where
    M: Copy + Eq + std::hash::Hash,
{
    fn evict_idle(&mut self, idle_timeout: Duration) {
        let now = Instant::now();
        self.loaded
            .retain(|_, loaded| now.duration_since(loaded.last_used) < idle_timeout);
    }

    fn evict_lru_until_below(&mut self, max_loaded: usize) {
        while self.loaded.len() >= max_loaded {
            let Some(model) = self
                .loaded
                .iter()
                .min_by_key(|(_, loaded)| loaded.last_used)
                .map(|(model, _)| *model)
            else {
                break;
            };
            self.loaded.remove(&model);
        }
    }
}

pub type AsrModelCache = ModelCache<AsrModel, Asr>;
pub type TtsModelCache = ModelCache<TtsModel, Tts>;

pub async fn ensure_available_models<M: ModelSpec>(
    label: &'static str,
    downloader: &ModelDownloader,
    models: &[M],
    dir: &PathBuf,
) -> anyhow::Result<usize> {
    for model in models {
        tracing::debug!(model = ?model, models_dir = %dir.display(), "ensuring {label} model is available");
        let path = downloader.download(*model, dir).await?;
        tracing::debug!(model = ?model, path = %path.display(), "{label} model cache ready");
    }
    Ok(models.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn registry(max_loaded: usize, idle_timeout: Duration) -> ModelRegistrySection<AsrModel> {
        ModelRegistrySection {
            default: AsrModel::Qwen3Asr06B,
            available: vec![AsrModel::Qwen3Asr06B, AsrModel::Qwen3Asr17B],
            idle_timeout,
            max_loaded,
            device: orchion::DevicePreference::Auto,
        }
    }

    #[tokio::test]
    async fn rejects_unavailable_model_without_loading() {
        let cache = ModelCache::<AsrModel, usize>::new(
            ModelRegistrySection {
                default: AsrModel::Qwen3Asr06B,
                available: vec![AsrModel::Qwen3Asr06B],
                idle_timeout: Duration::from_secs(60),
                max_loaded: 1,
                device: orchion::DevicePreference::Auto,
            },
            PathBuf::from("models"),
        );
        let loads = Arc::new(AtomicUsize::new(0));

        let result = cache
            .get_or_load(AsrModel::Qwen3Asr17B, |_, _| {
                let loads = Arc::clone(&loads);
                async move {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(1)
                }
            })
            .await
            .unwrap();

        assert_eq!(result, None);
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn returns_loaded_model_from_cache() {
        let cache = ModelCache::<AsrModel, usize>::new(
            registry(2, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let loads = Arc::new(AtomicUsize::new(0));

        let first = load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await;
        let second = load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await;

        assert_eq!(first, Some(1));
        assert_eq!(second, Some(1));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_model_when_full() {
        let cache = ModelCache::<AsrModel, usize>::new(
            registry(1, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(
            load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await,
            Some(1)
        );
        assert_eq!(
            load_counted(&cache, AsrModel::Qwen3Asr17B, &loads).await,
            Some(2)
        );
        assert_eq!(
            load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await,
            Some(3)
        );
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn cleanup_idle_unloads_inactive_models() {
        let cache = ModelCache::<AsrModel, usize>::new(
            registry(2, Duration::from_millis(1)),
            PathBuf::from("models"),
        );
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(
            load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await,
            Some(1)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
        cache.cleanup_idle().await;
        assert_eq!(
            load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await,
            Some(2)
        );
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    async fn load_counted(
        cache: &ModelCache<AsrModel, usize>,
        model: AsrModel,
        loads: &Arc<AtomicUsize>,
    ) -> Option<usize> {
        cache
            .get_or_load(model, |_, _| {
                let loads = Arc::clone(loads);
                async move { Ok(loads.fetch_add(1, Ordering::SeqCst) + 1) }
            })
            .await
            .unwrap()
    }
}
