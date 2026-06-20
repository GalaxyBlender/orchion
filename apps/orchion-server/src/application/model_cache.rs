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

#[derive(Clone, Copy)]
struct LoadedModelEntry<M> {
    model: M,
    last_used: Instant,
}

#[derive(Clone)]
pub struct GlobalModelCacheLimiter {
    max_loaded: usize,
    lock: Arc<Mutex<()>>,
}

impl GlobalModelCacheLimiter {
    pub fn new(max_loaded: usize) -> Self {
        Self {
            max_loaded,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn get_or_load<PrimaryM, PrimaryE, OtherM, OtherE, F, Fut>(
        &self,
        primary: &ModelCache<PrimaryM, PrimaryE>,
        other: &ModelCache<OtherM, OtherE>,
        model: PrimaryM,
        load: F,
    ) -> anyhow::Result<Option<PrimaryE>>
    where
        PrimaryM: ModelSpec + std::hash::Hash,
        PrimaryE: Clone,
        OtherM: ModelSpec + std::hash::Hash,
        OtherE: Clone,
        F: FnOnce(PrimaryM, PathBuf) -> Fut,
        Fut: Future<Output = anyhow::Result<PrimaryE>>,
    {
        if let Some(engine) = primary.get_loaded(model).await {
            return Ok(Some(engine));
        }

        let _guard = self.lock.lock().await;
        if let Some(engine) = primary.get_loaded(model).await {
            return Ok(Some(engine));
        }
        if !primary.is_available(model).await {
            return Ok(None);
        }
        self.evict_global_lru_before_load(primary, other).await;
        let engine = primary.get_or_load(model, load).await?;
        self.evict_global_lru(primary, other).await;
        Ok(engine)
    }

    async fn evict_global_lru_before_load<PrimaryM, PrimaryE, OtherM, OtherE>(
        &self,
        primary: &ModelCache<PrimaryM, PrimaryE>,
        other: &ModelCache<OtherM, OtherE>,
    ) where
        PrimaryM: ModelSpec + std::hash::Hash,
        PrimaryE: Clone,
        OtherM: ModelSpec + std::hash::Hash,
        OtherE: Clone,
    {
        while primary.loaded_len().await + other.loaded_len().await >= self.max_loaded {
            if !self.evict_global_lru_once(primary, other).await {
                break;
            }
        }
    }

    async fn evict_global_lru<PrimaryM, PrimaryE, OtherM, OtherE>(
        &self,
        primary: &ModelCache<PrimaryM, PrimaryE>,
        other: &ModelCache<OtherM, OtherE>,
    ) where
        PrimaryM: ModelSpec + std::hash::Hash,
        PrimaryE: Clone,
        OtherM: ModelSpec + std::hash::Hash,
        OtherE: Clone,
    {
        while primary.loaded_len().await + other.loaded_len().await > self.max_loaded {
            if !self.evict_global_lru_once(primary, other).await {
                break;
            }
        }
    }

    async fn evict_global_lru_once<PrimaryM, PrimaryE, OtherM, OtherE>(
        &self,
        primary: &ModelCache<PrimaryM, PrimaryE>,
        other: &ModelCache<OtherM, OtherE>,
    ) -> bool
    where
        PrimaryM: ModelSpec + std::hash::Hash,
        PrimaryE: Clone,
        OtherM: ModelSpec + std::hash::Hash,
        OtherE: Clone,
    {
        let primary_lru = primary
            .loaded_entries()
            .await
            .into_iter()
            .min_by_key(|entry| entry.last_used);
        let other_lru = other
            .loaded_entries()
            .await
            .into_iter()
            .min_by_key(|entry| entry.last_used);

        match (primary_lru, other_lru) {
            (Some(primary_entry), Some(other_entry))
                if primary_entry.last_used <= other_entry.last_used =>
            {
                primary.evict_loaded(primary_entry.model).await
            }
            (Some(_), Some(other_entry)) => other.evict_loaded(other_entry.model).await,
            (Some(primary_entry), None) => primary.evict_loaded(primary_entry.model).await,
            (None, Some(other_entry)) => other.evict_loaded(other_entry.model).await,
            (None, None) => false,
        }
    }
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

    async fn get_loaded(&self, model: M) -> Option<E> {
        let mut state = self.inner.lock().await;
        state.evict_idle(self.idle_timeout);
        let loaded = state.loaded.get_mut(&model)?;
        loaded.last_used = Instant::now();
        Some(loaded.engine.clone())
    }

    async fn is_available(&self, model: M) -> bool {
        self.inner.lock().await.available.contains(&model)
    }

    #[cfg(test)]
    pub async fn is_loaded(&self, model: M) -> bool {
        self.inner.lock().await.loaded.contains_key(&model)
    }

    async fn loaded_len(&self) -> usize {
        self.inner.lock().await.loaded.len()
    }

    async fn loaded_entries(&self) -> Vec<LoadedModelEntry<M>> {
        self.inner.lock().await.loaded_entries()
    }

    async fn evict_loaded(&self, model: M) -> bool {
        self.inner.lock().await.loaded.remove(&model).is_some()
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

    fn loaded_entries(&self) -> Vec<LoadedModelEntry<M>> {
        self.loaded
            .iter()
            .map(|(model, loaded)| LoadedModelEntry {
                model: *model,
                last_used: loaded.last_used,
            })
            .collect()
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

    fn tts_registry(max_loaded: usize, idle_timeout: Duration) -> ModelRegistrySection<TtsModel> {
        ModelRegistrySection {
            default: TtsModel::Qwen3Tts06BCustomVoice,
            available: vec![TtsModel::Qwen3Tts06BCustomVoice],
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

    #[tokio::test]
    async fn global_limiter_evicts_lru_across_model_categories() {
        let asr_cache = ModelCache::<AsrModel, usize>::new(
            registry(2, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let tts_cache = ModelCache::<TtsModel, usize>::new(
            tts_registry(2, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let limiter = GlobalModelCacheLimiter::new(1);
        let asr_loads = Arc::new(AtomicUsize::new(0));
        let tts_loads = Arc::new(AtomicUsize::new(0));

        let asr = limiter
            .get_or_load(&asr_cache, &tts_cache, AsrModel::Qwen3Asr06B, |_, _| {
                let asr_loads = Arc::clone(&asr_loads);
                async move { Ok(asr_loads.fetch_add(1, Ordering::SeqCst) + 1) }
            })
            .await
            .unwrap();

        assert_eq!(asr, Some(1));
        assert!(asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);

        let tts = limiter
            .get_or_load(
                &tts_cache,
                &asr_cache,
                TtsModel::Qwen3Tts06BCustomVoice,
                |_, _| {
                    let tts_loads = Arc::clone(&tts_loads);
                    async move { Ok(tts_loads.fetch_add(1, Ordering::SeqCst) + 1) }
                },
            )
            .await
            .unwrap();

        assert_eq!(tts, Some(1));
        assert!(!asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);
        assert!(tts_cache.is_loaded(TtsModel::Qwen3Tts06BCustomVoice).await);

        let asr = limiter
            .get_or_load(&asr_cache, &tts_cache, AsrModel::Qwen3Asr06B, |_, _| {
                let asr_loads = Arc::clone(&asr_loads);
                async move { Ok(asr_loads.fetch_add(1, Ordering::SeqCst) + 1) }
            })
            .await
            .unwrap();

        assert_eq!(asr, Some(2));
        assert!(asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);
        assert!(!tts_cache.is_loaded(TtsModel::Qwen3Tts06BCustomVoice).await);
        assert_eq!(asr_loads.load(Ordering::SeqCst), 2);
        assert_eq!(tts_loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn global_limiter_returns_loaded_model_without_waiting_for_cold_load() {
        let asr_cache = ModelCache::<AsrModel, usize>::new(
            registry(2, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let tts_cache = ModelCache::<TtsModel, usize>::new(
            tts_registry(2, Duration::from_secs(60)),
            PathBuf::from("models"),
        );
        let limiter = GlobalModelCacheLimiter::new(2);
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(
            limiter
                .get_or_load(&asr_cache, &tts_cache, AsrModel::Qwen3Asr06B, |_, _| {
                    let loads = Arc::clone(&loads);
                    async move { Ok(loads.fetch_add(1, Ordering::SeqCst) + 1) }
                })
                .await
                .unwrap(),
            Some(1)
        );

        let cold_limiter = limiter.clone();
        let cold_tts_cache = tts_cache.clone();
        let cold_asr_cache = asr_cache.clone();
        let cold_load = tokio::spawn(async move {
            cold_limiter
                .get_or_load(
                    &cold_tts_cache,
                    &cold_asr_cache,
                    TtsModel::Qwen3Tts06BCustomVoice,
                    |_, _| async move {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        Ok(10)
                    },
                )
                .await
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let start = Instant::now();
        let cached = limiter
            .get_or_load(
                &asr_cache,
                &tts_cache,
                AsrModel::Qwen3Asr06B,
                |_, _| async { Ok(99) },
            )
            .await
            .unwrap();

        assert_eq!(cached, Some(1));
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(cold_load.await.unwrap().unwrap(), Some(10));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
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
