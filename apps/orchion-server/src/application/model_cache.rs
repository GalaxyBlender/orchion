use orchion::{Asr, AsrModel, ModelDownloader, ModelSpec, Ocr, Tts, TtsModel};
use orchion_core::KnownOcrModel;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub(crate) trait CacheTracker: Send + Sync {
    fn loaded_len(&self) -> BoxFuture<'_, usize>;
    fn lru_entry(&self) -> BoxFuture<'_, Option<TrackedLoadedModel>>;
    fn evict_tracked(&self, key: String) -> BoxFuture<'_, bool>;
    fn cache_id(&self) -> &'static str;
}

pub(crate) trait CacheTrackerSet<'a> {
    fn into_trackers(self, target: &'a dyn CacheTracker) -> Vec<&'a dyn CacheTracker>;
}

#[derive(Debug, Clone)]
pub(crate) struct TrackedLoadedModel {
    cache_id: &'static str,
    key: String,
    last_used: Instant,
}

#[derive(Clone)]
pub struct ModelCache<M, E> {
    inner: Arc<Mutex<ModelCacheState<M, E>>>,
    cache_id: &'static str,
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

    pub(crate) async fn get_or_load<'a, M, E, C, F, Fut>(
        &self,
        target: &'a ModelCache<M, E>,
        all_caches: C,
        model: M,
        load: F,
    ) -> anyhow::Result<Option<E>>
    where
        M: ModelSpec + std::hash::Hash,
        E: Clone + Send,
        C: CacheTrackerSet<'a>,
        F: FnOnce(M, PathBuf) -> Fut,
        Fut: Future<Output = anyhow::Result<E>>,
    {
        if let Some(engine) = target.get_loaded(model).await {
            return Ok(Some(engine));
        }

        let _guard = self.lock.lock().await;
        if let Some(engine) = target.get_loaded(model).await {
            return Ok(Some(engine));
        }
        if !target.is_available(model).await {
            return Ok(None);
        }
        let all_caches = all_caches.into_trackers(target);
        validate_unique_cache_ids(all_caches.as_slice())?;
        self.evict_global_lru_before_load(all_caches.as_slice())
            .await;
        let engine = target.get_or_load(model, load).await?;
        self.evict_global_lru(all_caches.as_slice()).await;
        Ok(engine)
    }

    async fn evict_global_lru_before_load(&self, all_caches: &[&dyn CacheTracker]) {
        while loaded_len(all_caches).await >= self.max_loaded {
            if !self.evict_global_lru_once(all_caches).await {
                break;
            }
        }
    }

    async fn evict_global_lru(&self, all_caches: &[&dyn CacheTracker]) {
        while loaded_len(all_caches).await > self.max_loaded {
            if !self.evict_global_lru_once(all_caches).await {
                break;
            }
        }
    }

    async fn evict_global_lru_once(&self, all_caches: &[&dyn CacheTracker]) -> bool {
        let Some(lru) = lru_entry(all_caches).await else {
            return false;
        };
        for cache in all_caches {
            if cache.cache_id() == lru.cache_id {
                return cache.evict_tracked(lru.key).await;
            }
        }
        false
    }
}

impl<'a> CacheTrackerSet<'a> for &'a [&'a dyn CacheTracker] {
    fn into_trackers(self, _target: &'a dyn CacheTracker) -> Vec<&'a dyn CacheTracker> {
        self.to_vec()
    }
}

impl<'a, const N: usize> CacheTrackerSet<'a> for &'a [&'a dyn CacheTracker; N] {
    fn into_trackers(self, _target: &'a dyn CacheTracker) -> Vec<&'a dyn CacheTracker> {
        self.to_vec()
    }
}

impl<'a, M, E> CacheTrackerSet<'a> for &'a ModelCache<M, E>
where
    M: ModelSpec + std::hash::Hash,
    E: Clone + Send,
{
    fn into_trackers(self, target: &'a dyn CacheTracker) -> Vec<&'a dyn CacheTracker> {
        vec![target, self]
    }
}

async fn loaded_len(all_caches: &[&dyn CacheTracker]) -> usize {
    let mut total = 0;
    for cache in all_caches {
        total += cache.loaded_len().await;
    }
    total
}

fn validate_unique_cache_ids(all_caches: &[&dyn CacheTracker]) -> anyhow::Result<()> {
    let mut cache_ids = HashSet::new();
    for cache in all_caches {
        let cache_id = cache.cache_id();
        if !cache_ids.insert(cache_id) {
            anyhow::bail!("duplicate model cache id `{cache_id}`");
        }
    }
    Ok(())
}

async fn lru_entry(all_caches: &[&dyn CacheTracker]) -> Option<TrackedLoadedModel> {
    let mut lru = None;
    for cache in all_caches {
        let Some(entry) = cache.lru_entry().await else {
            continue;
        };
        if lru
            .as_ref()
            .is_none_or(|current: &TrackedLoadedModel| entry.last_used < current.last_used)
        {
            lru = Some(entry);
        }
    }
    lru
}

impl<M, E> ModelCache<M, E>
where
    M: ModelSpec + std::hash::Hash,
    E: Clone,
{
    pub fn new(
        cache_id: &'static str,
        available_models: Vec<M>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelCacheState {
                available: available_models,
                loaded: HashMap::new(),
                loading: HashMap::new(),
            })),
            cache_id,
            dir,
            idle_timeout,
            max_loaded,
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
            self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
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
            self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
            if let Some(loaded) = state.loaded.get_mut(&model) {
                loaded.last_used = Instant::now();
                return Ok(Some(loaded.engine.clone()));
            }
        }

        let engine = load(model, model.cache_path(&self.dir)).await?;
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
        self.log_unloaded_models(state.evict_lru_until_below(self.max_loaded), "cache limit");
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
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
    }

    async fn get_loaded(&self, model: M) -> Option<E> {
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
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

    pub const fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }

    const fn cache_id(&self) -> &'static str {
        self.cache_id
    }

    fn log_unloaded_models(&self, models: Vec<M>, reason: &'static str) {
        for model in models {
            tracing::info!(cache = self.cache_id, model = ?model, reason, "unloading model");
        }
    }
}

impl<M, E> CacheTracker for ModelCache<M, E>
where
    M: ModelSpec + std::hash::Hash,
    E: Clone + Send,
{
    fn loaded_len(&self) -> BoxFuture<'_, usize> {
        Box::pin(async move { self.inner.lock().await.loaded.len() })
    }

    fn lru_entry(&self) -> BoxFuture<'_, Option<TrackedLoadedModel>> {
        Box::pin(async move {
            self.inner
                .lock()
                .await
                .loaded
                .iter()
                .min_by_key(|(_, loaded)| loaded.last_used)
                .map(|(model, loaded)| TrackedLoadedModel {
                    cache_id: self.cache_id,
                    key: model.cache_key().to_string(),
                    last_used: loaded.last_used,
                })
        })
    }

    fn evict_tracked(&self, key: String) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            let mut state = self.inner.lock().await;
            let Some(model) = state
                .loaded
                .keys()
                .find(|model| model.cache_key() == key)
                .copied()
            else {
                return false;
            };
            let removed = state.loaded.remove(&model).is_some();
            if removed {
                tracing::info!(
                    cache = self.cache_id,
                    model = ?model,
                    reason = "global cache limit",
                    "unloading model"
                );
            }
            removed
        })
    }

    fn cache_id(&self) -> &'static str {
        self.cache_id()
    }
}

impl<M, E> ModelCacheState<M, E>
where
    M: Copy + Eq + std::hash::Hash,
{
    fn evict_idle(&mut self, idle_timeout: Duration) -> Vec<M> {
        let now = Instant::now();
        let evicted = self
            .loaded
            .iter()
            .filter_map(|(model, loaded)| {
                (now.duration_since(loaded.last_used) >= idle_timeout).then_some(*model)
            })
            .collect::<Vec<_>>();
        for model in &evicted {
            self.loaded.remove(model);
        }
        evicted
    }

    fn evict_lru_until_below(&mut self, max_loaded: usize) -> Vec<M> {
        let mut evicted = Vec::new();
        while self.loaded.len() >= max_loaded {
            let Some(model) = self
                .loaded
                .iter()
                .min_by_key(|(_, loaded)| loaded.last_used)
                .map(|(model, _)| *model)
            else {
                break;
            };
            if self.loaded.remove(&model).is_some() {
                evicted.push(model);
            }
        }
        evicted
    }
}

pub type AsrModelCache = ModelCache<AsrModel, Asr>;
pub type TtsModelCache = ModelCache<TtsModel, Tts>;
pub type OcrModelCache = ModelCache<KnownOcrModel, Ocr>;
pub type OcrVlModelCache = ModelCache<KnownOcrModel, Ocr>;

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
    use orchion_core::KnownOcrModel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn asr_cache(max_loaded: usize, idle_timeout: Duration) -> ModelCache<AsrModel, usize> {
        ModelCache::new(
            "asr",
            vec![AsrModel::Qwen3Asr06B, AsrModel::Qwen3Asr17B],
            idle_timeout,
            max_loaded,
            PathBuf::from("models"),
        )
    }

    fn tts_cache(max_loaded: usize, idle_timeout: Duration) -> ModelCache<TtsModel, usize> {
        ModelCache::new(
            "tts",
            vec![TtsModel::Qwen3Tts06BCustomVoice],
            idle_timeout,
            max_loaded,
            PathBuf::from("models"),
        )
    }

    fn ocr_cache(max_loaded: usize, idle_timeout: Duration) -> ModelCache<KnownOcrModel, usize> {
        ModelCache::new(
            "ocr",
            vec![KnownOcrModel::PpOcrV6Tiny],
            idle_timeout,
            max_loaded,
            PathBuf::from("models"),
        )
    }

    #[tokio::test]
    async fn rejects_unavailable_model_without_loading() {
        let cache = ModelCache::<AsrModel, usize>::new(
            "asr",
            vec![AsrModel::Qwen3Asr06B],
            Duration::from_secs(60),
            1,
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
        let cache = asr_cache(2, Duration::from_secs(60));
        let loads = Arc::new(AtomicUsize::new(0));

        let first = load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await;
        let second = load_counted(&cache, AsrModel::Qwen3Asr06B, &loads).await;

        assert_eq!(first, Some(1));
        assert_eq!(second, Some(1));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_model_when_full() {
        let cache = asr_cache(1, Duration::from_secs(60));
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
        let cache = asr_cache(2, Duration::from_millis(1));
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

    #[test]
    fn evict_idle_returns_unloaded_models() {
        let model = AsrModel::Qwen3Asr06B;
        let mut state = ModelCacheState {
            available: vec![model],
            loaded: HashMap::from([(
                model,
                LoadedModel {
                    engine: 1,
                    last_used: Instant::now() - Duration::from_secs(10),
                },
            )]),
            loading: HashMap::new(),
        };

        assert_eq!(state.evict_idle(Duration::from_secs(1)), vec![model]);
        assert!(state.loaded.is_empty());
    }

    #[tokio::test]
    async fn global_limiter_evicts_lru_across_model_categories() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(1);
        let asr_loads = Arc::new(AtomicUsize::new(0));
        let tts_loads = Arc::new(AtomicUsize::new(0));

        let asr = limiter
            .get_or_load(&asr_cache, &all_caches, AsrModel::Qwen3Asr06B, |_, _| {
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
                &all_caches,
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
            .get_or_load(&asr_cache, &all_caches, AsrModel::Qwen3Asr06B, |_, _| {
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
    async fn global_limiter_evicts_lru_across_three_model_categories() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let ocr_cache = ocr_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 3] = [&asr_cache, &tts_cache, &ocr_cache];
        let limiter = GlobalModelCacheLimiter::new(2);

        let asr = limiter
            .get_or_load(
                &asr_cache,
                &all_caches,
                AsrModel::Qwen3Asr06B,
                |_, _| async { Ok(1) },
            )
            .await
            .unwrap();

        assert_eq!(asr, Some(1));
        assert!(asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);

        let tts = limiter
            .get_or_load(
                &tts_cache,
                &all_caches,
                TtsModel::Qwen3Tts06BCustomVoice,
                |_, _| async { Ok(2) },
            )
            .await
            .unwrap();

        assert_eq!(tts, Some(2));
        assert!(asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);
        assert!(tts_cache.is_loaded(TtsModel::Qwen3Tts06BCustomVoice).await);

        let ocr = limiter
            .get_or_load(
                &ocr_cache,
                &all_caches,
                KnownOcrModel::PpOcrV6Tiny,
                |_, _| async { Ok(3) },
            )
            .await
            .unwrap();

        assert_eq!(ocr, Some(3));
        assert!(!asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);
        assert!(tts_cache.is_loaded(TtsModel::Qwen3Tts06BCustomVoice).await);
        assert!(ocr_cache.is_loaded(KnownOcrModel::PpOcrV6Tiny).await);
    }

    #[tokio::test]
    async fn global_limiter_rejects_duplicate_cache_ids() {
        let asr_cache = ModelCache::<AsrModel, usize>::new(
            "models",
            vec![AsrModel::Qwen3Asr06B],
            Duration::from_secs(60),
            2,
            PathBuf::from("models"),
        );
        let tts_cache = ModelCache::<TtsModel, usize>::new(
            "models",
            vec![TtsModel::Qwen3Tts06BCustomVoice],
            Duration::from_secs(60),
            2,
            PathBuf::from("models"),
        );
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(2);

        let error = limiter
            .get_or_load(
                &asr_cache,
                &all_caches,
                AsrModel::Qwen3Asr06B,
                |_, _| async { Ok(1) },
            )
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("duplicate model cache id"),
            "unexpected error: {error:#}"
        );
        assert!(!asr_cache.is_loaded(AsrModel::Qwen3Asr06B).await);
    }

    #[tokio::test]
    async fn global_limiter_returns_loaded_model_without_waiting_for_cold_load() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(2);
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(
            limiter
                .get_or_load(&asr_cache, &all_caches, AsrModel::Qwen3Asr06B, |_, _| {
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
            let all_caches: [&dyn CacheTracker; 2] = [&cold_asr_cache, &cold_tts_cache];
            cold_limiter
                .get_or_load(
                    &cold_tts_cache,
                    &all_caches,
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
                &all_caches,
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
