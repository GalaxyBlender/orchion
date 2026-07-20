use orchion::{Asr, AsrModel, KnownOcrModel, ModelDownloader, ModelSpec, Ocr, Tts, TtsModel};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::ops::Deref;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    active_leases: Arc<AtomicUsize>,
}

#[must_use = "the model lease must be held while the model is in use"]
pub struct ModelLease<E> {
    engine: E,
    active_leases: Arc<AtomicUsize>,
}

impl<E> ModelLease<E> {
    fn new(engine: E, active_leases: Arc<AtomicUsize>) -> Self {
        active_leases.fetch_add(1, Ordering::SeqCst);
        Self {
            engine,
            active_leases,
        }
    }
}

impl<E> ModelLease<E>
where
    E: Clone + Send + Sync + 'static,
{
    pub async fn run<T, F, Fut>(&self, operation: F) -> Result<T, tokio::task::JoinError>
    where
        T: Send + 'static,
        F: FnOnce(ModelLease<E>) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        let lease = self.clone();
        tokio::spawn(async move { operation(lease).await }).await
    }
}

impl<E: Clone> Clone for ModelLease<E> {
    fn clone(&self) -> Self {
        Self::new(self.engine.clone(), Arc::clone(&self.active_leases))
    }
}

impl<E> Deref for ModelLease<E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl<E: std::fmt::Debug> std::fmt::Debug for ModelLease<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.engine.fmt(formatter)
    }
}

impl<E> Drop for ModelLease<E> {
    fn drop(&mut self) {
        let previous = self.active_leases.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(previous > 0, "model lease count underflowed");
    }
}

impl<E: Clone> LoadedModel<E> {
    fn lease(&self) -> ModelLease<E> {
        ModelLease::new(self.engine.clone(), Arc::clone(&self.active_leases))
    }
}

impl<E> LoadedModel<E> {
    fn is_active(&self) -> bool {
        self.active_leases.load(Ordering::SeqCst) > 0
    }
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
    ) -> anyhow::Result<Option<ModelLease<E>>>
    where
        M: ModelSpec + std::hash::Hash,
        E: Clone + Send,
        C: CacheTrackerSet<'a>,
        F: FnOnce(M, PathBuf) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<E>> + Send + 'static,
        E: 'static,
    {
        if let Some(engine) = target.get_loaded(&model).await {
            return Ok(Some(engine));
        }

        let guard = Arc::clone(&self.lock).lock_owned().await;
        if let Some(engine) = target.get_loaded(&model).await {
            return Ok(Some(engine));
        }
        if !target.is_available(&model).await {
            return Ok(None);
        }
        let all_caches = all_caches.into_trackers(target);
        validate_unique_cache_ids(all_caches.as_slice())?;
        self.evict_global_lru_before_load(all_caches.as_slice())
            .await?;
        let target = target.clone();
        tokio::spawn(async move {
            let _guard = guard;
            target.get_or_load(model, load).await
        })
        .await
        .map_err(|error| anyhow::anyhow!("model load task failed: {error}"))?
    }

    async fn evict_global_lru_before_load(
        &self,
        all_caches: &[&dyn CacheTracker],
    ) -> anyhow::Result<()> {
        while loaded_len(all_caches).await >= self.max_loaded {
            if !self.evict_global_lru_once(all_caches).await {
                anyhow::bail!(
                    "global model cache capacity of {} is occupied by active models",
                    self.max_loaded
                );
            }
        }
        Ok(())
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

    pub async fn get_or_load<F, Fut>(
        &self,
        model: M,
        load: F,
    ) -> anyhow::Result<Option<ModelLease<E>>>
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
                return Ok(Some(loaded.lease()));
            }
            Arc::clone(
                state
                    .loading
                    .entry(model.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };

        let _load_guard = loading.lock().await;
        {
            let mut state = self.inner.lock().await;
            self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
            if let Some(loaded) = state.loaded.get_mut(&model) {
                loaded.last_used = Instant::now();
                return Ok(Some(loaded.lease()));
            }
            self.log_unloaded_models(state.evict_lru_until_below(self.max_loaded), "cache limit");
            if state.loaded.len() >= self.max_loaded {
                anyhow::bail!(
                    "model cache `{}` capacity of {} is occupied by active models",
                    self.cache_id,
                    self.max_loaded
                );
            }
        }

        let engine = load(model.clone(), model.cache_path(&self.dir)).await?;
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
        self.log_unloaded_models(state.evict_lru_until_below(self.max_loaded), "cache limit");
        if state.loaded.len() >= self.max_loaded {
            anyhow::bail!(
                "model cache `{}` capacity of {} is occupied by active models",
                self.cache_id,
                self.max_loaded
            );
        }
        let lease = state
            .loaded
            .entry(model)
            .or_insert_with(|| LoadedModel {
                engine,
                last_used: Instant::now(),
                active_leases: Arc::new(AtomicUsize::new(0)),
            })
            .lease();
        Ok(Some(lease))
    }

    pub async fn cleanup_idle(&self) {
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
    }

    async fn get_loaded(&self, model: &M) -> Option<ModelLease<E>> {
        let mut state = self.inner.lock().await;
        self.log_unloaded_models(state.evict_idle(self.idle_timeout), "idle timeout");
        let loaded = state.loaded.get_mut(model)?;
        loaded.last_used = Instant::now();
        Some(loaded.lease())
    }

    async fn is_available(&self, model: &M) -> bool {
        self.inner.lock().await.available.contains(model)
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
                .filter(|(_, loaded)| !loaded.is_active())
                .min_by_key(|(_, loaded)| loaded.last_used)
                .map(|(model, loaded)| TrackedLoadedModel {
                    cache_id: self.cache_id,
                    key: model.huggingface_repo().to_string(),
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
                .find(|model| model.huggingface_repo() == key)
                .cloned()
            else {
                return false;
            };
            if state.loaded.get(&model).is_some_and(LoadedModel::is_active) {
                return false;
            }
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
    M: Clone + Eq + std::hash::Hash,
{
    fn evict_idle(&mut self, idle_timeout: Duration) -> Vec<M> {
        let now = Instant::now();
        let evicted = self
            .loaded
            .iter()
            .filter_map(|(model, loaded)| {
                (!loaded.is_active() && now.duration_since(loaded.last_used) >= idle_timeout)
                    .then_some(model.clone())
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
                .filter(|(_, loaded)| !loaded.is_active())
                .min_by_key(|(_, loaded)| loaded.last_used)
                .map(|(model, _)| model.clone())
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
        let path = downloader.download(model.clone(), dir).await?;
        tracing::debug!(model = ?model, path = %path.display(), "{label} model cache ready");
    }
    Ok(models.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use orchion::KnownOcrModel;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    fn asr_model(value: &str) -> AsrModel {
        AsrModel::parse(value).unwrap()
    }

    fn tts_model(value: &str) -> TtsModel {
        TtsModel::parse(value).unwrap()
    }

    fn qwen_asr_06b() -> AsrModel {
        asr_model("Qwen/Qwen3-ASR-0.6B")
    }

    fn qwen_asr_17b() -> AsrModel {
        asr_model("Qwen/Qwen3-ASR-1.7B")
    }

    fn qwen_tts_custom_voice() -> TtsModel {
        tts_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    }

    fn asr_cache(max_loaded: usize, idle_timeout: Duration) -> ModelCache<AsrModel, usize> {
        ModelCache::new(
            "asr",
            vec![qwen_asr_06b(), qwen_asr_17b()],
            idle_timeout,
            max_loaded,
            PathBuf::from("models"),
        )
    }

    fn tts_cache(max_loaded: usize, idle_timeout: Duration) -> ModelCache<TtsModel, usize> {
        ModelCache::new(
            "tts",
            vec![qwen_tts_custom_voice()],
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
            vec![qwen_asr_06b()],
            Duration::from_secs(60),
            1,
            PathBuf::from("models"),
        );
        let loads = Arc::new(AtomicUsize::new(0));

        let result = cache
            .get_or_load(qwen_asr_17b(), |_, _| {
                let loads = Arc::clone(&loads);
                async move {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(1)
                }
            })
            .await
            .unwrap();

        assert!(result.is_none());
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn returns_loaded_model_from_cache() {
        let cache = asr_cache(2, Duration::from_secs(60));
        let loads = Arc::new(AtomicUsize::new(0));

        let first = load_counted(&cache, qwen_asr_06b(), &loads).await;
        let second = load_counted(&cache, qwen_asr_06b(), &loads).await;

        assert_eq!(first, Some(1));
        assert_eq!(second, Some(1));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_model_when_full() {
        let cache = asr_cache(1, Duration::from_secs(60));
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(1));
        assert_eq!(load_counted(&cache, qwen_asr_17b(), &loads).await, Some(2));
        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(3));
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn cleanup_idle_unloads_inactive_models() {
        let cache = asr_cache(2, Duration::from_millis(1));
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(1));
        tokio::time::sleep(Duration::from_millis(5)).await;
        cache.cleanup_idle().await;
        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(2));
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn evict_idle_returns_unloaded_models() {
        let model = qwen_asr_06b();
        let mut state = ModelCacheState {
            available: vec![model.clone()],
            loaded: HashMap::from([(
                model.clone(),
                LoadedModel {
                    engine: 1,
                    last_used: Instant::now() - Duration::from_secs(10),
                    active_leases: Arc::new(AtomicUsize::new(0)),
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

        let current_asr_loads = Arc::clone(&asr_loads);
        let asr = limiter
            .get_or_load(
                &asr_cache,
                &all_caches,
                qwen_asr_06b(),
                move |_, _| async move { Ok(current_asr_loads.fetch_add(1, Ordering::SeqCst) + 1) },
            )
            .await
            .unwrap();

        assert_eq!(asr.as_deref(), Some(&1));
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        drop(asr);

        let current_tts_loads = Arc::clone(&tts_loads);
        let tts = limiter
            .get_or_load(
                &tts_cache,
                &all_caches,
                qwen_tts_custom_voice(),
                move |_, _| async move { Ok(current_tts_loads.fetch_add(1, Ordering::SeqCst) + 1) },
            )
            .await
            .unwrap();

        assert_eq!(tts.as_deref(), Some(&1));
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
        drop(tts);

        let current_asr_loads = Arc::clone(&asr_loads);
        let asr = limiter
            .get_or_load(
                &asr_cache,
                &all_caches,
                qwen_asr_06b(),
                move |_, _| async move { Ok(current_asr_loads.fetch_add(1, Ordering::SeqCst) + 1) },
            )
            .await
            .unwrap();

        assert_eq!(asr.as_deref(), Some(&2));
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(!tts_cache.is_loaded(qwen_tts_custom_voice()).await);
        assert_eq!(asr_loads.load(Ordering::SeqCst), 2);
        assert_eq!(tts_loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn global_limiter_does_not_evict_an_active_model() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(1);

        let active_asr = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();

        let error = limiter
            .get_or_load(
                &tts_cache,
                &all_caches,
                qwen_tts_custom_voice(),
                |_, _| async { Ok(2) },
            )
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("model cache capacity"),
            "unexpected error: {error:#}"
        );
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(!tts_cache.is_loaded(qwen_tts_custom_voice()).await);

        drop(active_asr);
        assert!(
            limiter
                .get_or_load(
                    &tts_cache,
                    &all_caches,
                    qwen_tts_custom_voice(),
                    |_, _| async { Ok(2) },
                )
                .await
                .unwrap()
                .is_some()
        );
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
    }

    #[tokio::test]
    async fn detached_model_operation_keeps_lease_after_waiter_is_cancelled() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(1);
        let active_asr = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let completed = Arc::new(Notify::new());

        let waiter = tokio::spawn({
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let completed = Arc::clone(&completed);
            async move {
                active_asr
                    .run(move |lease| async move {
                        started.notify_one();
                        release.notified().await;
                        drop(lease);
                        completed.notify_one();
                    })
                    .await
            }
        });
        started.notified().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());

        let error = limiter
            .get_or_load(
                &tts_cache,
                &all_caches,
                qwen_tts_custom_voice(),
                |_, _| async { Ok(2) },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("model cache capacity"));

        release.notify_one();
        completed.notified().await;
        assert!(
            limiter
                .get_or_load(
                    &tts_cache,
                    &all_caches,
                    qwen_tts_custom_voice(),
                    |_, _| async { Ok(2) },
                )
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn cancelled_cold_load_waiter_does_not_release_global_capacity() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let limiter = GlobalModelCacheLimiter::new(1);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let loads = Arc::new(AtomicUsize::new(0));

        let waiter = tokio::spawn({
            let limiter = limiter.clone();
            let asr_cache = asr_cache.clone();
            let tts_cache = tts_cache.clone();
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let loads = Arc::clone(&loads);
            async move {
                let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
                limiter
                    .get_or_load(
                        &asr_cache,
                        &all_caches,
                        qwen_asr_06b(),
                        move |_, _| async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            started.notify_one();
                            release.notified().await;
                            Ok(1)
                        },
                    )
                    .await
            }
        });
        started.notified().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());

        let blocked_tts_load = async {
            let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
            limiter
                .get_or_load(
                    &tts_cache,
                    &all_caches,
                    qwen_tts_custom_voice(),
                    |_, _| async { Ok(2) },
                )
                .await
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(20), blocked_tts_load)
                .await
                .is_err()
        );
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !asr_cache.is_loaded(qwen_asr_06b()).await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn global_limiter_evicts_lru_across_three_model_categories() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let ocr_cache = ocr_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 3] = [&asr_cache, &tts_cache, &ocr_cache];
        let limiter = GlobalModelCacheLimiter::new(2);

        let asr = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap();

        assert_eq!(asr.as_deref(), Some(&1));
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        drop(asr);

        let tts = limiter
            .get_or_load(
                &tts_cache,
                &all_caches,
                qwen_tts_custom_voice(),
                |_, _| async { Ok(2) },
            )
            .await
            .unwrap();

        assert_eq!(tts.as_deref(), Some(&2));
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
        drop(tts);

        let ocr = limiter
            .get_or_load(
                &ocr_cache,
                &all_caches,
                KnownOcrModel::PpOcrV6Tiny,
                |_, _| async { Ok(3) },
            )
            .await
            .unwrap();

        assert_eq!(ocr.as_deref(), Some(&3));
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
        assert!(ocr_cache.is_loaded(KnownOcrModel::PpOcrV6Tiny).await);
    }

    #[tokio::test]
    async fn global_limiter_rejects_duplicate_cache_ids() {
        let asr_cache = ModelCache::<AsrModel, usize>::new(
            "models",
            vec![qwen_asr_06b()],
            Duration::from_secs(60),
            2,
            PathBuf::from("models"),
        );
        let tts_cache = ModelCache::<TtsModel, usize>::new(
            "models",
            vec![qwen_tts_custom_voice()],
            Duration::from_secs(60),
            2,
            PathBuf::from("models"),
        );
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(2);

        let error = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("duplicate model cache id"),
            "unexpected error: {error:#}"
        );
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
    }

    #[tokio::test]
    async fn global_limiter_returns_loaded_model_without_waiting_for_cold_load() {
        let asr_cache = asr_cache(2, Duration::from_secs(60));
        let tts_cache = tts_cache(2, Duration::from_secs(60));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(2);
        let loads = Arc::new(AtomicUsize::new(0));

        let current_loads = Arc::clone(&loads);
        assert_eq!(
            limiter
                .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), move |_, _| {
                    async move { Ok(current_loads.fetch_add(1, Ordering::SeqCst) + 1) }
                })
                .await
                .unwrap()
                .as_deref(),
            Some(&1)
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
                    qwen_tts_custom_voice(),
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
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(99)
            })
            .await
            .unwrap();

        assert_eq!(cached.as_deref(), Some(&1));
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(cold_load.await.unwrap().unwrap().as_deref(), Some(&10));
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
            .map(|lease| *lease)
    }
}
