use super::mark_owned_operation_dispatched;
use super::resource_policy::InferenceLimiter;
use orchion::{
    Asr, AsrModel, DeploymentSourcePlan, ModelDownloader, ModelSpec, ModelUrl, Ocr, OcrModel, Tts,
    TtsModel,
};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::ops::Deref;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, Semaphore, watch};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub type ModelProvisionFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<PathBuf>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvisioning {
    pub model_url: ModelUrl,
    pub source_intent: String,
    pub source_plan: Option<DeploymentSourcePlan>,
}

pub trait ModelProvisioner<M>: Send + Sync {
    fn provision(
        &self,
        model: M,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_>;
}

pub trait ModelCacheKey: Clone + std::fmt::Debug + Eq + Send + Sync + 'static {
    fn cache_path(&self, cache_dir: &std::path::Path) -> PathBuf;

    fn resolve_without_provisioner(
        &self,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        let path = self.cache_path(&models_dir);
        Box::pin(async move {
            if provisioning.is_some() {
                anyhow::bail!("this cache key does not support repository provisioning")
            }
            Ok(path)
        })
    }
}

impl<M: ModelSpec> ModelCacheKey for M {
    fn cache_path(&self, cache_dir: &std::path::Path) -> PathBuf {
        ModelSpec::cache_path(self, cache_dir)
    }

    fn resolve_without_provisioner(
        &self,
        provisioning: Option<ModelProvisioning>,
        models_dir: PathBuf,
    ) -> ModelProvisionFuture<'_> {
        let model = self.clone();
        Box::pin(async move {
            match provisioning {
                Some(provisioning) => match provisioning.source_plan.as_ref() {
                    Some(plan) => ModelDownloader::resolve_prepared_model_url_path_with_plan(
                        &model,
                        &provisioning.model_url,
                        &provisioning.source_intent,
                        plan,
                        &models_dir,
                    )
                    .map_err(anyhow::Error::from),
                    None => ModelDownloader::resolve_model_url_path(
                        &model,
                        &provisioning.model_url,
                        &provisioning.source_intent,
                        &models_dir,
                    )
                    .await
                    .map_err(anyhow::Error::from),
                },
                None => Ok(ModelSpec::cache_path(&model, &models_dir)),
            }
        })
    }
}

pub(crate) trait CacheTracker: Send + Sync {
    fn loaded_len(&self) -> BoxFuture<'_, usize>;
    fn lru_entry(&self) -> BoxFuture<'_, Option<TrackedLoadedModel>>;
    fn evict_tracked(&self, key: Box<dyn Any + Send + Sync>) -> BoxFuture<'_, bool>;
    fn cache_id(&self) -> &'static str;
    fn residency_domain(&self) -> ResidencyDomain;
    fn clone_tracker(&self) -> Arc<dyn CacheTracker>;
}

pub(crate) trait CacheTrackerSet {
    fn into_trackers(self, target: &dyn CacheTracker) -> Vec<Arc<dyn CacheTracker>>;
}

pub(crate) struct TrackedLoadedModel {
    cache_id: &'static str,
    key: Box<dyn Any + Send + Sync>,
    last_used: Instant,
}

#[derive(Clone)]
pub(crate) struct ResidencyDomain {
    version: Arc<watch::Sender<u64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelResidencyStatus {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

impl ResidencyDomain {
    pub(crate) fn new() -> Self {
        let (version, _) = watch::channel(0);
        Self {
            version: Arc::new(version),
        }
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }

    fn notify(&self) {
        self.version
            .send_modify(|version| *version = version.wrapping_add(1));
    }

    fn is_same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.version, &other.version)
    }
}

#[derive(Clone)]
pub struct ModelCache<M, E> {
    inner: Arc<Mutex<ModelCacheState<M, E>>>,
    capacity_lock: Arc<Mutex<()>>,
    pending_loads: Arc<StdMutex<HashSet<M>>>,
    cache_id: &'static str,
    dir: PathBuf,
    idle_timeout: Duration,
    max_loaded: usize,
    provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
    residency: ResidencyDomain,
}

struct ModelCacheState<M, E> {
    available: Vec<M>,
    provisioning: HashMap<M, ModelProvisioning>,
    loaded: HashMap<M, LoadedModel<E>>,
    loading: HashMap<M, Arc<Mutex<()>>>,
    draining: HashSet<M>,
    retiring: HashSet<M>,
    provisioned: HashMap<M, PathBuf>,
}

struct LoadedModel<E> {
    engine: E,
    last_used: Arc<StdMutex<Instant>>,
    active_leases: Arc<AtomicUsize>,
    run_permits: Arc<Semaphore>,
    residency: ResidencyDomain,
}

#[must_use = "the model lease must be held while the model is in use"]
pub struct ModelLease<E> {
    engine: E,
    last_used: Arc<StdMutex<Instant>>,
    active_leases: Arc<AtomicUsize>,
    run_permits: Arc<Semaphore>,
    residency: ResidencyDomain,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelOperationError {
    #[error("request was cancelled before model dispatch")]
    Cancelled,
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}

impl<E> ModelLease<E> {
    fn new(
        engine: E,
        last_used: Arc<StdMutex<Instant>>,
        active_leases: Arc<AtomicUsize>,
        run_permits: Arc<Semaphore>,
        residency: ResidencyDomain,
    ) -> Self {
        active_leases.fetch_add(1, Ordering::SeqCst);
        Self {
            engine,
            last_used,
            active_leases,
            run_permits,
            residency,
        }
    }
}

impl<E> ModelLease<E>
where
    E: Clone + Send + Sync + 'static,
{
    /// # Errors
    ///
    /// Returns [`ModelOperationError`] if the request is cancelled before dispatch or the owned
    /// operation task panics or is aborted.
    ///
    /// # Panics
    ///
    /// Panics if the model inference semaphore has been closed.
    pub async fn run<T, F, Fut>(&self, operation: F) -> Result<T, ModelOperationError>
    where
        T: Send + 'static,
        F: FnOnce(ModelLease<E>) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        let permit = Arc::clone(&self.run_permits)
            .acquire_owned()
            .await
            .expect("model inference semaphore must remain open");
        let lease = self.clone();
        if !mark_owned_operation_dispatched() {
            return Err(ModelOperationError::Cancelled);
        }
        Ok(tokio::spawn(async move {
            let _permit = permit;
            operation(lease).await
        })
        .await?)
    }

    pub(crate) async fn run_with_inference<T, F, Fut>(
        &self,
        inference: InferenceLimiter,
        operation: F,
    ) -> Result<T, ModelOperationError>
    where
        T: Send + 'static,
        F: FnOnce(ModelLease<E>) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        let model_permit = Arc::clone(&self.run_permits)
            .acquire_owned()
            .await
            .expect("model inference semaphore must remain open");
        let inference_permit = inference.acquire().await;
        let lease = self.clone();
        if !mark_owned_operation_dispatched() {
            return Err(ModelOperationError::Cancelled);
        }
        Ok(tokio::spawn(async move {
            let _model_permit = model_permit;
            let _inference_permit = inference_permit;
            operation(lease).await
        })
        .await?)
    }
}

impl<E: Clone> Clone for ModelLease<E> {
    fn clone(&self) -> Self {
        Self::new(
            self.engine.clone(),
            Arc::clone(&self.last_used),
            Arc::clone(&self.active_leases),
            Arc::clone(&self.run_permits),
            self.residency.clone(),
        )
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
        *self
            .last_used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
        let previous = self.active_leases.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(previous > 0, "model lease count underflowed");
        if previous == 1 {
            self.residency.notify();
        }
    }
}

impl<E: Clone> LoadedModel<E> {
    fn lease(&self) -> ModelLease<E> {
        ModelLease::new(
            self.engine.clone(),
            Arc::clone(&self.last_used),
            Arc::clone(&self.active_leases),
            Arc::clone(&self.run_permits),
            self.residency.clone(),
        )
    }
}

impl<E> LoadedModel<E> {
    fn touch(&self) {
        *self
            .last_used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
    }

    fn last_used(&self) -> Instant {
        *self
            .last_used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_active(&self) -> bool {
        self.active_leases.load(Ordering::SeqCst) > 0
    }
}

#[derive(Clone)]
pub struct GlobalModelCacheLimiter {
    max_loaded: usize,
    state: Arc<StdMutex<GlobalLimiterState>>,
    drained: Arc<Notify>,
    residency: ResidencyDomain,
}

#[derive(Default)]
struct GlobalLimiterState {
    closed: bool,
    cold_tasks: usize,
    reservations: usize,
}

struct ColdTaskGuard {
    limiter: GlobalModelCacheLimiter,
}

struct GlobalReservation {
    limiter: GlobalModelCacheLimiter,
}

struct LocalLoadReservation<M>
where
    M: Eq + std::hash::Hash,
{
    model: M,
    pending_loads: Arc<StdMutex<HashSet<M>>>,
    residency: ResidencyDomain,
}

impl GlobalModelCacheLimiter {
    #[must_use]
    pub fn new(max_loaded: usize) -> Self {
        Self::new_in_domain(max_loaded, ResidencyDomain::new())
    }

    pub(crate) fn new_in_domain(max_loaded: usize, residency: ResidencyDomain) -> Self {
        Self {
            max_loaded,
            state: Arc::new(StdMutex::new(GlobalLimiterState::default())),
            drained: Arc::new(Notify::new()),
            residency,
        }
    }

    pub(crate) async fn get_or_load<M, E, C, F, Fut>(
        &self,
        target: &ModelCache<M, E>,
        all_caches: C,
        model: M,
        load: F,
    ) -> anyhow::Result<Option<ModelLease<E>>>
    where
        M: ModelCacheKey + std::hash::Hash + 'static,
        E: Clone + Send + 'static,
        C: CacheTrackerSet,
        F: FnOnce(M, PathBuf) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<E>> + Send + 'static,
    {
        if let Some(engine) = target.get_loaded(&model).await {
            return Ok(Some(engine));
        }

        let all_caches = all_caches.into_trackers(target);
        validate_unique_cache_ids(all_caches.as_slice())?;
        self.validate_residency_domains(target, all_caches.as_slice())?;
        let task_guard = self.begin_cold_task()?;
        let limiter = self.clone();
        let target = target.clone();
        tokio::spawn(async move {
            let _task_guard = task_guard;
            if let Some(engine) = target.get_loaded(&model).await {
                return Ok(Some(engine));
            }
            if !target.is_available(&model).await {
                return Ok(None);
            }
            target
                .get_or_load_after(model, || limiter.reserve(all_caches.as_slice()), load)
                .await
        })
        .await
        .map_err(|error| anyhow::anyhow!("model load task failed: {error:#}"))?
    }

    fn begin_cold_task(&self) -> anyhow::Result<ColdTaskGuard> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            anyhow::bail!("global model cache limiter is closed");
        }
        state.cold_tasks += 1;
        Ok(ColdTaskGuard {
            limiter: self.clone(),
        })
    }

    async fn reserve(
        &self,
        all_caches: &[Arc<dyn CacheTracker>],
    ) -> anyhow::Result<GlobalReservation> {
        loop {
            let mut changes = self.residency.subscribe();
            let loaded = loaded_len(all_caches).await;
            {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if changes
                    .has_changed()
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?
                {
                    continue;
                }
                if loaded + state.reservations < self.max_loaded {
                    state.reservations += 1;
                    return Ok(GlobalReservation {
                        limiter: self.clone(),
                    });
                }
            }

            if self.evict_global_lru_once(all_caches).await {
                continue;
            }
            changes
                .changed()
                .await
                .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
        }
    }

    /// Fences new cold loads and waits for all cold-load tasks already accepted by this limiter.
    pub(crate) async fn close_and_drain(&self) {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
        }

        loop {
            let drained = self.drained.notified();
            tokio::pin!(drained);
            drained.as_mut().enable();
            if self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cold_tasks
                == 0
            {
                return;
            }
            drained.as_mut().await;
        }
    }

    async fn evict_global_lru_once(&self, all_caches: &[Arc<dyn CacheTracker>]) -> bool {
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

    fn validate_residency_domains<M, E>(
        &self,
        target: &ModelCache<M, E>,
        all_caches: &[Arc<dyn CacheTracker>],
    ) -> anyhow::Result<()> {
        if !self.residency.is_same(&target.residency)
            || all_caches
                .iter()
                .any(|cache| !self.residency.is_same(&cache.residency_domain()))
        {
            anyhow::bail!("global model cache limiter and caches must share a residency domain");
        }
        Ok(())
    }
}

impl Drop for ColdTaskGuard {
    fn drop(&mut self) {
        let mut state = self
            .limiter
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cold_tasks -= 1;
        let drained = state.cold_tasks == 0;
        drop(state);
        if drained {
            self.limiter.drained.notify_waiters();
        }
    }
}

impl Drop for GlobalReservation {
    fn drop(&mut self) {
        let mut state = self
            .limiter
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.reservations -= 1;
        drop(state);
        self.limiter.residency.notify();
    }
}

impl<M> Drop for LocalLoadReservation<M>
where
    M: Eq + std::hash::Hash,
{
    fn drop(&mut self) {
        self.pending_loads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.model);
        self.residency.notify();
    }
}

impl CacheTrackerSet for &[&dyn CacheTracker] {
    fn into_trackers(self, _target: &dyn CacheTracker) -> Vec<Arc<dyn CacheTracker>> {
        self.iter().map(|cache| cache.clone_tracker()).collect()
    }
}

impl<const N: usize> CacheTrackerSet for &[&dyn CacheTracker; N] {
    fn into_trackers(self, _target: &dyn CacheTracker) -> Vec<Arc<dyn CacheTracker>> {
        self.iter().map(|cache| cache.clone_tracker()).collect()
    }
}

impl<M, E> CacheTrackerSet for &ModelCache<M, E>
where
    M: ModelCacheKey + std::hash::Hash + 'static,
    E: Clone + Send + 'static,
{
    fn into_trackers(self, target: &dyn CacheTracker) -> Vec<Arc<dyn CacheTracker>> {
        vec![target.clone_tracker(), self.clone_tracker()]
    }
}

async fn loaded_len(all_caches: &[Arc<dyn CacheTracker>]) -> usize {
    let mut total = 0;
    for cache in all_caches {
        total += cache.loaded_len().await;
    }
    total
}

fn validate_unique_cache_ids(all_caches: &[Arc<dyn CacheTracker>]) -> anyhow::Result<()> {
    let mut cache_ids = HashSet::new();
    for cache in all_caches {
        let cache_id = cache.cache_id();
        if !cache_ids.insert(cache_id) {
            anyhow::bail!("duplicate model cache id `{cache_id}`");
        }
    }
    Ok(())
}

async fn lru_entry(all_caches: &[Arc<dyn CacheTracker>]) -> Option<TrackedLoadedModel> {
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
    M: ModelCacheKey + std::hash::Hash,
    E: Clone,
{
    #[must_use]
    pub fn new(
        cache_id: &'static str,
        available_models: Vec<M>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
    ) -> Self {
        Self::build(
            cache_id,
            available_models,
            HashMap::new(),
            idle_timeout,
            max_loaded,
            dir,
            None,
            ResidencyDomain::new(),
        )
    }

    #[must_use]
    pub fn new_with_provisioner<P>(
        cache_id: &'static str,
        available_models: Vec<M>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
        provisioner: Arc<P>,
    ) -> Self
    where
        P: ModelProvisioner<M> + 'static,
    {
        let provisioner: Arc<dyn ModelProvisioner<M>> = provisioner;
        Self::new_with_dyn_provisioner(
            cache_id,
            available_models,
            idle_timeout,
            max_loaded,
            dir,
            provisioner,
        )
    }

    pub(crate) fn new_with_dyn_provisioner(
        cache_id: &'static str,
        available_models: Vec<M>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
        provisioner: Arc<dyn ModelProvisioner<M>>,
    ) -> Self {
        Self::build(
            cache_id,
            available_models,
            HashMap::new(),
            idle_timeout,
            max_loaded,
            dir,
            Some(provisioner),
            ResidencyDomain::new(),
        )
    }

    pub(crate) fn new_in_domain(
        cache_id: &'static str,
        available_models: Vec<M>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
        provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
        residency: ResidencyDomain,
    ) -> Self {
        Self::build(
            cache_id,
            available_models,
            HashMap::new(),
            idle_timeout,
            max_loaded,
            dir,
            provisioner,
            residency,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "cache identity, policy, storage, provisioning, and residency are independent inputs"
    )]
    pub(crate) fn new_in_domain_with_provisioning(
        cache_id: &'static str,
        available_models: Vec<M>,
        provisioning: HashMap<M, ModelProvisioning>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
        provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
        residency: ResidencyDomain,
    ) -> Self {
        Self::build(
            cache_id,
            available_models,
            provisioning,
            idle_timeout,
            max_loaded,
            dir,
            provisioner,
            residency,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "centralizes construction for the public cache constructors"
    )]
    fn build(
        cache_id: &'static str,
        available_models: Vec<M>,
        provisioning: HashMap<M, ModelProvisioning>,
        idle_timeout: Duration,
        max_loaded: usize,
        dir: PathBuf,
        provisioner: Option<Arc<dyn ModelProvisioner<M>>>,
        residency: ResidencyDomain,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelCacheState {
                available: available_models,
                provisioning,
                loaded: HashMap::new(),
                loading: HashMap::new(),
                draining: HashSet::new(),
                retiring: HashSet::new(),
                provisioned: HashMap::new(),
            })),
            capacity_lock: Arc::new(Mutex::new(())),
            pending_loads: Arc::new(StdMutex::new(HashSet::new())),
            cache_id,
            dir,
            idle_timeout,
            max_loaded,
            provisioner,
            residency,
        }
    }

    /// # Errors
    ///
    /// Returns an error when model provisioning fails.
    pub async fn ensure_provisioned(&self, model: M) -> anyhow::Result<Option<PathBuf>> {
        let Some(loading) = self.loading_mutex(&model).await else {
            return Ok(None);
        };

        let _load_guard = loading.lock().await;
        self.provision_locked(model).await.map(Some)
    }

    /// # Errors
    ///
    /// Returns an error when provisioning, loading, or cache-capacity enforcement fails.
    pub async fn get_or_load<F, Fut>(
        &self,
        model: M,
        load: F,
    ) -> anyhow::Result<Option<ModelLease<E>>>
    where
        M: Send + 'static,
        E: Send + 'static,
        F: FnOnce(M, PathBuf) -> Fut,
        Fut: Future<Output = anyhow::Result<E>>,
    {
        self.get_or_load_after(model, || async { Ok(()) }, load)
            .await
    }

    async fn get_or_load_after<F, Fut, A, AFut, Admission>(
        &self,
        model: M,
        admission: A,
        load: F,
    ) -> anyhow::Result<Option<ModelLease<E>>>
    where
        M: Send + 'static,
        E: Send + 'static,
        F: FnOnce(M, PathBuf) -> Fut,
        Fut: Future<Output = anyhow::Result<E>>,
        A: FnOnce() -> AFut,
        AFut: Future<Output = anyhow::Result<Admission>>,
    {
        loop {
            self.cleanup_idle().await;
            let Some(loading) = self.loading_mutex(&model).await else {
                return Ok(None);
            };
            let load_guard = loading.lock_owned().await;
            let mut changes = self.residency.subscribe();
            {
                let mut state = self.inner.lock().await;
                if state.draining.contains(&model) || state.retiring.contains(&model) {
                    drop(state);
                    drop(load_guard);
                    changes
                        .changed()
                        .await
                        .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                    continue;
                }
                if let Some(loaded) = state.loaded.get_mut(&model) {
                    loaded.touch();
                    return Ok(Some(loaded.lease()));
                }
            }

            let capacity_guard = Arc::clone(&self.capacity_lock).lock_owned().await;
            let mut state = self.inner.lock().await;
            if state.draining.contains(&model) || state.retiring.contains(&model) {
                drop(state);
                drop(capacity_guard);
                drop(load_guard);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }
            if let Some(loaded) = state.loaded.get_mut(&model) {
                loaded.touch();
                return Ok(Some(loaded.lease()));
            }
            let (is_pending, at_capacity) = {
                let pending_loads = self
                    .pending_loads
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (
                    pending_loads.contains(&model),
                    state.resident_len() + pending_loads.len() >= self.max_loaded,
                )
            };
            if is_pending {
                drop(state);
                drop(capacity_guard);
                drop(load_guard);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }
            if at_capacity {
                let retired = state.retire_lru(self.max_loaded);
                drop(state);
                drop(capacity_guard);
                drop(load_guard);
                if retired.is_empty() {
                    changes
                        .changed()
                        .await
                        .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                } else {
                    self.destroy_retired(retired, "cache limit").await?;
                }
                continue;
            }
            self.pending_loads
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(model.clone());
            drop(state);
            drop(capacity_guard);
            drop(load_guard);

            let _local_reservation = LocalLoadReservation {
                model: model.clone(),
                pending_loads: Arc::clone(&self.pending_loads),
                residency: self.residency.clone(),
            };
            let _admission = admission().await?;
            let loading = self
                .loading_mutex(&model)
                .await
                .expect("reserved model must remain available");
            let load_guard = loading.lock_owned().await;
            let path = self.provision_locked(model.clone()).await?;
            let engine = load(model.clone(), path).await?;
            let lease = self.insert_loaded(model, engine).await;
            drop(load_guard);
            return Ok(Some(lease));
        }
    }

    async fn insert_loaded(&self, model: M, engine: E) -> ModelLease<E> {
        let _capacity_guard = self.capacity_lock.lock().await;
        let mut state = self.inner.lock().await;
        debug_assert!(state.resident_len() < self.max_loaded);
        let std::collections::hash_map::Entry::Vacant(entry) = state.loaded.entry(model) else {
            unreachable!("per-model load reservation prevents duplicate cache insertion");
        };
        let lease = entry
            .insert(LoadedModel {
                engine,
                last_used: Arc::new(StdMutex::new(Instant::now())),
                active_leases: Arc::new(AtomicUsize::new(0)),
                run_permits: Arc::new(Semaphore::new(1)),
                residency: self.residency.clone(),
            })
            .lease();
        drop(state);
        self.residency.notify();
        lease
    }

    async fn provision_locked(&self, model: M) -> anyhow::Result<PathBuf> {
        if let Some(path) = self.inner.lock().await.provisioned.get(&model).cloned() {
            return Ok(path);
        }
        let Some(provisioner) = self.provisioner.as_ref() else {
            let provisioning = self.inner.lock().await.provisioning.get(&model).cloned();
            return model
                .resolve_without_provisioner(provisioning, self.dir.clone())
                .await;
        };

        tracing::debug!(cache = self.cache_id, model = ?model, models_dir = %self.dir.display(), "ensuring model is available");
        let provisioning = self.inner.lock().await.provisioning.get(&model).cloned();
        let path = provisioner
            .provision(model.clone(), provisioning, self.dir.clone())
            .await?;
        self.inner
            .lock()
            .await
            .provisioned
            .insert(model.clone(), path.clone());
        tracing::debug!(cache = self.cache_id, model = ?model, path = %path.display(), "model cache ready");
        Ok(path)
    }

    pub async fn cleanup_idle(&self)
    where
        M: Send + 'static,
        E: Send + 'static,
    {
        let retired = self.inner.lock().await.retire_idle(self.idle_timeout);
        if let Err(error) = self.destroy_retired(retired, "idle timeout").await {
            tracing::error!(cache = self.cache_id, %error, "model destructor failed");
        }
    }

    async fn get_loaded(&self, model: &M) -> Option<ModelLease<E>>
    where
        M: Send + 'static,
        E: Send + 'static,
    {
        self.cleanup_idle().await;
        loop {
            let loading = self.loading_mutex(model).await?;
            let load_guard = loading.lock_owned().await;
            let mut changes = self.residency.subscribe();
            let mut state = self.inner.lock().await;
            if state.draining.contains(model) || state.retiring.contains(model) {
                drop(state);
                drop(load_guard);
                changes.changed().await.ok()?;
                continue;
            }
            let loaded = state.loaded.get_mut(model)?;
            loaded.touch();
            return Some(loaded.lease());
        }
    }

    async fn is_available(&self, model: &M) -> bool {
        self.inner.lock().await.available.contains(model)
    }

    async fn loading_mutex(&self, model: &M) -> Option<Arc<Mutex<()>>> {
        let mut state = self.inner.lock().await;
        if !state.available.contains(model) {
            return None;
        }
        Some(Arc::clone(
            state
                .loading
                .entry(model.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        ))
    }

    pub(crate) async fn status(&self, model: &M) -> Option<ModelResidencyStatus> {
        let state = self.inner.lock().await;
        if !state.available.contains(model) {
            return None;
        }
        if state.draining.contains(model) || state.retiring.contains(model) {
            return Some(ModelResidencyStatus::Unloading);
        }
        if state.loaded.contains_key(model) {
            return Some(ModelResidencyStatus::Loaded);
        }
        if self
            .pending_loads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(model)
        {
            return Some(ModelResidencyStatus::Loading);
        }
        let loading = state.loading.get(model).cloned();
        drop(state);
        Some(
            if loading.is_some_and(|loading| loading.try_lock().is_err()) {
                ModelResidencyStatus::Loading
            } else {
                ModelResidencyStatus::Unloaded
            },
        )
    }

    pub(crate) async fn next_idle_deadline(&self) -> Option<Instant> {
        let state = self.inner.lock().await;
        state
            .loaded
            .iter()
            .filter(|(model, loaded)| !state.draining.contains(*model) && !loaded.is_active())
            .filter_map(|(_, loaded)| loaded.last_used().checked_add(self.idle_timeout))
            .min()
    }

    #[cfg(test)]
    pub async fn is_loaded(&self, model: M) -> bool {
        self.inner.lock().await.loaded.contains_key(&model)
    }

    #[must_use]
    pub const fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }

    const fn cache_id(&self) -> &'static str {
        self.cache_id
    }
}

impl<M, E> ModelCache<M, E>
where
    M: ModelCacheKey + std::hash::Hash + Send + 'static,
    E: Clone + Send + 'static,
{
    async fn destroy_retired(
        &self,
        retired: Vec<(M, LoadedModel<E>)>,
        reason: &'static str,
    ) -> anyhow::Result<()> {
        if retired.is_empty() {
            return Ok(());
        }

        for (model, _) in &retired {
            tracing::info!(cache = self.cache_id, model = ?model, reason, "unloading model");
        }

        let cache = self.clone();
        tokio::spawn(async move {
            let models = retired
                .iter()
                .map(|(model, _)| model.clone())
                .collect::<Vec<_>>();
            let destruction = tokio::task::spawn_blocking(move || drop(retired)).await;

            let mut state = cache.inner.lock().await;
            for model in models {
                state.retiring.remove(&model);
            }
            drop(state);
            cache.residency.notify();

            destruction.map_err(|error| anyhow::anyhow!("model destructor failed: {error:#}"))
        })
        .await
        .map_err(|error| anyhow::anyhow!("model retirement task failed: {error:#}"))?
    }

    pub(crate) async fn unload(&self, model: M) -> anyhow::Result<Option<bool>> {
        let cache = self.clone();
        tokio::spawn(async move { cache.unload_owned(model).await })
            .await
            .map_err(|error| anyhow::anyhow!("model unload task failed: {error:#}"))?
    }

    pub(crate) async fn unload_many(&self, models: Vec<M>) -> anyhow::Result<Option<bool>> {
        let cache = self.clone();
        tokio::spawn(async move { cache.unload_many_owned(models).await })
            .await
            .map_err(|error| anyhow::anyhow!("model unload task failed: {error:#}"))?
    }

    async fn unload_many_owned(&self, models: Vec<M>) -> anyhow::Result<Option<bool>> {
        let available = self.inner.lock().await.available.clone();
        let mut seen = HashSet::new();
        let models = models
            .into_iter()
            .filter(|model| seen.insert(model.clone()))
            .filter(|model| available.contains(model))
            .collect::<Vec<_>>();
        if models.is_empty() {
            return Ok(None);
        }

        loop {
            let mut changes = self.residency.subscribe();
            let mut load_guards = Vec::with_capacity(models.len());
            for model in &models {
                let loading = self
                    .loading_mutex(model)
                    .await
                    .expect("model availability was checked");
                load_guards.push(loading.lock_owned().await);
            }
            let mut state = self.inner.lock().await;
            if models.iter().any(|model| {
                state.draining.contains(model)
                    || state.retiring.contains(model)
                    || self
                        .pending_loads
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .contains(model)
            }) {
                drop(state);
                drop(load_guards);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }
            state.draining.extend(models.iter().cloned());
            drop(state);
            drop(load_guards);
            break;
        }

        loop {
            let mut changes = self.residency.subscribe();
            let mut state = self.inner.lock().await;
            if models
                .iter()
                .filter_map(|model| state.loaded.get(model))
                .all(|loaded| !loaded.is_active())
            {
                let retired = models
                    .iter()
                    .filter_map(|model| state.retire(model).map(|loaded| (model.clone(), loaded)))
                    .collect::<Vec<_>>();
                for model in &models {
                    state.draining.remove(model);
                }
                drop(state);
                let retired_any = !retired.is_empty();
                self.destroy_retired(retired, "explicit request").await?;
                return Ok(Some(retired_any));
            }
            drop(state);
            changes
                .changed()
                .await
                .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
        }
    }

    async fn unload_owned(&self, model: M) -> anyhow::Result<Option<bool>> {
        if !self.is_available(&model).await {
            return Ok(None);
        }

        let mut owns_drain = false;
        loop {
            let mut changes = self.residency.subscribe();
            let loading = self
                .loading_mutex(&model)
                .await
                .expect("model availability was checked");
            let load_guard = loading.lock_owned().await;
            let mut state = self.inner.lock().await;
            if state.retiring.contains(&model) || (state.draining.contains(&model) && !owns_drain) {
                drop(state);
                drop(load_guard);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }
            if self
                .pending_loads
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(&model)
            {
                drop(state);
                drop(load_guard);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }
            let Some(is_active) = state.loaded.get(&model).map(LoadedModel::is_active) else {
                return Ok(Some(false));
            };
            state.draining.insert(model.clone());
            owns_drain = true;
            if is_active {
                drop(state);
                drop(load_guard);
                changes
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("model residency domain closed"))?;
                continue;
            }

            let Some(loaded) = state.retire(&model) else {
                return Err(anyhow::anyhow!("loaded model disappeared during unload"));
            };
            state.draining.remove(&model);
            drop(state);
            drop(load_guard);

            self.destroy_retired(vec![(model, loaded)], "explicit request")
                .await?;
            return Ok(Some(true));
        }
    }
}

impl<M, E> CacheTracker for ModelCache<M, E>
where
    M: ModelCacheKey + std::hash::Hash + Send + 'static,
    E: Clone + Send + 'static,
{
    fn loaded_len(&self) -> BoxFuture<'_, usize> {
        Box::pin(async move { self.inner.lock().await.resident_len() })
    }

    fn lru_entry(&self) -> BoxFuture<'_, Option<TrackedLoadedModel>> {
        Box::pin(async move {
            let state = self.inner.lock().await;
            state
                .loaded
                .iter()
                .filter(|(model, loaded)| !loaded.is_active() && !state.draining.contains(*model))
                .min_by_key(|(_, loaded)| loaded.last_used())
                .map(|(model, loaded)| TrackedLoadedModel {
                    cache_id: self.cache_id,
                    key: Box::new(model.clone()),
                    last_used: loaded.last_used(),
                })
        })
    }

    fn evict_tracked(&self, key: Box<dyn Any + Send + Sync>) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            let Ok(model) = key.downcast::<M>() else {
                return false;
            };
            let model = *model;
            let retired = {
                let mut state = self.inner.lock().await;
                if !state.loaded.contains_key(&model) {
                    return false;
                }
                if state.draining.contains(&model)
                    || state.loaded.get(&model).is_some_and(LoadedModel::is_active)
                {
                    return false;
                }
                state.retire(&model).map(|loaded| (model, loaded))
            };
            let Some(retired) = retired else {
                return false;
            };
            if let Err(error) = self
                .destroy_retired(vec![retired], "global cache limit")
                .await
            {
                tracing::error!(cache = self.cache_id, %error, "model destructor failed");
            }
            true
        })
    }

    fn cache_id(&self) -> &'static str {
        self.cache_id()
    }

    fn residency_domain(&self) -> ResidencyDomain {
        self.residency.clone()
    }

    fn clone_tracker(&self) -> Arc<dyn CacheTracker> {
        Arc::new(self.clone())
    }
}

impl<M, E> ModelCacheState<M, E>
where
    M: Clone + Eq + std::hash::Hash,
{
    fn resident_len(&self) -> usize {
        self.loaded.len() + self.retiring.len()
    }

    fn retire(&mut self, model: &M) -> Option<LoadedModel<E>> {
        let loaded = self.loaded.remove(model)?;
        self.retiring.insert(model.clone());
        Some(loaded)
    }

    fn retire_idle(&mut self, idle_timeout: Duration) -> Vec<(M, LoadedModel<E>)> {
        let now = Instant::now();
        let evicted = self
            .loaded
            .iter()
            .filter_map(|(model, loaded)| {
                (!self.draining.contains(model)
                    && !loaded.is_active()
                    && now.duration_since(loaded.last_used()) >= idle_timeout)
                    .then_some(model.clone())
            })
            .collect::<Vec<_>>();
        evicted
            .into_iter()
            .filter_map(|model| self.retire(&model).map(|loaded| (model, loaded)))
            .collect()
    }

    fn retire_lru(&mut self, max_loaded: usize) -> Vec<(M, LoadedModel<E>)> {
        if self.resident_len() < max_loaded {
            return Vec::new();
        }
        let Some(model) = self
            .loaded
            .iter()
            .filter(|(model, loaded)| !self.draining.contains(*model) && !loaded.is_active())
            .min_by_key(|(_, loaded)| loaded.last_used())
            .map(|(model, _)| model.clone())
        else {
            return Vec::new();
        };
        self.retire(&model)
            .map(|loaded| vec![(model, loaded)])
            .unwrap_or_default()
    }
}

pub type AsrModelCache = ModelCache<AsrModel, Asr>;
pub type TtsModelCache = ModelCache<TtsModel, Tts>;
pub type OcrModelCache = ModelCache<OcrModel, Ocr>;
pub type OcrVlModelCache = ModelCache<OcrModel, Ocr>;

#[deprecated(note = "use ModelCache lazy provisioning or a startup preload policy")]
#[allow(clippy::ptr_arg, reason = "preserves the previously public signature")]
/// Provisions each requested model sequentially using the supplied provisioner.
///
/// # Errors
///
/// Returns the first provisioning error.
pub async fn ensure_available_models<M>(
    label: &'static str,
    provisioner: &ModelDownloader,
    models: &[M],
    dir: &PathBuf,
) -> anyhow::Result<usize>
where
    M: ModelSpec,
{
    for model in models {
        tracing::debug!(model = ?model, models_dir = %dir.display(), "ensuring {label} model is available");
        let path = provisioner
            .provision(model.clone(), None, dir.clone())
            .await?;
        tracing::debug!(model = ?model, path = %path.display(), "{label} model cache ready");
    }
    Ok(models.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::resource_policy::ResourcePolicy;
    use orchion::{KnownOcrModel, ModelCategory, OcrModel};
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct DropProbe {
        cached_copy: bool,
        dropped_outside_state_lock: Arc<AtomicBool>,
        state: std::sync::Weak<Mutex<ModelCacheState<AsrModel, Self>>>,
    }

    impl Clone for DropProbe {
        fn clone(&self) -> Self {
            Self {
                cached_copy: false,
                dropped_outside_state_lock: Arc::clone(&self.dropped_outside_state_lock),
                state: self.state.clone(),
            }
        }
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            if !self.cached_copy {
                return;
            }
            let outside_lock = self
                .state
                .upgrade()
                .is_some_and(|state| state.try_lock().is_ok());
            self.dropped_outside_state_lock
                .store(outside_lock, Ordering::SeqCst);
        }
    }

    struct BlockingDropProbe {
        blocks_on_drop: bool,
        started: Arc<Notify>,
        release: Arc<(StdMutex<bool>, Condvar)>,
    }

    impl Clone for BlockingDropProbe {
        fn clone(&self) -> Self {
            Self {
                blocks_on_drop: false,
                started: Arc::clone(&self.started),
                release: Arc::clone(&self.release),
            }
        }
    }

    impl Drop for BlockingDropProbe {
        fn drop(&mut self) {
            if !self.blocks_on_drop {
                return;
            }
            self.started.notify_one();
            let (released, wake) = &*self.release;
            let guard = released
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(
                wake.wait_while(guard, |released| !*released)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }
    }

    fn asr_model(value: &str) -> AsrModel {
        AsrModel::parse(value).unwrap()
    }

    fn tts_model(value: &str) -> TtsModel {
        TtsModel::parse(value).unwrap()
    }

    fn qwen_asr_06b() -> AsrModel {
        asr_model("alibaba/qwen3-asr-0.6b")
    }

    fn qwen_asr_17b() -> AsrModel {
        asr_model("alibaba/qwen3-asr-1.7b")
    }

    fn qwen_tts_custom_voice() -> TtsModel {
        tts_model("alibaba/qwen3-tts-12hz-0.6b-customvoice")
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct CompositeModel {
        primary: AsrModel,
        variant: u8,
    }

    impl ModelSpec for CompositeModel {
        fn category(&self) -> ModelCategory {
            self.primary.category()
        }

        fn huggingface_repo(&self) -> &str {
            self.primary.huggingface_repo()
        }

        fn modelscope_repo(&self) -> &str {
            self.primary.modelscope_repo()
        }

        fn required_files(&self) -> &'static [&'static str] {
            self.primary.required_files()
        }
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

    #[tokio::test]
    async fn tracker_evicts_the_exact_composite_key_when_repositories_match() {
        let first = CompositeModel {
            primary: qwen_asr_06b(),
            variant: 1,
        };
        let second = CompositeModel {
            primary: qwen_asr_06b(),
            variant: 2,
        };
        let cache = ModelCache::new(
            "composite",
            vec![first.clone(), second.clone()],
            Duration::from_mins(1),
            2,
            PathBuf::from("models"),
        );
        drop(
            cache
                .get_or_load(first.clone(), |_, _| async { Ok(1) })
                .await
                .unwrap()
                .unwrap(),
        );
        drop(
            cache
                .get_or_load(second.clone(), |_, _| async { Ok(2) })
                .await
                .unwrap()
                .unwrap(),
        );

        let lru = cache.lru_entry().await.unwrap();
        assert_eq!(
            lru.key.downcast_ref::<CompositeModel>().unwrap().variant,
            first.variant
        );
        assert!(cache.evict_tracked(lru.key).await);
        assert!(!cache.is_loaded(first).await);
        assert!(cache.is_loaded(second).await);
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

    fn cache_in_domain<M>(
        cache_id: &'static str,
        available_models: Vec<M>,
        max_loaded: usize,
        residency: &ResidencyDomain,
    ) -> ModelCache<M, usize>
    where
        M: ModelCacheKey + std::hash::Hash,
    {
        ModelCache::new_in_domain(
            cache_id,
            available_models,
            Duration::from_mins(1),
            max_loaded,
            PathBuf::from("models"),
            None,
            residency.clone(),
        )
    }

    fn asr_cache_in(residency: &ResidencyDomain) -> ModelCache<AsrModel, usize> {
        cache_in_domain("asr", vec![qwen_asr_06b(), qwen_asr_17b()], 2, residency)
    }

    fn tts_cache_in(residency: &ResidencyDomain) -> ModelCache<TtsModel, usize> {
        cache_in_domain("tts", vec![qwen_tts_custom_voice()], 2, residency)
    }

    fn ocr_cache_in(residency: &ResidencyDomain) -> ModelCache<OcrModel, usize> {
        cache_in_domain(
            "ocr",
            vec![KnownOcrModel::PpOcrV6Tiny.into_model()],
            2,
            residency,
        )
    }

    #[tokio::test]
    async fn rejects_unavailable_model_without_loading() {
        let cache = ModelCache::<AsrModel, usize>::new(
            "asr",
            vec![qwen_asr_06b()],
            Duration::from_mins(1),
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
        let cache = asr_cache(2, Duration::from_mins(1));
        let loads = Arc::new(AtomicUsize::new(0));

        let first = load_counted(&cache, qwen_asr_06b(), &loads).await;
        let second = load_counted(&cache, qwen_asr_06b(), &loads).await;

        assert_eq!(first, Some(1));
        assert_eq!(second, Some(1));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_model_when_full() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let loads = Arc::new(AtomicUsize::new(0));

        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(1));
        assert_eq!(load_counted(&cache, qwen_asr_17b(), &loads).await, Some(2));
        assert_eq!(load_counted(&cache, qwen_asr_06b(), &loads).await, Some(3));
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn per_cache_capacity_waits_for_active_lease_then_resumes() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let active = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let loads = Arc::new(AtomicUsize::new(0));
        let waiter = tokio::spawn({
            let cache = cache.clone();
            let loads = Arc::clone(&loads);
            async move {
                cache
                    .get_or_load(qwen_asr_17b(), move |_, _| async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(2)
                    })
                    .await
            }
        });

        tokio::task::yield_now().await;
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert!(!waiter.is_finished());

        drop(active);
        let resumed = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .unwrap();

        assert_eq!(*resumed, 2);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(!cache.is_loaded(qwen_asr_06b()).await);
        assert!(cache.is_loaded(qwen_asr_17b()).await);
    }

    #[tokio::test]
    async fn cancelled_per_cache_capacity_waiter_does_not_load() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let active = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let loads = Arc::new(AtomicUsize::new(0));
        let waiter = tokio::spawn({
            let cache = cache.clone();
            let loads = Arc::clone(&loads);
            async move {
                cache
                    .get_or_load(qwen_asr_17b(), move |_, _| async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(2)
                    })
                    .await
            }
        });

        tokio::task::yield_now().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(active);
        tokio::task::yield_now().await;

        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert!(!cache.is_loaded(qwen_asr_17b()).await);
    }

    #[tokio::test]
    async fn residency_signal_observes_release_before_await() {
        let residency = ResidencyDomain::new();
        let mut changes = residency.subscribe();

        residency.notify();

        tokio::time::timeout(Duration::from_millis(50), changes.changed())
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn status_tracks_loaded_and_unloading_transitions() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = qwen_asr_06b();

        assert_eq!(
            cache.status(&qwen_asr_17b()).await,
            Some(ModelResidencyStatus::Unloaded)
        );
        assert_eq!(
            ModelCache::<AsrModel, usize>::new(
                "other",
                vec![],
                Duration::from_mins(1),
                1,
                PathBuf::from("models"),
            )
            .status(&model)
            .await,
            None
        );

        let active = cache
            .get_or_load(model.clone(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            cache.status(&model).await,
            Some(ModelResidencyStatus::Loaded)
        );

        let unload = tokio::spawn({
            let cache = cache.clone();
            let model = model.clone();
            async move { cache.unload(model).await.unwrap() }
        });
        wait_for_status(&cache, &model, ModelResidencyStatus::Unloading).await;

        drop(active);
        assert_eq!(unload.await.unwrap(), Some(true));
        assert_eq!(
            cache.status(&model).await,
            Some(ModelResidencyStatus::Unloaded)
        );
    }

    #[tokio::test]
    async fn unload_waits_for_active_lease_and_blocks_a_new_lease() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = qwen_asr_06b();
        let active = cache
            .get_or_load(model.clone(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let unload = tokio::spawn({
            let cache = cache.clone();
            let model = model.clone();
            async move { cache.unload(model).await.unwrap() }
        });
        wait_for_status(&cache, &model, ModelResidencyStatus::Unloading).await;

        let loads = Arc::new(AtomicUsize::new(0));
        let reload = tokio::spawn({
            let cache = cache.clone();
            let model = model.clone();
            let loads = Arc::clone(&loads);
            async move {
                cache
                    .get_or_load(model, move |_, _| async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(2)
                    })
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!unload.is_finished());
        assert!(!reload.is_finished());
        assert_eq!(loads.load(Ordering::SeqCst), 0);

        drop(active);
        assert_eq!(unload.await.unwrap(), Some(true));
        let reloaded = reload.await.unwrap().unwrap().unwrap();
        assert_eq!(*reloaded, 2);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unload_is_idempotent_for_unavailable_and_unloaded_models() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = qwen_asr_06b();
        let unavailable = asr_model("example/unavailable");

        assert_eq!(cache.unload(unavailable).await.unwrap(), None);
        assert_eq!(cache.unload(model.clone()).await.unwrap(), Some(false));
        drop(
            cache
                .get_or_load(model.clone(), |_, _| async { Ok(1) })
                .await
                .unwrap()
                .unwrap(),
        );
        assert_eq!(cache.unload(model.clone()).await.unwrap(), Some(true));
        assert_eq!(cache.unload(model).await.unwrap(), Some(false));
    }

    #[tokio::test]
    async fn unload_many_fences_every_model_until_the_family_is_retired() {
        let cache = asr_cache(2, Duration::from_mins(1));
        let first = qwen_asr_06b();
        let second = qwen_asr_17b();
        drop(
            cache
                .get_or_load(first.clone(), |_, _| async { Ok(1) })
                .await
                .unwrap()
                .unwrap(),
        );
        let active_second = cache
            .get_or_load(second.clone(), |_, _| async { Ok(2) })
            .await
            .unwrap()
            .unwrap();

        let unload = tokio::spawn({
            let cache = cache.clone();
            let first = first.clone();
            let second = second.clone();
            async move { cache.unload_many(vec![first, second]).await.unwrap() }
        });
        wait_for_status(&cache, &first, ModelResidencyStatus::Unloading).await;
        let reload = tokio::spawn({
            let cache = cache.clone();
            let first = first.clone();
            async move {
                cache
                    .get_or_load(first, |_, _| async { Ok(3) })
                    .await
                    .unwrap()
                    .unwrap()
            }
        });
        tokio::task::yield_now().await;

        assert!(!unload.is_finished());
        assert!(!reload.is_finished());
        drop(active_second);

        assert_eq!(unload.await.unwrap(), Some(true));
        assert_eq!(*reload.await.unwrap(), 3);
    }

    #[tokio::test]
    async fn cancelling_unload_caller_does_not_abandon_drain() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = qwen_asr_06b();
        let active = cache
            .get_or_load(model.clone(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let caller = tokio::spawn({
            let cache = cache.clone();
            let model = model.clone();
            async move { cache.unload(model).await.unwrap() }
        });
        wait_for_status(&cache, &model, ModelResidencyStatus::Unloading).await;

        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        drop(active);
        wait_for_status(&cache, &model, ModelResidencyStatus::Unloaded).await;
        assert!(!cache.is_loaded(model).await);
    }

    #[tokio::test]
    async fn unload_destroys_cached_engine_outside_state_lock() {
        let cache = ModelCache::<AsrModel, DropProbe>::new(
            "asr",
            vec![qwen_asr_06b()],
            Duration::from_mins(1),
            1,
            PathBuf::from("models"),
        );
        let model = qwen_asr_06b();
        let dropped_outside_state_lock = Arc::new(AtomicBool::new(false));
        let state = Arc::downgrade(&cache.inner);
        let probe = Arc::clone(&dropped_outside_state_lock);
        drop(
            cache
                .get_or_load(model.clone(), move |_, _| async move {
                    Ok(DropProbe {
                        cached_copy: true,
                        dropped_outside_state_lock: probe,
                        state,
                    })
                })
                .await
                .unwrap()
                .unwrap(),
        );

        assert_eq!(cache.unload(model).await.unwrap(), Some(true));
        assert!(dropped_outside_state_lock.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn idle_cleanup_destroys_cached_engine_outside_state_lock() {
        let cache = ModelCache::<AsrModel, DropProbe>::new(
            "asr",
            vec![qwen_asr_06b()],
            Duration::from_millis(1),
            1,
            PathBuf::from("models"),
        );
        let dropped_outside_state_lock = Arc::new(AtomicBool::new(false));
        let state = Arc::downgrade(&cache.inner);
        let probe = Arc::clone(&dropped_outside_state_lock);
        drop(
            cache
                .get_or_load(qwen_asr_06b(), move |_, _| async move {
                    Ok(DropProbe {
                        cached_copy: true,
                        dropped_outside_state_lock: probe,
                        state,
                    })
                })
                .await
                .unwrap()
                .unwrap(),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        cache.cleanup_idle().await;

        assert!(dropped_outside_state_lock.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn retiring_model_blocks_replacement_until_destructor_finishes() {
        let cache = ModelCache::<AsrModel, BlockingDropProbe>::new(
            "asr",
            vec![qwen_asr_06b(), qwen_asr_17b()],
            Duration::from_mins(1),
            1,
            PathBuf::from("models"),
        );
        let started = Arc::new(Notify::new());
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        drop(
            cache
                .get_or_load(qwen_asr_06b(), {
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    move |_, _| async move {
                        Ok(BlockingDropProbe {
                            blocks_on_drop: true,
                            started,
                            release,
                        })
                    }
                })
                .await
                .unwrap()
                .unwrap(),
        );

        let replacement_loads = Arc::new(AtomicUsize::new(0));
        let replacement = tokio::spawn({
            let cache = cache.clone();
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let replacement_loads = Arc::clone(&replacement_loads);
            async move {
                cache
                    .get_or_load(qwen_asr_17b(), move |_, _| async move {
                        replacement_loads.fetch_add(1, Ordering::SeqCst);
                        Ok(BlockingDropProbe {
                            blocks_on_drop: false,
                            started,
                            release,
                        })
                    })
                    .await
            }
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .unwrap();
        assert!(!replacement.is_finished());
        assert_eq!(replacement_loads.load(Ordering::SeqCst), 0);
        assert_eq!(cache.inner.lock().await.resident_len(), 1);
        assert_eq!(
            cache.status(&qwen_asr_06b()).await,
            Some(ModelResidencyStatus::Unloading)
        );

        let (released, wake) = &*release;
        *released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        wake.notify_one();
        let loaded = tokio::time::timeout(Duration::from_secs(1), replacement)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .unwrap();

        assert_eq!(replacement_loads.load(Ordering::SeqCst), 1);
        drop(loaded);
    }

    #[tokio::test]
    async fn same_model_reload_waits_for_retiring_destructor() {
        let cache = ModelCache::<AsrModel, BlockingDropProbe>::new(
            "asr",
            vec![qwen_asr_06b()],
            Duration::ZERO,
            1,
            PathBuf::from("models"),
        );
        let model = qwen_asr_06b();
        let started = Arc::new(Notify::new());
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        drop(
            cache
                .get_or_load(model.clone(), {
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    move |_, _| async move {
                        Ok(BlockingDropProbe {
                            blocks_on_drop: true,
                            started,
                            release,
                        })
                    }
                })
                .await
                .unwrap()
                .unwrap(),
        );

        let cleanup = tokio::spawn({
            let cache = cache.clone();
            async move { cache.cleanup_idle().await }
        });
        started.notified().await;
        let reloads = Arc::new(AtomicUsize::new(0));
        let reload = tokio::spawn({
            let cache = cache.clone();
            let reloads = Arc::clone(&reloads);
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            async move {
                cache
                    .get_or_load(model, move |_, _| async move {
                        reloads.fetch_add(1, Ordering::SeqCst);
                        Ok(BlockingDropProbe {
                            blocks_on_drop: false,
                            started,
                            release,
                        })
                    })
                    .await
            }
        });

        tokio::task::yield_now().await;
        assert!(!reload.is_finished());
        assert_eq!(reloads.load(Ordering::SeqCst), 0);

        release_blocking_drop(&release);
        cleanup.await.unwrap();
        let lease = tokio::time::timeout(Duration::from_secs(1), reload)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(reloads.load(Ordering::SeqCst), 1);
        drop(lease);
    }

    #[tokio::test]
    async fn unload_waits_for_existing_retirement_to_finish() {
        let cache = ModelCache::<AsrModel, BlockingDropProbe>::new(
            "asr",
            vec![qwen_asr_06b()],
            Duration::ZERO,
            1,
            PathBuf::from("models"),
        );
        let model = qwen_asr_06b();
        let started = Arc::new(Notify::new());
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        drop(
            cache
                .get_or_load(model.clone(), {
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    move |_, _| async move {
                        Ok(BlockingDropProbe {
                            blocks_on_drop: true,
                            started,
                            release,
                        })
                    }
                })
                .await
                .unwrap()
                .unwrap(),
        );

        let cleanup = tokio::spawn({
            let cache = cache.clone();
            async move { cache.cleanup_idle().await }
        });
        started.notified().await;
        let unload = tokio::spawn({
            let cache = cache.clone();
            async move { cache.unload(model).await }
        });

        tokio::task::yield_now().await;
        assert!(!unload.is_finished());
        release_blocking_drop(&release);
        cleanup.await.unwrap();
        assert_eq!(unload.await.unwrap().unwrap(), Some(false));
    }

    #[tokio::test]
    async fn idle_deadline_starts_at_last_lease_drop() {
        let idle_timeout = Duration::from_millis(200);
        let cache = asr_cache(1, idle_timeout);
        let lease = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let last_lease = lease.clone();

        drop(lease);
        assert_eq!(cache.next_idle_deadline().await, None);
        tokio::time::sleep(Duration::from_millis(10)).await;
        let before_drop = Instant::now();
        drop(last_lease);
        let after_drop = Instant::now();
        let deadline = cache.next_idle_deadline().await.unwrap();

        assert!(deadline >= before_drop.checked_add(idle_timeout).unwrap());
        assert!(deadline <= after_drop.checked_add(idle_timeout).unwrap());
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
    fn retire_idle_returns_unloaded_models() {
        let model = qwen_asr_06b();
        let mut state = ModelCacheState {
            available: vec![model.clone()],
            provisioning: HashMap::new(),
            loaded: HashMap::from([(
                model.clone(),
                LoadedModel {
                    engine: 1,
                    last_used: Arc::new(StdMutex::new(
                        Instant::now()
                            .checked_sub(Duration::from_secs(10))
                            .expect("test duration fits before the current instant"),
                    )),
                    active_leases: Arc::new(AtomicUsize::new(0)),
                    run_permits: Arc::new(Semaphore::new(1)),
                    residency: ResidencyDomain::new(),
                },
            )]),
            loading: HashMap::new(),
            draining: HashSet::new(),
            retiring: HashSet::new(),
            provisioned: HashMap::new(),
        };

        let retired = state.retire_idle(Duration::from_secs(1));
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].0, model);
        assert!(state.loaded.is_empty());
        assert!(state.retiring.contains(&model));
    }

    #[tokio::test]
    async fn global_limiter_evicts_lru_across_model_categories() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);
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
    async fn global_lru_destroys_cached_engine_outside_state_lock() {
        let residency = ResidencyDomain::new();
        let asr_cache = ModelCache::<AsrModel, DropProbe>::new_in_domain(
            "asr",
            vec![qwen_asr_06b()],
            Duration::from_mins(1),
            1,
            PathBuf::from("models"),
            None,
            residency.clone(),
        );
        let tts_cache = tts_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);
        let dropped_outside_state_lock = Arc::new(AtomicBool::new(false));
        let state = Arc::downgrade(&asr_cache.inner);
        let probe = Arc::clone(&dropped_outside_state_lock);
        let asr = limiter
            .get_or_load(
                &asr_cache,
                &all_caches,
                qwen_asr_06b(),
                move |_, _| async move {
                    Ok(DropProbe {
                        cached_copy: true,
                        dropped_outside_state_lock: probe,
                        state,
                    })
                },
            )
            .await
            .unwrap()
            .unwrap();
        drop(asr);

        drop(
            limiter
                .get_or_load(
                    &tts_cache,
                    &all_caches,
                    qwen_tts_custom_voice(),
                    |_, _| async { Ok(1) },
                )
                .await
                .unwrap()
                .unwrap(),
        );

        assert!(dropped_outside_state_lock.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn global_capacity_waits_for_active_lease_then_resumes() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);

        let active_asr = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();

        let tts_loads = Arc::new(AtomicUsize::new(0));
        let waiter = tokio::spawn({
            let limiter = limiter.clone();
            let asr_cache = asr_cache.clone();
            let tts_cache = tts_cache.clone();
            let tts_loads = Arc::clone(&tts_loads);
            async move {
                let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
                limiter
                    .get_or_load(
                        &tts_cache,
                        &all_caches,
                        qwen_tts_custom_voice(),
                        move |_, _| async move {
                            tts_loads.fetch_add(1, Ordering::SeqCst);
                            Ok(2)
                        },
                    )
                    .await
            }
        });
        tokio::task::yield_now().await;

        assert_eq!(tts_loads.load(Ordering::SeqCst), 0);
        assert!(!waiter.is_finished());
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(!tts_cache.is_loaded(qwen_tts_custom_voice()).await);

        drop(active_asr);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .is_some()
        );
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
    }

    #[tokio::test]
    async fn locally_blocked_cold_load_does_not_block_another_service() {
        let residency = ResidencyDomain::new();
        let asr_cache = cache_in_domain("asr", vec![qwen_asr_06b(), qwen_asr_17b()], 1, &residency);
        let tts_cache = cache_in_domain("tts", vec![qwen_tts_custom_voice()], 1, &residency);
        let limiter = GlobalModelCacheLimiter::new_in_domain(2, residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let active = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();
        let local_capacity = asr_cache.capacity_lock.lock().await;
        let blocked = tokio::spawn({
            let limiter = limiter.clone();
            let asr_cache = asr_cache.clone();
            let tts_cache = tts_cache.clone();
            async move {
                let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
                limiter
                    .get_or_load(&asr_cache, &all_caches, qwen_asr_17b(), |_, _| async {
                        Ok(2)
                    })
                    .await
            }
        });
        wait_for_status(&asr_cache, &qwen_asr_17b(), ModelResidencyStatus::Loading).await;

        let tts = tokio::time::timeout(
            Duration::from_secs(1),
            limiter.get_or_load(
                &tts_cache,
                &all_caches,
                qwen_tts_custom_voice(),
                |_, _| async { Ok(3) },
            ),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        assert_eq!(*tts, 3);
        assert!(!blocked.is_finished());

        drop(local_capacity);
        drop(active);
        drop(tts);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), blocked)
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn cancelled_model_operation_holds_lease_until_operation_finishes() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);
        let active_asr = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let finished = Arc::new(Notify::new());

        let waiter = tokio::spawn({
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let finished = Arc::clone(&finished);
            async move {
                active_asr
                    .run(move |lease| async move {
                        started.notify_one();
                        release.notified().await;
                        drop(lease);
                        finished.notify_one();
                    })
                    .await
                    .unwrap();
            }
        });
        started.notified().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());

        let blocked = tokio::spawn({
            let limiter = limiter.clone();
            let asr_cache = asr_cache.clone();
            let tts_cache = tts_cache.clone();
            async move {
                let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
                limiter
                    .get_or_load(
                        &tts_cache,
                        &all_caches,
                        qwen_tts_custom_voice(),
                        |_, _| async { Ok(2) },
                    )
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!blocked.is_finished());
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(!tts_cache.is_loaded(qwen_tts_custom_voice()).await);

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), finished.notified())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), blocked)
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn model_operations_are_serialized_per_loaded_model() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let first = tokio::spawn({
            let model = model.clone();
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            async move {
                model
                    .run(move |_| async move {
                        started.notify_one();
                        release.notified().await;
                    })
                    .await
                    .unwrap();
            }
        });
        started.notified().await;

        let second_started = Arc::new(Notify::new());
        let second = tokio::spawn({
            let model = model.clone();
            let second_started = Arc::clone(&second_started);
            async move {
                model
                    .run(move |_| async move { second_started.notify_one() })
                    .await
                    .unwrap();
            }
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), second_started.notified())
                .await
                .is_err()
        );

        release.notify_one();
        first.await.unwrap();
        second.await.unwrap();
    }

    #[tokio::test]
    async fn same_model_waiter_does_not_occupy_global_inference_capacity() {
        let cache = asr_cache(2, Duration::from_mins(1));
        let first_model = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let other_model = cache
            .get_or_load(qwen_asr_17b(), |_, _| async { Ok(2) })
            .await
            .unwrap()
            .unwrap();
        let inference = ResourcePolicy::new(2, 1, 1).inference_limiter();
        let first_started = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let first = tokio::spawn({
            let model = first_model.clone();
            let inference = inference.clone();
            let first_started = Arc::clone(&first_started);
            let release_first = Arc::clone(&release_first);
            async move {
                model
                    .run_with_inference(inference, move |_| async move {
                        first_started.notify_one();
                        release_first.notified().await;
                    })
                    .await
                    .unwrap();
            }
        });
        first_started.notified().await;

        let same_started = Arc::new(Notify::new());
        let same = tokio::spawn({
            let model = first_model.clone();
            let inference = inference.clone();
            let same_started = Arc::clone(&same_started);
            async move {
                model
                    .run_with_inference(inference, move |_| async move {
                        same_started.notify_one();
                    })
                    .await
                    .unwrap();
            }
        });
        tokio::task::yield_now().await;

        let other_started = Arc::new(Notify::new());
        let release_other = Arc::new(Notify::new());
        let other = tokio::spawn({
            let inference = inference.clone();
            let other_started = Arc::clone(&other_started);
            let release_other = Arc::clone(&release_other);
            async move {
                other_model
                    .run_with_inference(inference, move |_| async move {
                        other_started.notify_one();
                        release_other.notified().await;
                    })
                    .await
                    .unwrap();
            }
        });

        tokio::time::timeout(Duration::from_secs(1), other_started.notified())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), same_started.notified())
                .await
                .is_err()
        );

        release_first.notify_one();
        tokio::time::timeout(Duration::from_secs(1), same_started.notified())
            .await
            .unwrap();
        release_other.notify_one();
        first.await.unwrap();
        same.await.unwrap();
        other.await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_global_inference_waiter_never_starts_operation() {
        let cache = asr_cache(1, Duration::from_mins(1));
        let model = cache
            .get_or_load(qwen_asr_06b(), |_, _| async { Ok(1) })
            .await
            .unwrap()
            .unwrap();
        let policy = ResourcePolicy::new(1, 1, 1);
        let occupied = policy.acquire_inference().await;
        let started = Arc::new(AtomicBool::new(false));
        let waiter = tokio::spawn({
            let model = model.clone();
            let inference = policy.inference_limiter();
            let started = Arc::clone(&started);
            async move {
                model
                    .run_with_inference(inference, move |_| async move {
                        started.store(true, Ordering::SeqCst);
                    })
                    .await
            }
        });
        tokio::task::yield_now().await;

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(occupied);
        tokio::task::yield_now().await;
        assert!(!started.load(Ordering::SeqCst));

        tokio::time::timeout(
            Duration::from_secs(1),
            model.run_with_inference(policy.inference_limiter(), |_| async {}),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[tokio::test]
    async fn cancelled_cold_load_continues_as_a_shared_single_flight() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);
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

        let second_load = async {
            let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
            let loads = Arc::clone(&loads);
            limiter
                .get_or_load(
                    &asr_cache,
                    &all_caches,
                    qwen_asr_06b(),
                    move |_, _| async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(2)
                    },
                )
                .await
        };
        tokio::pin!(second_load);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut second_load)
                .await
                .is_err()
        );
        release.notify_one();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), second_load)
                .await
                .unwrap()
                .unwrap()
                .as_deref(),
            Some(&1)
        );
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(!tts_cache.is_loaded(qwen_tts_custom_voice()).await);
    }

    #[tokio::test]
    async fn limiter_close_fences_and_drains_queued_cold_load() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let ocr_cache = ocr_cache_in(&residency);
        let limiter = GlobalModelCacheLimiter::new_in_domain(1, residency);
        let all_caches: [&dyn CacheTracker; 3] = [&asr_cache, &tts_cache, &ocr_cache];
        let active = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap()
            .unwrap();
        let queued = tokio::spawn({
            let limiter = limiter.clone();
            let asr_cache = asr_cache.clone();
            let tts_cache = tts_cache.clone();
            let ocr_cache = ocr_cache.clone();
            async move {
                let all_caches: [&dyn CacheTracker; 3] = [&asr_cache, &tts_cache, &ocr_cache];
                limiter
                    .get_or_load(
                        &tts_cache,
                        &all_caches,
                        qwen_tts_custom_voice(),
                        |_, _| async { Ok(2) },
                    )
                    .await
            }
        });
        wait_for_status(
            &tts_cache,
            &qwen_tts_custom_voice(),
            ModelResidencyStatus::Loading,
        )
        .await;

        let drain = tokio::spawn({
            let limiter = limiter.clone();
            async move { limiter.close_and_drain().await }
        });
        while !limiter
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed
        {
            tokio::task::yield_now().await;
        }
        assert!(!drain.is_finished());
        let error = limiter
            .get_or_load(
                &ocr_cache,
                &all_caches,
                KnownOcrModel::PpOcrV6Tiny.into_model(),
                |_, _| async { Ok(3) },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("limiter is closed"));

        drop(active);
        let loaded = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), drain)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*loaded, 2);
    }

    #[tokio::test]
    async fn global_limiter_evicts_lru_across_three_model_categories() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let ocr_cache = ocr_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 3] = [&asr_cache, &tts_cache, &ocr_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(2, residency);

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
                KnownOcrModel::PpOcrV6Tiny.into_model(),
                |_, _| async { Ok(3) },
            )
            .await
            .unwrap();

        assert_eq!(ocr.as_deref(), Some(&3));
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
        assert!(tts_cache.is_loaded(qwen_tts_custom_voice()).await);
        assert!(
            ocr_cache
                .is_loaded(KnownOcrModel::PpOcrV6Tiny.into_model())
                .await
        );
    }

    #[tokio::test]
    async fn global_limiter_rejects_duplicate_cache_ids() {
        let residency = ResidencyDomain::new();
        let asr_cache = ModelCache::<AsrModel, usize>::new_in_domain(
            "models",
            vec![qwen_asr_06b()],
            Duration::from_mins(1),
            2,
            PathBuf::from("models"),
            None,
            residency.clone(),
        );
        let tts_cache = ModelCache::<TtsModel, usize>::new_in_domain(
            "models",
            vec![qwen_tts_custom_voice()],
            Duration::from_mins(1),
            2,
            PathBuf::from("models"),
            None,
            residency.clone(),
        );
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(2, residency);

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
    async fn global_limiter_rejects_caches_from_another_residency_domain() {
        let asr_cache = asr_cache(2, Duration::from_mins(1));
        let tts_cache = tts_cache(2, Duration::from_mins(1));
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new(2);

        let error = limiter
            .get_or_load(&asr_cache, &all_caches, qwen_asr_06b(), |_, _| async {
                Ok(1)
            })
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("share a residency domain"),
            "unexpected error: {error:#}"
        );
        assert!(!asr_cache.is_loaded(qwen_asr_06b()).await);
    }

    #[tokio::test]
    async fn global_limiter_returns_loaded_model_without_waiting_for_cold_load() {
        let residency = ResidencyDomain::new();
        let asr_cache = asr_cache_in(&residency);
        let tts_cache = tts_cache_in(&residency);
        let all_caches: [&dyn CacheTracker; 2] = [&asr_cache, &tts_cache];
        let limiter = GlobalModelCacheLimiter::new_in_domain(2, residency);
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

    async fn wait_for_status<M, E>(
        cache: &ModelCache<M, E>,
        model: &M,
        expected: ModelResidencyStatus,
    ) where
        M: ModelCacheKey + std::hash::Hash,
        E: Clone,
    {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if cache.status(model).await == Some(expected) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("model status transition timed out");
    }

    fn release_blocking_drop(release: &Arc<(StdMutex<bool>, Condvar)>) {
        let (released, wake) = &**release;
        *released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        wake.notify_one();
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
