use std::path::PathBuf;
use std::sync::Arc;

use crate::contract::{
    AdvancedRequest, AdvancedSemanticRequest, EmbeddingRequest, Error, Request, RuntimeConfig,
    SemanticRequest, SemanticTokenCountRequest, TokenCountRequest,
};

pub use crate::scheduler::{
    ChoiceGeneration, ChoiceReservation, Embedding, EmbeddingReservation, Generation,
    ReasoningControlAttempt, ReasoningControlCancellation, ReasoningControlHandle,
    ReasoningControlResult, Reservation,
};

#[derive(Debug)]
pub struct BackendOwner {
    _inner: Arc<crate::scheduler::BackendOwner>,
}

impl BackendOwner {
    pub fn acquire() -> Result<Arc<Self>, Error> {
        Ok(Arc::new(Self {
            _inner: crate::scheduler::BackendOwner::acquire()?,
        }))
    }
}

#[derive(Debug)]
pub struct Engine {
    inner: crate::scheduler::Engine,
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Engine {
    pub fn load(path: PathBuf, config: RuntimeConfig) -> Result<Self, Error> {
        crate::scheduler::Engine::load(path, config).map(Self::from_scheduler)
    }

    pub async fn reserve(
        &self,
        request: Request,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        self.inner.reserve(request, event_capacity).await
    }

    pub async fn reserve_semantic(
        &self,
        request: SemanticRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        self.inner.reserve_semantic(request, event_capacity).await
    }

    pub async fn reserve_advanced(
        &self,
        request: AdvancedRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        self.inner.reserve_advanced(request, event_capacity).await
    }

    pub async fn reserve_advanced_semantic(
        &self,
        request: AdvancedSemanticRequest,
        event_capacity: usize,
    ) -> Result<Reservation, Error> {
        self.inner
            .reserve_advanced_semantic(request, event_capacity)
            .await
    }

    pub async fn reserve_choices(
        &self,
        request: AdvancedRequest,
        event_capacity: usize,
    ) -> Result<ChoiceReservation, Error> {
        self.inner.reserve_choices(request, event_capacity).await
    }

    pub async fn reserve_choice_semantic(
        &self,
        request: AdvancedSemanticRequest,
        event_capacity: usize,
    ) -> Result<ChoiceReservation, Error> {
        self.inner
            .reserve_choice_semantic(request, event_capacity)
            .await
    }

    pub async fn reserve_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingReservation, Error> {
        self.inner.reserve_embedding(request).await
    }

    pub async fn count_tokens(&self, request: TokenCountRequest) -> Result<usize, Error> {
        self.inner.count_tokens(request).await
    }

    pub async fn count_semantic_tokens(
        &self,
        request: SemanticTokenCountRequest,
    ) -> Result<usize, Error> {
        self.inner.count_semantic_tokens(request).await
    }

    pub async fn generate(
        &self,
        request: Request,
        event_capacity: usize,
    ) -> Result<Generation, Error> {
        self.inner.generate(request, event_capacity).await
    }

    pub async fn generate_semantic(
        &self,
        request: SemanticRequest,
        event_capacity: usize,
    ) -> Result<Generation, Error> {
        self.inner.generate_semantic(request, event_capacity).await
    }

    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    pub(crate) fn from_scheduler(inner: crate::scheduler::Engine) -> Self {
        Self { inner }
    }
}
