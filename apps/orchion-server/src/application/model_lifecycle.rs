use super::RuntimeError;
pub use orchion_protocol::{ModelResidency, ModelService, ModelStatus};
use std::future::Future;
use std::pin::Pin;

pub type ModelStatusesFuture<'a> = Pin<Box<dyn Future<Output = Vec<ModelStatus>> + Send + 'a>>;
pub type ModelControlFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ModelStatus>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct ModelSelector {
    pub model: String,
    pub service: ModelService,
}

pub trait ModelLifecycleRuntime: Send + Sync {
    fn model_statuses(&self) -> ModelStatusesFuture<'_>;
    fn load_model(&self, selector: ModelSelector) -> ModelControlFuture<'_>;
    fn unload_model(&self, selector: ModelSelector) -> ModelControlFuture<'_>;
}
