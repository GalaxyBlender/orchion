use super::RuntimeError;
use std::future::Future;
use std::pin::Pin;

pub type ModelStatusesFuture<'a> = Pin<Box<dyn Future<Output = Vec<ModelStatus>> + Send + 'a>>;
pub type ModelControlFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ModelStatus>, RuntimeError>> + Send + 'a>>;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelService {
    Asr,
    Tts,
    Ocr,
    OcrVl,
}

#[derive(Debug, Clone)]
pub struct ModelSelector {
    pub model: String,
    pub service: ModelService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelResidency {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct ModelStatus {
    pub object: &'static str,
    pub id: String,
    pub service: ModelService,
    pub status: ModelResidency,
}

impl ModelStatus {
    #[must_use]
    pub fn new(id: impl Into<String>, service: ModelService, status: ModelResidency) -> Self {
        Self {
            object: "model_status",
            id: id.into(),
            service,
            status,
        }
    }
}

pub trait ModelLifecycleRuntime: Send + Sync {
    fn model_statuses(&self) -> ModelStatusesFuture<'_>;
    fn prewarm_model(&self, selector: ModelSelector) -> ModelControlFuture<'_>;
    fn unload_model(&self, selector: ModelSelector) -> ModelControlFuture<'_>;
}
