use super::ModelId;
use std::fmt;

pub const QWEN3_EMBEDDING_06B_GGUF_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";
pub const QWEN3_EMBEDDING_06B_GGUF_SIZE: u64 = 639_150_592;
pub const QWEN3_EMBEDDING_06B_GGUF_SHA256: &str =
    "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439";
pub const QWEN3_EMBEDDING_06B_DIMENSIONS: usize = 1024;

/// Logical identifier for one configured generation or embedding deployment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LlmModel(ModelId);

impl LlmModel {
    #[must_use]
    pub const fn new(id: ModelId) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn id(&self) -> &ModelId {
        &self.0
    }
}

impl fmt::Display for LlmModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
