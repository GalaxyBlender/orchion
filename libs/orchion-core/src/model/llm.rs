use super::ModelId;
use std::fmt;

/// Logical identifier for one configured text-generation deployment.
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
