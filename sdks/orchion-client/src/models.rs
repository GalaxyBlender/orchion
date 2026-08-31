use crate::client::decode_json;
use crate::{Client, ClientError};
use serde::{Deserialize, Serialize};

pub use orchion_protocol::{
    ModelControlRequest, ModelResidency, ModelService, ModelStatus, ModelStatusList,
};

/// Client for the models API.
pub struct ModelsClient<'a> {
    client: &'a Client,
}

impl<'a> ModelsClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Lists available models.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request cannot be sent or the response cannot be decoded.
    pub async fn list(&self) -> Result<ListModelsResponse, ClientError> {
        let response = self.client.get("/v1/models")?.send().await?;
        decode_json(response).await
    }

    /// Lists the runtime residency status of configured models.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request cannot be sent or the response cannot be decoded.
    pub async fn list_statuses(&self) -> Result<ModelStatusList, ClientError> {
        let response = self.client.get("/api/models/status")?.send().await?;
        decode_json(response).await
    }

    /// Loads a model into its service runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request cannot be sent or the response cannot be decoded.
    pub async fn load(&self, request: ModelControlRequest) -> Result<ModelStatus, ClientError> {
        let response = self
            .client
            .post("/api/models/load")?
            .json(&request)
            .send()
            .await?;
        decode_json(response).await
    }

    /// Unloads a model from its service runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request cannot be sent or the response cannot be decoded.
    pub async fn unload(&self, request: ModelControlRequest) -> Result<ModelStatus, ClientError> {
        let response = self
            .client
            .post("/api/models/unload")?
            .json(&request)
            .send()
            .await?;
        decode_json(response).await
    }
}

/// Response returned by the models list endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// Model metadata returned by the API.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<ModelCapability>,
}

/// Top-level model capability type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Asr,
    Tts,
    Ocr,
    Llm,
}

/// Capability supported by a configured model deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    AsrTranscription,
    AsrStreaming,
    TtsVoiceCloning,
    TtsPresetSpeakers,
    TtsVoiceDesign,
    OcrText,
    OcrLayout,
    OcrTableStructure,
    OcrVisionLanguage,
    OcrMarkdown,
    OcrHtml,
    LlmChat,
    LlmResponses,
    LlmStreaming,
}
