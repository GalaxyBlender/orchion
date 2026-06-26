use crate::client::decode_json;
use crate::{Client, ClientError};
use serde::{Deserialize, Serialize};

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
    pub subtype: Option<ModelSubtype>,
}

/// Top-level model capability type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Asr,
    Tts,
    Ocr,
}

/// Model subtype reported by the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSubtype {
    Standard,
    Vl,
    Layout,
    PresetVoice,
    VoiceClone,
    VoiceDesign,
}
