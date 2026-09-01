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

    /// Retrieves one configured public model without inspecting runtime residency.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the model is empty, transport fails, or decoding fails.
    pub async fn retrieve(&self, model: &str) -> Result<ModelObject, ClientError> {
        if model.is_empty() {
            return Err(ClientError::build_request("model must not be empty"));
        }
        let response = self
            .client
            .get_with_path_segment("/v1/models/", model)?
            .send()
            .await?;
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
    #[serde(default)]
    pub capability_details: Option<LlmCapabilityDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LlmCapabilityDetails {
    pub max_choices: usize,
    pub max_top_logprobs: usize,
    pub legacy_max_logprobs: usize,
    pub strict_json_schema: bool,
    pub runtime_template_validation: bool,
}

/// Top-level model capability type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelType {
    Asr,
    Tts,
    Ocr,
    Llm,
    Unknown(String),
}

/// Capability supported by a configured model deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    LlmEmbeddings,
    LlmCompletions,
    LlmInputTokens,
    LlmTools,
    LlmParallelTools,
    LlmJsonObject,
    LlmJsonSchema,
    LlmLogprobs,
    LlmLogitBias,
    LlmMultipleChoices,
    LlmReasoning,
    LlmVision,
    LlmResumableStreaming,
    LlmReasoningControl,
    Unknown(String),
}

macro_rules! string_enum_serde {
    ($type:ty, {$($variant:path => $value:literal),+ $(,)?}) => {
        impl Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let value = match self {
                    $($variant => $value,)+
                    Self::Unknown(value) => value,
                };
                serializer.serialize_str(value)
            }
        }

        impl<'de> Deserialize<'de> for $type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Ok(match value.as_str() {
                    $($value => $variant,)+
                    _ => Self::Unknown(value),
                })
            }
        }
    };
}

string_enum_serde!(ModelType, {
    ModelType::Asr => "asr",
    ModelType::Tts => "tts",
    ModelType::Ocr => "ocr",
    ModelType::Llm => "llm",
});

string_enum_serde!(ModelCapability, {
    ModelCapability::AsrTranscription => "asr_transcription",
    ModelCapability::AsrStreaming => "asr_streaming",
    ModelCapability::TtsVoiceCloning => "tts_voice_cloning",
    ModelCapability::TtsPresetSpeakers => "tts_preset_speakers",
    ModelCapability::TtsVoiceDesign => "tts_voice_design",
    ModelCapability::OcrText => "ocr_text",
    ModelCapability::OcrLayout => "ocr_layout",
    ModelCapability::OcrTableStructure => "ocr_table_structure",
    ModelCapability::OcrVisionLanguage => "ocr_vision_language",
    ModelCapability::OcrMarkdown => "ocr_markdown",
    ModelCapability::OcrHtml => "ocr_html",
    ModelCapability::LlmChat => "llm_chat",
    ModelCapability::LlmResponses => "llm_responses",
    ModelCapability::LlmStreaming => "llm_streaming",
    ModelCapability::LlmEmbeddings => "llm_embeddings",
    ModelCapability::LlmCompletions => "llm_completions",
    ModelCapability::LlmInputTokens => "llm_input_tokens",
    ModelCapability::LlmTools => "llm_tools",
    ModelCapability::LlmParallelTools => "llm_parallel_tools",
    ModelCapability::LlmJsonObject => "llm_json_object",
    ModelCapability::LlmJsonSchema => "llm_json_schema",
    ModelCapability::LlmLogprobs => "llm_logprobs",
    ModelCapability::LlmLogitBias => "llm_logit_bias",
    ModelCapability::LlmMultipleChoices => "llm_multiple_choices",
        ModelCapability::LlmReasoning => "llm_reasoning",
        ModelCapability::LlmReasoningControl => "llm_reasoning_control",
    ModelCapability::LlmVision => "llm_vision",
    ModelCapability::LlmResumableStreaming => "llm_resumable_streaming",
});
