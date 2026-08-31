use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ModelService {
    Asr,
    Tts,
    Ocr,
    OcrVl,
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ModelResidency {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ModelStatus {
    pub object: String,
    pub id: String,
    pub service: ModelService,
    pub status: ModelResidency,
}

impl ModelStatus {
    #[must_use]
    pub fn new(id: impl Into<String>, service: ModelService, status: ModelResidency) -> Self {
        Self {
            object: "model_status".to_string(),
            id: id.into(),
            service,
            status,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ModelStatusList {
    pub object: String,
    pub data: Vec<ModelStatus>,
}

impl ModelStatusList {
    #[must_use]
    pub fn new(data: Vec<ModelStatus>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct ModelControlRequest {
    pub model: String,
    pub service: ModelService,
}

impl ModelControlRequest {
    #[must_use]
    pub fn new(model: impl Into<String>, service: ModelService) -> Self {
        Self {
            model: model.into(),
            service,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_lifecycle_types_round_trip() {
        for service in [
            ModelService::Asr,
            ModelService::Tts,
            ModelService::Ocr,
            ModelService::OcrVl,
            ModelService::Llm,
        ] {
            let encoded = serde_json::to_string(&service).unwrap();
            assert_eq!(
                serde_json::from_str::<ModelService>(&encoded).unwrap(),
                service
            );
        }

        for residency in [
            ModelResidency::Unloaded,
            ModelResidency::Loading,
            ModelResidency::Loaded,
            ModelResidency::Unloading,
        ] {
            let encoded = serde_json::to_string(&residency).unwrap();
            assert_eq!(
                serde_json::from_str::<ModelResidency>(&encoded).unwrap(),
                residency
            );
        }

        let status = ModelStatus::new("model-id", ModelService::Llm, ModelResidency::Loaded);
        let list = ModelStatusList::new(vec![status.clone()]);
        let request = ModelControlRequest {
            model: "model-id".to_string(),
            service: ModelService::Llm,
        };

        assert_eq!(
            serde_json::from_value::<ModelStatus>(json!({
                "object": "model_status",
                "id": "model-id",
                "service": "llm",
                "status": "loaded"
            }))
            .unwrap(),
            status
        );
        assert_eq!(
            serde_json::from_value::<ModelStatusList>(serde_json::to_value(&list).unwrap())
                .unwrap(),
            list
        );
        assert_eq!(
            serde_json::from_value::<ModelControlRequest>(serde_json::to_value(&request).unwrap())
                .unwrap(),
            request
        );
    }
}
