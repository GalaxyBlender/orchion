use crate::api::http_shared::authorize;
use crate::api::openai::{ApiError, ModelList, ModelObject, ModelSubtype, ModelType};
use crate::application::model_lifecycle::{ModelSelector, ModelService, ModelStatus};
use crate::application::{ServerApplication, UseCaseError};
use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

pub(super) async fn list_models<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let policy = state.api_policy();
    let mut data = Vec::new();
    if let Some(asr) = &policy.asr {
        data.extend(
            asr.models
                .iter()
                .cloned()
                .map(|model| ModelObject::new(model, ModelType::Asr, None)),
        );
    }
    if let Some(tts_models) = &policy.tts_models {
        data.extend(tts_models.iter().cloned().map(|model| {
            let subtype = tts_model_subtype(&model);
            ModelObject::new(model, ModelType::Tts, subtype)
        }));
    }
    if let Some(ocr) = &policy.ocr {
        data.extend(ocr.models.iter().map(|id| {
            ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Standard))
        }));
        data.extend(ocr.layout_models.iter().map(|id| {
            ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
        }));
    }
    if let Some(ocr_vl) = &policy.ocr_vl {
        data.extend(
            ocr_vl.models.iter().map(|id| {
                ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Vl))
            }),
        );
        data.extend(ocr_vl.layout_models.iter().map(|id| {
            ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
        }));
    }
    dedupe_model_objects(&mut data);
    Ok(Json(ModelList {
        object: "list",
        data,
    }))
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ModelStatusList {
    pub object: &'static str,
    pub data: Vec<ModelStatus>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct ModelControlRequest {
    pub model: String,
    pub service: ModelService,
}

pub(super) async fn list_model_statuses<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
) -> Result<Json<ModelStatusList>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    Ok(Json(ModelStatusList {
        object: "list",
        data: state.model_statuses().await,
    }))
}

pub(super) async fn load_model<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    request: Result<Json<ModelControlRequest>, JsonRejection>,
) -> Result<Json<ModelStatus>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let request = parse_control_request(request)?;
    let model = request.model.clone();
    let status = state
        .load_model(ModelSelector {
            model: request.model,
            service: request.service,
        })
        .await
        .map_err(|error| ApiError::from(UseCaseError::from(error)))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    Ok(Json(status))
}

pub(super) async fn unload_model<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    request: Result<Json<ModelControlRequest>, JsonRejection>,
) -> Result<Json<ModelStatus>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let request = parse_control_request(request)?;
    let model = request.model.clone();
    let status = state
        .unload_model(ModelSelector {
            model: request.model,
            service: request.service,
        })
        .await
        .map_err(|error| ApiError::from(UseCaseError::from(error)))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    Ok(Json(status))
}

fn parse_control_request(
    request: Result<Json<ModelControlRequest>, JsonRejection>,
) -> Result<ModelControlRequest, ApiError> {
    request
        .map(|Json(request)| request)
        .map_err(|error| ApiError::invalid_request(error.to_string(), None, Some("invalid_json")))
}

fn tts_model_subtype(model: &orchion::TtsModel) -> Option<ModelSubtype> {
    if model.supports_preset_speakers() {
        Some(ModelSubtype::PresetVoice)
    } else if model.supports_voice_design() {
        Some(ModelSubtype::VoiceDesign)
    } else if model.supports_voice_cloning() {
        Some(ModelSubtype::VoiceClone)
    } else {
        None
    }
}

fn dedupe_model_objects(models: &mut Vec<ModelObject>) {
    let mut seen = HashSet::new();
    models.retain(|model| seen.insert(model.id.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_models_without_registered_capabilities_have_no_subtype() {
        let model = orchion::TtsModel::parse("Acme/Experimental-TTS").unwrap();

        assert!(tts_model_subtype(&model).is_none());
    }
}
