use crate::api::http_shared::authorize;
use crate::api::openai::{ApiError, ModelList, ModelObject, ModelType};
use crate::application::model_lifecycle::{ModelSelector, ModelService, ModelStatus};
use crate::application::{ServerApplication, UseCaseError};
use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub(super) async fn list_models<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let data = state
        .model_catalog()
        .await
        .into_iter()
        .map(|model| {
            let model_type = match model.service {
                ModelService::Asr => ModelType::Asr,
                ModelService::Tts => ModelType::Tts,
                ModelService::Ocr | ModelService::OcrVl => ModelType::Ocr,
                ModelService::Llm => ModelType::Llm,
            };
            ModelObject::new(
                model.id.to_string(),
                model.name,
                model_type,
                model.capabilities,
            )
        })
        .collect();
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
