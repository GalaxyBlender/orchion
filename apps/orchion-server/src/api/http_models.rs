use crate::api::http_shared::authorize;
use crate::api::openai::{ApiError, ModelList, ModelObject, ModelSubtype, ModelType};
use crate::infrastructure::orchion::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use std::collections::HashSet;
use std::sync::Arc;

pub(super) async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError> {
    authorize(&state, &headers)?;
    let mut data = Vec::new();
    if state.config().services.asr.enabled {
        data.extend(
            state
                .config()
                .services
                .asr
                .available_models
                .iter()
                .cloned()
                .map(|model| ModelObject::new(model, ModelType::Asr, None)),
        );
    }
    if state.config().services.tts.enabled {
        data.extend(
            state
                .config()
                .services
                .tts
                .available_models
                .iter()
                .cloned()
                .map(|model| {
                    let subtype = tts_model_subtype(&model);
                    ModelObject::new(model, ModelType::Tts, Some(subtype))
                }),
        );
    }
    if state.config().services.ocr.active() {
        data.extend(
            state
                .config()
                .services
                .ocr
                .available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Standard))
                }),
        );
        data.extend(
            state
                .config()
                .services
                .ocr
                .layout_available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
                }),
        );
    }
    if state.config().services.ocr_vl.active() {
        data.extend(
            state
                .config()
                .services
                .ocr_vl
                .available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Vl))
                }),
        );
        data.extend(
            state
                .config()
                .services
                .ocr_vl
                .layout_available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
                }),
        );
    }
    dedupe_model_objects(&mut data);
    Ok(Json(ModelList {
        object: "list",
        data,
    }))
}

fn tts_model_subtype(model: &orchion::TtsModel) -> ModelSubtype {
    if model.supports_preset_speakers() {
        ModelSubtype::PresetVoice
    } else if model.supports_voice_design() {
        ModelSubtype::VoiceDesign
    } else {
        ModelSubtype::VoiceClone
    }
}

fn dedupe_model_objects(models: &mut Vec<ModelObject>) {
    let mut seen = HashSet::new();
    models.retain(|model| seen.insert(model.id.clone()));
}
