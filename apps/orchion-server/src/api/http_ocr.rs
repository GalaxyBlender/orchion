use crate::api::http_shared::{
    authorize, parse_multipart_value, read_text_field, write_multipart_file_to_temp_file,
};
use crate::api::openai::{ApiError, OcrApiFormat, OcrJsonResponse};
use crate::infrastructure::orchion::AppState;
use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use orchion::{KnownOcrModel, ModelId, OcrOptions, OcrResponseFormat, OcrTask};
use std::sync::Arc;

pub(super) async fn create_ocr(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let mut image_file = None;
    let mut model = None;
    let mut response_format = None;
    let mut task = OcrTask::Ocr;
    let mut layout_model = None;
    let mut max_tokens = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                image_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "response_format" => {
                let value = read_text_field(field, "response_format").await?;
                response_format = Some(OcrApiFormat::try_from(value.as_str())?);
            }
            "task" => {
                let value = read_text_field(field, "task").await?;
                task = parse_ocr_task(&value)?;
            }
            "layout_model" => {
                let value = read_text_field(field, "layout_model").await?;
                layout_model = Some(parse_ocr_model_id(&value, "layout_model")?);
            }
            "max_tokens" => {
                let value: usize = parse_multipart_value(field, "max_tokens").await?;
                if value == 0 {
                    return Err(ApiError::invalid_request(
                        "`max_tokens` must be greater than 0",
                        Some("max_tokens"),
                        Some("invalid_multipart_field"),
                    ));
                }
                max_tokens = Some(value);
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let (image_file, image_bytes) = image_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if image_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded OCR file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }

    let choice = resolve_ocr_service_choice(&state, model.as_deref(), response_format)?;
    let response_format = resolve_ocr_response_format(&state, choice, response_format);
    validate_ocr_parameters(
        choice,
        response_format,
        task,
        layout_model.as_ref(),
        max_tokens,
        &state.config().services.ocr,
        &state.config().services.ocr_vl,
    )?;
    let ocr = match choice {
        OcrServiceChoice::Ocr { model } => state.ocr(model).await,
        OcrServiceChoice::OcrVl { model } => state.ocr_vl(model).await,
    }
    .map_err(|error| {
        tracing::error!(error = %format_args!("{error:#}"), "failed to load OCR runtime");
        ApiError::internal(error.to_string())
    })?
    .ok_or_else(|| ApiError::model_not_available(choice.model().id()))?;

    let options = OcrOptions {
        response_format: OcrResponseFormat::from(response_format),
        task,
        layout_model: resolve_ocr_layout_model(&state, choice, layout_model),
        max_tokens,
    };
    let result = ocr.recognize_file_with(image_file.path(), options).await?;

    Ok(match response_format {
        OcrApiFormat::Json => Json(OcrJsonResponse {
            model: result.model.to_string(),
            format: response_format,
            text: result.text,
            markdown: result.markdown,
            html: result.html,
            regions: result.regions,
            layout_blocks: result.layout_blocks,
            usage: result.usage,
        })
        .into_response(),
        OcrApiFormat::Text => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            result.text,
        )
            .into_response(),
        OcrApiFormat::Markdown => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/markdown; charset=utf-8"),
            )],
            result.markdown.unwrap_or(result.text),
        )
            .into_response(),
        OcrApiFormat::Html => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            )],
            result.html.unwrap_or(result.text),
        )
            .into_response(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OcrServiceChoice {
    Ocr { model: KnownOcrModel },
    OcrVl { model: KnownOcrModel },
}

impl OcrServiceChoice {
    fn ocr(model: KnownOcrModel) -> Result<Self, ApiError> {
        if !model.is_traditional_ocr() {
            return Err(invalid_ocr_model_kind(model, "traditional OCR"));
        }
        Ok(Self::Ocr { model })
    }

    fn ocr_vl(model: KnownOcrModel) -> Result<Self, ApiError> {
        if !model.is_ocr_vl() {
            return Err(invalid_ocr_model_kind(model, "OCR-VL"));
        }
        Ok(Self::OcrVl { model })
    }

    pub(super) const fn model(self) -> KnownOcrModel {
        match self {
            Self::Ocr { model } | Self::OcrVl { model } => model,
        }
    }

    pub(super) const fn is_ocr_vl(self) -> bool {
        matches!(self, Self::OcrVl { .. })
    }
}

pub(super) fn resolve_ocr_service_choice(
    state: &AppState,
    model: Option<&str>,
    response_format: Option<OcrApiFormat>,
) -> Result<OcrServiceChoice, ApiError> {
    if let Some(model) = model {
        return resolve_explicit_ocr_model(state, model);
    }

    let ocr_active = state.config().services.ocr.active();
    let ocr_vl_active = state.config().services.ocr_vl.active();
    match (ocr_active, ocr_vl_active) {
        (true, false) => OcrServiceChoice::ocr(default_ocr_choice(state, false)?),
        (false, true) => OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?),
        (true, true) => resolve_default_ocr_model(state, response_format),
        (false, false) => Err(ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )),
    }
}

fn resolve_explicit_ocr_model(state: &AppState, model: &str) -> Result<OcrServiceChoice, ApiError> {
    let model_id = ModelId::parse(model).map_err(|_| ApiError::model_not_available(model))?;
    let ocr_match = state.config().services.ocr.active()
        && state
            .config()
            .services
            .ocr
            .available_models
            .contains(&model_id);
    let ocr_vl_match = state.config().services.ocr_vl.active()
        && state
            .config()
            .services
            .ocr_vl
            .available_models
            .contains(&model_id);

    match (ocr_match, ocr_vl_match) {
        (true, true) => OcrServiceChoice::ocr(known_traditional_ocr_model(&model_id, model)?),
        (true, false) => OcrServiceChoice::ocr(known_traditional_ocr_model(&model_id, model)?),
        (false, true) => OcrServiceChoice::ocr_vl(known_ocr_vl_model(&model_id, model)?),
        (false, false) => Err(ApiError::model_not_available(model)),
    }
}

fn resolve_default_ocr_model(
    state: &AppState,
    response_format: Option<OcrApiFormat>,
) -> Result<OcrServiceChoice, ApiError> {
    let prefer_ocr_vl = matches!(
        response_format,
        Some(OcrApiFormat::Markdown | OcrApiFormat::Html)
    );
    if prefer_ocr_vl {
        if effective_default_ocr_model(state, true).is_some() {
            OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?)
        } else {
            OcrServiceChoice::ocr(default_ocr_choice(state, false)?)
        }
    } else if effective_default_ocr_model(state, false).is_some() {
        OcrServiceChoice::ocr(default_ocr_choice(state, false)?)
    } else {
        OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?)
    }
}

pub(super) fn resolve_ocr_response_format(
    state: &AppState,
    choice: OcrServiceChoice,
    response_format: Option<OcrApiFormat>,
) -> OcrApiFormat {
    response_format.unwrap_or_else(|| match choice {
        OcrServiceChoice::Ocr { .. } => OcrApiFormat::from(state.config().services.ocr.format),
        OcrServiceChoice::OcrVl { .. } => OcrApiFormat::from(state.config().services.ocr_vl.format),
    })
}

pub(super) fn resolve_ocr_layout_model(
    _state: &AppState,
    _choice: OcrServiceChoice,
    layout_model: Option<ModelId>,
) -> Option<ModelId> {
    layout_model
}

impl From<OcrResponseFormat> for OcrApiFormat {
    fn from(format: OcrResponseFormat) -> Self {
        match format {
            OcrResponseFormat::Json => Self::Json,
            OcrResponseFormat::Text => Self::Text,
            OcrResponseFormat::Markdown => Self::Markdown,
            OcrResponseFormat::Html => Self::Html,
        }
    }
}

fn default_ocr_choice(state: &AppState, ocr_vl: bool) -> Result<KnownOcrModel, ApiError> {
    let Some(default_model) = effective_default_ocr_model(state, ocr_vl) else {
        return Err(ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        ));
    };
    if ocr_vl {
        known_ocr_vl_model(default_model, default_model.as_str())
    } else {
        known_traditional_ocr_model(default_model, default_model.as_str())
    }
}

fn effective_default_ocr_model(state: &AppState, ocr_vl: bool) -> Option<&ModelId> {
    if ocr_vl {
        let service = &state.config().services.ocr_vl;
        if !service.active() {
            return None;
        }
        service
            .default_model
            .as_ref()
            .or_else(|| service.available_models.first())
    } else {
        let service = &state.config().services.ocr;
        if !service.active() {
            return None;
        }
        service
            .default_model
            .as_ref()
            .or_else(|| service.available_models.first())
    }
}

fn known_traditional_ocr_model(
    model_id: &ModelId,
    raw_model: &str,
) -> Result<KnownOcrModel, ApiError> {
    KnownOcrModel::from_traditional_model_id(model_id)
        .map_err(|_| ApiError::model_not_available(raw_model))
}

fn known_ocr_vl_model(model_id: &ModelId, raw_model: &str) -> Result<KnownOcrModel, ApiError> {
    KnownOcrModel::from_ocr_vl_model_id(model_id)
        .map_err(|_| ApiError::model_not_available(raw_model))
}

fn invalid_ocr_model_kind(model: KnownOcrModel, service: &str) -> ApiError {
    ApiError::invalid_request(
        format!(
            "model `{}` is not compatible with the {service} service",
            model.id()
        ),
        Some("model"),
        Some("invalid_ocr_model_kind"),
    )
}

pub(super) fn validate_ocr_parameters(
    choice: OcrServiceChoice,
    response_format: OcrApiFormat,
    task: OcrTask,
    layout_model: Option<&ModelId>,
    max_tokens: Option<usize>,
    ocr_service: &crate::settings::OcrServiceSection,
    ocr_vl_service: &crate::settings::OcrVlServiceSection,
) -> Result<(), ApiError> {
    let model = choice.model();
    if matches!(response_format, OcrApiFormat::Markdown | OcrApiFormat::Html)
        && !model.supports_markdown()
        && layout_model.is_none()
    {
        return Err(ApiError::invalid_request(
            "selected OCR model does not support structured response format",
            Some("response_format"),
            Some("unsupported_response_format"),
        ));
    }
    if choice.is_ocr_vl() {
        if let Some(layout_model) = layout_model {
            validate_ocr_layout_model(layout_model)?;
            validate_configured_layout_model(
                &ocr_vl_service.layout_available_models,
                layout_model,
                "OCR-VL",
            )?;
        }
        return Ok(());
    }
    if let Some(layout_model) = layout_model {
        validate_ocr_layout_model(layout_model)?;
        validate_configured_layout_model(
            &ocr_service.layout_available_models,
            layout_model,
            "OCR",
        )?;
    }
    if task != OcrTask::Ocr || max_tokens.is_some() {
        return Err(ApiError::invalid_request(
            "selected OCR model does not support OCR-VL parameters",
            None,
            Some("unsupported_ocr_parameter"),
        ));
    }
    Ok(())
}

fn validate_ocr_layout_model(layout_model: &ModelId) -> Result<(), ApiError> {
    KnownOcrModel::from_layout_model_id(layout_model)
        .map(|_| ())
        .map_err(|_| {
            ApiError::invalid_request(
                "`layout_model` must be PaddlePaddle/PP-DocLayoutV3",
                Some("layout_model"),
                Some("invalid_ocr_model_kind"),
            )
        })
}

fn validate_configured_layout_model(
    available_models: &[ModelId],
    layout_model: &ModelId,
    service_name: &str,
) -> Result<(), ApiError> {
    if available_models.contains(layout_model) {
        return Ok(());
    }
    Err(ApiError::invalid_request(
        format!("`layout_model` is not configured for the {service_name} service"),
        Some("layout_model"),
        Some("model_not_available"),
    ))
}

pub(super) fn parse_ocr_task(value: &str) -> Result<OcrTask, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "ocr" => Ok(OcrTask::Ocr),
        "table" => Ok(OcrTask::Table),
        "formula" => Ok(OcrTask::Formula),
        "chart" => Ok(OcrTask::Chart),
        "spotting" => Ok(OcrTask::Spotting),
        "seal" => Ok(OcrTask::Seal),
        _ => Err(ApiError::invalid_request(
            "unsupported OCR task; supported tasks are ocr, table, formula, chart, spotting, and seal",
            Some("task"),
            Some("unsupported_ocr_parameter"),
        )),
    }
}

fn parse_ocr_model_id(value: &str, param: &'static str) -> Result<ModelId, ApiError> {
    ModelId::parse(value).map_err(|_| {
        ApiError::invalid_request(
            format!("invalid `{param}`; expected vendor/name"),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}
