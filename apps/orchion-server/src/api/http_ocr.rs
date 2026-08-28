use crate::api::http_shared::{
    authorize, parse_multipart_value, read_text_field, run_owned, write_multipart_file_to_temp_file,
};
use crate::api::openai::{ApiError, OcrApiFormat, OcrJsonResponse};
use crate::application::ServerApplication;
use crate::application::ocr::{OcrCommand, recognize};
use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use orchion::{ModelId, OcrResponseFormat, OcrTask};
use std::sync::Arc;

#[allow(
    clippy::too_many_lines,
    reason = "multipart parsing and cleanup form one request transaction"
)]
pub(super) async fn create_ocr<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
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

    let image_path = image_file.path().to_path_buf();
    let operation_state = Arc::clone(&state);
    let output = run_owned(async move {
        let _image_file = image_file;
        recognize(
            operation_state.as_ref(),
            OcrCommand {
                image_path,
                model,
                response_format: response_format.map(OcrResponseFormat::from),
                task,
                layout_model,
                max_tokens,
            },
        )
        .await
        .map_err(ApiError::from)
    })
    .await?;

    let response_format = OcrApiFormat::from(output.format);
    let result = output.result;
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
            result
                .html
                .unwrap_or_else(|| escaped_text_html(&result.text)),
        )
            .into_response(),
    })
}

fn escaped_text_html(text: &str) -> String {
    let mut html = String::from("<pre>");
    for character in text.chars() {
        match character {
            '&' => html.push_str("&amp;"),
            '<' => html.push_str("&lt;"),
            '>' => html.push_str("&gt;"),
            _ => html.push(character),
        }
    }
    html.push_str("</pre>");
    html
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
