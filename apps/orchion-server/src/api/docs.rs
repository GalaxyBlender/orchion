use crate::api::openai::{
    ErrorBody, ModelList, OcrApiFormat, OcrJsonResponse, SpeechRequest, TranscriptionJson,
    TranscriptionVerboseJson,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(healthz_doc, list_models_doc, create_speech_doc, create_transcription_doc, create_ocr_doc),
    components(schemas(SpeechRequest, ErrorBody, ModelList, TranscriptionJson, TranscriptionVerboseJson, OcrJsonResponse, OcrApiFormat)),
    tags(
        (name = "audio", description = "OpenAI-compatible audio APIs"),
        (name = "ocr", description = "OCR and OCR-VL APIs"),
        (name = "models", description = "OpenAI-compatible model APIs")
    )
)]
struct ApiDoc;

pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/docs").url("/openapi/v1.json", ApiDoc::openapi())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_includes_ocr_path_and_schemas() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();

        assert!(spec["paths"]["/v1/ocr"]["post"].is_object());
        assert!(spec["components"]["schemas"]["OcrJsonResponse"].is_object());
        assert!(spec["components"]["schemas"]["OcrApiFormat"].is_object());
    }
}

#[utoipa::path(
    get,
    path = "/healthz",
    responses((status = 200, description = "Server health", body = String))
)]
#[allow(dead_code)]
async fn healthz_doc() {}

#[utoipa::path(
    get,
    path = "/v1/models",
    responses(
        (status = 200, description = "Configured model list", body = ModelList),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "models"
)]
#[allow(dead_code)]
async fn list_models_doc() {}

#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body(
        content = SpeechRequest,
        content_type = "application/json",
        description = "JSON speech synthesis. Voice clone requests use multipart/form-data on the same endpoint."
    ),
    responses(
        (status = 200, description = "Generated speech audio", content_type = "application/octet-stream", body = Vec<u8>),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "audio"
)]
#[allow(dead_code)]
async fn create_speech_doc() {}

#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    request_body(content = String, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Transcription JSON", body = TranscriptionJson),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "audio"
)]
#[allow(dead_code)]
async fn create_transcription_doc() {}

#[utoipa::path(
    post,
    path = "/v1/ocr",
    request_body(
        content = String,
        content_type = "multipart/form-data",
        description = "POST /v1/ocr accepts multipart/form-data with file, optional model, response_format, task, layout_model, and max_tokens fields. Response formats are json, text, markdown, and html. Model IDs use {vendor}/{name}. Traditional metal maps to CoreML; OCR-VL metal maps to Candle Metal."
    ),
    responses(
        (status = 200, description = "OCR response. JSON requests return OcrJsonResponse; text requests return text/plain; markdown requests return text/markdown; html requests return text/html.", body = OcrJsonResponse),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "ocr"
)]
#[allow(dead_code)]
async fn create_ocr_doc() {}
