use crate::openai::{ErrorBody, SpeechRequest, TranscriptionJson, TranscriptionVerboseJson};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(healthz_doc, create_speech_doc, create_transcription_doc),
    components(schemas(SpeechRequest, ErrorBody, TranscriptionJson, TranscriptionVerboseJson)),
    tags((name = "audio", description = "OpenAI-compatible audio APIs"))
)]
struct ApiDoc;

pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi())
}

#[utoipa::path(
    get,
    path = "/healthz",
    responses((status = 200, description = "Server health", body = String))
)]
#[allow(dead_code)]
async fn healthz_doc() {}

#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechRequest,
    responses(
        (status = 200, description = "Generated speech audio", content_type = "audio/wav", body = Vec<u8>),
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
