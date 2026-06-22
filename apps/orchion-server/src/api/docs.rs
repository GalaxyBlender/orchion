use crate::api::openai::{
    ErrorBody, ModelList, OcrApiFormat, OcrJsonResponse, SpeechRequest, TranscriptionJson,
    TranscriptionVerboseJson,
};
use orchion_docs::PdfImageFormat;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(healthz_doc, list_models_doc, create_speech_doc, create_transcription_doc, create_ocr_doc, create_pdf_images_doc),
    components(schemas(SpeechRequest, ErrorBody, ModelList, TranscriptionJson, TranscriptionVerboseJson, OcrJsonResponse, OcrApiFormat, PdfImageFormat, PdfImagesMultipartRequest)),
    tags(
        (name = "audio", description = "OpenAI-compatible audio APIs"),
        (name = "ocr", description = "OCR and OCR-VL APIs"),
        (name = "pdf", description = "PDF rendering APIs"),
        (name = "models", description = "OpenAI-compatible model APIs")
    )
)]
struct ApiDoc;

#[derive(ToSchema)]
#[allow(dead_code)]
struct PdfImagesMultipartRequest {
    /// PDF file to render as page images.
    #[schema(value_type = String, format = Binary, content_media_type = "application/pdf")]
    file: String,
    response_format: Option<PdfImageFormat>,
    #[schema(example = "1,3-5")]
    pages: Option<String>,
    #[schema(example = 1.0, minimum = 0.1, maximum = 4.0)]
    scale: Option<f32>,
}

pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/docs").url("/openapi/v1.json", ApiDoc::openapi())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn openapi_includes_ocr_path_and_schemas() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();

        assert!(spec["paths"]["/v1/ocr"]["post"].is_object());
        assert!(spec["components"]["schemas"]["OcrJsonResponse"].is_object());
        assert!(spec["components"]["schemas"]["OcrApiFormat"].is_object());
    }

    #[test]
    fn openapi_includes_model_type_schema() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        let model_object = &spec["components"]["schemas"]["ModelObject"];

        assert_eq!(
            model_object["properties"]["type"]["$ref"],
            "#/components/schemas/ModelType"
        );
        assert_eq!(
            spec["components"]["schemas"]["ModelType"]["enum"],
            serde_json::json!(["asr", "tts", "ocr"])
        );
        let subtype_schema = &model_object["properties"]["subtype"];
        let subtype_ref = subtype_schema["$ref"].as_str().or_else(|| {
            ["anyOf", "allOf", "oneOf"].iter().find_map(|key| {
                subtype_schema[*key]
                    .as_array()
                    .and_then(|schemas| schemas.iter().find_map(|schema| schema["$ref"].as_str()))
            })
        });
        assert_eq!(subtype_ref, Some("#/components/schemas/ModelSubtype"));
        assert_eq!(
            spec["components"]["schemas"]["ModelSubtype"]["enum"],
            serde_json::json!([
                "standard",
                "vl",
                "layout",
                "preset_voice",
                "voice_clone",
                "voice_design"
            ])
        );
    }

    #[test]
    fn openapi_includes_pdf_images_path_and_schemas() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        let pdf_images_post = &spec["paths"]["/v1/pdf/images"]["post"];
        let multipart_schema =
            &pdf_images_post["requestBody"]["content"]["multipart/form-data"]["schema"];
        let request_schema = &spec["components"]["schemas"]["PdfImagesMultipartRequest"];
        let request_properties = &request_schema["properties"];

        assert!(pdf_images_post.is_object());
        assert!(schema_references(
            multipart_schema,
            "PdfImagesMultipartRequest"
        ));
        assert!(request_schema.is_object());
        assert!(
            request_schema["required"]
                .as_array()
                .is_some_and(|required| required.iter().any(|field| field == "file"))
        );
        assert_eq!(request_properties["file"]["type"], "string");
        assert_eq!(request_properties["file"]["format"], "binary");
        assert!(
            request_properties["file"]["description"]
                .as_str()
                .is_some_and(|description| description.contains("PDF"))
        );
        assert!(schema_has_type(&request_properties["scale"], "number"));
        assert!(!schema_has_type(&request_properties["scale"], "string"));
        assert_eq!(request_properties["scale"]["minimum"], 0.1);
        assert_eq!(request_properties["scale"]["maximum"], 4.0);
        assert!(schema_references(
            &request_properties["response_format"],
            "PdfImageFormat"
        ));
        assert!(spec["components"]["schemas"]["PdfImageFormat"].is_object());
        assert_eq!(
            spec["components"]["schemas"]["PdfImageFormat"]["enum"][0],
            "png"
        );
        assert_eq!(
            spec["components"]["schemas"]["PdfImageFormat"]["enum"][1],
            "jpeg"
        );
        assert_eq!(
            spec["components"]["schemas"]["PdfImageFormat"]["enum"][2],
            "webp"
        );
        assert!(pdf_images_post["responses"]["200"]["content"]["application/zip"].is_object());
    }

    fn schema_references(schema: &Value, schema_name: &str) -> bool {
        if schema
            .get("$ref")
            .and_then(Value::as_str)
            .is_some_and(|reference| reference.ends_with(&format!("/{schema_name}")))
        {
            return true;
        }

        match schema {
            Value::Array(items) => items
                .iter()
                .any(|item| schema_references(item, schema_name)),
            Value::Object(fields) => fields
                .values()
                .any(|value| schema_references(value, schema_name)),
            _ => false,
        }
    }

    fn schema_has_type(schema: &Value, expected_type: &str) -> bool {
        match &schema["type"] {
            Value::String(schema_type) => schema_type == expected_type,
            Value::Array(schema_types) => schema_types
                .iter()
                .any(|schema_type| schema_type == expected_type),
            _ => false,
        }
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

#[utoipa::path(
    post,
    path = "/v1/pdf/images",
    request_body(
        content = PdfImagesMultipartRequest,
        content_type = "multipart/form-data",
        description = "POST /v1/pdf/images accepts multipart/form-data with a required PDF file and optional response_format (png, jpeg, or webp), pages (for example 1,3-5), and scale (0.1..=4.0) fields."
    ),
    responses(
        (status = 200, description = "ZIP archive of rendered PDF page images", content_type = "application/zip", body = Vec<u8>),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "pdf"
)]
#[allow(dead_code)]
async fn create_pdf_images_doc() {}
