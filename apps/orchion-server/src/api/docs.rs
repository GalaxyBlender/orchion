use crate::api::activity::{
    ActivityEntry, ActivityOperation, ActivityOutcome, ActivityPage, ActivityState,
    ActivitySummary, ActivityTransport,
};
use crate::api::http::{ReadinessReason, ReadinessReasonCode, ReadinessResponse};
use crate::api::http_llm::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionSseEvent,
    ChatReasoningControlRequest, ChatReasoningControlResponse, CompletionRequest,
    CompletionResponse, CompletionSseEvent, EmbeddingsRequest, EmbeddingsResponse,
    ResponsesInputTokensResponse, ResponsesRequest, ResponsesResponse, ResponsesSseEvent,
};
use crate::api::http_models::{ModelControlRequest, ModelStatusList};
use crate::api::http_streams::{StreamLookupRequest, StreamLookupResponse};
use crate::api::llm_streams::{StreamMetadata, StreamProtocol, StreamStatus};
use crate::api::openai::{
    ErrorBody, ModelList, OcrApiFormat, OcrJsonResponse, SpeechRequest, TranscriptionJson,
    TranscriptionVerboseJson,
};
use crate::application::model_lifecycle::ModelStatus;
use orchion::OcrTask;
use orchion::docs::PdfImageFormat;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::openapi::{Content, Ref, RefOr};
use utoipa::{Modify, OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(healthz_doc, readyz_doc, metrics_doc, list_models_doc, retrieve_model_doc, list_model_statuses_doc, load_model_doc, unload_model_doc, create_completion_doc, create_chat_completion_doc, control_chat_completion_doc, create_response_doc, get_stream_doc, delete_stream_doc, lookup_streams_doc, count_response_input_tokens_doc, create_embeddings_doc, create_speech_doc, create_transcription_doc, create_ocr_doc, create_pdf_images_doc, list_activity_doc, activity_events_doc),
    components(schemas(ReadinessResponse, ReadinessReason, ReadinessReasonCode, CompletionRequest, CompletionResponse, CompletionSseEvent, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionSseEvent, ChatReasoningControlRequest, ChatReasoningControlResponse, ResponsesRequest, ResponsesResponse, ResponsesSseEvent, StreamLookupRequest, StreamLookupResponse, StreamMetadata, StreamProtocol, StreamStatus, ResponsesInputTokensResponse, EmbeddingsRequest, EmbeddingsResponse, SpeechRequest, ErrorBody, ModelList, ModelStatusList, ModelStatus, ModelControlRequest, TranscriptionJson, TranscriptionVerboseJson, OcrJsonResponse, OcrApiFormat, OcrTask, OcrMultipartRequest, PdfImageFormat, PdfImagesMultipartRequest, ActivityPage, ActivityEntry, ActivitySummary, ActivityState, ActivityTransport, ActivityOperation, ActivityOutcome)),
    modifiers(&BearerAuth),
    tags(
        (name = "audio", description = "OpenAI-compatible audio APIs"),
        (name = "llm", description = "Text-only OpenAI-compatible subset"),
        (name = "ocr", description = "OCR and OCR-VL APIs"),
        (name = "pdf", description = "PDF rendering APIs"),
        (name = "models", description = "Model discovery and runtime lifecycle APIs"),
        (name = "activity", description = "Live and retained request metadata"),
        (name = "observability", description = "Liveness, readiness, and OpenMetrics")
    )
)]
struct ApiDoc;

struct BearerAuth;

impl Modify for BearerAuth {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("API key")
                        .build(),
                ),
            );
        }
        add_json_response_content(openapi, "/v1/chat/completions", "ChatCompletionResponse");
        add_json_response_content(openapi, "/v1/completions", "CompletionResponse");
        add_json_response_content(openapi, "/v1/responses", "ResponsesResponse");
        add_json_response_content(openapi, "/v1/embeddings", "EmbeddingsResponse");
    }
}

fn add_json_response_content(openapi: &mut utoipa::openapi::OpenApi, path: &str, schema: &str) {
    let Some(operation) = openapi
        .paths
        .paths
        .get_mut(path)
        .and_then(|item| item.post.as_mut())
    else {
        return;
    };
    let Some(RefOr::T(response)) = operation.responses.responses.get_mut("200") else {
        return;
    };
    response.content.insert(
        "application/json".to_string(),
        Content::new(Some(Ref::from_schema_name(schema))),
    );
}

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

#[derive(ToSchema)]
#[allow(dead_code)]
struct OcrMultipartRequest {
    #[schema(value_type = String, format = Binary, content_media_type = "application/octet-stream")]
    file: String,
    #[schema(example = "paddlepaddle/pp-ocrv6-medium")]
    model: String,
    response_format: Option<OcrApiFormat>,
    task: Option<OcrTask>,
    #[schema(minimum = 1)]
    max_tokens: Option<usize>,
}

#[must_use]
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
        assert_eq!(
            spec["components"]["schemas"]["OcrMultipartRequest"]["required"],
            serde_json::json!(["file", "model"])
        );
        assert_eq!(
            spec["paths"]["/v1/ocr"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
                ["$ref"],
            "#/components/schemas/OcrMultipartRequest"
        );
        assert_eq!(
            spec["components"]["schemas"]["OcrTask"]["enum"],
            serde_json::json!(["ocr", "table", "formula", "chart", "spotting", "seal"])
        );
        assert_eq!(
            spec["components"]["schemas"]["OcrMultipartRequest"]["properties"]["max_tokens"]["minimum"],
            1
        );
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
            serde_json::json!(["asr", "tts", "ocr", "llm"])
        );
        assert!(model_object["properties"].get("subtype").is_none());
        assert!(model_object["properties"]["name"].is_object());
        assert_eq!(
            spec["components"]["schemas"]["ModelCapability"]["enum"],
            serde_json::json!([
                "asr_transcription",
                "asr_streaming",
                "tts_voice_cloning",
                "tts_preset_speakers",
                "tts_voice_design",
                "ocr_text",
                "ocr_layout",
                "ocr_table_structure",
                "ocr_vision_language",
                "ocr_markdown",
                "ocr_html",
                "llm_chat",
                "llm_responses",
                "llm_streaming",
                "llm_embeddings",
                "llm_completions",
                "llm_input_tokens",
                "llm_tools",
                "llm_parallel_tools",
                "llm_json_object",
                "llm_json_schema",
                "llm_logprobs",
                "llm_logit_bias",
                "llm_multiple_choices",
                "llm_reasoning",
                "llm_vision",
                "llm_resumable_streaming",
                "llm_reasoning_control"
            ])
        );
    }

    #[test]
    fn openapi_includes_resumable_stream_paths_and_schemas() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        assert!(spec["paths"]["/v1/stream"]["get"].is_object());
        assert!(spec["paths"]["/v1/stream"]["delete"].is_object());
        assert!(spec["paths"]["/v1/streams/lookup"]["post"].is_object());
        assert!(spec["components"]["schemas"]["LlmStreamMetadata"].is_object());
        assert!(
            spec["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["headers"]
                ["X-Orchion-Stream-ID"]
                .is_object()
        );
        assert_eq!(
            spec["paths"]["/v1/stream"]["get"]["responses"]["409"]["content"]["application/json"]["schema"]
                ["$ref"],
            "#/components/schemas/ErrorBody"
        );
    }

    #[test]
    fn openapi_includes_model_lifecycle_paths_and_schemas() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();

        assert!(spec["paths"]["/api/models/status"]["get"].is_object());
        assert!(spec["paths"]["/api/models/load"]["post"].is_object());
        assert!(spec["paths"]["/api/models/unload"]["post"].is_object());
        assert!(spec["paths"].get("/v1/models/status").is_none());
        assert!(spec["paths"].get("/v1/models/prewarm").is_none());
        assert!(spec["paths"].get("/v1/models/unload").is_none());
        for (path, method) in [
            ("/api/models/status", "get"),
            ("/api/models/load", "post"),
            ("/api/models/unload", "post"),
        ] {
            assert_eq!(
                spec["paths"][path][method]["security"][0]["bearer_auth"],
                serde_json::json!([]),
                "{method} {path}"
            );
        }
        assert!(spec["components"]["schemas"]["ModelStatus"].is_object());
        assert!(spec["components"]["schemas"]["ModelControlRequest"].is_object());
    }

    #[test]
    fn openapi_documents_readiness_and_authenticated_openmetrics() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        assert!(spec["paths"]["/readyz"]["get"]["responses"]["200"].is_object());
        assert!(spec["paths"]["/readyz"]["get"]["responses"]["503"].is_object());
        assert_eq!(
            spec["paths"]["/metrics"]["get"]["security"][0]["bearer_auth"],
            serde_json::json!([])
        );
        assert!(spec["components"]["schemas"]["ReadinessResponse"].is_object());
        assert!(spec["components"]["schemas"]["ReadinessReasonCode"].is_object());
    }

    #[test]
    fn openapi_includes_embeddings_contract() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        assert!(spec["paths"]["/v1/embeddings"]["post"].is_object());
        assert!(spec["components"]["schemas"]["EmbeddingsRequest"].is_object());
        assert!(spec["components"]["schemas"]["EmbeddingsResponse"].is_object());
        assert_eq!(
            spec["paths"]["/v1/embeddings"]["post"]["security"][0]["bearer_auth"],
            serde_json::json!([])
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

    #[test]
    fn openapi_includes_activity_endpoints_and_contract() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();

        assert!(spec["paths"]["/api/activity"]["get"].is_object());
        assert!(spec["paths"]["/api/activity/events"]["get"].is_object());
        assert!(spec["components"]["schemas"]["ActivityPage"].is_object());
        assert_eq!(
            spec["components"]["securitySchemes"]["bearer_auth"]["scheme"],
            "bearer"
        );
        assert_eq!(
            spec["paths"]["/api/activity"]["get"]["security"][0]["bearer_auth"],
            serde_json::json!([])
        );
        assert_eq!(
            spec["components"]["schemas"]["ActivityOutcome"]["enum"],
            serde_json::json!([
                "success",
                "client_error",
                "server_error",
                "cancelled",
                "disconnected",
                "timeout",
                "resource_exhausted"
            ])
        );
        assert_eq!(
            spec["components"]["schemas"]["ActivityOperation"]["enum"],
            serde_json::json!([
                "asr",
                "asr_stream",
                "tts",
                "ocr",
                "pdf",
                "chat",
                "responses",
                "embeddings",
                "completions",
                "input_tokens"
            ])
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "keeps the generated LLM OpenAPI contract assertions in one inspection"
    )]
    fn openapi_includes_text_llm_subset() {
        let spec = serde_json::to_value(ApiDoc::openapi()).unwrap();
        assert!(spec["paths"]["/v1/chat/completions"]["post"].is_object());
        assert!(spec["paths"]["/v1/chat/completions/control"]["post"].is_object());
        assert!(spec["paths"]["/v1/completions"]["post"].is_object());
        assert!(spec["paths"]["/v1/responses"]["post"].is_object());
        assert!(spec["paths"]["/v1/responses/input_tokens"]["post"].is_object());
        assert!(spec["components"]["schemas"]["ChatCompletionRequest"].is_object());
        assert!(spec["components"]["schemas"]["ChatReasoningControlRequest"].is_object());
        assert!(spec["components"]["schemas"]["ChatReasoningControlResponse"].is_object());
        assert!(
            spec["components"]["schemas"]["ChatCompletionRequest"]["properties"]
                ["reasoning_control"]
                .is_object()
        );
        assert!(spec["components"]["schemas"]["ResponsesRequest"].is_object());
        assert_eq!(
            spec["components"]["schemas"]["ResponseImageDetail"]["enum"],
            serde_json::json!(["auto"])
        );
        assert!(
            spec["components"]["schemas"]["CompletionStreamChunk"]["properties"]["timings"]
                .is_object()
        );
        for path in ["/v1/completions", "/v1/chat/completions", "/v1/responses"] {
            assert_eq!(
                spec["paths"][path]["post"]["security"][0]["bearer_auth"],
                serde_json::json!([])
            );
            let content = &spec["paths"][path]["post"]["responses"]["200"]["content"];
            assert!(content["application/json"].is_object(), "{path}: {content}");
            assert!(
                content["text/event-stream"].is_object(),
                "{path}: {content}"
            );
        }
        assert!(schema_references(
            &spec["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"]["text/event-stream"]
                ["schema"],
            "ChatCompletionSseEvent"
        ));
        assert!(schema_references(
            &spec["paths"]["/v1/responses"]["post"]["responses"]["200"]["content"]["text/event-stream"]
                ["schema"],
            "ResponsesSseEvent"
        ));
        assert!(spec["components"]["schemas"]["ChatCompletionSseEvent"]["oneOf"].is_array());
        assert!(spec["components"]["schemas"]["ResponsesSseEvent"]["oneOf"].is_array());
        let timing_schema = &spec["components"]["schemas"]["TimingObject"];
        for field in [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ] {
            assert!(timing_schema["properties"][field].is_object(), "{field}");
        }
        assert!(schema_references(
            &spec["components"]["schemas"]["ChatCompletionResponse"]["properties"]["timings"],
            "TimingObject"
        ));
        assert!(schema_references(
            &spec["components"]["schemas"]["ResponsesResponse"]["properties"]["timings"],
            "TimingObject"
        ));
        assert!(schema_references(
            &spec["components"]["schemas"]["ChatCompletionSseEvent"],
            "ErrorBody"
        ));
        assert_eq!(
            spec["components"]["schemas"]["ResponsesSseEvent"]["oneOf"]
                .as_array()
                .unwrap()
                .len(),
            17
        );

        assert!(schema_contains_number(
            &spec["components"]["schemas"]["ResponsesRequest"]["properties"]["max_output_tokens"],
            "minimum",
            16
        ));

        let error_schema = &spec["components"]["schemas"]["ErrorObject"];
        for field in ["param", "code"] {
            assert!(
                error_schema["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value == field)
            );
            assert!(schema_contains_type(
                &error_schema["properties"][field],
                "null"
            ));
        }
        for (schema, field) in [
            ("ChatChoice", "logprobs"),
            ("AssistantMessage", "refusal"),
            ("ChatCompletionStreamChunk", "usage"),
            ("LlmStreamErrorEvent", "code"),
            ("LlmStreamErrorEvent", "param"),
        ] {
            let component = &spec["components"]["schemas"][schema];
            assert!(
                component["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value == field),
                "{schema}.{field} is not required"
            );
            assert!(
                schema_contains_type(&component["properties"][field], "null"),
                "{schema}.{field} is not nullable"
            );
        }

        let mut event_types = Vec::new();
        for (schema, event_type, fields) in [
            (
                "ResponseCreatedSseEvent",
                "response.created",
                &["response"][..],
            ),
            (
                "ResponseInProgressSseEvent",
                "response.in_progress",
                &["response"][..],
            ),
            (
                "ResponseOutputItemAddedSseEvent",
                "response.output_item.added",
                &["output_index", "item"][..],
            ),
            (
                "ResponseContentPartAddedSseEvent",
                "response.content_part.added",
                &["item_id", "output_index", "content_index", "part"][..],
            ),
            (
                "ResponseOutputTextDeltaSseEvent",
                "response.output_text.delta",
                &[
                    "item_id",
                    "output_index",
                    "content_index",
                    "delta",
                    "logprobs",
                ][..],
            ),
            (
                "ResponseOutputTextDoneSseEvent",
                "response.output_text.done",
                &[
                    "item_id",
                    "output_index",
                    "content_index",
                    "text",
                    "logprobs",
                ][..],
            ),
            (
                "ResponseContentPartDoneSseEvent",
                "response.content_part.done",
                &["item_id", "output_index", "content_index", "part"][..],
            ),
            (
                "ResponseOutputItemDoneSseEvent",
                "response.output_item.done",
                &["output_index", "item"][..],
            ),
            (
                "ResponseCompletedSseEvent",
                "response.completed",
                &["response"][..],
            ),
            (
                "ResponseIncompleteSseEvent",
                "response.incomplete",
                &["response"][..],
            ),
            (
                "LlmStreamErrorEvent",
                "error",
                &["code", "message", "param"][..],
            ),
            (
                "ResponseFunctionCallArgumentsDeltaSseEvent",
                "response.function_call_arguments.delta",
                &["item_id", "output_index", "delta"][..],
            ),
            (
                "ResponseFunctionCallArgumentsDoneSseEvent",
                "response.function_call_arguments.done",
                &["item_id", "output_index", "arguments"][..],
            ),
            (
                "ResponseReasoningSummaryPartAddedSseEvent",
                "response.reasoning_summary_part.added",
                &["item_id", "output_index", "summary_index", "part"][..],
            ),
            (
                "ResponseReasoningSummaryTextDeltaSseEvent",
                "response.reasoning_summary_text.delta",
                &["item_id", "output_index", "summary_index", "delta"][..],
            ),
            (
                "ResponseReasoningSummaryTextDoneSseEvent",
                "response.reasoning_summary_text.done",
                &["item_id", "output_index", "summary_index", "text"][..],
            ),
            (
                "ResponseReasoningSummaryPartDoneSseEvent",
                "response.reasoning_summary_part.done",
                &["item_id", "output_index", "summary_index", "part"][..],
            ),
        ] {
            let component = &spec["components"]["schemas"][schema];
            let required = component["required"].as_array().unwrap();
            assert!(required.iter().any(|value| value == "type"), "{schema}");
            assert!(
                required.iter().any(|value| value == "sequence_number"),
                "{schema}"
            );
            for field in fields {
                assert!(
                    required.iter().any(|value| value == field),
                    "{schema}.{field}"
                );
            }
            let discriminator_values = schema_enum_strings(&component["properties"]["type"]);
            assert_eq!(discriminator_values, [event_type], "{schema}.type");
            assert!(!event_types.contains(&event_type), "duplicate {event_type}");
            event_types.push(event_type);
            assert!(schema_references(
                &spec["components"]["schemas"]["ResponsesSseEvent"],
                schema
            ));
        }
        assert_eq!(event_types.len(), 17);

        let response_schema = &spec["components"]["schemas"]["ResponsesResponse"];
        let required = response_schema["required"].as_array().unwrap();
        for field in [
            "error",
            "incomplete_details",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "temperature",
            "top_p",
            "usage",
        ] {
            assert!(
                required.iter().any(|value| value == field),
                "missing {field}"
            );
            assert!(
                schema_contains_type(&response_schema["properties"][field], "null"),
                "{field} is not nullable: {}",
                response_schema["properties"][field]
            );
        }
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

    fn schema_contains_type(schema: &Value, expected_type: &str) -> bool {
        if schema_has_type(schema, expected_type) {
            return true;
        }
        match schema {
            Value::Array(items) => items
                .iter()
                .any(|item| schema_contains_type(item, expected_type)),
            Value::Object(fields) => fields
                .values()
                .any(|value| schema_contains_type(value, expected_type)),
            _ => false,
        }
    }

    fn schema_contains_number(schema: &Value, key: &str, expected: u64) -> bool {
        if schema.get(key).and_then(Value::as_u64) == Some(expected) {
            return true;
        }
        match schema {
            Value::Array(items) => items
                .iter()
                .any(|item| schema_contains_number(item, key, expected)),
            Value::Object(fields) => fields
                .values()
                .any(|value| schema_contains_number(value, key, expected)),
            _ => false,
        }
    }

    fn schema_enum_strings(schema: &Value) -> Vec<&str> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return values.iter().filter_map(Value::as_str).collect();
        }
        match schema {
            Value::Array(items) => items.iter().flat_map(schema_enum_strings).collect(),
            Value::Object(fields) => fields.values().flat_map(schema_enum_strings).collect(),
            _ => Vec::new(),
        }
    }
}

#[utoipa::path(
    get,
    path = "/healthz",
    responses((status = 200, description = "Server health", body = String))
)]
#[allow(dead_code)]
fn healthz_doc() {}

#[utoipa::path(
    get,
    path = "/readyz",
    responses(
        (status = 200, description = "Server is ready", body = ReadinessResponse),
        (status = 503, description = "Server is not ready", body = ReadinessResponse)
    ),
    tag = "observability"
)]
#[allow(dead_code)]
fn readyz_doc() {}

#[utoipa::path(
    get,
    path = "/metrics",
    responses(
        (status = 200, description = "OpenMetrics 1.0 exposition", body = String, content_type = "application/openmetrics-text"),
        (status = 401, description = "Bearer authentication failed", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "observability"
)]
#[allow(dead_code)]
fn metrics_doc() {}

#[utoipa::path(
    get,
    path = "/v1/models",
    responses(
        (status = 200, description = "Configured model list", body = ModelList),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "models"
)]
#[allow(dead_code)]
fn list_models_doc() {}

#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    params(("model" = String, Path, description = "Configured public model ID")),
    responses(
        (status = 200, description = "Configured model", body = crate::api::openai::ModelObject),
        (status = 404, description = "Model not found", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "models"
)]
#[allow(dead_code)]
fn retrieve_model_doc() {}

#[utoipa::path(
    get,
    path = "/api/models/status",
    responses(
        (status = 200, description = "Configured model runtime residency", body = ModelStatusList),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "models"
)]
#[allow(dead_code)]
fn list_model_statuses_doc() {}

#[utoipa::path(
    post,
    path = "/api/models/load",
    request_body = ModelControlRequest,
    responses(
        (status = 200, description = "Loaded model runtime", body = ModelStatus),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "models"
)]
#[allow(dead_code)]
fn load_model_doc() {}

#[utoipa::path(
    post,
    path = "/api/models/unload",
    request_body = ModelControlRequest,
    responses(
        (status = 200, description = "Unloaded model runtime", body = ModelStatus),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "models"
)]
#[allow(dead_code)]
fn unload_model_doc() {}

#[utoipa::path(
    post,
    path = "/v1/completions",
    params(("X-Orchion-Resumable" = Option<bool>, Header, description = "Set to true with stream=true to retain a bounded resumable stream")),
    request_body = CompletionRequest,
    responses(
        (status = 200, description = "Legacy text completion JSON", body = CompletionResponse, content_type = "application/json"),
        (status = 200, description = "Legacy text completion SSE", body = CompletionSseEvent, content_type = "text/event-stream", headers(
            ("X-Orchion-Stream-ID" = String, description = "Present for opt-in resumable streams"),
            ("X-Orchion-Stream-TTL-Seconds" = u64, description = "Retention TTL for an opt-in resumable stream")
        )),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn create_completion_doc() {}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    params(("X-Orchion-Resumable" = Option<bool>, Header, description = "Set to true with stream=true to retain a bounded resumable stream")),
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Indexed rich chat completion JSON", body = ChatCompletionResponse, content_type = "application/json"),
        (status = 200, description = "Indexed rich chat completion SSE", body = ChatCompletionSseEvent, content_type = "text/event-stream", headers(
            ("X-Orchion-Completion-ID" = String, description = "Stable unguessable completion ID used by reasoning control"),
            ("X-Orchion-Stream-ID" = String, description = "Present for opt-in resumable streams"),
            ("X-Orchion-Stream-TTL-Seconds" = u64, description = "Retention TTL for an opt-in resumable stream")
        )),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn create_chat_completion_doc() {}

#[utoipa::path(
    post,
    path = "/v1/chat/completions/control",
    request_body = ChatReasoningControlRequest,
    responses(
        (status = 200, description = "Bounded reasoning control result", body = ChatReasoningControlResponse),
        (status = 400, description = "Malformed control request", body = ErrorBody),
        (status = 503, description = "Control channel unavailable", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn control_chat_completion_doc() {}

#[utoipa::path(
    post,
    path = "/v1/responses",
    params(("X-Orchion-Resumable" = Option<bool>, Header, description = "Set to true with stream=true to retain a bounded resumable stream")),
    request_body = ResponsesRequest,
    responses(
        (status = 200, description = "Stateless dynamic-item response JSON", body = ResponsesResponse, content_type = "application/json"),
        (status = 200, description = "Stateless dynamic-item lifecycle SSE; this server terminates with response.completed, response.incomplete, or error (response.failed/cancelled are not currently emitted)", body = ResponsesSseEvent, content_type = "text/event-stream", headers(
            ("X-Orchion-Stream-ID" = String, description = "Present for opt-in resumable streams"),
            ("X-Orchion-Stream-TTL-Seconds" = u64, description = "Retention TTL for an opt-in resumable stream")
        )),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn create_response_doc() {}

#[utoipa::path(
    get,
    path = "/v1/stream",
    params(
        ("stream_id" = String, Query, description = "Resumable stream ID"),
        ("follow" = Option<bool>, Query, description = "Follow new events after replay; defaults to true"),
        ("Last-Event-ID" = Option<u64>, Header, description = "Replay strictly after this decimal SSE event ID")
    ),
    responses(
        (status = 200, description = "Replay followed by optional live SSE events", content_type = "text/event-stream", body = String),
        (status = 400, body = ErrorBody), (status = 404, body = ErrorBody),
        (status = 409, body = ErrorBody), (status = 429, body = ErrorBody), (status = 503, body = ErrorBody)
    ),
    security(("bearer_auth" = [])), tag = "llm"
)]
#[allow(dead_code)]
fn get_stream_doc() {}

#[utoipa::path(
    delete, path = "/v1/stream",
    params(("stream_id" = String, Query, description = "Resumable stream ID")),
    responses((status = 204, description = "Stream deleted or not visible to this principal"), (status = 400, body = ErrorBody)),
    security(("bearer_auth" = [])), tag = "llm"
)]
#[allow(dead_code)]
fn delete_stream_doc() {}

#[utoipa::path(
    post, path = "/v1/streams/lookup", request_body = StreamLookupRequest,
    responses((status = 200, body = StreamLookupResponse), (status = 400, body = ErrorBody)),
    security(("bearer_auth" = [])), tag = "llm"
)]
#[allow(dead_code)]
fn lookup_streams_doc() {}

#[utoipa::path(
    post,
    path = "/v1/responses/input_tokens",
        request_body = ResponsesRequest,
    responses(
        (status = 200, description = "Responses prepared input token count", body = ResponsesInputTokensResponse),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn count_response_input_tokens_doc() {}

#[utoipa::path(
    post,
    path = "/v1/embeddings",
    request_body = EmbeddingsRequest,
    responses(
        (status = 200, description = "OpenAI-compatible embedding list", body = EmbeddingsResponse),
        (status = 400, description = "OpenAI-compatible error", body = ErrorBody),
        (status = 500, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    security(("bearer_auth" = [])),
    tag = "llm"
)]
#[allow(dead_code)]
fn create_embeddings_doc() {}

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
fn create_speech_doc() {}

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
fn create_transcription_doc() {}

#[utoipa::path(
    post,
    path = "/v1/ocr",
    request_body(
        content = OcrMultipartRequest,
        content_type = "multipart/form-data",
        description = "POST /v1/ocr accepts multipart/form-data with required file and model fields plus optional response_format, task, and max_tokens fields. Structured response formats are available when the selected deployment reports the matching capability. Response formats are json, text, markdown, and html. Model IDs use {vendor}/{name}. Traditional metal maps to CoreML; OCR-VL metal maps to Candle Metal."
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
fn create_ocr_doc() {}

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
fn create_pdf_images_doc() {}

#[utoipa::path(
    get,
    path = "/api/activity",
    security(("bearer_auth" = [])),
    params(
        ("limit" = Option<usize>, Query, description = "History rows to return, from 1 to 200"),
        ("before" = Option<String>, Query, description = "Return completed requests with IDs older than this cursor"),
        ("operation" = Option<ActivityOperation>, Query, description = "Filter by operation"),
        ("outcome" = Option<ActivityOutcome>, Query, description = "Filter by completion outcome"),
        ("model" = Option<String>, Query, description = "Filter by exact model ID")
    ),
    responses(
        (status = 200, description = "Current in-flight requests and retained history", body = ActivityPage),
        (status = 400, description = "Invalid query", body = ErrorBody),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "activity"
)]
#[allow(dead_code)]
fn list_activity_doc() {}

#[utoipa::path(
    get,
    path = "/api/activity/events",
    security(("bearer_auth" = [])),
    responses(
        (status = 200, description = "Server-sent events named snapshot, started, updated, completed, or reset", content_type = "text/event-stream", body = String),
        (status = 401, description = "OpenAI-compatible error", body = ErrorBody)
    ),
    tag = "activity"
)]
#[allow(dead_code)]
fn activity_events_doc() {}
