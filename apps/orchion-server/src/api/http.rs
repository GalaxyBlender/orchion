use crate::api::http_audio::{create_speech, create_transcription, create_transcription_ws};
use crate::api::http_models::list_models;
use crate::api::http_ocr::create_ocr;
use crate::api::http_pdf_images::create_pdf_images;
use crate::api::{docs, ui};
use crate::infrastructure::orchion::AppState;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::header::LOCATION;
use axum::http::{HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

pub fn router(state: Arc<AppState>) -> Router {
    router_with_ui_routes(state, ui::routes())
}

pub fn router_with_ui_routes(state: Arc<AppState>, ui_routes: Router<Arc<AppState>>) -> Router {
    let max_upload_size = state.config().server.max_upload_size;
    let mut router = Router::new()
        .route("/", get(root_redirect))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/pdf/images", post(create_pdf_images));

    if state.config().services.tts.enabled {
        router = router.route("/v1/audio/speech", post(create_speech));
    }
    if state.config().services.asr.enabled {
        router = router
            .route("/v1/audio/transcriptions", post(create_transcription))
            .route(
                "/v1/audio/transcriptions/stream",
                get(create_transcription_ws),
            );
    }
    if state.config().services.ocr.active() || state.config().services.ocr_vl.active() {
        router = router.route("/v1/ocr", post(create_ocr));
    }

    router
        .merge(ui_routes)
        .merge(docs::swagger_ui())
        .layer(DefaultBodyLimit::max(max_upload_size))
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::DEBUG))
                .on_request(DefaultOnRequest::new().level(Level::DEBUG))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
}

async fn root_redirect() -> impl IntoResponse {
    (
        StatusCode::FOUND,
        [(LOCATION, HeaderValue::from_static("/ui"))],
    )
}

async fn healthz() -> &'static str {
    "ok"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::http_audio::parse_timestamp_granularities;
    use crate::api::http_ocr::{
        OcrServiceChoice, resolve_ocr_layout_model, resolve_ocr_max_tokens,
        resolve_ocr_response_format, resolve_ocr_service_choice, validate_ocr_parameters,
    };
    use crate::api::http_shared::multipart_file_suffix;
    use crate::api::openai::OcrApiFormat;
    use crate::settings::ServerConfig;
    use axum::body::Body;
    use axum::http::header::CONTENT_TYPE;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use http_body_util::BodyExt;
    use orchion::{AsrModel, KnownOcrModel, ModelId, OcrResponseFormat, OcrTask, TtsModel};
    use serde_json::Value;
    use std::time::Duration;
    use tower::ServiceExt;

    #[test]
    fn parse_timestamp_granularities_accepts_segment() {
        let values = vec!["segment".to_string()];

        assert!(parse_timestamp_granularities(&values).unwrap());
    }

    #[test]
    fn parse_timestamp_granularities_rejects_word() {
        let values = vec!["segment".to_string(), "word".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_timestamp_granularity")
        );
    }

    #[test]
    fn parse_timestamp_granularities_rejects_unknown_value() {
        let values = vec!["sentence".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("timestamp_granularities")
        );
    }

    #[test]
    fn multipart_file_suffix_uses_supported_mime_type() {
        assert_eq!(multipart_file_suffix(Some("image/png")), ".png");
        assert_eq!(multipart_file_suffix(Some("image/jpeg")), ".jpg");
        assert_eq!(multipart_file_suffix(Some("application/pdf")), ".pdf");
        assert_eq!(multipart_file_suffix(Some("video/mp4")), ".mp4");
        assert_eq!(multipart_file_suffix(Some("text/plain")), "");
        assert_eq!(multipart_file_suffix(None), "");
    }

    #[tokio::test]
    async fn ocr_route_is_absent_when_ocr_services_are_inactive() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn websocket_transcription_route_uses_stream_suffix() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });

        let old_response = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/audio/transcriptions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let stream_response = router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(old_response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_ne!(stream_response.status(), StatusCode::NOT_FOUND);
        assert_ne!(stream_response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn multipart_ocr_requires_file() {
        let boundary = "orchion-ocr-missing-file";
        let body = multipart_body(boundary, &[("model", "PaddlePaddle/PP-OCRv6_tiny")], None);

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "missing_required_parameter");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_empty_file() {
        let boundary = "orchion-ocr-empty-file";
        let body = multipart_body(
            boundary,
            &[("model", "PaddlePaddle/PP-OCRv6_tiny")],
            Some(("file", "empty.png", b"")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_invalid_response_format() {
        let boundary = "orchion-ocr-invalid-format";
        let body = multipart_body(
            boundary,
            &[("response_format", "verbose_json")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "unsupported_response_format");
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_unknown_model() {
        let boundary = "orchion-ocr-unknown-model";
        let body = multipart_body(
            boundary,
            &[("model", "Acme/Experimental-OCR")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "model_not_available");
        assert_eq!(body["error"]["param"], "model");
    }

    #[tokio::test]
    async fn pdf_images_route_requires_file() {
        let boundary = "orchion-pdf-images-missing-file";
        let body = multipart_body(boundary, &[("response_format", "png")], None);

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "missing_required_parameter");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_empty_file() {
        let boundary = "orchion-pdf-images-empty-file";
        let body = multipart_body(boundary, &[], Some(("file", "empty.pdf", b"")));

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_non_pdf_file() {
        let boundary = "orchion-pdf-images-non-pdf-file";
        let body = multipart_body(boundary, &[], Some(("file", "document.png", b"image")));

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_response_format() {
        let boundary = "orchion-pdf-images-invalid-format";
        let body = multipart_body(
            boundary,
            &[("response_format", "gif")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_pages() {
        let boundary = "orchion-pdf-images-invalid-pages";
        let body = multipart_body(
            boundary,
            &[("pages", "2-1")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "pages");
    }

    #[tokio::test]
    async fn pdf_images_route_rejects_invalid_scale() {
        let boundary = "orchion-pdf-images-invalid-scale";
        let body = multipart_body(
            boundary,
            &[("scale", "4.1")],
            Some(("file", "document.pdf", b"%PDF-1.7\n")),
        );

        let response = post_pdf_images(test_state(false, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "scale");
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_for_default_markdown() {
        let state = test_state(true, true);

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_for_default_html() {
        let state = test_state(true, true);

        let choice = resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Html)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_defaults_ocr_vl_format_when_only_ocr_vl_is_active() {
        let state = test_state(false, true);

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();
        let response_format = resolve_ocr_response_format(&state, choice, None);

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Markdown);
    }

    #[test]
    fn resolve_ocr_service_choice_defaults_to_ocr_config_format_when_ocr_is_active() {
        let state = test_state_with_config(true, true, |config| {
            config.services.ocr.format = OcrResponseFormat::Text;
        });

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();
        let response_format = resolve_ocr_response_format(&state, choice, None);

        assert_eq!(choice.model(), KnownOcrModel::PpOcrV6Tiny);
        assert!(!choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Text);
    }

    #[test]
    fn resolve_ocr_service_choice_keeps_explicit_markdown_preference_for_ocr_vl() {
        let state = test_state(true, true);

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();
        let response_format =
            resolve_ocr_response_format(&state, choice, Some(OcrApiFormat::Markdown));

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Markdown);
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_fallback_for_default_markdown() {
        let state = test_state_with_config(true, true, |config| {
            let ocr_vl_model = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
            config.services.ocr_vl.default_model = None;
            config.services.ocr_vl.available_models = vec![ocr_vl_model];
        });

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_fallback_for_default_format() {
        let state = test_state_with_config(true, true, |config| {
            config.services.ocr.default_model = None;
        });

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PpOcrV6Tiny);
        assert!(!choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_layout_model_does_not_use_ocr_vl_layout_default_without_request_value() {
        let state = test_state_with_config(false, true, |config| {
            let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
            config.services.ocr_vl.layout_default_model = Some(layout_model);
        });
        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        let resolved = resolve_ocr_layout_model(&state, choice, None);

        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_ocr_layout_model_does_not_use_ocr_layout_default_without_request_value() {
        let state = test_state_with_config(true, false, |config| {
            let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
            config.services.ocr.layout_default_model = Some(layout_model);
        });
        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        let resolved = resolve_ocr_layout_model(&state, choice, None);

        assert_eq!(resolved, None);
    }

    #[test]
    fn validate_ocr_parameters_allows_markdown_for_traditional_layout() {
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let state = test_state_with_config(true, false, |config| {
            config.services.ocr.layout_available_models = vec![layout_model.clone()];
        });
        let choice = OcrServiceChoice::Ocr {
            model: KnownOcrModel::PpOcrV6Tiny,
        };

        validate_ocr_parameters(
            choice,
            OcrApiFormat::Markdown,
            OcrTask::Ocr,
            Some(&layout_model),
            None,
            &state.config().services.ocr,
            &state.config().services.ocr_vl,
        )
        .unwrap();
    }

    #[test]
    fn validate_ocr_parameters_rejects_unconfigured_traditional_layout() {
        let state = test_state(true, false);
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let choice = OcrServiceChoice::Ocr {
            model: KnownOcrModel::PpOcrV6Tiny,
        };

        let error = validate_ocr_parameters(
            choice,
            OcrApiFormat::Json,
            OcrTask::Ocr,
            Some(&layout_model),
            None,
            &state.config().services.ocr,
            &state.config().services.ocr_vl,
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
        assert_eq!(error.error.param.as_deref(), Some("layout_model"));
    }

    #[test]
    fn omitted_ocr_vl_max_tokens_uses_configured_limit() {
        let state = test_state_with_config(false, true, |config| {
            config.services.ocr_vl.max_tokens = 64;
        });
        let choice = OcrServiceChoice::OcrVl {
            model: KnownOcrModel::PaddleOcrVl16,
        };

        assert_eq!(
            resolve_ocr_max_tokens(choice, None, &state.config().services.ocr_vl),
            Some(64)
        );
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_unknown_explicit_model() {
        let state = test_state(true, false);

        let error = resolve_ocr_service_choice(
            &state,
            Some("Acme/Experimental-OCR"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_traditional_model_not_configured_for_ocr_vl_service() {
        let state = test_state(false, true);

        let error = resolve_ocr_service_choice(
            &state,
            Some("PaddlePaddle/PP-OCRv6_tiny"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
        assert_eq!(error.error.param.as_deref(), Some("model"));
    }

    async fn post_ocr(state: Arc<AppState>, boundary: &str, body: Vec<u8>) -> Response {
        router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn post_pdf_images(state: Arc<AppState>, boundary: &str, body: Vec<u8>) -> Response {
        router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/pdf/images")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn multipart_body(
        boundary: &str,
        fields: &[(&str, &str)],
        file: Option<(&str, &str, &[u8])>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        if let Some((field_name, file_name, file_bytes)) = file {
            let content_type = if file_name.to_ascii_lowercase().ends_with(".pdf") {
                "application/pdf"
            } else {
                "image/png"
            };
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_name}\"\r\nContent-Type: {content_type}\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(file_bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn test_state(ocr_active: bool, ocr_vl_active: bool) -> Arc<AppState> {
        test_state_with_config(ocr_active, ocr_vl_active, |_| {})
    }

    fn test_state_with_config(
        ocr_active: bool,
        ocr_vl_active: bool,
        configure: impl FnOnce(&mut ServerConfig),
    ) -> Arc<AppState> {
        let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config.services.ocr.enabled = ocr_active;
        config.services.ocr.default_model =
            Some(ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap());
        config.services.ocr.available_models = vec![
            ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap(),
            ModelId::parse("PaddlePaddle/PP-OCRv6_small").unwrap(),
        ];
        config.services.ocr_vl.enabled = ocr_vl_active;
        config.services.ocr_vl.default_model =
            Some(ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap());
        config.services.ocr_vl.available_models =
            vec![ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()];
        config.services.asr.available_models =
            vec![AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap()];
        config.services.tts.available_models =
            vec![TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").unwrap()];
        config.services.asr.idle_timeout = Duration::from_secs(600);
        config.services.tts.idle_timeout = Duration::from_secs(600);
        configure(&mut config);

        Arc::new(AppState::from_prepared_config(config).unwrap())
    }
}
