use crate::api::http_audio::{create_speech, create_transcription, create_transcription_ws};
use crate::api::http_models::list_models;
use crate::api::http_ocr::create_ocr;
use crate::api::http_pdf_images::create_pdf_images;
use crate::api::http_shared::origin_is_allowed;
use crate::api::{docs, ui};
use crate::application::ServerApplication;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::header::LOCATION;
use axum::http::{HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::cors::{AllowHeaders, AllowOrigin, Any, CorsLayer};
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

pub fn router<S>(state: Arc<S>) -> Router
where
    S: ServerApplication,
{
    router_with_ui_routes(state, ui::routes::<S>())
}

pub fn router_with_ui_routes<S>(state: Arc<S>, ui_routes: Router<Arc<S>>) -> Router
where
    S: ServerApplication,
{
    let policy = state.api_policy();
    let max_upload_size = policy.max_upload_size;
    let cors = cors_layer(&policy.cors_allowed_origins);
    let tts_enabled = policy.tts_models.is_some();
    let asr_enabled = policy.asr.is_some();
    let ocr_enabled = policy.ocr.is_some() || policy.ocr_vl.is_some();
    let mut router = Router::new()
        .route("/", get(root_redirect))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models::<S>))
        .route("/v1/pdf/images", post(create_pdf_images::<S>));

    if tts_enabled {
        router = router.route("/v1/audio/speech", post(create_speech::<S>));
    }
    if asr_enabled {
        router = router
            .route("/v1/audio/transcriptions", post(create_transcription::<S>))
            .route(
                "/v1/audio/transcriptions/stream",
                get(create_transcription_ws::<S>),
            );
    }
    if ocr_enabled {
        router = router.route("/v1/ocr", post(create_ocr::<S>));
    }

    router
        .merge(ui_routes)
        .merge(docs::swagger_ui())
        .layer(DefaultBodyLimit::max(max_upload_size))
        .layer(cors)
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::DEBUG))
                .on_request(DefaultOnRequest::new().level(Level::DEBUG))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
}

fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(Any);
    if allowed_origins == ["*"] {
        layer.allow_origin(Any)
    } else {
        let allowed_origins = allowed_origins.to_vec();
        layer.allow_origin(AllowOrigin::predicate(move |origin, _| {
            origin_is_allowed(&allowed_origins, origin)
        }))
    }
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
    use crate::api::http_shared::multipart_file_suffix;
    use crate::infrastructure::orchion::AppState;
    use crate::settings::ServerConfig;
    use axum::body::Body;
    use axum::http::header::{
        ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
        ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, CONTENT_TYPE, ORIGIN,
    };
    use axum::http::{Method, Request, StatusCode};
    use axum::response::Response;
    use futures_util::SinkExt;
    use http_body_util::BodyExt;
    use orchion::{AsrModel, ModelId, TtsModel};
    use serde_json::Value;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio_tungstenite::{connect_async, tungstenite::Message};
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
    async fn cors_allows_every_origin_by_default() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
    }

    #[tokio::test]
    async fn cors_only_echoes_configured_origins() {
        let state = test_state_with_config(false, false, |config| {
            config.server.cors_allowed_origins = vec!["https://app.example.com".to_string()];
        });
        let allowed = router_with_ui_routes(state.clone(), Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let denied = router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header(ORIGIN, "https://other.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            allowed.headers()[ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://app.example.com"
        );
        assert!(!denied.headers().contains_key(ACCESS_CONTROL_ALLOW_ORIGIN));
    }

    #[tokio::test]
    async fn cors_handles_authorization_preflight_requests() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/healthz")
                    .header(ORIGIN, "https://app.example.com")
                    .header(ACCESS_CONTROL_REQUEST_METHOD, "GET")
                    .header(ACCESS_CONTROL_REQUEST_HEADERS, "authorization")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        assert!(
            response
                .headers()
                .contains_key(ACCESS_CONTROL_ALLOW_METHODS)
        );
        assert_eq!(
            response.headers()[ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization"
        );
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
    async fn disconnected_websocket_leaves_the_inference_queue() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_concurrent_inference = 1;
            config.server.max_websocket_connections = 1;
        });
        let inference = state.acquire_inference().await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(Arc::clone(&state), Router::new());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket
            .send(Message::Text(
                r#"{"type":"start","model":"Qwen/Qwen3-ASR-0.6B","input_audio_format":"pcm_s16le","sample_rate":16000}"#
                    .into(),
            ))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let Some(permit) = state.try_acquire_websocket() else {
                    break;
                };
                drop(permit);
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        socket.send(Message::Close(None)).await.unwrap();
        drop(socket);
        let websocket = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(permit) = state.try_acquire_websocket() {
                    break permit;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(websocket);

        drop(inference);
        tokio::time::timeout(Duration::from_secs(1), state.acquire_inference())
            .await
            .unwrap();
        server.abort();
        let _ = server.await;
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
        config.services.asr.idle_timeout = Duration::from_mins(10);
        config.services.tts.idle_timeout = Duration::from_mins(10);
        configure(&mut config);

        Arc::new(AppState::from_prepared_config(config).unwrap())
    }
}
