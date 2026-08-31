use crate::api::activity::{ActivityHub, track_activity};
use crate::api::http_activity::{activity_events, list_activity};
use crate::api::http_audio::{create_speech, create_transcription, create_transcription_ws};
use crate::api::http_llm::{create_chat_completion, create_response};
use crate::api::http_models::{list_model_statuses, list_models, load_model, unload_model};
use crate::api::http_ocr::create_ocr;
use crate::api::http_pdf_images::create_pdf_images;
use crate::api::http_shared::origin_is_allowed;
use crate::api::{docs, ui};
use crate::application::ServerApplication;
use axum::extract::DefaultBodyLimit;
use axum::http::header::LOCATION;
use axum::http::{HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Router, middleware};
use std::sync::Arc;
use tokio::sync::watch;
use tower_http::cors::{AllowHeaders, AllowOrigin, Any, CorsLayer};
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

#[derive(Clone, Debug)]
pub struct ServerShutdown {
    signal: watch::Sender<bool>,
}

impl ServerShutdown {
    #[must_use]
    pub fn new() -> Self {
        let (signal, _) = watch::channel(false);
        Self { signal }
    }

    pub fn trigger(&self) {
        self.signal.send_replace(true);
    }

    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.signal.subscribe();
        if *receiver.borrow() {
            return;
        }
        let _ = receiver.changed().await;
    }
}

impl Default for ServerShutdown {
    fn default() -> Self {
        Self::new()
    }
}

pub fn router<S>(state: Arc<S>) -> Router
where
    S: ServerApplication,
{
    router_with_shutdown(state, ServerShutdown::new())
}

pub fn router_with_shutdown<S>(state: Arc<S>, shutdown: ServerShutdown) -> Router
where
    S: ServerApplication,
{
    router_with_ui_routes_and_shutdown(state, ui::routes::<S>(), shutdown)
}

pub fn router_with_ui_routes<S>(state: Arc<S>, ui_routes: Router<Arc<S>>) -> Router
where
    S: ServerApplication,
{
    router_with_ui_routes_and_shutdown(state, ui_routes, ServerShutdown::new())
}

fn router_with_ui_routes_and_shutdown<S>(
    state: Arc<S>,
    ui_routes: Router<Arc<S>>,
    shutdown: ServerShutdown,
) -> Router
where
    S: ServerApplication,
{
    let policy = state.api_policy();
    let max_upload_size = policy.max_upload_size;
    let cors = cors_layer(&policy.cors_allowed_origins);
    let tts_enabled = policy.tts_models.is_some();
    let asr_enabled = policy.asr.is_some();
    let ocr_enabled = policy.ocr_enabled;
    let llm_enabled = policy.llm_enabled;
    let activity = ActivityHub::new(policy.activity);
    let mut router = Router::new()
        .route("/", get(root_redirect))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models::<S>))
        .route("/api/models/status", get(list_model_statuses::<S>))
        .route("/api/models/load", post(load_model::<S>))
        .route("/api/models/unload", post(unload_model::<S>))
        .route("/v1/pdf/images", post(create_pdf_images::<S>))
        .route("/api/activity", get(list_activity::<S>))
        .route("/api/activity/events", get(activity_events::<S>));

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
    if llm_enabled {
        router = router
            .route("/v1/chat/completions", post(create_chat_completion::<S>))
            .route("/v1/responses", post(create_response::<S>));
    }

    router
        .merge(ui_routes)
        .merge(docs::swagger_ui())
        .layer(DefaultBodyLimit::max(max_upload_size))
        .layer(cors)
        .with_state(state)
        .layer(middleware::from_fn_with_state(
            activity.clone(),
            track_activity,
        ))
        .layer(Extension(shutdown))
        .layer(Extension(activity))
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
    use crate::api::openai::ApiError;
    use crate::infrastructure::orchion::AppState;
    use crate::settings::ServerConfig;
    use crate::settings::{
        ChatTemplateConfig, LlmGenerationConfig, LlmModelDeployment, LlmRuntimeConfig,
        ModelDeployment, OcrModelDeployment,
    };
    use axum::body::Body;
    use axum::extract::connect_info::ConnectInfo;
    use axum::http::header::{
        ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
        ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, AUTHORIZATION, CONTENT_TYPE,
        ORIGIN, USER_AGENT,
    };
    use axum::http::{Method, Request, StatusCode};
    use axum::response::Response;
    use futures_util::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use orchion::llm_test_support::scripted_context_limit_llm_engine;
    use orchion::{
        AsrModel, ModelCategory, ModelDownloader, ModelId, ModelSpec, ModelUrl, ModelUrlSource,
        OcrModel, OcrModelKind, TtsModel,
    };
    use serde_json::Value;
    use std::net::SocketAddr;
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
    async fn activity_excludes_model_and_control_requests() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        for route in ["/v1/models", "/api/models/status"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(route).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        for route in ["/api/models/load", "/api/models/unload"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri(route)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(r#"{"model":"Private/Control-Sentinel"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["summary"]["retained"], 0);
        assert!(activity["history"].as_array().unwrap().is_empty());
        assert!(!activity.to_string().contains("Private/Control-Sentinel"));
    }

    #[tokio::test]
    async fn activity_excludes_unmatched_v1_routes() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/not-a-route?secret=sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["summary"]["retained"], 0);
        assert!(!activity.to_string().contains("sentinel"));
    }

    #[tokio::test]
    async fn activity_records_auth_errors_without_leaking_credentials() {
        let boundary = "activity-auth";
        let body = multipart_body(
            boundary,
            &[("model", "alibaba/qwen3-asr-0.6b")],
            Some(("file", "audio.wav", b"audio")),
        );
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("activity-secret-sentinel".to_string());
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(AUTHORIZATION, "Bearer wrong-secret-sentinel")
                    .header(USER_AGENT, "orchion-test-agent/1.0")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .extension(ConnectInfo(
                        "203.0.113.7:4242".parse::<SocketAddr>().unwrap(),
                    ))
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let live = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .header(AUTHORIZATION, "Bearer activity-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let live = json_body(live).await;
        assert_eq!(live["active"][0]["address"], "203.0.113.7");
        assert_eq!(live["active"][0]["user_agent"], "orchion-test-agent/1.0");
        denied.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .header(AUTHORIZATION, "Bearer activity-secret-sentinel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;
        let serialized = activity.to_string();

        assert_eq!(activity["history"][0]["http_status"], 401);
        assert_eq!(activity["history"][0]["error_code"], "invalid_api_key");
        assert!(activity["history"][0].get("address").is_none());
        assert!(activity["history"][0].get("user_agent").is_none());
        assert!(!serialized.contains("activity-secret-sentinel"));
        assert!(!serialized.contains("wrong-secret-sentinel"));
    }

    #[tokio::test]
    async fn activity_sanitizes_internal_error_details_for_server_errors() {
        let app = router_with_ui_routes(
            test_state(false, false),
            Router::new().route(
                "/v1/audio/speech",
                post(|| async {
                    Err::<(), ApiError>(ApiError::internal(
                        "model load failed at /tmp/private/model.gguf: sentinel",
                    ))
                }),
            ),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/speech")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let response_body = json_body(response).await;
        assert_eq!(response_body["error"]["message"], "internal server error");
        assert!(
            !response_body
                .to_string()
                .contains("/tmp/private/model.gguf")
        );

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["history"][0]["error_code"], "internal_error");
        assert_eq!(
            activity["history"][0]["error_message"],
            "internal server error"
        );
        assert!(!activity.to_string().contains("/tmp/private/model.gguf"));
        assert!(!activity.to_string().contains("sentinel"));
    }

    #[tokio::test]
    async fn activity_query_auth_precedes_validation() {
        let state = test_state_with_config(false, false, |config| {
            config.auth.api_key = Some("activity-query-secret".to_string());
        });
        let app = router_with_ui_routes(state, Router::new());
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/activity?limit=invalid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let invalid = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity?limit=invalid")
                    .header(AUTHORIZATION, "Bearer activity-query-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(invalid).await["error"]["code"],
            "invalid_activity_query"
        );
    }

    #[tokio::test]
    async fn activity_does_not_record_unavailable_client_model_values() {
        let boundary = "activity-model-privacy";
        let body = multipart_body(
            boundary,
            &[("model", "Private/Client-Sentinel")],
            Some(("file", "audio.wav", b"audio")),
        );
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert!(activity["history"][0].get("model").is_none());
        assert!(!activity.to_string().contains("Private/Client-Sentinel"));
    }

    #[tokio::test]
    async fn llm_unknown_model_is_not_recorded_before_allowlist_validation() {
        let root = tempfile::tempdir().unwrap();
        let gguf = root.path().join("main.gguf");
        let mmproj = root.path().join("mmproj.gguf");
        std::fs::write(&gguf, "test gguf fixture").unwrap();
        std::fs::write(&mmproj, "test mmproj fixture").unwrap();
        let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
        let models_dir = config.models.dir.clone();
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        let id = ModelId::parse("qwen/test").unwrap();
        config.services.llm.enabled = true;
        config.services.llm.default_model = Some(id.clone());
        config.services.llm.models = vec![LlmModelDeployment {
            id,
            name: None,
            model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
            mmproj_model: Some(
                orchion::ModelUrl::parse(&format!("file://{}", mmproj.display())).unwrap(),
            ),
            runtime: LlmRuntimeConfig::default(),
            chat_template: ChatTemplateConfig::default(),
            generation: LlmGenerationConfig::default(),
        }];
        let state = AppState::load(config).await.unwrap();
        let manifest = serde_json::from_str::<Value>(
            &std::fs::read_to_string(find_deployment_manifest(&models_dir).unwrap()).unwrap(),
        )
        .unwrap();
        let manifest_text = manifest.to_string();
        assert!(manifest_text.contains("llm_model"));
        assert!(manifest_text.contains("llm_mmproj"));
        assert!(manifest_text.contains("sha256"));
        assert!(manifest_text.contains("size"));
        assert!(!manifest_text.contains("runtime="));
        assert!(!manifest_text.contains("generation="));
        assert!(
            !manifest["source_intent"]
                .as_str()
                .unwrap()
                .contains("qwen/test")
        );
        assert!(manifest_text.contains(&gguf.to_string_lossy().to_string()));
        assert!(manifest_text.contains(&mmproj.to_string_lossy().to_string()));
        let lock = std::fs::read_to_string(models_dir.join("orchion-models.lock")).unwrap();
        assert!(lock.contains("llm_model"));
        assert!(lock.contains("llm_mmproj"));
        assert!(!models_dir.join(".orchion/blobs").exists());
        let app = router_with_ui_routes(state, Router::new());
        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let listed = json_body(listed).await;
        assert_eq!(listed["data"][0]["id"], "qwen/test");
        assert_eq!(listed["data"][0]["type"], "llm");
        assert_eq!(
            listed["data"][0]["capabilities"],
            serde_json::json!(["llm_chat", "llm_responses", "llm_streaming"])
        );
        assert!(!listed.to_string().contains("main.gguf"));
        let response = app.clone().oneshot(Request::builder()
            .method(Method::POST).uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"model":"Private/Prompt-Sentinel","messages":[{"role":"user","content":"secret prompt"}]}"#)).unwrap())
            .await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();
        let activity = wait_for_activity_history(app).await;
        assert!(activity["history"][0].get("model").is_none());
        let serialized = activity.to_string();
        assert!(!serialized.contains("Private/Prompt-Sentinel"));
        assert!(!serialized.contains("secret prompt"));
    }

    #[tokio::test]
    async fn llm_context_limit_is_json_4xx_before_sync_or_stream_headers() {
        for stream in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let gguf = root.path().join("main.gguf");
            std::fs::write(&gguf, "test gguf fixture").unwrap();
            let mut config = ServerConfig::default_for_exe(&root.path().join("orchion-server"));
            config.services.asr.enabled = false;
            config.services.tts.enabled = false;
            let id = ModelId::parse("qwen/test").unwrap();
            config.services.llm.enabled = true;
            config.services.llm.default_model = Some(id.clone());
            config.services.llm.models = vec![LlmModelDeployment {
                id,
                name: None,
                model: orchion::ModelUrl::parse(&format!("file://{}", gguf.display())).unwrap(),
                mmproj_model: None,
                runtime: LlmRuntimeConfig::default(),
                chat_template: ChatTemplateConfig::default(),
                generation: LlmGenerationConfig::default(),
            }];
            let state = AppState::load_with_test_llm_engine(
                config,
                scripted_context_limit_llm_engine(100, 50, 128),
            )
            .await
            .unwrap();
            let response = router_with_ui_routes(state.clone(), Router::new())
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/chat/completions")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "model":"qwen/test",
                                "messages":[{"role":"user","content":"long prompt"}],
                                "stream":stream
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_ne!(response.headers()[CONTENT_TYPE], "text/event-stream");
            let body = json_body(response).await;
            assert_eq!(body["error"]["code"], "context_length_exceeded");
            assert_eq!(body["error"]["param"], "max_completion_tokens");

            let response = router_with_ui_routes(state.clone(), Router::new())
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/v1/responses")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "model":"qwen/test",
                                "input":"long prompt",
                                "store":false,
                                "stream":stream
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_ne!(response.headers()[CONTENT_TYPE], "text/event-stream");
            let body = json_body(response).await;
            assert_eq!(body["error"]["code"], "context_length_exceeded");
            assert_eq!(body["error"]["param"], "max_output_tokens");
            state.shutdown().await;
        }
    }

    #[tokio::test]
    async fn disabled_activity_returns_an_empty_disabled_page() {
        let state = test_state_with_config(false, false, |config| {
            config.activity.enabled = false;
        });
        let app = router_with_ui_routes(state, Router::new());
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["enabled"], false);
        assert_eq!(activity["summary"]["retained"], 0);
        assert!(activity["history"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn activity_sse_begins_with_a_snapshot() {
        let app = router_with_ui_routes(test_state(false, false), Router::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let mut body = response.into_body();
        let frame = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let data = frame.into_data().unwrap();
        let text = String::from_utf8(data.to_vec()).unwrap();

        assert!(text.contains("event: snapshot"));
        assert!(text.contains("\"active\":[]"));
    }

    #[tokio::test]
    async fn activity_sse_closes_on_server_shutdown() {
        let shutdown = ServerShutdown::new();
        let app = router_with_ui_routes_and_shutdown(
            test_state(false, false),
            Router::new(),
            shutdown.clone(),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body();
        let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap();
        assert!(first.is_some());

        shutdown.trigger();

        let end = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap();
        assert!(end.is_none());
    }

    #[tokio::test]
    async fn activity_records_oversized_multipart_rejections_as_client_errors() {
        let boundary = "activity-limit";
        let body = multipart_body(
            boundary,
            &[("model", "alibaba/qwen3-asr-0.6b")],
            Some(("file", "audio.wav", &[0_u8; 64])),
        );
        let content_length = body.len();
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_upload_size = 32;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .header(axum::http::header::CONTENT_LENGTH, content_length)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        response.into_body().collect().await.unwrap();

        let activity = app
            .oneshot(
                Request::builder()
                    .uri("/api/activity")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let activity = json_body(activity).await;

        assert_eq!(activity["history"][0]["http_status"], 400, "{activity}");
        assert_eq!(activity["history"][0]["outcome"], "client_error");
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
    async fn failed_websocket_upgrade_remains_http_activity() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let app = router_with_ui_routes(state, Router::new());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/audio/transcriptions/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        response.into_body().collect().await.unwrap();

        let activity = wait_for_activity_history(app).await;
        assert_eq!(activity["history"][0]["transport"], "http");
    }

    #[tokio::test]
    async fn websocket_disconnect_before_start_is_recorded_and_releases_pending_capacity() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_pending_websocket_connections = 1;
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(Arc::clone(&state), Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();

        socket.send(Message::Close(None)).await.unwrap();
        drop(socket);
        let pending = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(permit) = state.try_acquire_pending_websocket() {
                    break permit;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(pending);

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "disconnected");
        assert_eq!(activity["history"][0]["transport"], "websocket");

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn websocket_activity_counts_input_rejected_before_ready() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
            config.server.max_concurrent_inference = 1;
        });
        let inference = state.acquire_inference().await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(Arc::clone(&state), Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket
            .send(Message::Text(
                r#"{"type":"start","model":"alibaba/qwen3-asr-0.6b","input_audio_format":"pcm_s16le","sample_rate":16000}"#
                    .into(),
            ))
            .await
            .unwrap();
        socket
            .send(Message::Binary(vec![1_u8, 2, 3, 4].into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap();

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "client_error");
        assert_eq!(activity["history"][0]["error_code"], "invalid_stream_state");
        assert_eq!(activity["history"][0]["input_bytes"], 4);

        drop(socket);
        drop(inference);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn websocket_activity_counts_a_rejected_binary_first_frame() {
        let state = test_state_with_config(false, false, |config| {
            config.services.asr.enabled = true;
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = router_with_ui_routes(state, Router::new());
        let activity_app = app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut socket, _) =
            connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
                .await
                .unwrap();
        socket
            .send(Message::Binary(vec![1_u8, 2, 3].into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap();

        let activity = wait_for_activity_history(activity_app).await;
        assert_eq!(activity["history"][0]["outcome"], "client_error");
        assert_eq!(
            activity["history"][0]["error_code"],
            "missing_start_message"
        );
        assert_eq!(activity["history"][0]["input_bytes"], 3);

        drop(socket);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn multipart_ocr_requires_file() {
        let boundary = "orchion-ocr-missing-file";
        let body = multipart_body(boundary, &[("model", "paddlepaddle/pp-ocrv6-tiny")], None);

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
            &[("model", "paddlepaddle/pp-ocrv6-tiny")],
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

    async fn wait_for_activity_history(app: Router) -> Value {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri("/api/activity")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let activity = json_body(response).await;
                if activity["summary"]["retained"].as_u64().unwrap_or_default() > 0 {
                    return activity;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap()
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

    fn find_deployment_manifest(root: &std::path::Path) -> Option<std::path::PathBuf> {
        for entry in std::fs::read_dir(root).ok()? {
            let path = entry.ok()?.path();
            if path.is_dir() {
                if let Some(manifest) = find_deployment_manifest(&path) {
                    return Some(manifest);
                }
            } else if path
                .file_name()
                .is_some_and(|name| name == "orchion-deployment.json")
            {
                return Some(path);
            }
        }
        None
    }

    fn test_state(ocr_active: bool, ocr_vl_active: bool) -> Arc<AppState> {
        test_state_with_config(ocr_active, ocr_vl_active, |_| {})
    }

    fn test_state_with_config(
        ocr_active: bool,
        ocr_vl_active: bool,
        configure: impl FnOnce(&mut ServerConfig),
    ) -> Arc<AppState> {
        let root = tempfile::tempdir().unwrap().keep();
        let mut config = ServerConfig::default_for_exe(&root.join("orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config.services.ocr.enabled = ocr_active;
        config.services.ocr.default_model =
            Some(ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap());
        config.services.ocr.models = ["paddlepaddle/pp-ocrv6-tiny", "paddlepaddle/pp-ocrv6-small"]
            .into_iter()
            .map(|id| {
                OcrModelDeployment::from_runtime(OcrModel::new(
                    ModelId::parse(id).unwrap(),
                    OcrModelKind::TraditionalOcr,
                ))
            })
            .collect();
        config.services.ocr_vl.enabled = ocr_vl_active;
        config.services.ocr_vl.default_model =
            Some(ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap());
        config.services.ocr_vl.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap(),
            OcrModelKind::OcrVl,
        ))];
        config.services.asr.models = vec![ModelDeployment::from_asr_runtime(
            AsrModel::parse("alibaba/qwen3-asr-0.6b").unwrap(),
        )];
        config.services.tts.models = vec![ModelDeployment::from_tts_runtime(
            TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap(),
        )];
        config.services.asr.idle_timeout = Duration::from_mins(10);
        config.services.tts.idle_timeout = Duration::from_mins(10);
        configure(&mut config);
        if config.services.asr.enabled {
            for deployment in &config.services.asr.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "huggingface",
                );
            }
        }
        if config.services.tts.enabled {
            for deployment in &config.services.tts.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "huggingface",
                );
            }
        }
        if config.services.ocr.enabled {
            for deployment in &config.services.ocr.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "modelscope",
                );
            }
        }
        if config.services.ocr_vl.enabled {
            for deployment in &config.services.ocr_vl.models {
                write_ready_fixture(
                    &config.models.dir,
                    &deployment.runtime,
                    &deployment.model,
                    "modelscope",
                );
            }
        }

        Arc::new(AppState::from_prepared_config(config).unwrap())
    }

    fn write_ready_fixture<M: ModelSpec>(
        models_dir: &std::path::Path,
        model: &M,
        model_url: &ModelUrl,
        neutral_source: &str,
    ) {
        if model_url.source() == ModelUrlSource::File {
            return;
        }
        let locator_repo = format!(
            "{}/{}",
            model_url.owner().unwrap(),
            model_url.repository().unwrap()
        );
        let source = match model_url.source() {
            ModelUrlSource::HuggingFace => "huggingface",
            ModelUrlSource::ModelScope => "modelscope",
            ModelUrlSource::Neutral => neutral_source,
            ModelUrlSource::File => unreachable!(),
        };
        let target = if matches!(model.category(), ModelCategory::Ocr | ModelCategory::OcrVl) {
            ModelSpec::cache_path(model, models_dir)
        } else {
            models_dir.join(&locator_repo)
        };
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("fixture.json"), "{}").unwrap();
        let cache_repo = target
            .strip_prefix(models_dir)
            .unwrap()
            .to_string_lossy()
            .to_string();
        let repo_id = if matches!(model.category(), ModelCategory::Ocr | ModelCategory::OcrVl) {
            if source == "modelscope" {
                model.modelscope_repo()
            } else {
                model.huggingface_repo()
            }
        } else {
            locator_repo.as_str()
        };
        let requested_revision = if source == "modelscope" {
            "master"
        } else {
            "main"
        };
        let resolved_revision = "1111111111111111111111111111111111111111";
        let mut repository_identities = ModelDownloader::model_artifact_plan(model, model_url)
            .unwrap()
            .into_iter()
            .map(|artifact| artifact.repository)
            .collect::<Vec<_>>();
        repository_identities.sort();
        repository_identities.dedup();
        let repositories = repository_identities
            .iter()
            .map(|identity| {
                let actual_repo = if identity == model.huggingface_repo() && source == "modelscope"
                {
                    model.modelscope_repo()
                } else {
                    identity
                };
                serde_json::json!({
                    "identity": identity,
                    "repo_id": actual_repo,
                    "requested_revision": requested_revision,
                    "resolved_revision": resolved_revision,
                })
            })
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "schema_version": 3,
            "source": source,
            "repo_id": repo_id,
            "downloaded_repos": repositories.iter().map(|repository| repository["repo_id"].clone()).collect::<Vec<_>>(),
            "revision": requested_revision,
            "resolved_revision": resolved_revision,
            "repositories": repositories,
            "layout": "model-hub-native",
            "files": [{
                "repo": cache_repo,
                "path": "fixture.json",
                "file_type": "file",
                "size": 2,
                "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            }],
        });
        std::fs::write(
            target.join(".orchion-ready.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
}
