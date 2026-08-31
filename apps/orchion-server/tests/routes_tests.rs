use axum::body::Body;
use axum::http::{Request, StatusCode, header, header::AUTHORIZATION};
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use orchion::{AsrModel, ModelId, ModelUrl, OcrModel, OcrModelKind, TtsModel};
use orchion_client::{Client as OrchionClient, ClientConfig as OrchionClientConfig};
use orchion_protocol::{AsrStreamEvent, AsrStreamInputAudioFormat, AsrStreamStartMessage};
use orchion_server::api::ui;
use orchion_server::config::{
    ModelDeployment, OcrModelDeployment, ServerConfig, TableStructureConfig, TableStructureType,
};
use orchion_server::routes::{router, router_with_ui_routes};
use orchion_server::state::AppState;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tower::ServiceExt;

#[tokio::test]
async fn models_endpoint_returns_configured_models() {
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["object"], "list");
    let ids = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "alibaba/qwen3-asr-0.6b",
            "alibaba/qwen3-asr-1.7b",
            "alibaba/qwen3-tts-12hz-0.6b-customvoice",
            "alibaba/qwen3-tts-12hz-0.6b-base",
            "alibaba/qwen3-tts-12hz-1.7b-voicedesign",
        ]
    );
    assert!(body["data"].as_array().unwrap().iter().all(|model| {
        model["object"] == "model"
            && model["created"] == 0
            && model["owned_by"] == "orchion"
            && model.get("model").is_none()
            && model.get("path").is_none()
            && model.get("digest").is_none()
    }));
    assert_eq!(model_type(&body, "alibaba/qwen3-asr-0.6b"), "asr");
    assert_eq!(
        model_capabilities(&body, "alibaba/qwen3-asr-0.6b"),
        vec!["asr_transcription", "asr_streaming"]
    );
    assert_eq!(model_type(&body, "alibaba/qwen3-asr-1.7b"), "asr");
    assert_eq!(
        model_capabilities(&body, "alibaba/qwen3-asr-1.7b"),
        vec!["asr_transcription", "asr_streaming"]
    );
    assert_eq!(
        model_type(&body, "alibaba/qwen3-tts-12hz-0.6b-customvoice"),
        "tts"
    );
    assert_eq!(
        model_capabilities(&body, "alibaba/qwen3-tts-12hz-0.6b-customvoice"),
        vec!["tts_preset_speakers"]
    );
    assert_eq!(model_type(&body, "alibaba/qwen3-tts-12hz-0.6b-base"), "tts");
    assert_eq!(
        model_capabilities(&body, "alibaba/qwen3-tts-12hz-0.6b-base"),
        vec!["tts_voice_cloning"]
    );
    assert_eq!(
        model_type(&body, "alibaba/qwen3-tts-12hz-1.7b-voicedesign"),
        "tts"
    );
    assert_eq!(
        model_capabilities(&body, "alibaba/qwen3-tts-12hz-1.7b-voicedesign"),
        vec!["tts_voice_design"]
    );
}

#[tokio::test]
async fn rust_client_matches_server_discovery_and_residency_contracts() {
    let (address, server) = start_websocket_test_server().await;
    let config = OrchionClientConfig::new(format!("http://{address}"))
        .unwrap()
        .with_api_key("secret");
    let client = OrchionClient::from_config(config).unwrap();

    client.health().check().await.unwrap();
    let models = client.models().list().await.unwrap();
    assert_eq!(models.object, "list");
    assert!(
        models
            .data
            .iter()
            .any(|model| model.id == "alibaba/qwen3-asr-0.6b")
    );

    let statuses = client.models().list_statuses().await.unwrap();
    assert_eq!(statuses.object, "list");
    assert!(
        statuses
            .data
            .iter()
            .any(|status| status.id == "alibaba/qwen3-asr-0.6b")
    );

    server.abort();
}

#[tokio::test]
async fn models_endpoint_returns_optional_deployment_display_name() {
    let state = test_state_with_services_config(None, true, false, |config| {
        config.services.asr.models[0].name = Some("Fast transcription".to_string());
    });
    let response = router(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    let models = body["data"].as_array().unwrap();
    assert_eq!(models[0]["name"], "Fast transcription");
    assert_eq!(models[1]["name"], "Qwen3-ASR 1.7B");
}

#[tokio::test]
async fn unregistered_speech_deployments_publish_only_conservative_base_capabilities() {
    let state = test_state_with_services_config(None, true, true, |config| {
        let asr = asr_model("Acme/Private-ASR");
        config.services.asr.default_model = asr.clone();
        config.services.asr.models = vec![ModelDeployment::from_asr_runtime(asr)];

        let tts = tts_model("Acme/Private-TTS-Base");
        config.services.tts.default_model = tts.clone();
        config.services.tts.models = vec![ModelDeployment::from_tts_runtime(tts)];
    });
    let response = router(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(
        model_capabilities(&body, "Acme/Private-ASR"),
        vec!["asr_transcription"]
    );
    assert_eq!(
        model_capabilities(&body, "Acme/Private-TTS-Base"),
        Vec::<&str>::new()
    );
}

#[tokio::test]
async fn model_status_endpoint_reports_runtime_residency_by_service() {
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .uri("/api/models/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["object"], "list");
    let statuses = body["data"].as_array().unwrap();
    assert_eq!(statuses.len(), 5);
    assert!(
        statuses
            .iter()
            .all(|model| { model["object"] == "model_status" && model["status"] == "unloaded" })
    );
    assert_eq!(statuses[0]["id"], "alibaba/qwen3-asr-0.6b");
    assert_eq!(statuses[0]["service"], "asr");
    assert_eq!(statuses[2]["service"], "tts");
}

#[tokio::test]
async fn model_unload_is_idempotent_for_configured_unloaded_model() {
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/unload")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"alibaba/qwen3-asr-0.6b","service":"asr"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["id"], "alibaba/qwen3-asr-0.6b");
    assert_eq!(body["service"], "asr");
    assert_eq!(body["status"], "unloaded");
}

#[tokio::test]
async fn model_control_rejects_wrong_service_and_invalid_json() {
    let app = router(test_state(None));
    let wrong_service = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/load")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"alibaba/qwen3-asr-0.6b","service":"ocr"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wrong_service.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(wrong_service).await["error"]["code"],
        "model_not_available"
    );

    let invalid_json = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/unload")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_json.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(invalid_json).await["error"]["code"],
        "invalid_json"
    );
}

#[tokio::test]
async fn legacy_model_control_routes_are_removed() {
    let app = router(test_state(None));
    for (method, route) in [
        ("GET", "/v1/models/status"),
        ("POST", "/v1/models/prewarm"),
        ("POST", "/v1/models/unload"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(route)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"model":"alibaba/qwen3-asr-0.6b","service":"asr"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{route}");
    }
}

#[tokio::test]
async fn models_endpoint_excludes_disabled_services() {
    let response = router(test_state_with_services(None, true, false))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    let ids = model_ids(&body);
    assert_eq!(
        ids,
        vec!["alibaba/qwen3-asr-0.6b", "alibaba/qwen3-asr-1.7b"]
    );
}

#[tokio::test]
async fn models_endpoint_is_empty_when_all_services_are_disabled() {
    let response = router(test_state_with_services(None, false, false))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["object"], "list");
    assert!(body["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn models_endpoint_includes_active_ocr_model_ids() {
    let response = router(test_state_with_ocr_services(None))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    let ids = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"paddlepaddle/pp-ocrv6-tiny".to_string()));
    assert!(ids.contains(&"paddlepaddle/paddleocr-vl-1.6".to_string()));
    assert_eq!(model_type(&body, "paddlepaddle/pp-ocrv6-tiny"), "ocr");
    assert_eq!(
        model_capabilities(&body, "paddlepaddle/pp-ocrv6-tiny"),
        vec!["ocr_text"]
    );
    assert_eq!(model_type(&body, "paddlepaddle/paddleocr-vl-1.6"), "ocr");
    assert_eq!(
        model_capabilities(&body, "paddlepaddle/paddleocr-vl-1.6"),
        vec!["ocr_text", "ocr_vision_language"]
    );
}

#[tokio::test]
async fn models_endpoint_exposes_layout_as_primary_deployment_capabilities() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.models[0] = config.services.ocr.models[0]
            .clone()
            .with_supported_layout();
        config.services.ocr_vl.models[0] = config.services.ocr_vl.models[0]
            .clone()
            .with_supported_layout();
    });

    let response = router(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    let ids = model_ids(&body);
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&"PaddlePaddle/PP-DocLayoutV3"));
    assert_eq!(
        model_capabilities(&body, "paddlepaddle/pp-ocrv6-tiny"),
        vec!["ocr_text", "ocr_layout", "ocr_markdown"]
    );
    assert_eq!(
        model_capabilities(&body, "paddlepaddle/paddleocr-vl-1.6"),
        vec![
            "ocr_text",
            "ocr_layout",
            "ocr_vision_language",
            "ocr_markdown",
            "ocr_html"
        ]
    );
}

#[tokio::test]
async fn models_endpoint_does_not_expose_unloaded_table_structure() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.models[0] = config.services.ocr.models[0]
            .clone()
            .with_supported_layout();
        config.services.ocr.models[0].table_structure = Some(TableStructureConfig {
            model: ModelUrl::parse("//Acme/Table/table.onnx").unwrap(),
            dictionary: ModelUrl::parse("//Acme/Table/table_dict.txt").unwrap(),
            table_type: TableStructureType::Wired,
            score_threshold: 0.5,
            max_structure_length: 500,
        });
    });
    let response = router(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = json_body(response).await;
    assert_eq!(model_ids(&body).len(), 2);
    assert_eq!(
        model_capabilities(&body, "paddlepaddle/pp-ocrv6-tiny"),
        vec!["ocr_text", "ocr_layout", "ocr_markdown"]
    );
    let model = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "paddlepaddle/pp-ocrv6-tiny")
        .unwrap();
    assert!(model.get("table_structure").is_none());
    assert!(model.get("artifacts").is_none());
}

#[tokio::test]
async fn speech_route_is_absent_when_tts_is_disabled() {
    let response = router(test_state_with_services(None, true, false))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"alibaba/qwen3-tts-12hz-0.6b-customvoice",
                        "input":"hello",
                        "voice":"alloy"
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn transcription_route_is_absent_when_asr_is_disabled() {
    let response = router(test_state_with_services(None, false, true))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn v1_routes_require_bearer_auth_when_api_key_is_configured() {
    let response = router(test_state(Some("secret")))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn v1_routes_accept_matching_bearer_auth() {
    let response = router(test_state(Some("secret")))
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header(AUTHORIZATION, "Bearer secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn model_management_routes_enforce_configured_bearer_auth() {
    let app = router(test_state(Some("secret")));
    for (method, route, authorized_status) in [
        ("GET", "/api/models/status", StatusCode::OK),
        ("POST", "/api/models/load", StatusCode::BAD_REQUEST),
        ("POST", "/api/models/unload", StatusCode::BAD_REQUEST),
    ] {
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(route)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED, "{route}");

        let authorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(route)
                    .header(AUTHORIZATION, "Bearer secret")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), authorized_status, "{route}");
    }
}

#[tokio::test]
async fn transcription_websocket_accepts_api_key_in_start_message() {
    let (address, server) = start_websocket_test_server().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
            .await
            .unwrap();

    let mut start = AsrStreamStartMessage::new("Acme/Missing-ASR", AsrStreamInputAudioFormat::Mp3);
    start.api_key = Some("secret".to_string());
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            start.to_text().unwrap().into(),
        ))
        .await
        .unwrap();
    let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
    let event = AsrStreamEvent::from_text(&event).unwrap();

    assert!(matches!(
        event,
        AsrStreamEvent::Error { error }
            if error.code.as_deref() == Some("model_not_available")
    ));
    server.abort();
}

#[tokio::test]
async fn transcription_websocket_rejects_invalid_start_message_api_key() {
    let (address, server) = start_websocket_test_server().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
            .await
            .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            r#"{"type":"start","model":"Acme/Missing-ASR","api_key":"wrong","input_audio_format":"mp3"}"#
                .into(),
        ))
        .await
        .unwrap();
    let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
    let event = serde_json::from_str::<Value>(&event).unwrap();

    assert_eq!(event["type"], "error");
    assert_eq!(event["error"]["code"], "invalid_api_key");
    server.abort();
}

#[tokio::test]
async fn transcription_websocket_keeps_supporting_bearer_auth() {
    let (address, server) = start_websocket_test_server().await;
    let mut request = format!("ws://{address}/v1/audio/transcriptions/stream")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert(AUTHORIZATION, "Bearer secret".parse().unwrap());
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            r#"{"type":"start","model":"Acme/Missing-ASR","input_audio_format":"mp3"}"#.into(),
        ))
        .await
        .unwrap();
    let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
    let event = serde_json::from_str::<Value>(&event).unwrap();

    assert_eq!(event["type"], "error");
    assert_eq!(event["error"]["code"], "model_not_available");
    server.abort();
}

#[tokio::test]
async fn transcription_websocket_rejects_invalid_options_before_model_load() {
    let (address, server) = start_websocket_test_server().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{address}/v1/audio/transcriptions/stream"))
            .await
            .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            r#"{"type":"start","model":"Acme/Missing-ASR","api_key":"secret","input_audio_format":"mp3","chunk_size_sec":31}"#
                .into(),
        ))
        .await
        .unwrap();
    let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
    let event = serde_json::from_str::<Value>(&event).unwrap();

    assert_eq!(event["type"], "error");
    assert_eq!(event["error"]["code"], "invalid_chunk_size");
    server.abort();
}

#[tokio::test]
async fn healthz_does_not_require_bearer_auth() {
    let response = router(test_state(Some("secret")))
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn root_redirects_to_ui() {
    let response = router(test_state(None))
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FOUND);
    assert_eq!(response.headers().get(header::LOCATION).unwrap(), "/ui");
}

#[tokio::test]
async fn root_redirect_does_not_require_bearer_auth() {
    let response = router(test_state(Some("secret")))
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FOUND);
    assert_eq!(response.headers().get(header::LOCATION).unwrap(), "/ui");
}

#[tokio::test]
async fn ui_root_serves_index_from_dist() {
    let dist_dir = create_test_dist("ui_root_serves_index_from_dist", "orchion-ui");
    let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert!(content_type.contains("text/html"));
    let body = text_body(response).await;
    assert!(body.contains("orchion-ui"));

    remove_test_dist(&dist_dir);
}

#[tokio::test]
async fn ui_spa_route_falls_back_to_index() {
    let dist_dir = create_test_dist("ui_spa_route_falls_back_to_index", "spa-fallback");
    let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
        .oneshot(
            Request::builder()
                .uri("/ui/tts")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await;
    assert!(body.contains("spa-fallback"));

    remove_test_dist(&dist_dir);
}

#[tokio::test]
async fn ui_missing_dist_returns_actionable_error() {
    let dist_dir = unique_dist_dir("ui_missing_dist_returns_actionable_error");
    let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = text_body(response).await;
    assert!(body.contains("web/dist was not found"));
    assert!(body.contains("bun install && bun run build"));
    assert!(body.contains("debug build"));
}

#[tokio::test]
async fn ui_routes_are_public_when_v1_auth_is_configured() {
    let dist_dir = create_test_dist(
        "ui_routes_are_public_when_v1_auth_is_configured",
        "public-ui",
    );
    let response =
        router_with_ui_routes(test_state(Some("secret")), ui::routes_from_path(&dist_dir))
            .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
            .await
            .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await;
    assert!(body.contains("public-ui"));

    remove_test_dist(&dist_dir);
}

#[tokio::test]
async fn ui_rejects_encoded_backslash_traversal_asset_path() {
    let workspace_dir = unique_dist_dir("ui_rejects_encoded_backslash_traversal_asset_path");
    let dist_dir = workspace_dir.join("dist");
    create_dist(&dist_dir, "safe-index");
    fs::write(dist_dir.join(r"..\outside.txt"), "escaped-disk-file").unwrap();

    let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
        .oneshot(
            Request::builder()
                .uri("/ui/..%5Coutside.txt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = text_body(response).await;
    assert!(!body.contains("escaped-disk-file"));

    remove_test_dist(&workspace_dir);
}

#[tokio::test]
async fn ui_rejects_encoded_windows_drive_style_asset_path() {
    let dist_dir = create_test_dist(
        "ui_rejects_encoded_windows_drive_style_asset_path",
        "safe-index",
    );
    #[cfg(not(windows))]
    {
        fs::create_dir_all(dist_dir.join("C:")).unwrap();
        fs::write(dist_dir.join("C:/outside.txt"), "drive-style-disk-file").unwrap();
    }

    let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
        .oneshot(
            Request::builder()
                .uri("/ui/C%3A/outside.txt")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = text_body(response).await;
    assert!(!body.contains("drive-style-disk-file"));

    remove_test_dist(&dist_dir);
}

#[tokio::test]
async fn ui_rejects_unsafe_asset_path_segments() {
    let workspace_dir = unique_dist_dir("ui_rejects_unsafe_asset_path_segments");
    let dist_dir = workspace_dir.join("dist");
    create_dist(&dist_dir, "safe-index");
    fs::create_dir_all(dist_dir.join("assets")).unwrap();
    fs::write(dist_dir.join("assets/app.js"), "asset-disk-file").unwrap();
    fs::write(dist_dir.join("outside.txt"), "dot-dot-disk-file").unwrap();

    let absolute_file = workspace_dir.join("absolute-outside.txt");
    fs::write(&absolute_file, "absolute-disk-file").unwrap();
    let encoded_absolute_path = absolute_file
        .to_string_lossy()
        .replace('\\', "%5C")
        .replace('/', "%2F")
        .replace(':', "%3A")
        .replace(' ', "%20");

    let cases = [
        (
            "absolute path",
            format!("/ui/{encoded_absolute_path}"),
            "absolute-disk-file",
        ),
        (
            "empty segment",
            "/ui/assets//app.js".to_string(),
            "asset-disk-file",
        ),
        (
            "dot segment",
            "/ui/assets/%2E/app.js".to_string(),
            "asset-disk-file",
        ),
        (
            "dot-dot segment",
            "/ui/assets/%2E%2E/outside.txt".to_string(),
            "dot-dot-disk-file",
        ),
    ];

    for (name, uri, unsafe_marker) in cases {
        let response = router_with_ui_routes(test_state(None), ui::routes_from_path(&dist_dir))
            .oneshot(
                Request::builder()
                    .uri(uri.as_str())
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{name}");
        let body = text_body(response).await;
        assert!(!body.contains(unsafe_marker), "{name}");
    }

    remove_test_dist(&workspace_dir);
}

#[tokio::test]
async fn json_speech_rejects_voice_clone() {
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"alibaba/qwen3-tts-12hz-0.6b-customvoice",
                        "input":"hello",
                        "voice":"clone",
                        "reference_audio":"/server/reference.wav",
                        "reference_text":"hello",
                        "response_format":"wav"
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "unsupported_voice_input");
    assert_eq!(body["error"]["param"], "voice");
}

#[tokio::test]
async fn speech_rejects_max_length_above_service_limit_before_model_load() {
    let state = test_state_with_services_config(None, false, true, |config| {
        config.services.tts.max_length = 4;
    });
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"alibaba/qwen3-tts-12hz-0.6b-customvoice",
                        "input":"hello",
                        "voice":"ryan",
                        "max_length":5
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "max_length");
    assert_eq!(body["error"]["code"], "max_length_exceeded");
}

#[tokio::test]
async fn ocr_vl_rejects_max_tokens_above_service_limit_before_model_load() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr_vl.max_tokens = 4;
    });
    let boundary = "orchion-ocr-vl-token-limit";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/paddleocr-vl-1.6"),
            ("max_tokens", "5"),
        ],
        "file",
        "input.png",
        b"image",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "max_tokens");
    assert_eq!(body["error"]["code"], "max_tokens_exceeded");
}

#[tokio::test]
async fn traditional_ocr_rejects_html_before_model_load() {
    let boundary = "orchion-traditional-ocr-html";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/pp-ocrv6-tiny"),
            ("response_format", "html"),
        ],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(test_state_with_ocr_services(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "response_format");
    assert_eq!(body["error"]["code"], "unsupported_response_format");
}

#[tokio::test]
async fn ocr_rejects_invalid_image_before_model_load() {
    let boundary = "orchion-invalid-ocr-image";
    let body = multipart_body(
        boundary,
        &[("model", "paddlepaddle/pp-ocrv6-tiny")],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(test_state_with_ocr_services(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "file");
    assert_eq!(body["error"]["code"], "invalid_file");
}

#[tokio::test]
async fn ocr_rejects_images_above_configured_pixel_limit_before_model_load() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.max_pixels = 1;
    });
    let boundary = "orchion-ocr-pixel-limit";
    let body = multipart_body(
        boundary,
        &[("model", "paddlepaddle/pp-ocrv6-tiny")],
        "file",
        "input.ppm",
        b"P6\n2 1\n255\n\0\0\0\0\0\0",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "file");
    assert_eq!(body["error"]["code"], "invalid_file");
    assert!(body["error"]["message"].as_str().unwrap().contains("pixel"));
}

#[tokio::test]
async fn ocr_requires_an_explicit_model() {
    let boundary = "orchion-ocr-model-required";
    let body = multipart_body(boundary, &[], "file", "input.png", b"not an image");
    let response = router(test_state_with_ocr_services(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "model");
    assert_eq!(body["error"]["code"], "missing_required_parameter");
}

#[tokio::test]
async fn traditional_ocr_markdown_uses_the_deployment_layout_model() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.models[0] = config.services.ocr.models[0]
            .clone()
            .with_supported_layout();
    });
    let boundary = "orchion-ocr-default-layout";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/pp-ocrv6-tiny"),
            ("response_format", "markdown"),
        ],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "file");
    assert_eq!(body["error"]["code"], "invalid_file");
}

#[tokio::test]
async fn traditional_ocr_html_is_rejected_even_with_a_deployment_layout_model() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.models[0] = config.services.ocr.models[0]
            .clone()
            .with_supported_layout();
    });
    let boundary = "orchion-ocr-traditional-html";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/pp-ocrv6-tiny"),
            ("response_format", "html"),
        ],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "response_format");
    assert_eq!(body["error"]["code"], "unsupported_response_format");
}

#[tokio::test]
async fn ocr_rejects_the_removed_layout_model_request_parameter() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr.models[0] = config.services.ocr.models[0]
            .clone()
            .with_supported_layout();
    });
    let boundary = "orchion-ocr-explicit-layout";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/pp-ocrv6-tiny"),
            ("layout_model", "PaddlePaddle/PP-DocLayoutV3"),
            ("response_format", "markdown"),
        ],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "layout_model");
    assert_eq!(body["error"]["code"], "unsupported_ocr_parameter");
}

#[tokio::test]
async fn ocr_vl_rejects_structured_format_without_layout_before_model_load() {
    let state = test_state_with_ocr_services_config(None, |config| {
        config.services.ocr_vl.models[0].layout_model = None;
        config.services.ocr_vl.models[0].layout_runtime = None;
    });
    let boundary = "orchion-ocr-vl-layout-required";
    let body = multipart_body(
        boundary,
        &[
            ("model", "paddlepaddle/paddleocr-vl-1.6"),
            ("response_format", "markdown"),
        ],
        "file",
        "input.png",
        b"not an image",
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "response_format");
    assert_eq!(body["error"]["code"], "unsupported_response_format");
}

#[tokio::test]
async fn transcription_rejects_decoded_audio_above_duration_limit_before_model_load() {
    let state = test_state_with_services_config(None, true, false, |config| {
        config.services.asr.max_audio_duration = Duration::from_millis(1);
    });
    let boundary = "orchion-asr-duration-limit";
    let body = multipart_body(
        boundary,
        &[("model", "alibaba/qwen3-asr-0.6b")],
        "file",
        "input.wav",
        &wav_bytes(),
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_audio");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("sample limit")
    );
}

#[tokio::test]
async fn transcription_rejects_unsupported_prompt_parameter() {
    let state = test_state_with_services_config(None, true, false, |_| {});
    let boundary = "orchion-asr-unsupported-prompt";
    let body = multipart_body(
        boundary,
        &[("prompt", "previous context")],
        "file",
        "input.wav",
        &wav_bytes(),
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["param"], "prompt");
    assert_eq!(body["error"]["code"], "unsupported_parameter");
}

#[tokio::test]
async fn voice_clone_rejects_reference_audio_above_duration_limit_before_model_load() {
    let state = test_state_with_services_config(None, false, true, |config| {
        config.services.tts.max_reference_audio_duration = Duration::from_millis(1);
    });
    let boundary = "orchion-tts-reference-duration-limit";
    let body = multipart_body(
        boundary,
        &[
            ("model", "alibaba/qwen3-tts-12hz-0.6b-base"),
            ("input", "hello"),
            ("voice", "clone"),
            ("reference_text", "hello"),
        ],
        "reference_audio",
        "reference.wav",
        &wav_bytes(),
    );
    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_audio");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("sample limit")
    );
}

#[tokio::test]
async fn multipart_speech_accepts_uploaded_voice_clone_audio() {
    let boundary = "orchion-test-boundary";
    let body = multipart_body(
        boundary,
        &[
            ("model", "not-a-model"),
            ("input", "hello"),
            ("voice", "clone"),
            ("reference_text", "hello"),
            ("response_format", "wav"),
        ],
        "reference_audio",
        "reference.wav",
        &wav_bytes(),
    );
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "model_not_available");
    assert_eq!(body["error"]["param"], "model");
}

#[tokio::test]
async fn multipart_speech_rejects_unknown_model_before_reference_audio_decode() {
    let boundary = "orchion-unknown-model-before-audio";
    let body = multipart_body(
        boundary,
        &[
            ("model", "not-a-model"),
            ("input", "hello"),
            ("voice", "clone"),
            ("reference_text", "hello"),
            ("response_format", "wav"),
        ],
        "reference_audio",
        "reference.wav",
        b"not an audio file",
    );
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "model_not_available");
    assert_eq!(body["error"]["param"], "model");
}

#[tokio::test]
async fn multipart_speech_rejects_invalid_reference_audio_before_inference() {
    let boundary = "orchion-invalid-reference-audio";
    let body = multipart_body(
        boundary,
        &[
            ("model", "alibaba/qwen3-tts-12hz-0.6b-base"),
            ("input", "hello"),
            ("voice", "clone"),
            ("reference_text", "hello"),
            ("response_format", "wav"),
        ],
        "reference_audio",
        "reference.mp3",
        b"not an audio file",
    );
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "invalid_audio");
    assert_eq!(body["error"]["param"], "reference_audio");
}

async fn json_body(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn model_ids(body: &Value) -> Vec<&str> {
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap())
        .collect()
}

fn model_type<'a>(body: &'a Value, expected_id: &str) -> &'a str {
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == expected_id)
        .unwrap_or_else(|| panic!("model `{expected_id}` was not returned"))["type"]
        .as_str()
        .unwrap()
}

fn model_capabilities<'a>(body: &'a Value, expected_id: &str) -> Vec<&'a str> {
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == expected_id)
        .unwrap_or_else(|| panic!("model `{expected_id}` was not returned"))["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|capability| capability.as_str().unwrap())
        .collect()
}

async fn text_body(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file_field: &str,
    file_name: &str,
    file_bytes: &[u8],
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
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{file_name}\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

fn wav_bytes() -> Vec<u8> {
    let samples = (0_u16..2_400)
        .map(|index| {
            let phase = f32::from(index) / 24_000.0 * 440.0 * std::f32::consts::TAU;
            test_sample_to_i16(phase.sin() * 0.25)
        })
        .collect::<Vec<_>>();
    let data_len = samples
        .len()
        .checked_mul(2)
        .and_then(|length| u32::try_from(length).ok())
        .expect("test WAV data length fits u32");
    let mut bytes = Vec::with_capacity(
        44 + usize::try_from(data_len).expect("test WAV data length fits usize"),
    );
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&24_000_u32.to_le_bytes());
    bytes.extend_from_slice(&48_000_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn test_sample_to_i16(sample: f32) -> i16 {
    let scaled = sample.clamp(-1.0, 1.0) * f32::from(i16::MAX);
    assert!(scaled.is_finite(), "test sample must be finite");

    // The finite clamped value is in the i16 range; test encoding intentionally truncates it.
    #[allow(clippy::cast_possible_truncation)]
    {
        scaled as i16
    }
}

fn test_state(api_key: Option<&str>) -> Arc<AppState> {
    test_state_with_services(api_key, true, true)
}

async fn start_websocket_test_server() -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router(test_state(Some("secret"))))
            .await
            .unwrap();
    });
    (address, server)
}

fn test_state_with_ocr_services(api_key: Option<&str>) -> Arc<AppState> {
    test_state_with_ocr_services_config(api_key, |_| {})
}

fn test_state_with_ocr_services_config(
    api_key: Option<&str>,
    configure: impl FnOnce(&mut ServerConfig),
) -> Arc<AppState> {
    test_state_with_services_config(api_key, false, false, |config| {
        config.services.ocr.enabled = true;
        config.services.ocr.default_model =
            Some(ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap());
        config.services.ocr.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap(),
            OcrModelKind::TraditionalOcr,
        ))];
        config.services.ocr_vl.enabled = true;
        config.services.ocr_vl.default_model =
            Some(ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap());
        config.services.ocr_vl.models = vec![OcrModelDeployment::from_runtime(OcrModel::new(
            ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap(),
            OcrModelKind::OcrVl,
        ))];
        configure(config);
    })
}

fn test_state_with_services(
    api_key: Option<&str>,
    asr_enabled: bool,
    tts_enabled: bool,
) -> Arc<AppState> {
    test_state_with_services_config(api_key, asr_enabled, tts_enabled, |_| {})
}

fn test_state_with_services_config(
    api_key: Option<&str>,
    asr_enabled: bool,
    tts_enabled: bool,
    configure: impl FnOnce(&mut ServerConfig),
) -> Arc<AppState> {
    let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
    config.auth.api_key = api_key.map(str::to_string);
    config.services.asr.enabled = asr_enabled;
    config.services.tts.enabled = tts_enabled;
    config.services.asr.models = [
        asr_model("alibaba/qwen3-asr-0.6b"),
        asr_model("alibaba/qwen3-asr-1.7b"),
    ]
    .into_iter()
    .map(ModelDeployment::from_asr_runtime)
    .collect();
    config.services.asr.idle_timeout = Duration::from_mins(10);
    config.services.asr.max_loaded = 2;
    config.services.tts.models = vec![
        tts_model("alibaba/qwen3-tts-12hz-0.6b-customvoice"),
        tts_model("alibaba/qwen3-tts-12hz-0.6b-base"),
        tts_model("alibaba/qwen3-tts-12hz-1.7b-voicedesign"),
    ]
    .into_iter()
    .map(ModelDeployment::from_tts_runtime)
    .collect();
    config.services.tts.idle_timeout = Duration::from_mins(10);
    config.services.tts.max_loaded = 2;
    configure(&mut config);
    Arc::new(AppState::from_prepared_config(config).unwrap())
}

fn asr_model(value: &str) -> AsrModel {
    AsrModel::parse(value).unwrap()
}

fn tts_model(value: &str) -> TtsModel {
    TtsModel::parse(value).unwrap()
}

fn create_test_dist(test_name: &str, marker: &str) -> PathBuf {
    let dist_dir = unique_dist_dir(test_name);
    create_dist(&dist_dir, marker);
    dist_dir
}

fn create_dist(dist_dir: &Path, marker: &str) {
    fs::create_dir_all(dist_dir).unwrap();
    fs::write(
        dist_dir.join("index.html"),
        format!("<!doctype html><html><body>{marker}</body></html>"),
    )
    .unwrap();
}

fn unique_dist_dir(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("orchion-{test_name}-{nanos}"))
}

fn remove_test_dist(dist_dir: &Path) {
    let _ = fs::remove_dir_all(dist_dir);
}
