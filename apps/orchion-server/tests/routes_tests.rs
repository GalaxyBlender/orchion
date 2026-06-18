use axum::body::Body;
use axum::http::{Request, StatusCode, header::AUTHORIZATION};
use http_body_util::BodyExt;
use orchion::{AsrModel, TtsModel};
use orchion_server::config::ServerConfig;
use orchion_server::model_cache::{AsrModelCache, TtsModelCache};
use orchion_server::routes::router;
use orchion_server::state::AppState;
use serde_json::Value;
use std::sync::Arc;
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
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "qwen3-tts-0.6b-custom-voice",
            "qwen3-tts-1.7b-voice-design",
        ]
    );
    assert!(body["data"].as_array().unwrap().iter().all(|model| {
        model["object"] == "model" && model["created"] == 0 && model["owned_by"] == "orchion"
    }));
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
async fn json_speech_rejects_voice_clone() {
    let response = router(test_state(None))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"qwen3-tts-0.6b-custom-voice",
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
        b"RIFFtestWAVE",
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

async fn json_body(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
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

fn test_state(api_key: Option<&str>) -> Arc<AppState> {
    let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
    config.auth.api_key = api_key.map(str::to_string);
    config.models.asr.available = vec![AsrModel::Qwen3Asr06B, AsrModel::Qwen3Asr17B];
    config.models.tts.available = vec![
        TtsModel::Qwen3Tts06BCustomVoice,
        TtsModel::Qwen3Tts17BVoiceDesign,
    ];
    let asr_models = AsrModelCache::new(config.models.asr.clone(), config.models.dir.clone());
    let tts_models = TtsModelCache::new(config.models.tts.clone(), config.models.dir.clone());
    Arc::new(AppState {
        config,
        asr_models,
        tts_models,
    })
}
