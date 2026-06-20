use axum::body::Body;
use axum::http::{Request, StatusCode, header, header::AUTHORIZATION};
use http_body_util::BodyExt;
use orchion::{AsrModel, TtsModel};
use orchion_server::api::ui;
use orchion_server::config::ServerConfig;
use orchion_server::model_cache::{AsrModelCache, GlobalModelCacheLimiter, TtsModelCache};
use orchion_server::routes::{router, router_with_ui_routes};
use orchion_server::state::AppState;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
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
async fn multipart_speech_rejects_invalid_reference_audio_before_inference() {
    let boundary = "orchion-invalid-reference-audio";
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
    let samples = [0_i16; 128];
    let data_len = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);
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
    let global_models = GlobalModelCacheLimiter::new(config.models.max_loaded);
    Arc::new(AppState {
        config,
        asr_models,
        tts_models,
        global_models,
    })
}

fn create_test_dist(test_name: &str, marker: &str) -> PathBuf {
    let dist_dir = unique_dist_dir(test_name);
    create_dist(&dist_dir, marker);
    dist_dir
}

fn create_dist(dist_dir: &Path, marker: &str) {
    fs::create_dir_all(&dist_dir).unwrap();
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
