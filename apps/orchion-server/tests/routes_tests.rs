use axum::body::Body;
use axum::http::{Request, StatusCode, header, header::AUTHORIZATION};
use http_body_util::BodyExt;
use orchion::{AsrModel, ModelId, TtsModel};
use orchion_server::api::ui;
use orchion_server::config::ServerConfig;
use orchion_server::model_cache::{
    AsrModelCache, GlobalModelCacheLimiter, OcrModelCache, OcrVlModelCache, TtsModelCache,
};
use orchion_server::routes::{router, router_with_ui_routes};
use orchion_server::state::AppState;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
            "Qwen/Qwen3-ASR-0.6B",
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ]
    );
    assert!(body["data"].as_array().unwrap().iter().all(|model| {
        model["object"] == "model" && model["created"] == 0 && model["owned_by"] == "orchion"
    }));
    assert_eq!(model_type(&body, "Qwen/Qwen3-ASR-0.6B"), "asr");
    assert_eq!(model_subtype(&body, "Qwen/Qwen3-ASR-0.6B"), None);
    assert_eq!(model_type(&body, "Qwen/Qwen3-ASR-1.7B"), "asr");
    assert_eq!(model_subtype(&body, "Qwen/Qwen3-ASR-1.7B"), None);
    assert_eq!(
        model_type(&body, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        "tts"
    );
    assert_eq!(
        model_subtype(&body, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        Some("preset_voice")
    );
    assert_eq!(
        model_type(&body, "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "tts"
    );
    assert_eq!(
        model_subtype(&body, "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        Some("voice_clone")
    );
    assert_eq!(
        model_type(&body, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        "tts"
    );
    assert_eq!(
        model_subtype(&body, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        Some("voice_design")
    );
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
    assert_eq!(ids, vec!["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]);
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
    assert!(ids.contains(&"PaddlePaddle/PP-OCRv6_tiny".to_string()));
    assert!(ids.contains(&"PaddlePaddle/PaddleOCR-VL-1.6".to_string()));
    assert_eq!(model_type(&body, "PaddlePaddle/PP-OCRv6_tiny"), "ocr");
    assert_eq!(
        model_subtype(&body, "PaddlePaddle/PP-OCRv6_tiny"),
        Some("standard")
    );
    assert_eq!(model_type(&body, "PaddlePaddle/PaddleOCR-VL-1.6"), "ocr");
    assert_eq!(
        model_subtype(&body, "PaddlePaddle/PaddleOCR-VL-1.6"),
        Some("vl")
    );
}

#[tokio::test]
async fn models_endpoint_includes_configured_ocr_layout_model_ids() {
    let mut state = test_state_with_ocr_services(None);
    let state_mut = Arc::get_mut(&mut state).unwrap();
    let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
    state_mut.config.services.ocr.layout_available_models = vec![layout_model.clone()];
    state_mut.config.services.ocr_vl.layout_available_models = vec![layout_model];

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
    assert!(ids.contains(&"PaddlePaddle/PP-DocLayoutV3"));
    assert_eq!(
        ids.iter()
            .filter(|id| **id == "PaddlePaddle/PP-DocLayoutV3")
            .count(),
        1
    );
    assert_eq!(
        model_type(&body, "PaddlePaddle/PP-DocLayoutV3"),
        "ocr"
    );
    assert_eq!(
        model_subtype(&body, "PaddlePaddle/PP-DocLayoutV3"),
        Some("layout")
    );
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
                        "model":"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
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
                        "model":"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
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
            ("model", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
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

fn model_subtype<'a>(body: &'a Value, expected_id: &str) -> Option<&'a str> {
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == expected_id)
        .unwrap_or_else(|| panic!("model `{expected_id}` was not returned"))["subtype"]
        .as_str()
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
    let samples = (0..2_400)
        .map(|index| {
            let phase = index as f32 / 24_000.0 * 440.0 * std::f32::consts::TAU;
            (phase.sin() * f32::from(i16::MAX) * 0.25) as i16
        })
        .collect::<Vec<_>>();
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
    test_state_with_services(api_key, true, true)
}

fn test_state_with_ocr_services(api_key: Option<&str>) -> Arc<AppState> {
    let mut state = test_state_with_services(api_key, false, false);
    let state_mut = Arc::get_mut(&mut state).unwrap();
    state_mut.config.services.ocr.enabled = true;
    state_mut.config.services.ocr.default_model =
        Some(ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap());
    state_mut.config.services.ocr.available_models =
        vec![ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap()];
    state_mut.config.services.ocr_vl.enabled = true;
    state_mut.config.services.ocr_vl.default_model =
        Some(ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap());
    state_mut.config.services.ocr_vl.available_models =
        vec![ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()];
    state
}

fn test_state_with_services(
    api_key: Option<&str>,
    asr_enabled: bool,
    tts_enabled: bool,
) -> Arc<AppState> {
    let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
    config.auth.api_key = api_key.map(str::to_string);
    config.services.asr.enabled = asr_enabled;
    config.services.tts.enabled = tts_enabled;
    config.services.asr.available_models = vec![AsrModel::Qwen3Asr06B, AsrModel::Qwen3Asr17B];
    config.services.asr.idle_timeout = Duration::from_secs(600);
    config.services.asr.max_loaded = 2;
    config.services.tts.available_models = vec![
        TtsModel::Qwen3Tts06BCustomVoice,
        TtsModel::Qwen3Tts06BBase,
        TtsModel::Qwen3Tts17BVoiceDesign,
    ];
    config.services.tts.idle_timeout = Duration::from_secs(600);
    config.services.tts.max_loaded = 2;
    let asr_models = AsrModelCache::new(
        "asr",
        config.services.asr.available_models.clone(),
        config.services.asr.idle_timeout,
        config.services.asr.max_loaded,
        config.models.dir.clone(),
    );
    let tts_models = TtsModelCache::new(
        "tts",
        config.services.tts.available_models.clone(),
        config.services.tts.idle_timeout,
        config.services.tts.max_loaded,
        config.models.dir.clone(),
    );
    let ocr_models = OcrModelCache::new(
        "ocr",
        Vec::new(),
        config.services.ocr.idle_timeout,
        config.services.ocr.max_loaded,
        config.models.dir.clone(),
    );
    let ocr_vl_models = OcrVlModelCache::new(
        "ocr-vl",
        Vec::new(),
        config.services.ocr_vl.idle_timeout,
        config.services.ocr_vl.max_loaded,
        config.models.dir.clone(),
    );
    let global_models = GlobalModelCacheLimiter::new(config.models.max_loaded);
    Arc::new(AppState {
        config,
        asr_models,
        tts_models,
        ocr_models,
        ocr_vl_models,
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
