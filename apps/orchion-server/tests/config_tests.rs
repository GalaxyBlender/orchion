use orchion::{DevicePreference, ModelId, ModelUrlSource, OcrResponseFormat};
use orchion_server::config::{ConfigError, ModelSource, ServerConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

fn exe() -> &'static Path {
    Path::new("/tmp/orchion-server")
}

#[test]
fn defaults_are_executable_relative_and_use_neutral_deployments() {
    let exe_path = Path::new("/tmp/orchion/bin/orchion-server");
    let config = ServerConfig::default_for_exe(exe_path);

    assert_eq!(
        config.server.bind,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9090)
    );
    assert_eq!(
        config.config_path,
        exe_path.parent().unwrap().join("config.toml")
    );
    assert_eq!(config.models.dir, exe_path.parent().unwrap().join("models"));
    assert_eq!(config.models.source, ModelSource::Auto);
    assert_eq!(config.services.asr.models.len(), 1);
    assert_eq!(
        config.services.asr.models[0].id.as_str(),
        "Qwen/Qwen3-ASR-0.6B"
    );
    assert_eq!(
        config.services.asr.models[0].model.source(),
        ModelUrlSource::Neutral
    );
    assert_eq!(
        config.services.tts.models[0].id.as_str(),
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    );
    assert!(config.services.ocr.models.is_empty());
    assert!(config.services.ocr_vl.models.is_empty());
    assert_eq!(config.services.ocr.format, OcrResponseFormat::Json);
    assert_eq!(config.services.ocr_vl.format, OcrResponseFormat::Markdown);
}

#[test]
fn deployment_arrays_parse_and_replace_defaults() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "Qwen/Qwen3-ASR-1.7B"

[[services.asr.models]]
id = "Qwen/Qwen3-ASR-1.7B"
name = "  Qwen3 ASR 1.7B  "
model = "ms://Qwen/Qwen3-ASR-1.7B"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(config.services.asr.models.len(), 1);
    let deployment = &config.services.asr.models[0];
    assert_eq!(deployment.id.as_str(), "Qwen/Qwen3-ASR-1.7B");
    assert_eq!(deployment.runtime.as_str(), deployment.id.as_str());
    assert_eq!(deployment.name.as_deref(), Some("Qwen3 ASR 1.7B"));
    assert_eq!(deployment.display_name(), "Qwen3 ASR 1.7B");
    assert_eq!(deployment.model.as_str(), "ms://Qwen/Qwen3-ASR-1.7B");
    assert_eq!(deployment.model.source(), ModelUrlSource::ModelScope);
}

#[test]
fn omitted_name_falls_back_to_model_id_name_segment() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.tts.models]]
id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
model = "//Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.tts.models[0].display_name(),
        "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    );
}

#[test]
fn blank_names_are_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
name = "   "
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidModelName { .. }));
}

#[test]
fn enabled_services_require_nonempty_models() {
    for section in ["asr", "tts", "ocr", "ocr-vl"] {
        let document = format!("[services.{section}]\nenabled = true\nmodels = []");
        let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
        assert!(matches!(
            error,
            ConfigError::ServiceEnabledWithoutModels { .. }
        ));
    }
}

#[test]
fn defaults_must_match_exactly_one_local_deployment_even_when_disabled() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = false
default_model = "Qwen/Qwen3-ASR-1.7B"

[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DefaultModelUnavailable { .. }));
    assert!(error.to_string().contains("services.asr.models"));
}

#[test]
fn model_ids_are_globally_unique_including_disabled_services() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "//Qwen/Qwen3-ASR-0.6B"

[[services.tts.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));

    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"

[[services.ocr-vl.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));
}

#[test]
fn duplicate_ids_within_one_disabled_service_are_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = false

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));
}

#[test]
fn configured_ids_must_match_the_service_runtime_category() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "PaddlePaddle/PaddleOCR-VL-1.6"
[[services.asr.models]]
id = "PaddlePaddle/PaddleOCR-VL-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedRuntimeModel { .. }));

    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr.models]]
id = "PaddlePaddle/PaddleOCR-VL-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedRuntimeModel { .. }));
}

#[test]
fn unregistered_speech_ids_remain_valid_runtime_models() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "Acme/Private-ASR"
[[services.asr.models]]
id = "Acme/Private-ASR"
model = "hf://Acme/Private-ASR-Package"

[services.tts]
default_model = "Acme/Private-TTS"
[[services.tts.models]]
id = "Acme/Private-TTS"
model = "ms://Acme/Private-TTS-Package"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.asr.models[0].runtime.as_str(),
        "Acme/Private-ASR"
    );
    assert_eq!(
        config.services.tts.models[0].runtime.as_str(),
        "Acme/Private-TTS"
    );
}

#[test]
fn repository_runtime_rejects_exact_file_model_locator() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "//Qwen/Qwen3-ASR-0.6B/model.safetensors"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::UnsupportedModelLocator { .. }));
}

#[test]
fn ocr_deployment_accepts_supported_layout_recipe_and_projects_runtime_id() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap();

    let deployment = &config.services.ocr.models[0];
    assert_eq!(deployment.model.source(), ModelUrlSource::HuggingFace);
    assert_eq!(
        deployment.layout_model.as_ref().unwrap().as_str(),
        "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
    );
    assert_eq!(
        config.services.ocr.layout_ids(),
        [ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap()]
    );
    assert_eq!(
        config
            .services
            .ocr
            .default_layout_runtime()
            .unwrap()
            .id()
            .as_str(),
        "PaddlePaddle/PP-DocLayoutV3"
    );
}

#[test]
fn unsupported_layout_recipe_is_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr-vl.models]]
id = "PaddlePaddle/PaddleOCR-VL-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
layout_model = "//Acme/Layout/model.onnx"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedLayoutModel { .. }));
}

#[test]
fn local_layout_artifact_uses_the_supported_layout_runtime_recipe() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.ocr-vl.models]]
id = "PaddlePaddle/PaddleOCR-VL-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
layout_model = "file:///tmp/layout.onnx"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.ocr_vl.models[0]
            .layout_runtime
            .as_ref()
            .unwrap()
            .id()
            .as_str(),
        "PaddlePaddle/PP-DocLayoutV3"
    );
}

#[test]
fn separate_ocr_deployments_accept_distinct_layout_locators() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "hf://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"

[[services.ocr.models]]
id = "PaddlePaddle/PP-OCRv6_small"
model = "//PaddlePaddle/PP-OCRv6_small"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.ocr.models[0]
            .layout_model
            .as_ref()
            .unwrap()
            .source(),
        ModelUrlSource::HuggingFace
    );
    assert_eq!(
        config.services.ocr.models[1]
            .layout_model
            .as_ref()
            .unwrap()
            .source(),
        ModelUrlSource::ModelScope
    );
}

#[test]
fn layout_model_is_rejected_for_speech_deployments() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "//Qwen/Qwen3-ASR-0.6B"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::ParseToml(_)));
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn all_removed_config_fields_are_rejected_without_aliases() {
    for field in [
        "available_models = []",
        "layout_available_models = []",
        "layout_default_model = \"PaddlePaddle/PP-DocLayoutV3\"",
    ] {
        let section = if field.starts_with("available") {
            "asr"
        } else {
            "ocr"
        };
        let document = format!("[services.{section}]\n{field}");
        let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{field}");
    }
}

#[test]
fn model_url_validation_runs_during_toml_deserialization() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "Qwen/Qwen3-ASR-0.6B"
model = "https://huggingface.co/Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("invalid model URL"));
}

#[test]
fn general_server_and_service_overrides_still_parse() {
    let config = ServerConfig::from_toml_str(
        r#"
[server]
bind = "0.0.0.0:9000"
max_upload_size = "64M"
max_concurrent_inference = 4

[models]
dir = "cache/models"
source = "modelscope"
max_loaded = 3

[services.asr]
device = "cuda0"
idle_timeout = "5m"
stream_chunk_size = 1.5

[services.tts]
format = "mp3"
max_length = 1024
"#,
        Path::new("/opt/orchion/orchion-server"),
    )
    .unwrap();

    assert_eq!(config.server.bind.port(), 9000);
    assert_eq!(config.server.max_upload_size, 64 * 1024 * 1024);
    assert_eq!(config.models.source, ModelSource::ModelScope);
    assert_eq!(config.services.asr.device, DevicePreference::Cuda(Some(0)));
    assert_eq!(config.services.asr.idle_timeout, Duration::from_mins(5));
    assert_eq!(config.services.tts.format, "mp3");
    assert_eq!(config.services.tts.max_length, 1024);
}

#[test]
fn shipped_config_uses_the_new_contract() {
    let document = include_str!("../config.toml");
    let config = ServerConfig::from_toml_str(document, exe()).unwrap();

    assert!(config.services.asr.enabled);
    assert_eq!(config.services.asr.models.len(), 2);
    assert_eq!(config.services.tts.models.len(), 5);
    assert_eq!(config.services.ocr.models.len(), 5);
    assert_eq!(config.services.ocr_vl.models.len(), 2);
    assert_eq!(config.services.ocr.layout_ids().len(), 1);
}

#[test]
fn malformed_limits_and_unknown_fields_remain_rejected() {
    assert!(ServerConfig::from_toml_str("[models]\nmax_loaded = 0", exe()).is_err());
    assert!(ServerConfig::from_toml_str("[services.tts]\nmax_length = 0", exe()).is_err());
    let error = ServerConfig::from_toml_str("[services.tts]\nvoice = \"ryan\"", exe()).unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}
