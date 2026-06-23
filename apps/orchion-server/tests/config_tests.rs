use orchion::{AsrModel, DevicePreference, ModelId, OcrResponseFormat, TtsModel};
use orchion_server::config::{ConfigError, ModelSource, ServerConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

fn asr_model(value: &str) -> AsrModel {
    AsrModel::parse(value).unwrap()
}

fn tts_model(value: &str) -> TtsModel {
    TtsModel::parse(value).unwrap()
}

#[test]
fn defaults_are_executable_relative() {
    let exe_path = std::path::Path::new("/tmp/orchion/bin/orchion-server");
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
    assert_eq!(config.models.max_loaded, 2);
    assert!(!config.services.asr.enabled);
    assert_eq!(
        config.services.asr.default_model,
        asr_model("Qwen/Qwen3-ASR-0.6B")
    );
    assert_eq!(
        config.services.asr.available_models,
        vec![asr_model("Qwen/Qwen3-ASR-0.6B")]
    );
    assert_eq!(config.services.asr.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.services.asr.max_loaded, 1);
    assert_eq!(config.services.asr.device, DevicePreference::Auto);
    assert!(!config.services.tts.enabled);
    assert_eq!(
        config.services.tts.default_model,
        tts_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    );
    assert_eq!(
        config.services.tts.available_models,
        vec![tts_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")]
    );
    assert_eq!(config.services.tts.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.services.tts.max_loaded, 1);
    assert_eq!(config.services.tts.device, DevicePreference::Auto);
    assert!(!config.services.ocr.enabled);
    assert_eq!(config.services.ocr.default_model, None);
    assert!(config.services.ocr.available_models.is_empty());
    assert_eq!(config.services.ocr.layout_default_model, None);
    assert!(config.services.ocr.layout_available_models.is_empty());
    assert_eq!(config.services.ocr.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.services.ocr.max_loaded, 1);
    assert_eq!(config.services.ocr.device, DevicePreference::Auto);
    assert_eq!(config.services.ocr.format, OcrResponseFormat::Json);
    assert!(!config.services.ocr.active());
    assert!(!config.services.ocr_vl.enabled);
    assert_eq!(config.services.ocr_vl.default_model, None);
    assert!(config.services.ocr_vl.available_models.is_empty());
    assert_eq!(config.services.ocr_vl.layout_default_model, None);
    assert!(config.services.ocr_vl.layout_available_models.is_empty());
    assert_eq!(
        config.services.ocr_vl.idle_timeout,
        Duration::from_secs(600)
    );
    assert_eq!(config.services.ocr_vl.max_loaded, 1);
    assert_eq!(config.services.ocr_vl.device, DevicePreference::Auto);
    assert_eq!(config.services.ocr_vl.format, OcrResponseFormat::Markdown);
    assert!(!config.services.ocr_vl.active());
    assert_eq!(config.auth.api_key, None);
    assert_eq!(config.server.max_upload_size, 30 * 1024 * 1024);
}

#[test]
fn enabled_ocr_requires_available_models() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
"#,
        exe_path,
    )
    .unwrap_err();

    match error {
        ConfigError::ServiceEnabledWithoutModels { section } => {
            assert_eq!(section, "services.ocr");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn active_ocr_default_must_be_available_models_member() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_small"
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("services.ocr.available_models"));
    assert!(matches!(error, ConfigError::DefaultModelUnavailable { .. }));
}

#[test]
fn invalid_ocr_model_id_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr-vl]
enabled = true
default_model = "bad id with spaces"
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
"#,
        exe_path,
    )
    .unwrap_err();

    match error {
        ConfigError::InvalidModelId { section, value } => {
            assert_eq!(section, "services.ocr-vl.default_model");
            assert_eq!(value, "bad id with spaces");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn ocr_html_response_format_is_parsed() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
format = "html"
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.services.ocr.format, OcrResponseFormat::Html);
}

#[test]
fn ocr_vl_toml_overrides_are_parsed() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr-vl]
enabled = true
default_model = "PaddlePaddle/PaddleOCR-VL-1.6"
available_models = ["PaddlePaddle/PaddleOCR-VL-1.5", "PaddlePaddle/PaddleOCR-VL-1.6"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]
idle_timeout = "2m"
max_loaded = 1
device = "cpu"
format = "text"
"#,
        exe_path,
    )
    .unwrap();

    assert!(config.services.ocr_vl.active());
    assert_eq!(
        config.services.ocr_vl.default_model,
        Some(ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap())
    );
    assert_eq!(
        config.services.ocr_vl.available_models,
        vec![
            ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.5").unwrap(),
            ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap(),
        ]
    );
    assert_eq!(
        config.services.ocr_vl.layout_default_model,
        Some(ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap())
    );
    assert_eq!(
        config.services.ocr_vl.layout_available_models,
        vec![ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap()]
    );
    assert_eq!(
        config.services.ocr_vl.idle_timeout,
        Duration::from_secs(120)
    );
    assert_eq!(config.services.ocr_vl.device, DevicePreference::Cpu);
    assert_eq!(config.services.ocr_vl.format, OcrResponseFormat::Text);
}

#[test]
fn active_ocr_rejects_ocr_vl_models() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidOcrModelKind { .. }));
    assert!(error.to_string().contains("traditional OCR model"));
}

#[test]
fn flat_layout_fields_are_parsed_for_ocr_and_ocr_vl() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]

[services.ocr-vl]
enabled = true
default_model = "PaddlePaddle/PaddleOCR-VL-1.6"
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]
"#,
        exe_path,
    )
    .unwrap();

    let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
    assert_eq!(
        config.services.ocr.layout_default_model.as_ref(),
        Some(&layout_model)
    );
    assert_eq!(
        config.services.ocr.layout_available_models,
        vec![layout_model.clone()]
    );
    assert_eq!(
        config.services.ocr_vl.layout_default_model.as_ref(),
        Some(&layout_model)
    );
    assert_eq!(
        config.services.ocr_vl.layout_available_models,
        vec![layout_model]
    );
}

#[test]
fn active_ocr_rejects_layout_model_in_main_available_models() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
available_models = ["PaddlePaddle/PP-OCRv6_tiny", "PaddlePaddle/PP-DocLayoutV3"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidOcrModelKind { .. }));
    assert!(error.to_string().contains("traditional OCR model"));
}

#[test]
fn active_ocr_accepts_layout_available_models_and_default() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]
"#,
        exe_path,
    )
    .unwrap();

    let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
    assert_eq!(
        config.services.ocr.layout_default_model.as_ref(),
        Some(&layout_model)
    );
    assert_eq!(
        config.services.ocr.layout_available_models,
        vec![layout_model]
    );
}

#[test]
fn active_ocr_vl_rejects_traditional_models() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr-vl]
enabled = true
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidOcrModelKind { .. }));
    assert!(error.to_string().contains("OCR-VL model"));
}

#[test]
fn active_ocr_vl_rejects_wrong_layout_model() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr-vl]
enabled = true
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
layout_default_model = "PaddlePaddle/PP-OCRv6_tiny"
layout_available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidOcrModelKind { .. }));
    assert!(error.to_string().contains("PaddlePaddle/PP-DocLayoutV3"));
}

#[test]
fn active_ocr_vl_layout_default_must_be_available_models_member() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr-vl]
enabled = true
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = []
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DefaultModelUnavailable { .. }));
    assert!(
        error
            .to_string()
            .contains("services.ocr-vl.layout_available_models")
    );
}

#[test]
fn toml_overrides_model_registry_and_services() {
    let exe_path = std::path::Path::new("/opt/orchion/orchion-server");
    let document = r#"
[server]
bind = "0.0.0.0:9000"
max_upload_size = "64M"

[models]
dir = "cache/models"
source = "modelscope"
max_loaded = 3

[services.asr]
enabled = true
default_model = "Qwen/Qwen3-ASR-1.7B"
available_models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
idle_timeout = "5m"
max_loaded = 2
device = "cuda0"

[services.tts]
enabled = true
default_model = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
available_models = ["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]
idle_timeout = "30s"
max_loaded = 1
device = "cuda:1"
format = "mp3"

[auth]
api_key = "test-secret"
"#;

    let config = ServerConfig::from_toml_str(document, exe_path).unwrap();

    assert_eq!(config.server.bind.port(), 9000);
    assert_eq!(config.server.max_upload_size, 64 * 1024 * 1024);
    assert_eq!(
        config.models.dir,
        exe_path.parent().unwrap().join("cache/models")
    );
    assert_eq!(config.models.source, ModelSource::ModelScope);
    assert_eq!(config.models.max_loaded, 3);
    assert!(config.services.asr.enabled);
    assert_eq!(
        config.services.asr.default_model,
        asr_model("Qwen/Qwen3-ASR-1.7B")
    );
    assert_eq!(
        config.services.asr.available_models,
        vec![
            asr_model("Qwen/Qwen3-ASR-0.6B"),
            asr_model("Qwen/Qwen3-ASR-1.7B"),
        ]
    );
    assert_eq!(config.services.asr.idle_timeout, Duration::from_secs(300));
    assert_eq!(config.services.asr.max_loaded, 2);
    assert_eq!(config.services.asr.device, DevicePreference::Cuda(Some(0)));
    assert!(config.services.tts.enabled);
    assert_eq!(
        config.services.tts.default_model,
        tts_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    );
    assert_eq!(
        config.services.tts.available_models,
        vec![
            tts_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            tts_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        ]
    );
    assert_eq!(config.services.tts.idle_timeout, Duration::from_secs(30));
    assert_eq!(config.services.tts.max_loaded, 1);
    assert_eq!(config.services.tts.device, DevicePreference::Cuda(Some(1)));
    assert_eq!(config.services.tts.format, "mp3");
    assert_eq!(config.auth.api_key.as_deref(), Some("test-secret"));
}

#[test]
fn device_aliases_are_parsed_in_services() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
device = "metal0"

[services.tts]
device = "cuda"
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.services.asr.device, DevicePreference::Metal);
    assert_eq!(config.services.tts.device, DevicePreference::Cuda(None));
}

#[test]
fn malformed_device_config_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
device = "cuda:"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("invalid services.asr.device"));
}

#[test]
fn empty_api_key_disables_authentication() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[auth]
api_key = ""
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.auth.api_key, None);
}

#[test]
fn enabled_service_default_must_be_available_models_member() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "Qwen/Qwen3-ASR-1.7B"
available_models = ["Qwen/Qwen3-ASR-0.6B"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("services.asr.available_models"));
}

#[test]
fn disabled_service_can_keep_default_outside_available_models() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = false
default_model = "Qwen/Qwen3-ASR-1.7B"
available_models = ["Qwen/Qwen3-ASR-0.6B"]
"#,
        exe_path,
    )
    .unwrap();

    assert!(!config.services.asr.enabled);
    assert_eq!(
        config.services.asr.default_model,
        asr_model("Qwen/Qwen3-ASR-1.7B")
    );
    assert_eq!(
        config.services.asr.available_models,
        vec![asr_model("Qwen/Qwen3-ASR-0.6B")]
    );
}

#[test]
fn invalid_model_cache_limits_are_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.tts]
max_loaded = 0
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("max_loaded"));
}

#[test]
fn invalid_global_model_cache_limit_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models]
max_loaded = 0
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("invalid models.max_loaded"));
}

#[test]
fn upload_size_units_are_parsed() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[server]
max_upload_size = "512K"
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.server.max_upload_size, 512 * 1024);
}

#[test]
fn invalid_upload_size_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[server]
max_upload_size = "huge"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("invalid upload size"));
}

#[test]
fn invalid_asr_model_id_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
available_models = ["not-a-model"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("invalid ASR model id"));
}

#[test]
fn invalid_short_asr_and_tts_model_ids_are_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
available_models = ["qwen3-asr-0.6b"]
"#,
        exe_path,
    )
    .unwrap_err();
    assert!(error.to_string().contains("invalid ASR model id"));

    let error = ServerConfig::from_toml_str(
        r#"
[services.tts]
available_models = ["qwen3-tts-0.6b-custom-voice"]
"#,
        exe_path,
    )
    .unwrap_err();
    assert!(error.to_string().contains("invalid TTS model id"));
}

#[test]
fn custom_asr_and_tts_model_ids_are_accepted() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "Acme/New-ASR"
available_models = ["Acme/New-ASR"]

[services.tts]
default_model = "Acme/New-TTS"
available_models = ["Acme/New-TTS"]
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.services.asr.default_model.as_str(), "Acme/New-ASR");
    assert_eq!(config.services.tts.default_model.as_str(), "Acme/New-TTS");
}

#[test]
fn tts_voice_and_language_are_request_only() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[services.tts]
voice = "ryan"
language = "english"
format = "wav"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn old_model_service_sections_are_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models.asr]
default = "Qwen/Qwen3-ASR-0.6B"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown field"));
}
