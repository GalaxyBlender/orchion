use orchion::{AsrModel, DevicePreference, TtsModel};
use orchion_server::config::{ModelSource, ServerConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

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
    assert_eq!(config.models.asr.default, AsrModel::Qwen3Asr06B);
    assert_eq!(config.models.asr.available, vec![AsrModel::Qwen3Asr06B]);
    assert_eq!(config.models.asr.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.models.asr.max_loaded, 1);
    assert_eq!(config.models.asr.device, DevicePreference::Auto);
    assert_eq!(config.models.tts.default, TtsModel::Qwen3Tts06BCustomVoice);
    assert_eq!(
        config.models.tts.available,
        vec![TtsModel::Qwen3Tts06BCustomVoice]
    );
    assert_eq!(config.models.tts.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.models.tts.max_loaded, 1);
    assert_eq!(config.models.tts.device, DevicePreference::Auto);
    assert_eq!(config.auth.api_key, None);
    assert_eq!(config.server.max_upload_size, 30 * 1024 * 1024);
}

#[test]
fn toml_overrides_model_registries_and_defaults() {
    let exe_path = std::path::Path::new("/opt/orchion/orchion-server");
    let document = r#"
[server]
bind = "0.0.0.0:9000"
max_upload_size = "64M"

[models]
dir = "cache/models"
source = "modelscope"
max_loaded = 3

[models.asr]
default = "qwen3-asr-1.7b"
available = ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]
idle_timeout = "5m"
max_loaded = 2
device = "cuda0"

[models.tts]
default = "qwen3-tts-1.7b-voice-design"
available = ["qwen3-tts-0.6b-custom-voice", "qwen3-tts-1.7b-voice-design"]
idle_timeout = "30s"
max_loaded = 1
device = "cuda:1"

[auth]
api_key = "test-secret"

[defaults.tts]
format = "wav"
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
    assert_eq!(config.models.asr.default, AsrModel::Qwen3Asr17B);
    assert_eq!(
        config.models.asr.available,
        vec![AsrModel::Qwen3Asr06B, AsrModel::Qwen3Asr17B]
    );
    assert_eq!(config.models.asr.idle_timeout, Duration::from_secs(300));
    assert_eq!(config.models.asr.max_loaded, 2);
    assert_eq!(config.models.asr.device, DevicePreference::Cuda(Some(0)));
    assert_eq!(config.models.tts.default, TtsModel::Qwen3Tts17BVoiceDesign);
    assert_eq!(
        config.models.tts.available,
        vec![
            TtsModel::Qwen3Tts06BCustomVoice,
            TtsModel::Qwen3Tts17BVoiceDesign,
        ]
    );
    assert_eq!(config.models.tts.idle_timeout, Duration::from_secs(30));
    assert_eq!(config.models.tts.max_loaded, 1);
    assert_eq!(config.models.tts.device, DevicePreference::Cuda(Some(1)));
    assert_eq!(config.auth.api_key.as_deref(), Some("test-secret"));
    assert_eq!(config.defaults.tts.format, "wav");
}

#[test]
fn device_aliases_are_parsed_in_model_registries() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let config = ServerConfig::from_toml_str(
        r#"
[models.asr]
device = "metal0"

[models.tts]
device = "cuda"
"#,
        exe_path,
    )
    .unwrap();

    assert_eq!(config.models.asr.device, DevicePreference::Metal);
    assert_eq!(config.models.tts.device, DevicePreference::Cuda(None));
}

#[test]
fn malformed_device_config_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models.asr]
device = "cuda:"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("invalid models.asr.device"));
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
fn model_default_must_be_available() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models.asr]
default = "qwen3-asr-1.7b"
available = ["qwen3-asr-0.6b"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("default ASR model"));
}

#[test]
fn invalid_model_cache_limits_are_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models.tts]
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
fn unknown_model_name_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models]

[models.asr]
available = ["not-a-model"]
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown ASR model"));
}

#[test]
fn tts_voice_and_language_are_request_only() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[defaults.tts]
voice = "ryan"
language = "english"
format = "wav"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown field"));
}
