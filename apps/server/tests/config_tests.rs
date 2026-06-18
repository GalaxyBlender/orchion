use orchion::{AsrModel, TtsModel};
use orchion_server::config::{ModelSource, ServerConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

#[test]
fn defaults_are_executable_relative() {
    let exe_path = std::path::Path::new("/tmp/orchion/bin/orchion-server");
    let config = ServerConfig::default_for_exe(exe_path);

    assert_eq!(
        config.server.bind,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080)
    );
    assert_eq!(
        config.config_path,
        exe_path.parent().unwrap().join("config.toml")
    );
    assert_eq!(config.models.dir, exe_path.parent().unwrap().join("models"));
    assert_eq!(config.models.source, ModelSource::Auto);
    assert_eq!(config.models.asr, AsrModel::Qwen3Asr06B);
    assert_eq!(config.models.tts, TtsModel::Qwen3Tts06BCustomVoice);
}

#[test]
fn toml_overrides_models_and_defaults() {
    let exe_path = std::path::Path::new("/opt/orchion/orchion-server");
    let document = r#"
[server]
bind = "0.0.0.0:9000"

[models]
dir = "cache/models"
source = "modelscope"
asr = "qwen3-asr-1.7b"
tts = "qwen3-tts-1.7b-voice-design"

[defaults.tts]
voice = "design"
language = "chinese"
format = "wav"
"#;

    let config = ServerConfig::from_toml_str(document, exe_path).unwrap();

    assert_eq!(config.server.bind.port(), 9000);
    assert_eq!(
        config.models.dir,
        exe_path.parent().unwrap().join("cache/models")
    );
    assert_eq!(config.models.source, ModelSource::ModelScope);
    assert_eq!(config.models.asr, AsrModel::Qwen3Asr17B);
    assert_eq!(config.models.tts, TtsModel::Qwen3Tts17BVoiceDesign);
    assert_eq!(config.defaults.tts.voice, "design");
    assert_eq!(config.defaults.tts.language.as_deref(), Some("chinese"));
}

#[test]
fn unknown_model_name_is_rejected() {
    let exe_path = std::path::Path::new("/tmp/orchion-server");
    let error = ServerConfig::from_toml_str(
        r#"
[models]
asr = "not-a-model"
"#,
        exe_path,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown ASR model"));
}
