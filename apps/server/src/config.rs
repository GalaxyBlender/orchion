use orchion::{AsrModel, DownloadSource, TtsModel};
use serde::Deserialize;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    Auto,
    HuggingFace,
    ModelScope,
}

impl From<ModelSource> for DownloadSource {
    fn from(source: ModelSource) -> Self {
        match source {
            ModelSource::Auto => Self::Auto,
            ModelSource::HuggingFace => Self::HuggingFace,
            ModelSource::ModelScope => Self::ModelScope,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub config_path: PathBuf,
    pub server: ServerSection,
    pub models: ModelsSection,
    pub defaults: DefaultsSection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSection {
    pub bind: SocketAddr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsSection {
    pub dir: PathBuf,
    pub source: ModelSource,
    pub asr: AsrModel,
    pub tts: TtsModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultsSection {
    pub tts: TtsDefaults,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsDefaults {
    pub voice: String,
    pub language: Option<String>,
    pub format: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config `{path}`: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config TOML: {0}")]
    ParseToml(#[from] toml::de::Error),
    #[error("invalid server bind address `{value}`: {source}")]
    InvalidBind {
        value: String,
        source: std::net::AddrParseError,
    },
    #[error("unknown model source `{0}`; expected auto, huggingface, or modelscope")]
    UnknownModelSource(String),
    #[error("unknown ASR model `{0}`")]
    UnknownAsrModel(String),
    #[error("unknown TTS model `{0}`")]
    UnknownTtsModel(String),
}

impl ServerConfig {
    #[must_use]
    pub fn default_for_exe(exe_path: &Path) -> Self {
        let exe_dir = exe_path.parent().unwrap_or_else(|| Path::new("."));
        Self {
            config_path: exe_dir.join("config.toml"),
            server: ServerSection {
                bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080),
            },
            models: ModelsSection {
                dir: exe_dir.join("models"),
                source: ModelSource::Auto,
                asr: AsrModel::Qwen3Asr06B,
                tts: TtsModel::Qwen3Tts06BCustomVoice,
            },
            defaults: DefaultsSection {
                tts: TtsDefaults {
                    voice: "ryan".to_string(),
                    language: Some("english".to_string()),
                    format: "wav".to_string(),
                },
            },
        }
    }

    pub fn load(config_path: Option<PathBuf>) -> Result<Self, ConfigError> {
        let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("orchion-server"));
        let default = Self::default_for_exe(&exe_path);
        let path = config_path.unwrap_or_else(|| default.config_path.clone());
        if !path.exists() {
            return Ok(Self {
                config_path: path,
                ..default
            });
        }
        let document = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
            path: path.clone(),
            source,
        })?;
        let mut config = Self::from_toml_str(&document, &exe_path)?;
        config.config_path = path;
        Ok(config)
    }

    pub fn from_toml_str(document: &str, exe_path: &Path) -> Result<Self, ConfigError> {
        let raw = toml::from_str::<RawConfig>(document)?;
        let mut config = Self::default_for_exe(exe_path);
        let exe_dir = exe_path.parent().unwrap_or_else(|| Path::new("."));

        if let Some(server) = raw.server {
            if let Some(bind) = server.bind {
                config.server.bind = bind.parse().map_err(|source| ConfigError::InvalidBind {
                    value: bind,
                    source,
                })?;
            }
        }

        if let Some(models) = raw.models {
            if let Some(dir) = models.dir {
                config.models.dir = resolve_exe_relative(exe_dir, dir);
            }
            if let Some(source) = models.source {
                config.models.source = parse_model_source(&source)?;
            }
            if let Some(asr) = models.asr {
                config.models.asr = parse_asr_model(&asr)?;
            }
            if let Some(tts) = models.tts {
                config.models.tts = parse_tts_model(&tts)?;
            }
        }

        if let Some(defaults) = raw.defaults {
            if let Some(tts) = defaults.tts {
                if let Some(voice) = tts.voice {
                    config.defaults.tts.voice = voice;
                }
                if let Some(language) = tts.language {
                    config.defaults.tts.language = Some(language);
                }
                if let Some(format) = tts.format {
                    config.defaults.tts.format = format;
                }
            }
        }

        Ok(config)
    }
}

pub fn parse_asr_model(value: &str) -> Result<AsrModel, ConfigError> {
    match normalize_identifier(value).as_str() {
        "qwen3-asr-0.6b" | "qwen/qwen3-asr-0.6b" => Ok(AsrModel::Qwen3Asr06B),
        "qwen3-asr-1.7b" | "qwen/qwen3-asr-1.7b" => Ok(AsrModel::Qwen3Asr17B),
        _ => Err(ConfigError::UnknownAsrModel(value.to_string())),
    }
}

pub fn parse_tts_model(value: &str) -> Result<TtsModel, ConfigError> {
    match normalize_identifier(value).as_str() {
        "qwen3-tts-0.6b-base" | "qwen/qwen3-tts-12hz-0.6b-base" => Ok(TtsModel::Qwen3Tts06BBase),
        "qwen3-tts-0.6b-custom-voice" | "qwen/qwen3-tts-12hz-0.6b-customvoice" => {
            Ok(TtsModel::Qwen3Tts06BCustomVoice)
        }
        "qwen3-tts-1.7b-base" | "qwen/qwen3-tts-12hz-1.7b-base" => Ok(TtsModel::Qwen3Tts17BBase),
        "qwen3-tts-1.7b-custom-voice" | "qwen/qwen3-tts-12hz-1.7b-customvoice" => {
            Ok(TtsModel::Qwen3Tts17BCustomVoice)
        }
        "qwen3-tts-1.7b-voice-design" | "qwen/qwen3-tts-12hz-1.7b-voicedesign" => {
            Ok(TtsModel::Qwen3Tts17BVoiceDesign)
        }
        _ => Err(ConfigError::UnknownTtsModel(value.to_string())),
    }
}

fn parse_model_source(value: &str) -> Result<ModelSource, ConfigError> {
    match normalize_identifier(value).as_str() {
        "auto" => Ok(ModelSource::Auto),
        "huggingface" | "hf" => Ok(ModelSource::HuggingFace),
        "modelscope" | "ms" => Ok(ModelSource::ModelScope),
        _ => Err(ConfigError::UnknownModelSource(value.to_string())),
    }
}

fn resolve_exe_relative(exe_dir: &Path, value: impl Into<PathBuf>) -> PathBuf {
    let path = value.into();
    if path.is_absolute() {
        path
    } else {
        exe_dir.join(path)
    }
}

fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    server: Option<RawServer>,
    models: Option<RawModels>,
    defaults: Option<RawDefaults>,
}

#[derive(Debug, Deserialize)]
struct RawServer {
    bind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawModels {
    dir: Option<PathBuf>,
    source: Option<String>,
    asr: Option<String>,
    tts: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawDefaults {
    tts: Option<RawTtsDefaults>,
}

#[derive(Debug, Deserialize)]
struct RawTtsDefaults {
    voice: Option<String>,
    language: Option<String>,
    format: Option<String>,
}
