use orchion::{AsrModel, DevicePreference, DownloadSource, ModelSpec, TtsModel};
use serde::Deserialize;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

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
    pub auth: AuthSection,
    pub defaults: DefaultsSection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSection {
    pub bind: SocketAddr,
    pub max_upload_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsSection {
    pub dir: PathBuf,
    pub source: ModelSource,
    pub max_loaded: usize,
    pub asr: ModelRegistrySection<AsrModel>,
    pub tts: ModelRegistrySection<TtsModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistrySection<M> {
    pub default: M,
    pub available: Vec<M>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSection {
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultsSection {
    pub tts: TtsDefaults,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsDefaults {
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
    #[error("invalid upload size `{value}`: {message}")]
    InvalidUploadSize { value: String, message: String },
    #[error("unknown model source `{0}`; expected auto, huggingface, or modelscope")]
    UnknownModelSource(String),
    #[error("unknown ASR model `{0}`")]
    UnknownAsrModel(String),
    #[error("unknown TTS model `{0}`")]
    UnknownTtsModel(String),
    #[error("invalid duration `{value}`: {message}")]
    InvalidDuration { value: String, message: String },
    #[error("invalid {section}.max_loaded `{value}`: value must be greater than zero")]
    InvalidMaxLoaded { section: &'static str, value: usize },
    #[error(
        "invalid {section}.device `{value}`; expected auto, cpu, metal, metal0, cuda, cuda0, cuda:0, ..."
    )]
    InvalidDevice {
        section: &'static str,
        value: String,
    },
    #[error("default {category} model `{default}` must be included in {section}.available")]
    DefaultModelUnavailable {
        category: &'static str,
        section: &'static str,
        default: String,
    },
}

impl ServerConfig {
    #[must_use]
    pub fn default_for_exe(exe_path: &Path) -> Self {
        let exe_dir = exe_path.parent().unwrap_or_else(|| Path::new("."));
        Self {
            config_path: exe_dir.join("config.toml"),
            server: ServerSection {
                bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9090),
                max_upload_size: 30 * 1024 * 1024,
            },
            models: ModelsSection {
                dir: exe_dir.join("models"),
                source: ModelSource::Auto,
                max_loaded: 2,
                asr: ModelRegistrySection {
                    default: AsrModel::Qwen3Asr06B,
                    available: vec![AsrModel::Qwen3Asr06B],
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                },
                tts: ModelRegistrySection {
                    default: TtsModel::Qwen3Tts06BCustomVoice,
                    available: vec![TtsModel::Qwen3Tts06BCustomVoice],
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                },
            },
            auth: AuthSection { api_key: None },
            defaults: DefaultsSection {
                tts: TtsDefaults {
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
            if let Some(max_upload_size) = server.max_upload_size {
                config.server.max_upload_size = parse_upload_size(&max_upload_size)?;
            }
        }

        if let Some(models) = raw.models {
            if let Some(dir) = models.dir {
                config.models.dir = resolve_exe_relative(exe_dir, dir);
            }
            if let Some(source) = models.source {
                config.models.source = parse_model_source(&source)?;
            }
            if let Some(max_loaded) = models.max_loaded {
                if max_loaded == 0 {
                    return Err(ConfigError::InvalidMaxLoaded {
                        section: "models",
                        value: max_loaded,
                    });
                }
                config.models.max_loaded = max_loaded;
            }
            if let Some(asr) = models.asr {
                config.models.asr = parse_asr_registry(asr, config.models.asr)?;
            }
            if let Some(tts) = models.tts {
                config.models.tts = parse_tts_registry(tts, config.models.tts)?;
            }
        }

        if let Some(defaults) = raw.defaults {
            if let Some(tts) = defaults.tts {
                if let Some(format) = tts.format {
                    config.defaults.tts.format = format;
                }
            }
        }

        if let Some(auth) = raw.auth {
            if let Some(api_key) = auth.api_key {
                let api_key = api_key.trim();
                config.auth.api_key = if api_key.is_empty() {
                    None
                } else {
                    Some(api_key.to_string())
                };
            }
        }

        Ok(config)
    }
}

fn parse_asr_registry(
    raw: RawModelRegistry,
    mut registry: ModelRegistrySection<AsrModel>,
) -> Result<ModelRegistrySection<AsrModel>, ConfigError> {
    let available = raw.available;
    if let Some(default) = raw.default {
        registry.default = parse_asr_model(&default)?;
    }
    if let Some(available) = available {
        registry.available = available
            .iter()
            .map(String::as_str)
            .map(parse_asr_model)
            .collect::<Result<Vec<_>, _>>()?;
    } else {
        registry.available = vec![registry.default];
    }
    if let Some(device) = raw.device {
        registry.device = parse_device_preference("models.asr", &device)?;
    }
    apply_registry_limits(
        "models.asr",
        raw.idle_timeout,
        raw.max_loaded,
        &mut registry,
    )?;
    ensure_default_available(
        "ASR",
        "models.asr",
        registry.default.cache_key(),
        registry.available.contains(&registry.default),
    )?;
    Ok(registry)
}

fn parse_tts_registry(
    raw: RawModelRegistry,
    mut registry: ModelRegistrySection<TtsModel>,
) -> Result<ModelRegistrySection<TtsModel>, ConfigError> {
    let available = raw.available;
    if let Some(default) = raw.default {
        registry.default = parse_tts_model(&default)?;
    }
    if let Some(available) = available {
        registry.available = available
            .iter()
            .map(String::as_str)
            .map(parse_tts_model)
            .collect::<Result<Vec<_>, _>>()?;
    } else {
        registry.available = vec![registry.default];
    }
    if let Some(device) = raw.device {
        registry.device = parse_device_preference("models.tts", &device)?;
    }
    apply_registry_limits(
        "models.tts",
        raw.idle_timeout,
        raw.max_loaded,
        &mut registry,
    )?;
    ensure_default_available(
        "TTS",
        "models.tts",
        registry.default.cache_key(),
        registry.available.contains(&registry.default),
    )?;
    Ok(registry)
}

fn apply_registry_limits<M>(
    section: &'static str,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    registry: &mut ModelRegistrySection<M>,
) -> Result<(), ConfigError> {
    if let Some(idle_timeout) = idle_timeout {
        registry.idle_timeout = parse_duration(&idle_timeout)?;
    }
    if let Some(max_loaded) = max_loaded {
        if max_loaded == 0 {
            return Err(ConfigError::InvalidMaxLoaded {
                section,
                value: max_loaded,
            });
        }
        registry.max_loaded = max_loaded;
    }
    Ok(())
}

fn ensure_default_available(
    category: &'static str,
    section: &'static str,
    default: &str,
    available: bool,
) -> Result<(), ConfigError> {
    if available {
        Ok(())
    } else {
        Err(ConfigError::DefaultModelUnavailable {
            category,
            section,
            default: default.to_string(),
        })
    }
}

fn parse_duration(value: &str) -> Result<Duration, ConfigError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ConfigError::InvalidDuration {
            value: value.to_string(),
            message: "value must not be empty".to_string(),
        });
    }
    let split_at = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, unit) = value.split_at(split_at);
    let amount = digits
        .parse::<u64>()
        .map_err(|error| ConfigError::InvalidDuration {
            value: value.to_string(),
            message: error.to_string(),
        })?;
    if amount == 0 {
        return Err(ConfigError::InvalidDuration {
            value: value.to_string(),
            message: "value must be greater than zero".to_string(),
        });
    }
    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 60 * 60,
        _ => {
            return Err(ConfigError::InvalidDuration {
                value: value.to_string(),
                message: "expected seconds, minutes, or hours".to_string(),
            });
        }
    };
    amount
        .checked_mul(multiplier)
        .map(Duration::from_secs)
        .ok_or_else(|| ConfigError::InvalidDuration {
            value: value.to_string(),
            message: "value is too large".to_string(),
        })
}

fn parse_upload_size(value: &str) -> Result<usize, ConfigError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ConfigError::InvalidUploadSize {
            value: value.to_string(),
            message: "value must not be empty".to_string(),
        });
    }

    let (digits, multiplier) = match value.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&value[..value.len() - 1], 1024_usize),
        Some(b'm' | b'M') => (&value[..value.len() - 1], 1024_usize * 1024),
        Some(b'g' | b'G') => (&value[..value.len() - 1], 1024_usize * 1024 * 1024),
        Some(_) => (value, 1),
        None => unreachable!("empty value handled above"),
    };
    let amount =
        digits
            .trim()
            .parse::<usize>()
            .map_err(|error| ConfigError::InvalidUploadSize {
                value: value.to_string(),
                message: error.to_string(),
            })?;
    if amount == 0 {
        return Err(ConfigError::InvalidUploadSize {
            value: value.to_string(),
            message: "value must be greater than zero".to_string(),
        });
    }
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| ConfigError::InvalidUploadSize {
            value: value.to_string(),
            message: "value is too large".to_string(),
        })
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

fn parse_device_preference(
    section: &'static str,
    value: &str,
) -> Result<DevicePreference, ConfigError> {
    value
        .parse::<DevicePreference>()
        .map_err(|_| ConfigError::InvalidDevice {
            section,
            value: value.to_string(),
        })
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
#[serde(deny_unknown_fields)]
struct RawConfig {
    server: Option<RawServer>,
    models: Option<RawModels>,
    auth: Option<RawAuth>,
    defaults: Option<RawDefaults>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServer {
    bind: Option<String>,
    max_upload_size: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModels {
    dir: Option<PathBuf>,
    source: Option<String>,
    max_loaded: Option<usize>,
    asr: Option<RawModelRegistry>,
    tts: Option<RawModelRegistry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelRegistry {
    default: Option<String>,
    available: Option<Vec<String>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuth {
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefaults {
    tts: Option<RawTtsDefaults>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTtsDefaults {
    format: Option<String>,
}
