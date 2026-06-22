use orchion::{
    AsrModel, DevicePreference, DownloadSource, ModelId, ModelSpec, OcrResponseFormat, TtsModel,
};
use orchion_core::{KnownOcrModel, OcrModelKind};
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
    pub services: ServicesSection,
    pub auth: AuthSection,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServicesSection {
    pub asr: ModelServiceSection<AsrModel>,
    pub tts: TtsServiceSection,
    pub ocr: OcrServiceSection,
    pub ocr_vl: OcrVlServiceSection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelServiceSection<M> {
    pub enabled: bool,
    pub default_model: M,
    pub available_models: Vec<M>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsServiceSection {
    pub enabled: bool,
    pub default_model: TtsModel,
    pub available_models: Vec<TtsModel>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrServiceSection {
    pub enabled: bool,
    pub default_model: Option<ModelId>,
    pub available_models: Vec<ModelId>,
    pub layout_default_model: Option<ModelId>,
    pub layout_available_models: Vec<ModelId>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: OcrResponseFormat,
}

impl OcrServiceSection {
    #[must_use]
    pub fn active(&self) -> bool {
        self.enabled && !self.available_models.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrVlServiceSection {
    pub enabled: bool,
    pub default_model: Option<ModelId>,
    pub available_models: Vec<ModelId>,
    pub layout_default_model: Option<ModelId>,
    pub layout_available_models: Vec<ModelId>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: OcrResponseFormat,
}

impl OcrVlServiceSection {
    #[must_use]
    pub fn active(&self) -> bool {
        self.enabled && !self.available_models.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSection {
    pub api_key: Option<String>,
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
    #[error("invalid {section} model id `{value}`; expected vendor/name")]
    InvalidModelId {
        section: &'static str,
        value: String,
    },
    #[error("invalid {section} `{value}`; expected json, text, markdown, or html")]
    InvalidOcrFormat {
        section: &'static str,
        value: String,
    },
    #[error("{section} is enabled but available_models is empty")]
    ServiceEnabledWithoutModels { section: &'static str },
    #[error("invalid {section} model `{model}`: expected {expected}")]
    InvalidOcrModelKind {
        section: &'static str,
        model: String,
        expected: &'static str,
    },
    #[error("default {category} model `{default}` must be included in {section}.available_models")]
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
            },
            services: ServicesSection {
                asr: ModelServiceSection {
                    enabled: false,
                    default_model: AsrModel::Qwen3Asr06B,
                    available_models: vec![AsrModel::Qwen3Asr06B],
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                },
                tts: TtsServiceSection {
                    enabled: false,
                    default_model: TtsModel::Qwen3Tts06BCustomVoice,
                    available_models: vec![TtsModel::Qwen3Tts06BCustomVoice],
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: "wav".to_string(),
                },
                ocr: OcrServiceSection {
                    enabled: false,
                    default_model: None,
                    available_models: Vec::new(),
                    layout_default_model: None,
                    layout_available_models: Vec::new(),
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: OcrResponseFormat::Json,
                },
                ocr_vl: OcrVlServiceSection {
                    enabled: false,
                    default_model: None,
                    available_models: Vec::new(),
                    layout_default_model: None,
                    layout_available_models: Vec::new(),
                    idle_timeout: Duration::from_secs(600),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: OcrResponseFormat::Markdown,
                },
            },
            auth: AuthSection { api_key: None },
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
        }

        if let Some(services) = raw.services {
            if let Some(asr) = services.asr {
                config.services.asr = parse_asr_service(asr, config.services.asr)?;
            }
            if let Some(tts) = services.tts {
                config.services.tts = parse_tts_service(tts, config.services.tts)?;
            }
            if let Some(ocr) = services.ocr {
                config.services.ocr = parse_ocr_service(ocr, config.services.ocr)?;
            }
            if let Some(ocr_vl) = services.ocr_vl {
                config.services.ocr_vl = parse_ocr_vl_service(ocr_vl, config.services.ocr_vl)?;
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

fn parse_asr_service(
    raw: RawModelService,
    mut service: ModelServiceSection<AsrModel>,
) -> Result<ModelServiceSection<AsrModel>, ConfigError> {
    let available_models = raw.available_models;
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = parse_asr_model(&default_model)?;
    }
    if let Some(available_models) = available_models {
        service.available_models = available_models
            .iter()
            .map(String::as_str)
            .map(parse_asr_model)
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.asr", &device)?;
    }
    apply_service_limits(
        "services.asr",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled {
        ensure_default_available(
            "ASR",
            "services.asr",
            service.default_model.cache_key(),
            service.available_models.contains(&service.default_model),
        )?;
    }
    Ok(service)
}

fn parse_tts_service(
    raw: RawTtsService,
    mut service: TtsServiceSection,
) -> Result<TtsServiceSection, ConfigError> {
    let available_models = raw.available_models;
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = parse_tts_model(&default_model)?;
    }
    if let Some(available_models) = available_models {
        service.available_models = available_models
            .iter()
            .map(String::as_str)
            .map(parse_tts_model)
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.tts", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = format;
    }
    apply_service_limits(
        "services.tts",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled {
        ensure_default_available(
            "TTS",
            "services.tts",
            service.default_model.cache_key(),
            service.available_models.contains(&service.default_model),
        )?;
    }
    Ok(service)
}

fn parse_ocr_service(
    raw: RawOcrService,
    mut service: OcrServiceSection,
) -> Result<OcrServiceSection, ConfigError> {
    let available_models = raw.available_models;
    let layout_available_models = raw.layout_available_models;
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = Some(parse_model_id(
            "services.ocr.default_model",
            &default_model,
        )?);
    }
    if let Some(available_models) = available_models {
        service.available_models =
            parse_model_ids("services.ocr.available_models", &available_models)?;
    }
    if let Some(layout_default_model) = raw.layout_default_model {
        service.layout_default_model = Some(parse_model_id(
            "services.ocr.layout_default_model",
            &layout_default_model,
        )?);
    }
    if let Some(layout_available_models) = layout_available_models {
        service.layout_available_models = parse_model_ids(
            "services.ocr.layout_available_models",
            &layout_available_models,
        )?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.ocr", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = parse_ocr_format("services.ocr.format", &format)?;
    }
    apply_service_limits(
        "services.ocr",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    validate_ocr_service(
        "OCR",
        "services.ocr",
        service.enabled,
        service.default_model.as_ref(),
        &service.available_models,
        validate_traditional_ocr_model,
    )?;
    validate_ocr_layout_config(
        "OCR layout",
        "services.ocr.layout_available_models",
        service.enabled,
        service.layout_default_model.as_ref(),
        &service.layout_available_models,
    )?;
    Ok(service)
}

fn parse_ocr_vl_service(
    raw: RawOcrVlService,
    mut service: OcrVlServiceSection,
) -> Result<OcrVlServiceSection, ConfigError> {
    let available_models = raw.available_models;
    let layout_available_models = raw.layout_available_models;
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = Some(parse_model_id(
            "services.ocr-vl.default_model",
            &default_model,
        )?);
    }
    if let Some(available_models) = available_models {
        service.available_models =
            parse_model_ids("services.ocr-vl.available_models", &available_models)?;
    }
    if let Some(layout_default_model) = raw.layout_default_model {
        service.layout_default_model = Some(parse_model_id(
            "services.ocr-vl.layout_default_model",
            &layout_default_model,
        )?);
    }
    if let Some(layout_available_models) = layout_available_models {
        service.layout_available_models = parse_model_ids(
            "services.ocr-vl.layout_available_models",
            &layout_available_models,
        )?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.ocr-vl", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = parse_ocr_format("services.ocr-vl.format", &format)?;
    }
    apply_service_limits(
        "services.ocr-vl",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    validate_ocr_service(
        "OCR-VL",
        "services.ocr-vl",
        service.enabled,
        service.default_model.as_ref(),
        &service.available_models,
        validate_ocr_vl_model,
    )?;
    validate_ocr_layout_config(
        "OCR-VL layout",
        "services.ocr-vl.layout_available_models",
        service.enabled,
        service.layout_default_model.as_ref(),
        &service.layout_available_models,
    )?;
    Ok(service)
}

fn validate_ocr_service(
    category: &'static str,
    section: &'static str,
    enabled: bool,
    default_model: Option<&ModelId>,
    available_models: &[ModelId],
    validate_model: fn(&'static str, &ModelId) -> Result<(), ConfigError>,
) -> Result<(), ConfigError> {
    if !enabled {
        return Ok(());
    }
    if available_models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels { section });
    }
    if let Some(default_model) = default_model {
        ensure_default_available(
            category,
            section,
            default_model.as_str(),
            available_models.contains(default_model),
        )?;
    }
    for model in available_models {
        validate_model(section, model)?;
    }
    Ok(())
}

fn validate_traditional_ocr_model(
    section: &'static str,
    model: &ModelId,
) -> Result<(), ConfigError> {
    match known_ocr_model(section, model)?.kind() {
        OcrModelKind::TraditionalOcr => Ok(()),
        OcrModelKind::Layout | OcrModelKind::OcrVl => Err(ConfigError::InvalidOcrModelKind {
            section,
            model: model.to_string(),
            expected: "traditional OCR model",
        }),
    }
}

fn validate_ocr_layout_config(
    category: &'static str,
    section: &'static str,
    enabled: bool,
    default_model: Option<&ModelId>,
    available_models: &[ModelId],
) -> Result<(), ConfigError> {
    if !enabled {
        return Ok(());
    }
    if let Some(default_model) = default_model {
        ensure_default_available(
            category,
            section,
            default_model.as_str(),
            available_models.contains(default_model),
        )?;
    }
    for model in available_models {
        validate_layout_model(section, model)?;
    }
    Ok(())
}

fn validate_ocr_vl_model(section: &'static str, model: &ModelId) -> Result<(), ConfigError> {
    match known_ocr_model(section, model)?.kind() {
        OcrModelKind::OcrVl => Ok(()),
        OcrModelKind::TraditionalOcr | OcrModelKind::Layout => {
            Err(ConfigError::InvalidOcrModelKind {
                section,
                model: model.to_string(),
                expected: "OCR-VL model",
            })
        }
    }
}

fn validate_layout_model(section: &'static str, model: &ModelId) -> Result<(), ConfigError> {
    let known = known_ocr_model(section, model)?;
    if known == KnownOcrModel::PpDocLayoutV3 {
        return Ok(());
    }
    Err(ConfigError::InvalidOcrModelKind {
        section,
        model: model.to_string(),
        expected: "PaddlePaddle/PP-DocLayoutV3",
    })
}

fn known_ocr_model(section: &'static str, model: &ModelId) -> Result<KnownOcrModel, ConfigError> {
    KnownOcrModel::from_model_id(model).map_err(|_| ConfigError::InvalidOcrModelKind {
        section,
        model: model.to_string(),
        expected: "supported OCR model",
    })
}

fn apply_service_limits(
    section: &'static str,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    service_idle_timeout: &mut Duration,
    service_max_loaded: &mut usize,
) -> Result<(), ConfigError> {
    if let Some(idle_timeout) = idle_timeout {
        *service_idle_timeout = parse_duration(&idle_timeout)?;
    }
    if let Some(max_loaded) = max_loaded {
        if max_loaded == 0 {
            return Err(ConfigError::InvalidMaxLoaded {
                section,
                value: max_loaded,
            });
        }
        *service_max_loaded = max_loaded;
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

fn parse_model_ids(section: &'static str, values: &[String]) -> Result<Vec<ModelId>, ConfigError> {
    values
        .iter()
        .map(|value| parse_model_id(section, value))
        .collect()
}

fn parse_model_id(section: &'static str, value: &str) -> Result<ModelId, ConfigError> {
    ModelId::parse(value).map_err(|_| ConfigError::InvalidModelId {
        section,
        value: value.to_string(),
    })
}

fn parse_ocr_format(section: &'static str, value: &str) -> Result<OcrResponseFormat, ConfigError> {
    match value {
        "json" => Ok(OcrResponseFormat::Json),
        "text" => Ok(OcrResponseFormat::Text),
        "markdown" => Ok(OcrResponseFormat::Markdown),
        "html" => Ok(OcrResponseFormat::Html),
        _ => Err(ConfigError::InvalidOcrFormat {
            section,
            value: value.to_string(),
        }),
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
    services: Option<RawServices>,
    auth: Option<RawAuth>,
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
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServices {
    asr: Option<RawModelService>,
    tts: Option<RawTtsService>,
    ocr: Option<RawOcrService>,
    #[serde(rename = "ocr-vl")]
    ocr_vl: Option<RawOcrVlService>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelService {
    enabled: Option<bool>,
    default_model: Option<String>,
    available_models: Option<Vec<String>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTtsService {
    enabled: Option<bool>,
    default_model: Option<String>,
    available_models: Option<Vec<String>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOcrService {
    enabled: Option<bool>,
    default_model: Option<String>,
    available_models: Option<Vec<String>>,
    layout_default_model: Option<String>,
    layout_available_models: Option<Vec<String>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOcrVlService {
    enabled: Option<bool>,
    default_model: Option<String>,
    available_models: Option<Vec<String>>,
    layout_default_model: Option<String>,
    layout_available_models: Option<Vec<String>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuth {
    api_key: Option<String>,
}
