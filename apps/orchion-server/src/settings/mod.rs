use orchion::{
    AsrModel, DevicePreference, DownloadSource, KnownOcrModel, ModelCategory, ModelId, ModelSpec,
    ModelUrl, OcrModel, OcrModelKind, OcrResponseFormat, TtsModel, model_descriptor,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::hash::Hash;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_ASR_STREAM_TARGET_SEGMENT: Duration = Duration::from_secs(12);
pub const DEFAULT_ASR_STREAM_MAX_SEGMENT: Duration = Duration::from_mins(2);
pub const DEFAULT_ASR_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
pub const DEFAULT_ASR_STREAM_MAX_DURATION: Duration = Duration::from_hours(2);
pub const DEFAULT_ASR_MAX_AUDIO_DURATION: Duration = Duration::from_mins(30);
pub const DEFAULT_TTS_MAX_LENGTH: usize = 2048;
pub const DEFAULT_TTS_MAX_REFERENCE_AUDIO_DURATION: Duration = Duration::from_mins(5);
pub const DEFAULT_OCR_VL_MAX_TOKENS: usize = 4096;
pub const DEFAULT_OCR_MAX_PIXELS: u64 = 100_000_000;
pub const DEFAULT_LLM_MAX_TOKENS: usize = 4096;
pub const DEFAULT_MAX_CONCURRENT_INFERENCE: usize = 2;
pub const DEFAULT_MAX_WEBSOCKET_CONNECTIONS: usize = 64;
pub const DEFAULT_MAX_PENDING_WEBSOCKET_CONNECTIONS: usize = 16;
pub const DEFAULT_MAX_WEBSOCKET_MESSAGE_SIZE: usize = 2 * 1024 * 1024;
pub const MAX_ACTIVITY_HISTORY_CAPACITY: usize = 10_000;
pub const CORS_ALLOWED_ORIGINS_ENV: &str = "CORS_ALLOWED_ORIGINS";

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

#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    pub config_path: PathBuf,
    pub server: ServerSection,
    pub activity: ActivitySection,
    pub models: ModelsSection,
    pub services: ServicesSection,
    pub auth: AuthSection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivitySection {
    pub enabled: bool,
    pub history_capacity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSection {
    pub bind: SocketAddr,
    pub cors_allowed_origins: Vec<String>,
    pub max_upload_size: usize,
    pub max_pdf_pages: usize,
    pub max_pdf_pixels: u64,
    pub max_pdf_output_size: usize,
    pub max_concurrent_inference: usize,
    pub max_websocket_connections: usize,
    pub max_pending_websocket_connections: usize,
    pub max_websocket_message_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsSection {
    pub dir: PathBuf,
    pub source: ModelSource,
    pub max_loaded: usize,
    pub verify_file_integrity: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServicesSection {
    pub asr: AsrServiceSection,
    pub tts: TtsServiceSection,
    pub ocr: OcrServiceSection,
    pub ocr_vl: OcrVlServiceSection,
    pub llm: LlmServiceSection,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrServiceSection {
    pub enabled: bool,
    pub default_model: AsrModel,
    pub models: Vec<ModelDeployment<AsrModel>>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub stream_chunk_size: f32,
    pub stream_target_segment: Duration,
    pub stream_max_segment: Duration,
    pub stream_idle_timeout: Duration,
    pub stream_max_duration: Duration,
    pub max_audio_duration: Duration,
}

impl AsrServiceSection {
    #[must_use]
    pub fn runtime_models(&self) -> Vec<AsrModel> {
        self.models
            .iter()
            .map(|model| model.runtime.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelServiceSection<M> {
    pub enabled: bool,
    pub default_model: M,
    pub models: Vec<ModelDeployment<M>>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsServiceSection {
    pub enabled: bool,
    pub default_model: TtsModel,
    pub models: Vec<ModelDeployment<TtsModel>>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: String,
    pub max_length: usize,
    pub max_reference_audio_duration: Duration,
}

impl TtsServiceSection {
    #[must_use]
    pub fn runtime_models(&self) -> Vec<TtsModel> {
        self.models
            .iter()
            .map(|model| model.runtime.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrServiceSection {
    pub enabled: bool,
    pub default_model: Option<ModelId>,
    pub models: Vec<OcrModelDeployment>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: OcrResponseFormat,
    pub max_pixels: u64,
}

impl OcrServiceSection {
    #[must_use]
    pub fn active(&self) -> bool {
        self.enabled && !self.models.is_empty()
    }

    #[must_use]
    pub fn model_ids(&self) -> Vec<ModelId> {
        self.models.iter().map(|model| model.id.clone()).collect()
    }

    #[must_use]
    pub fn layout_ids(&self) -> Vec<ModelId> {
        deployment_layout_ids(&self.models)
    }

    #[must_use]
    pub fn layout_ids_for(&self, id: &ModelId) -> Vec<ModelId> {
        deployment_layout_ids_for(id, &self.models)
    }

    #[must_use]
    pub fn model_layouts(&self) -> Vec<(ModelId, ModelId)> {
        deployment_model_layouts(&self.models)
    }

    #[must_use]
    pub fn default_layout_runtime(&self) -> Option<&OcrModel> {
        deployment_default_layout(self.default_model.as_ref(), &self.models)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrVlServiceSection {
    pub enabled: bool,
    pub default_model: Option<ModelId>,
    pub models: Vec<OcrModelDeployment>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
    pub device: DevicePreference,
    pub format: OcrResponseFormat,
    pub max_tokens: usize,
    pub max_pixels: u64,
}

impl OcrVlServiceSection {
    #[must_use]
    pub fn active(&self) -> bool {
        self.enabled && !self.models.is_empty()
    }

    #[must_use]
    pub fn model_ids(&self) -> Vec<ModelId> {
        self.models.iter().map(|model| model.id.clone()).collect()
    }

    #[must_use]
    pub fn layout_ids(&self) -> Vec<ModelId> {
        deployment_layout_ids(&self.models)
    }

    #[must_use]
    pub fn layout_ids_for(&self, id: &ModelId) -> Vec<ModelId> {
        deployment_layout_ids_for(id, &self.models)
    }

    #[must_use]
    pub fn model_layouts(&self) -> Vec<(ModelId, ModelId)> {
        deployment_model_layouts(&self.models)
    }

    #[must_use]
    pub fn default_layout_runtime(&self) -> Option<&OcrModel> {
        deployment_default_layout(self.default_model.as_ref(), &self.models)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmServiceSection {
    pub enabled: bool,
    pub default_model: Option<ModelId>,
    pub models: Vec<LlmModelDeployment>,
    pub idle_timeout: Duration,
    pub max_loaded: usize,
}

impl LlmServiceSection {
    #[must_use]
    pub fn active(&self) -> bool {
        self.enabled && !self.models.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LlmContextSize {
    Model,
    Tokens(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LlmGpuLayers {
    All,
    Count(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LlmRuntimeConfig {
    pub context_size: LlmContextSize,
    pub gpu_layers: LlmGpuLayers,
    pub parallel_sequences: u32,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub threads: i32,
    pub request_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub queue_timeout: Duration,
    pub generation_timeout: Duration,
}

impl Default for LlmRuntimeConfig {
    fn default() -> Self {
        Self {
            context_size: LlmContextSize::Model,
            gpu_layers: LlmGpuLayers::All,
            parallel_sequences: 1,
            batch_size: 512,
            micro_batch_size: 512,
            threads: 0,
            request_queue_capacity: 8,
            event_queue_capacity: 16,
            queue_timeout: Duration::from_secs(30),
            generation_timeout: Duration::from_mins(5),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatTemplateEngine {
    LlamaCpp,
    Jinja,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChatTemplateConfig {
    pub engine: ChatTemplateEngine,
    pub template: Option<String>,
    pub enable_thinking: bool,
}

impl Default for ChatTemplateConfig {
    fn default() -> Self {
        Self {
            engine: ChatTemplateEngine::LlamaCpp,
            template: None,
            enable_thinking: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmGenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
}

impl Default for LlmGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
        }
    }
}

impl PartialEq for LlmGenerationConfig {
    fn eq(&self, other: &Self) -> bool {
        self.max_tokens == other.max_tokens
            && self.temperature.to_bits() == other.temperature.to_bits()
            && self.top_p.to_bits() == other.top_p.to_bits()
            && self.top_k == other.top_k
            && self.min_p.to_bits() == other.min_p.to_bits()
            && self.presence_penalty.to_bits() == other.presence_penalty.to_bits()
            && self.frequency_penalty.to_bits() == other.frequency_penalty.to_bits()
            && self.repeat_penalty.to_bits() == other.repeat_penalty.to_bits()
    }
}
impl Eq for LlmGenerationConfig {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmModelDeployment {
    pub id: ModelId,
    pub name: Option<String>,
    pub model: ModelUrl,
    pub mmproj_model: Option<ModelUrl>,
    pub runtime: LlmRuntimeConfig,
    pub chat_template: ChatTemplateConfig,
    pub generation: LlmGenerationConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDeployment<M> {
    pub id: ModelId,
    pub name: Option<String>,
    pub model: ModelUrl,
    pub runtime: M,
}

impl<M> ModelDeployment<M> {
    #[must_use]
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or_else(|| self.id.name())
    }
}

impl ModelDeployment<AsrModel> {
    /// Builds a deployment from a built-in ASR runtime.
    ///
    /// # Panics
    ///
    /// Panics if the runtime's built-in model identifier violates the model ID or URL grammar.
    #[must_use]
    pub fn from_asr_runtime(runtime: AsrModel) -> Self {
        let id = ModelId::parse(runtime.as_str()).expect("ASR runtime contains a valid model id");
        let model = ModelUrl::parse(&format!("//{}", runtime.as_str()))
            .expect("ASR runtime id forms a valid neutral model URL");
        Self {
            id,
            name: None,
            model,
            runtime,
        }
    }
}

impl ModelDeployment<TtsModel> {
    /// Builds a deployment from a built-in TTS runtime.
    ///
    /// # Panics
    ///
    /// Panics if the runtime's built-in model identifier violates the model ID or URL grammar.
    #[must_use]
    pub fn from_tts_runtime(runtime: TtsModel) -> Self {
        let id = ModelId::parse(runtime.as_str()).expect("TTS runtime contains a valid model id");
        let model = ModelUrl::parse(&format!("//{}", runtime.as_str()))
            .expect("TTS runtime id forms a valid neutral model URL");
        Self {
            id,
            name: None,
            model,
            runtime,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableStructureType {
    Wired,
    Wireless,
}

impl TableStructureType {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Wired => "wired",
            Self::Wireless => "wireless",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableStructureConfig {
    pub model: ModelUrl,
    pub dictionary: ModelUrl,
    pub table_type: TableStructureType,
    pub score_threshold: f32,
    pub max_structure_length: usize,
}

impl PartialEq for TableStructureConfig {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.dictionary == other.dictionary
            && self.table_type == other.table_type
            && self.score_threshold.to_bits() == other.score_threshold.to_bits()
            && self.max_structure_length == other.max_structure_length
    }
}

impl Eq for TableStructureConfig {}

impl std::hash::Hash for TableStructureConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.dictionary.hash(state);
        self.table_type.hash(state);
        self.score_threshold.to_bits().hash(state);
        self.max_structure_length.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrModelDeployment {
    pub id: ModelId,
    pub name: Option<String>,
    pub model: ModelUrl,
    pub layout_model: Option<ModelUrl>,
    pub table_structure: Option<TableStructureConfig>,
    pub runtime: OcrModel,
    pub layout_runtime: Option<OcrModel>,
}

impl OcrModelDeployment {
    #[must_use]
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or_else(|| self.id.name())
    }

    /// Builds a deployment from a built-in OCR runtime.
    ///
    /// # Panics
    ///
    /// Panics if the runtime's built-in model identifier cannot form a neutral model URL.
    #[must_use]
    pub fn from_runtime(runtime: OcrModel) -> Self {
        let id = runtime.id().clone();
        let model = ModelUrl::parse(&format!("//{id}"))
            .expect("OCR runtime id forms a valid neutral model URL");
        Self {
            id,
            name: None,
            model,
            layout_model: None,
            table_structure: None,
            runtime,
            layout_runtime: None,
        }
    }

    /// Adds Orchion's built-in document-layout model.
    ///
    /// # Panics
    ///
    /// Panics if the compile-time built-in layout model URL is invalid.
    #[must_use]
    pub fn with_supported_layout(mut self) -> Self {
        self.layout_model = Some(
            ModelUrl::parse("//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx")
                .expect("supported layout URL is valid"),
        );
        self.layout_runtime = Some(KnownOcrModel::PpDocLayoutV3.into_model());
        self
    }
}

fn deployment_layout_ids(models: &[OcrModelDeployment]) -> Vec<ModelId> {
    let mut ids = Vec::new();
    for id in models
        .iter()
        .filter_map(|model| model.layout_runtime.as_ref().map(orchion::OcrModel::id))
    {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    ids
}

fn deployment_layout_ids_for(id: &ModelId, models: &[OcrModelDeployment]) -> Vec<ModelId> {
    models
        .iter()
        .filter(|model| model.id == *id)
        .filter_map(|model| {
            model
                .layout_runtime
                .as_ref()
                .map(|layout| layout.id().clone())
        })
        .collect()
}

fn deployment_model_layouts(models: &[OcrModelDeployment]) -> Vec<(ModelId, ModelId)> {
    models
        .iter()
        .filter_map(|model| {
            model
                .layout_runtime
                .as_ref()
                .map(|layout| (model.id.clone(), layout.id().clone()))
        })
        .collect()
}

fn deployment_default_layout<'a>(
    default: Option<&ModelId>,
    models: &'a [OcrModelDeployment],
) -> Option<&'a OcrModel> {
    let default = default?;
    models
        .iter()
        .find(|model| model.id == *default)
        .and_then(|model| model.layout_runtime.as_ref())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSection {
    pub api_key: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    #[error("invalid CORS origin `{value}`: {message}")]
    InvalidCorsOrigin { value: String, message: String },
    #[error("invalid CORS origins: wildcard `*` cannot be combined with specific origins")]
    CorsWildcardWithSpecificOrigins,
    #[error("environment variable `{key}` contains non-Unicode data")]
    InvalidEnvironmentVariable { key: &'static str },
    #[error("invalid upload size `{value}`: {message}")]
    InvalidUploadSize { value: String, message: String },
    #[error("unknown model source `{0}`; expected auto, huggingface, or modelscope")]
    UnknownModelSource(String),
    #[error("invalid ASR model id `{0}`; expected vendor/name")]
    InvalidAsrModelId(String),
    #[error("invalid TTS model id `{0}`; expected vendor/name")]
    InvalidTtsModelId(String),
    #[error("invalid duration `{value}`: {message}")]
    InvalidDuration { value: String, message: String },
    #[error("invalid {section}.max_loaded `{value}`: value must be greater than zero")]
    InvalidMaxLoaded { section: &'static str, value: usize },
    #[error("invalid {section}.{field} `{value}`: value must be greater than zero")]
    InvalidGenerationLimit {
        section: &'static str,
        field: &'static str,
        value: usize,
    },
    #[error("invalid {section}.{field} `{value}`: value must be greater than zero")]
    InvalidResourceLimit {
        section: &'static str,
        field: &'static str,
        value: String,
    },
    #[error(
        "invalid activity.history_capacity `{value}`: value must not exceed {MAX_ACTIVITY_HISTORY_CAPACITY}"
    )]
    InvalidActivityHistoryCapacity { value: usize },
    #[error(
        "invalid {section}.stream_chunk_size `{value}`: value must be finite and greater than zero"
    )]
    InvalidChunkSize { section: &'static str, value: f32 },
    #[error("invalid {section}.{field} `{value}`: {message}")]
    InvalidStreamSegmentDuration {
        section: &'static str,
        field: &'static str,
        value: String,
        message: String,
    },
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
    #[error("invalid services.tts.format `{value}`; expected wav, mp3, aac, opus, flac, or pcm")]
    InvalidTtsFormat { value: String },
    #[error("{section} is enabled but models is empty")]
    ServiceEnabledWithoutModels { section: &'static str },
    #[error(
        "default {category} model `{default}` must match exactly one entry in {section}.models"
    )]
    DefaultModelUnavailable {
        category: &'static str,
        section: &'static str,
        default: String,
    },
    #[error("invalid {section}.format `{format}`: format is not supported by this service")]
    UnsupportedOcrDefaultFormat {
        section: &'static str,
        format: &'static str,
    },
    #[error(
        "invalid {section}.models name for `{model}`: name must be omitted or trimmed nonblank text"
    )]
    InvalidModelName {
        section: &'static str,
        model: String,
    },
    #[error("model id `{id}` is configured more than once across services")]
    DuplicateModelId { id: String },
    #[error("configured {section} model id `{id}` is not a supported {expected} runtime model")]
    UnsupportedRuntimeModel {
        section: &'static str,
        id: String,
        expected: &'static str,
    },
    #[error("invalid {section}.models layout_model `{value}`: expected a supported layout recipe")]
    UnsupportedLayoutModel {
        section: &'static str,
        value: String,
    },
    #[error(
        "invalid {section}.models.table_structure.table_type `{value}`; expected wired or wireless"
    )]
    InvalidTableStructureType {
        section: &'static str,
        value: String,
    },
    #[error(
        "invalid {section}.models.table_structure.score_threshold `{value}`: expected a finite value in [0, 1]"
    )]
    InvalidTableStructureThreshold {
        section: &'static str,
        value: String,
    },
    #[error(
        "invalid {section}.models.table_structure.max_structure_length `0`: value must be greater than zero"
    )]
    InvalidTableStructureLength { section: &'static str },
    #[error("invalid {section}.models table_structure: layout_model must also be configured")]
    TableStructureRequiresLayout { section: &'static str },
    #[error(
        "invalid {section}.models table_structure: only traditional OCR deployments support table structure recognition"
    )]
    UnsupportedTableStructurePipeline { section: &'static str },
    #[error("invalid {section}.models model locator `{value}`: {message}")]
    UnsupportedModelLocator {
        section: &'static str,
        value: String,
        message: &'static str,
    },
    #[error("invalid services.llm.models.{field} `{value}`: {message}")]
    InvalidLlmSetting {
        field: &'static str,
        value: String,
        message: &'static str,
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
                cors_allowed_origins: vec!["*".to_string()],
                max_upload_size: 30 * 1024 * 1024,
                max_pdf_pages: 100,
                max_pdf_pixels: 200_000_000,
                max_pdf_output_size: 100 * 1024 * 1024,
                max_concurrent_inference: DEFAULT_MAX_CONCURRENT_INFERENCE,
                max_websocket_connections: DEFAULT_MAX_WEBSOCKET_CONNECTIONS,
                max_pending_websocket_connections: DEFAULT_MAX_PENDING_WEBSOCKET_CONNECTIONS,
                max_websocket_message_size: DEFAULT_MAX_WEBSOCKET_MESSAGE_SIZE,
            },
            activity: ActivitySection {
                enabled: true,
                history_capacity: 500,
            },
            models: ModelsSection {
                dir: exe_dir.join("models"),
                source: ModelSource::Auto,
                max_loaded: 2,
                verify_file_integrity: false,
            },
            services: ServicesSection {
                asr: AsrServiceSection {
                    enabled: false,
                    default_model: default_asr_model(),
                    models: vec![default_asr_deployment()],
                    idle_timeout: Duration::from_mins(10),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    stream_chunk_size: 2.0,
                    stream_target_segment: DEFAULT_ASR_STREAM_TARGET_SEGMENT,
                    stream_max_segment: DEFAULT_ASR_STREAM_MAX_SEGMENT,
                    stream_idle_timeout: DEFAULT_ASR_STREAM_IDLE_TIMEOUT,
                    stream_max_duration: DEFAULT_ASR_STREAM_MAX_DURATION,
                    max_audio_duration: DEFAULT_ASR_MAX_AUDIO_DURATION,
                },
                tts: TtsServiceSection {
                    enabled: false,
                    default_model: default_tts_model(),
                    models: vec![default_tts_deployment()],
                    idle_timeout: Duration::from_mins(10),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: "wav".to_string(),
                    max_length: DEFAULT_TTS_MAX_LENGTH,
                    max_reference_audio_duration: DEFAULT_TTS_MAX_REFERENCE_AUDIO_DURATION,
                },
                ocr: OcrServiceSection {
                    enabled: false,
                    default_model: None,
                    models: Vec::new(),
                    idle_timeout: Duration::from_mins(10),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: OcrResponseFormat::Json,
                    max_pixels: DEFAULT_OCR_MAX_PIXELS,
                },
                ocr_vl: OcrVlServiceSection {
                    enabled: false,
                    default_model: None,
                    models: Vec::new(),
                    idle_timeout: Duration::from_mins(10),
                    max_loaded: 1,
                    device: DevicePreference::Auto,
                    format: OcrResponseFormat::Markdown,
                    max_tokens: DEFAULT_OCR_VL_MAX_TOKENS,
                    max_pixels: DEFAULT_OCR_MAX_PIXELS,
                },
                llm: LlmServiceSection {
                    enabled: false,
                    default_model: None,
                    models: Vec::new(),
                    idle_timeout: Duration::from_mins(10),
                    max_loaded: 1,
                },
            },
            auth: AuthSection { api_key: None },
        }
    }

    /// Validates model deployment invariants for parsed or programmatically constructed config.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when a service deployment is inconsistent or unsupported.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_model_deployments(self)
    }

    /// # Errors
    ///
    /// Returns [`ConfigError`] when the configuration cannot be read or validated.
    pub fn load(config_path: Option<PathBuf>) -> Result<Self, ConfigError> {
        let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("orchion-server"));
        let default = Self::default_for_exe(&exe_path);
        let explicit_path = config_path.is_some();
        let path = config_path.unwrap_or_else(|| default.config_path.clone());
        let mut config = if !explicit_path && !path.exists() {
            Self {
                config_path: path,
                ..default
            }
        } else {
            let document = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                path: path.clone(),
                source,
            })?;
            let mut config = Self::from_toml_str(&document, &exe_path)?;
            config.config_path = path;
            config
        };
        apply_cors_environment_override(&mut config)?;
        Ok(config)
    }

    /// # Errors
    ///
    /// Returns [`ConfigError`] when the TOML document is malformed or invalid.
    #[allow(
        clippy::too_many_lines,
        reason = "all config overrides are validated in one parsing transaction"
    )]
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
            if let Some(cors_allowed_origins) = server.cors_allowed_origins {
                config.server.cors_allowed_origins =
                    parse_cors_allowed_origins(cors_allowed_origins)?;
            }
            if let Some(max_upload_size) = server.max_upload_size {
                config.server.max_upload_size = parse_upload_size(&max_upload_size)?;
            }
            if let Some(max_pdf_pages) = server.max_pdf_pages {
                if max_pdf_pages == 0 {
                    return Err(ConfigError::InvalidResourceLimit {
                        section: "server",
                        field: "max_pdf_pages",
                        value: max_pdf_pages.to_string(),
                    });
                }
                config.server.max_pdf_pages = max_pdf_pages;
            }
            if let Some(max_pdf_pixels) = server.max_pdf_pixels {
                if max_pdf_pixels == 0 {
                    return Err(ConfigError::InvalidResourceLimit {
                        section: "server",
                        field: "max_pdf_pixels",
                        value: max_pdf_pixels.to_string(),
                    });
                }
                config.server.max_pdf_pixels = max_pdf_pixels;
            }
            if let Some(max_pdf_output_size) = server.max_pdf_output_size {
                config.server.max_pdf_output_size = parse_upload_size(&max_pdf_output_size)?;
            }
            if let Some(max_concurrent_inference) = server.max_concurrent_inference {
                config.server.max_concurrent_inference = validate_nonzero_resource_limit(
                    "server",
                    "max_concurrent_inference",
                    max_concurrent_inference,
                )?;
            }
            if let Some(max_websocket_connections) = server.max_websocket_connections {
                config.server.max_websocket_connections = validate_nonzero_resource_limit(
                    "server",
                    "max_websocket_connections",
                    max_websocket_connections,
                )?;
            }
            if let Some(max_pending_websocket_connections) =
                server.max_pending_websocket_connections
            {
                config.server.max_pending_websocket_connections = validate_nonzero_resource_limit(
                    "server",
                    "max_pending_websocket_connections",
                    max_pending_websocket_connections,
                )?;
            }
            if let Some(max_websocket_message_size) = server.max_websocket_message_size {
                config.server.max_websocket_message_size = validate_nonzero_resource_limit(
                    "server",
                    "max_websocket_message_size",
                    max_websocket_message_size,
                )?;
            }
        }

        if let Some(activity) = raw.activity {
            if let Some(enabled) = activity.enabled {
                config.activity.enabled = enabled;
            }
            if let Some(history_capacity) = activity.history_capacity {
                if history_capacity > MAX_ACTIVITY_HISTORY_CAPACITY {
                    return Err(ConfigError::InvalidActivityHistoryCapacity {
                        value: history_capacity,
                    });
                }
                config.activity.history_capacity = history_capacity;
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
            if let Some(verify_file_integrity) = models.verify_file_integrity {
                config.models.verify_file_integrity = verify_file_integrity;
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
            if let Some(llm) = services.llm {
                config.services.llm = parse_llm_service(llm, config.services.llm)?;
            }
        }

        if let Some(auth) = raw.auth
            && let Some(api_key) = auth.api_key
        {
            let api_key = api_key.trim();
            config.auth.api_key = if api_key.is_empty() {
                None
            } else {
                Some(api_key.to_string())
            };
        }

        config.validate()?;

        Ok(config)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps validation for one configuration section together"
)]
fn parse_asr_service(
    raw: RawModelService,
    mut service: AsrServiceSection,
) -> Result<AsrServiceSection, ConfigError> {
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = parse_asr_model(&default_model)?;
    }
    if let Some(models) = raw.models {
        service.models = models
            .into_iter()
            .map(|model| parse_asr_deployment("services.asr", model))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.asr", &device)?;
    }
    if let Some(stream_chunk_size) = raw.stream_chunk_size {
        if !stream_chunk_size.is_finite() || stream_chunk_size <= 0.0 {
            return Err(ConfigError::InvalidChunkSize {
                section: "services.asr",
                value: stream_chunk_size,
            });
        }
        service.stream_chunk_size = stream_chunk_size;
    }
    if let Some(stream_target_segment) = raw.stream_target_segment {
        service.stream_target_segment = parse_stream_segment_duration(
            "services.asr",
            "stream_target_segment",
            &stream_target_segment,
        )?;
    }
    if let Some(stream_max_segment) = raw.stream_max_segment {
        service.stream_max_segment = parse_stream_segment_duration(
            "services.asr",
            "stream_max_segment",
            &stream_max_segment,
        )?;
    }
    if let Some(stream_idle_timeout) = raw.stream_idle_timeout {
        service.stream_idle_timeout = parse_stream_segment_duration(
            "services.asr",
            "stream_idle_timeout",
            &stream_idle_timeout,
        )?;
    }
    if let Some(stream_max_duration) = raw.stream_max_duration {
        service.stream_max_duration = parse_stream_segment_duration(
            "services.asr",
            "stream_max_duration",
            &stream_max_duration,
        )?;
    }
    if let Some(max_audio_duration) = raw.max_audio_duration {
        service.max_audio_duration = parse_stream_segment_duration(
            "services.asr",
            "max_audio_duration",
            &max_audio_duration,
        )?;
    }
    if service.stream_target_segment > service.stream_max_segment {
        return Err(ConfigError::InvalidStreamSegmentDuration {
            section: "services.asr",
            field: "stream_target_segment",
            value: format_duration_for_error(service.stream_target_segment),
            message: "value must be no greater than stream_max_segment".to_string(),
        });
    }
    if service.stream_idle_timeout > service.stream_max_duration {
        return Err(ConfigError::InvalidStreamSegmentDuration {
            section: "services.asr",
            field: "stream_idle_timeout",
            value: format_duration_for_error(service.stream_idle_timeout),
            message: "value must be no greater than stream_max_duration".to_string(),
        });
    }
    apply_service_limits(
        "services.asr",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.asr",
        });
    }
    Ok(service)
}

fn parse_tts_service(
    raw: RawTtsService,
    mut service: TtsServiceSection,
) -> Result<TtsServiceSection, ConfigError> {
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = parse_tts_model(&default_model)?;
    }
    if let Some(models) = raw.models {
        service.models = models
            .into_iter()
            .map(|model| parse_tts_deployment("services.tts", model))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.tts", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = parse_tts_format(&format)?;
    }
    if let Some(max_length) = raw.max_length {
        if max_length == 0 {
            return Err(ConfigError::InvalidGenerationLimit {
                section: "services.tts",
                field: "max_length",
                value: max_length,
            });
        }
        service.max_length = max_length;
    }
    if let Some(max_reference_audio_duration) = raw.max_reference_audio_duration {
        service.max_reference_audio_duration = parse_stream_segment_duration(
            "services.tts",
            "max_reference_audio_duration",
            &max_reference_audio_duration,
        )?;
    }
    apply_service_limits(
        "services.tts",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.tts",
        });
    }
    Ok(service)
}

fn parse_tts_format(value: &str) -> Result<String, ConfigError> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "wav" | "mp3" | "aac" | "opus" | "flac" | "pcm" => Ok(normalized),
        _ => Err(ConfigError::InvalidTtsFormat {
            value: value.to_string(),
        }),
    }
}

fn parse_ocr_service(
    raw: RawOcrService,
    mut service: OcrServiceSection,
) -> Result<OcrServiceSection, ConfigError> {
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = Some(parse_model_id(
            "services.ocr.default_model",
            &default_model,
        )?);
    }
    if let Some(models) = raw.models {
        service.models = models
            .into_iter()
            .map(|model| parse_ocr_deployment("services.ocr", model, OcrModelKind::TraditionalOcr))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.ocr", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = parse_ocr_format("services.ocr.format", &format)?;
    }
    if let Some(max_pixels) = raw.max_pixels {
        service.max_pixels = validate_ocr_max_pixels("services.ocr", max_pixels)?;
    }
    apply_service_limits(
        "services.ocr",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.ocr",
        });
    }
    if service.enabled && service.format == OcrResponseFormat::Html {
        return Err(ConfigError::UnsupportedOcrDefaultFormat {
            section: "services.ocr",
            format: "html",
        });
    }
    Ok(service)
}

fn parse_ocr_vl_service(
    raw: RawOcrVlService,
    mut service: OcrVlServiceSection,
) -> Result<OcrVlServiceSection, ConfigError> {
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = Some(parse_model_id(
            "services.ocr-vl.default_model",
            &default_model,
        )?);
    }
    if let Some(models) = raw.models {
        service.models = models
            .into_iter()
            .map(|model| parse_ocr_deployment("services.ocr-vl", model, OcrModelKind::OcrVl))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if let Some(device) = raw.device {
        service.device = parse_device_preference("services.ocr-vl", &device)?;
    }
    if let Some(format) = raw.format {
        service.format = parse_ocr_format("services.ocr-vl.format", &format)?;
    }
    if let Some(max_tokens) = raw.max_tokens {
        if max_tokens == 0 {
            return Err(ConfigError::InvalidGenerationLimit {
                section: "services.ocr-vl",
                field: "max_tokens",
                value: max_tokens,
            });
        }
        service.max_tokens = max_tokens;
    }
    if let Some(max_pixels) = raw.max_pixels {
        service.max_pixels = validate_ocr_max_pixels("services.ocr-vl", max_pixels)?;
    }
    apply_service_limits(
        "services.ocr-vl",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.ocr-vl",
        });
    }
    Ok(service)
}

fn parse_llm_service(
    raw: RawLlmService,
    mut service: LlmServiceSection,
) -> Result<LlmServiceSection, ConfigError> {
    if let Some(enabled) = raw.enabled {
        service.enabled = enabled;
    }
    if let Some(default_model) = raw.default_model {
        service.default_model = Some(parse_model_id(
            "services.llm.default_model",
            &default_model,
        )?);
    }
    if let Some(models) = raw.models {
        service.models = models
            .into_iter()
            .map(parse_llm_deployment)
            .collect::<Result<_, _>>()?;
    }
    apply_service_limits(
        "services.llm",
        raw.idle_timeout,
        raw.max_loaded,
        &mut service.idle_timeout,
        &mut service.max_loaded,
    )?;
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.llm",
        });
    }
    Ok(service)
}

fn parse_llm_deployment(raw: RawLlmModelDeployment) -> Result<LlmModelDeployment, ConfigError> {
    let id = parse_model_id("services.llm", &raw.id)?;
    Ok(LlmModelDeployment {
        name: parse_deployment_name("services.llm", &id, raw.name)?,
        id,
        model: raw.model,
        mmproj_model: raw.mmproj_model,
        runtime: parse_llm_runtime(raw.runtime.unwrap_or_default())?,
        chat_template: parse_chat_template(raw.chat_template.unwrap_or_default())?,
        generation: parse_llm_generation(&raw.generation.unwrap_or_default())?,
    })
}

fn parse_llm_runtime(raw: RawLlmRuntime) -> Result<LlmRuntimeConfig, ConfigError> {
    let mut runtime = LlmRuntimeConfig::default();
    if let Some(value) = raw.context_size {
        runtime.context_size = match value {
            RawContextSize::Name(value) if value == "model" => LlmContextSize::Model,
            RawContextSize::Tokens(value) if value > 0 => LlmContextSize::Tokens(value),
            value => {
                return Err(invalid_llm(
                    "runtime.context_size",
                    format!("{value:?}"),
                    "expected `model` or a positive integer",
                ));
            }
        };
    }
    if let Some(value) = raw.gpu_layers {
        runtime.gpu_layers = match value {
            RawGpuLayers::Name(value) if value == "all" => LlmGpuLayers::All,
            RawGpuLayers::Count(value) => LlmGpuLayers::Count(value),
            RawGpuLayers::Name(value) => {
                return Err(invalid_llm(
                    "runtime.gpu_layers",
                    format!("{value:?}"),
                    "expected `all` or a nonnegative integer",
                ));
            }
        };
    }
    if let Some(value) = raw.parallel_sequences {
        runtime.parallel_sequences = value;
    }
    if runtime.parallel_sequences != 1 {
        return Err(invalid_llm(
            "runtime.parallel_sequences",
            runtime.parallel_sequences.to_string(),
            "the text tracer supports exactly one parallel sequence",
        ));
    }
    if let Some(value) = raw.batch_size {
        runtime.batch_size = nonzero_u32("runtime.batch_size", value)?;
    }
    if let Some(value) = raw.micro_batch_size {
        runtime.micro_batch_size = nonzero_u32("runtime.micro_batch_size", value)?;
    }
    if let Some(value) = raw.threads {
        if value < 0 {
            return Err(invalid_llm(
                "runtime.threads",
                value.to_string(),
                "value must be nonnegative",
            ));
        }
        runtime.threads = value;
    }
    if let Some(value) = raw.request_queue_capacity {
        runtime.request_queue_capacity = nonzero_usize("runtime.request_queue_capacity", value)?;
    }
    if let Some(value) = raw.event_queue_capacity {
        runtime.event_queue_capacity = nonzero_usize("runtime.event_queue_capacity", value)?;
    }
    if let Some(value) = raw.queue_timeout {
        runtime.queue_timeout = parse_duration(&value)?;
    }
    if let Some(value) = raw.generation_timeout {
        runtime.generation_timeout = parse_duration(&value)?;
    }
    Ok(runtime)
}

fn nonzero_u32(field: &'static str, value: u32) -> Result<u32, ConfigError> {
    if value == 0 {
        Err(invalid_llm(
            field,
            value.to_string(),
            "value must be greater than zero",
        ))
    } else {
        Ok(value)
    }
}

fn nonzero_usize(field: &'static str, value: usize) -> Result<usize, ConfigError> {
    if value == 0 {
        Err(invalid_llm(
            field,
            value.to_string(),
            "value must be greater than zero",
        ))
    } else {
        Ok(value)
    }
}

fn parse_chat_template(raw: RawChatTemplate) -> Result<ChatTemplateConfig, ConfigError> {
    let engine = match raw.engine.as_deref().unwrap_or("llama_cpp") {
        "llama_cpp" => ChatTemplateEngine::LlamaCpp,
        "jinja" => ChatTemplateEngine::Jinja,
        value => {
            return Err(invalid_llm(
                "chat_template.engine",
                value.to_string(),
                "expected `llama_cpp` or the supported text-subset `jinja` engine",
            ));
        }
    };
    let template = raw.template.map(|value| value.trim().to_string());
    if template.as_ref().is_some_and(String::is_empty) {
        return Err(invalid_llm(
            "chat_template.template",
            String::new(),
            "override must be nonblank",
        ));
    }
    Ok(ChatTemplateConfig {
        engine,
        template,
        enable_thinking: raw.enable_thinking.unwrap_or(false),
    })
}

fn parse_llm_generation(raw: &RawLlmGeneration) -> Result<LlmGenerationConfig, ConfigError> {
    let mut value = LlmGenerationConfig::default();
    if let Some(v) = raw.max_tokens {
        value.max_tokens = v;
    }
    if let Some(v) = raw.temperature {
        value.temperature = v;
    }
    if let Some(v) = raw.top_p {
        value.top_p = v;
    }
    if let Some(v) = raw.top_k {
        value.top_k = v;
    }
    if let Some(v) = raw.min_p {
        value.min_p = v;
    }
    if let Some(v) = raw.presence_penalty {
        value.presence_penalty = v;
    }
    if let Some(v) = raw.frequency_penalty {
        value.frequency_penalty = v;
    }
    if let Some(v) = raw.repeat_penalty {
        value.repeat_penalty = v;
    }
    validate_llm_generation(&value)?;
    Ok(value)
}

fn validate_llm_generation(value: &LlmGenerationConfig) -> Result<(), ConfigError> {
    if value.max_tokens < 16 || value.max_tokens > DEFAULT_LLM_MAX_TOKENS {
        return Err(invalid_llm(
            "generation.max_tokens",
            value.max_tokens.to_string(),
            "expected 16..=4096",
        ));
    }
    if !value.temperature.is_finite() || !(0.0..=2.0).contains(&value.temperature) {
        return Err(invalid_llm(
            "generation.temperature",
            value.temperature.to_string(),
            "expected a finite value in [0, 2]",
        ));
    }
    if !value.top_p.is_finite() || value.top_p <= 0.0 || value.top_p > 1.0 {
        return Err(invalid_llm(
            "generation.top_p",
            value.top_p.to_string(),
            "expected a finite value in (0, 1]",
        ));
    }
    if value.top_k < 0 {
        return Err(invalid_llm(
            "generation.top_k",
            value.top_k.to_string(),
            "expected a nonnegative integer",
        ));
    }
    if !value.min_p.is_finite() || !(0.0..=1.0).contains(&value.min_p) {
        return Err(invalid_llm(
            "generation.min_p",
            value.min_p.to_string(),
            "expected a finite value in [0, 1]",
        ));
    }
    for (field, penalty) in [
        ("generation.presence_penalty", value.presence_penalty),
        ("generation.frequency_penalty", value.frequency_penalty),
    ] {
        if !penalty.is_finite() || !(-2.0..=2.0).contains(&penalty) {
            return Err(invalid_llm(
                field,
                penalty.to_string(),
                "expected a finite value in [-2, 2]",
            ));
        }
    }
    if !value.repeat_penalty.is_finite() || value.repeat_penalty <= 0.0 {
        return Err(invalid_llm(
            "generation.repeat_penalty",
            value.repeat_penalty.to_string(),
            "expected a finite value greater than zero",
        ));
    }
    Ok(())
}

fn invalid_llm(field: &'static str, value: String, message: &'static str) -> ConfigError {
    ConfigError::InvalidLlmSetting {
        field,
        value,
        message,
    }
}

fn validate_ocr_max_pixels(section: &'static str, max_pixels: u64) -> Result<u64, ConfigError> {
    if max_pixels == 0 {
        return Err(ConfigError::InvalidResourceLimit {
            section,
            field: "max_pixels",
            value: max_pixels.to_string(),
        });
    }
    Ok(max_pixels)
}

fn validate_nonzero_resource_limit(
    section: &'static str,
    field: &'static str,
    value: usize,
) -> Result<usize, ConfigError> {
    if value == 0 {
        return Err(ConfigError::InvalidResourceLimit {
            section,
            field,
            value: value.to_string(),
        });
    }
    Ok(value)
}

fn apply_cors_environment_override(config: &mut ServerConfig) -> Result<(), ConfigError> {
    let value = match std::env::var(CORS_ALLOWED_ORIGINS_ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(()),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(ConfigError::InvalidEnvironmentVariable {
                key: CORS_ALLOWED_ORIGINS_ENV,
            });
        }
    };
    apply_cors_allowed_origins_override(config, &value)?;
    Ok(())
}

fn apply_cors_allowed_origins_override(
    config: &mut ServerConfig,
    value: &str,
) -> Result<(), ConfigError> {
    config.server.cors_allowed_origins =
        parse_cors_allowed_origins(value.split(',').map(str::to_string))?;
    Ok(())
}

fn parse_cors_allowed_origins(
    values: impl IntoIterator<Item = String>,
) -> Result<Vec<String>, ConfigError> {
    let origins = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .map(|value| normalize_cors_origin(&value))
        .collect::<Result<Vec<_>, _>>()?;
    if origins.is_empty() {
        return Err(ConfigError::InvalidCorsOrigin {
            value: String::new(),
            message: "at least one origin is required".to_string(),
        });
    }
    if origins.len() > 1 && origins.iter().any(|origin| origin == "*") {
        return Err(ConfigError::CorsWildcardWithSpecificOrigins);
    }
    Ok(origins)
}

fn normalize_cors_origin(value: &str) -> Result<String, ConfigError> {
    if value.is_empty() {
        return Err(invalid_cors_origin(value, "value must not be empty"));
    }
    if matches!(value, "*" | "null") {
        return Ok(value.to_string());
    }
    let url =
        url::Url::parse(value).map_err(|error| invalid_cors_origin(value, &error.to_string()))?;
    if url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_cors_origin(
            value,
            "expected scheme://host[:port] without credentials, path, query, or fragment",
        ));
    }
    let origin = url.origin().ascii_serialization();
    if origin == "null" {
        return Err(invalid_cors_origin(
            value,
            "opaque origins must be configured as `null`",
        ));
    }
    Ok(origin)
}

fn invalid_cors_origin(value: &str, message: &str) -> ConfigError {
    ConfigError::InvalidCorsOrigin {
        value: value.to_string(),
        message: message.to_string(),
    }
}

fn parse_asr_deployment(
    section: &'static str,
    raw: RawModelDeployment,
) -> Result<ModelDeployment<AsrModel>, ConfigError> {
    let id = parse_model_id(section, &raw.id)?;
    Ok(ModelDeployment {
        runtime: parse_asr_model(id.as_str())?,
        name: parse_deployment_name(section, &id, raw.name)?,
        id,
        model: raw.model,
    })
}

fn parse_tts_deployment(
    section: &'static str,
    raw: RawModelDeployment,
) -> Result<ModelDeployment<TtsModel>, ConfigError> {
    let id = parse_model_id(section, &raw.id)?;
    Ok(ModelDeployment {
        runtime: parse_tts_model(id.as_str())?,
        name: parse_deployment_name(section, &id, raw.name)?,
        id,
        model: raw.model,
    })
}

fn parse_ocr_deployment(
    section: &'static str,
    raw: RawOcrModelDeployment,
    kind: OcrModelKind,
) -> Result<OcrModelDeployment, ConfigError> {
    let id = parse_model_id(section, &raw.id)?;
    let layout_runtime = raw
        .layout_model
        .as_ref()
        .map(|url| resolve_layout_recipe(section, url))
        .transpose()?;
    let table_structure = raw
        .table_structure
        .map(|table| parse_table_structure(section, table))
        .transpose()?;
    Ok(OcrModelDeployment {
        runtime: OcrModel::new(id.clone(), kind),
        layout_runtime,
        name: parse_deployment_name(section, &id, raw.name)?,
        id,
        model: raw.model,
        layout_model: raw.layout_model,
        table_structure,
    })
}

fn parse_table_structure(
    section: &'static str,
    raw: RawTableStructure,
) -> Result<TableStructureConfig, ConfigError> {
    let table_type = match raw.table_type.as_str() {
        "wired" => TableStructureType::Wired,
        "wireless" => TableStructureType::Wireless,
        _ => {
            return Err(ConfigError::InvalidTableStructureType {
                section,
                value: raw.table_type,
            });
        }
    };
    let table = TableStructureConfig {
        model: raw.model,
        dictionary: raw.dictionary,
        table_type,
        score_threshold: raw.score_threshold.unwrap_or(0.5),
        max_structure_length: raw.max_structure_length.unwrap_or(500),
    };
    validate_table_structure_config(section, &table)?;
    Ok(table)
}

fn validate_table_structure_config(
    section: &'static str,
    table: &TableStructureConfig,
) -> Result<(), ConfigError> {
    if !table.score_threshold.is_finite() || !(0.0..=1.0).contains(&table.score_threshold) {
        return Err(ConfigError::InvalidTableStructureThreshold {
            section,
            value: table.score_threshold.to_string(),
        });
    }
    if table.max_structure_length == 0 {
        return Err(ConfigError::InvalidTableStructureLength { section });
    }
    Ok(())
}

fn resolve_layout_recipe(section: &'static str, url: &ModelUrl) -> Result<OcrModel, ConfigError> {
    if url.source() == orchion::ModelUrlSource::File {
        return Ok(KnownOcrModel::PpDocLayoutV3.into_model());
    }
    if url.owner() == Some("PaddlePaddle")
        && url.repository() == Some("PP-DocLayoutV3_onnx")
        && url.path() == Some("inference.onnx")
    {
        return Ok(KnownOcrModel::PpDocLayoutV3.into_model());
    }
    Err(ConfigError::UnsupportedLayoutModel {
        section,
        value: url.to_string(),
    })
}

fn parse_deployment_name(
    section: &'static str,
    id: &ModelId,
    name: Option<String>,
) -> Result<Option<String>, ConfigError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(ConfigError::InvalidModelName {
            section,
            model: id.to_string(),
        });
    }
    Ok(Some(trimmed.to_string()))
}

fn validate_runtime_category(
    section: &'static str,
    id: &ModelId,
    category: ModelCategory,
    expected: &'static str,
) -> Result<(), ConfigError> {
    if let Some(descriptor) = model_descriptor(id.as_str()) {
        if descriptor.category == category {
            return Ok(());
        }
        return Err(ConfigError::UnsupportedRuntimeModel {
            section,
            id: id.to_string(),
            expected,
        });
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "validates cross-service deployment IDs, runtime categories, and defaults together"
)]
fn validate_model_deployments(config: &ServerConfig) -> Result<(), ConfigError> {
    let mut ids = HashSet::new();
    for id in config
        .services
        .asr
        .models
        .iter()
        .map(|model| &model.id)
        .chain(config.services.tts.models.iter().map(|model| &model.id))
        .chain(config.services.ocr.models.iter().map(|model| &model.id))
        .chain(config.services.ocr_vl.models.iter().map(|model| &model.id))
        .chain(config.services.llm.models.iter().map(|model| &model.id))
    {
        if !ids.insert(id) {
            return Err(ConfigError::DuplicateModelId { id: id.to_string() });
        }
    }

    validate_service_models(
        "services.asr",
        config.services.asr.enabled,
        &config.services.asr.models,
        ModelCategory::Asr,
        "ASR",
    )?;
    validate_service_models(
        "services.tts",
        config.services.tts.enabled,
        &config.services.tts.models,
        ModelCategory::Tts,
        "TTS",
    )?;
    validate_ocr_deployments(
        "services.ocr",
        config.services.ocr.enabled,
        &config.services.ocr.models,
        OcrModelKind::TraditionalOcr,
        "traditional OCR",
    )?;
    validate_ocr_deployments(
        "services.ocr-vl",
        config.services.ocr_vl.enabled,
        &config.services.ocr_vl.models,
        OcrModelKind::OcrVl,
        "OCR-VL",
    )?;
    validate_llm_deployments(&config.services.llm)?;

    ensure_default_available(
        "ASR",
        "services.asr",
        config.services.asr.default_model.as_str(),
        config
            .services
            .asr
            .models
            .iter()
            .filter(|model| model.id.as_str() == config.services.asr.default_model.as_str())
            .count()
            == 1,
    )?;
    ensure_default_available(
        "TTS",
        "services.tts",
        config.services.tts.default_model.as_str(),
        config
            .services
            .tts
            .models
            .iter()
            .filter(|model| model.id.as_str() == config.services.tts.default_model.as_str())
            .count()
            == 1,
    )?;
    validate_optional_default(
        "OCR",
        "services.ocr",
        config.services.ocr.default_model.as_ref(),
        &config.services.ocr.models,
    )?;
    validate_optional_default(
        "OCR-VL",
        "services.ocr-vl",
        config.services.ocr_vl.default_model.as_ref(),
        &config.services.ocr_vl.models,
    )?;
    if let Some(default) = config.services.llm.default_model.as_ref() {
        ensure_default_available(
            "LLM",
            "services.llm",
            default.as_str(),
            config
                .services
                .llm
                .models
                .iter()
                .filter(|model| model.id == *default)
                .count()
                == 1,
        )?;
    } else if config.services.llm.enabled {
        return Err(ConfigError::DefaultModelUnavailable {
            category: "LLM",
            section: "services.llm",
            default: "<missing>".to_string(),
        });
    }
    Ok(())
}

fn validate_llm_deployments(service: &LlmServiceSection) -> Result<(), ConfigError> {
    if service.enabled && service.models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels {
            section: "services.llm",
        });
    }
    for deployment in &service.models {
        validate_deployment_name("services.llm", &deployment.id, deployment.name.as_deref())?;
        validate_exact_gguf(
            &deployment.model,
            "LLM main GGUF must use an exact-file locator",
        )?;
        if let Some(mmproj) = &deployment.mmproj_model {
            validate_exact_gguf(mmproj, "LLM mmproj GGUF must use an exact-file locator")?;
        }
        validate_llm_generation(&deployment.generation)?;
        if deployment.runtime.parallel_sequences != 1 {
            return Err(invalid_llm(
                "runtime.parallel_sequences",
                deployment.runtime.parallel_sequences.to_string(),
                "the text tracer supports exactly one parallel sequence",
            ));
        }
    }
    Ok(())
}

fn validate_exact_gguf(url: &ModelUrl, message: &'static str) -> Result<(), ConfigError> {
    let exact = url.source() == orchion::ModelUrlSource::File
        || url
            .path()
            .is_some_and(|path| path.to_ascii_lowercase().ends_with(".gguf"));
    if exact {
        return Ok(());
    }
    Err(ConfigError::UnsupportedModelLocator {
        section: "services.llm",
        value: url.to_string(),
        message,
    })
}

fn validate_service_models<M>(
    section: &'static str,
    enabled: bool,
    models: &[ModelDeployment<M>],
    category: ModelCategory,
    expected: &'static str,
) -> Result<(), ConfigError>
where
    M: ModelSpec,
{
    if enabled && models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels { section });
    }
    for deployment in models {
        validate_deployment_name(section, &deployment.id, deployment.name.as_deref())?;
        validate_runtime_category(section, &deployment.id, category, expected)?;
        if deployment.runtime.huggingface_repo() != deployment.id.as_str() {
            return Err(ConfigError::UnsupportedRuntimeModel {
                section,
                id: deployment.id.to_string(),
                expected,
            });
        }
        validate_repository_model_locator(section, &deployment.model)?;
    }
    Ok(())
}

fn validate_ocr_deployments(
    section: &'static str,
    enabled: bool,
    models: &[OcrModelDeployment],
    kind: OcrModelKind,
    expected: &'static str,
) -> Result<(), ConfigError> {
    if enabled && models.is_empty() {
        return Err(ConfigError::ServiceEnabledWithoutModels { section });
    }
    for deployment in models {
        validate_deployment_name(section, &deployment.id, deployment.name.as_deref())?;
        let known_kind_matches =
            KnownOcrModel::from_model_id(&deployment.id).is_ok_and(|known| known.kind() == kind);
        if deployment.runtime.id() != &deployment.id || !known_kind_matches {
            return Err(ConfigError::UnsupportedRuntimeModel {
                section,
                id: deployment.id.to_string(),
                expected,
            });
        }
        validate_ocr_model_locator(section, deployment, kind)?;
        if let Some(table) = &deployment.table_structure {
            if kind != OcrModelKind::TraditionalOcr {
                return Err(ConfigError::UnsupportedTableStructurePipeline { section });
            }
            if deployment.layout_model.is_none() {
                return Err(ConfigError::TableStructureRequiresLayout { section });
            }
            validate_table_structure_config(section, table)?;
            for url in [&table.model, &table.dictionary] {
                if url.source() != orchion::ModelUrlSource::File && url.path().is_none() {
                    return Err(ConfigError::UnsupportedModelLocator {
                        section,
                        value: url.to_string(),
                        message: "table structure artifacts must use exact-file locators",
                    });
                }
            }
        }
        match (&deployment.layout_model, &deployment.layout_runtime) {
            (Some(url), Some(runtime)) if runtime.known() == Some(KnownOcrModel::PpDocLayoutV3) => {
                resolve_layout_recipe(section, url)?;
            }
            (None, None) => {}
            (Some(url), _) => {
                return Err(ConfigError::UnsupportedLayoutModel {
                    section,
                    value: url.to_string(),
                });
            }
            (None, Some(_)) => {
                return Err(ConfigError::UnsupportedLayoutModel {
                    section,
                    value: "<missing>".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn validate_repository_model_locator(
    section: &'static str,
    url: &ModelUrl,
) -> Result<(), ConfigError> {
    if url.source() != orchion::ModelUrlSource::File && url.path().is_some() {
        return Err(ConfigError::UnsupportedModelLocator {
            section,
            value: url.to_string(),
            message: "this repository-based runtime does not accept an exact-file model locator",
        });
    }
    Ok(())
}

fn validate_ocr_model_locator(
    section: &'static str,
    deployment: &OcrModelDeployment,
    kind: OcrModelKind,
) -> Result<(), ConfigError> {
    let url = &deployment.model;
    if url.source() == orchion::ModelUrlSource::File {
        if kind == OcrModelKind::TraditionalOcr {
            return Err(ConfigError::UnsupportedModelLocator {
                section,
                value: url.to_string(),
                message: "traditional OCR local packages are not supported by the current multi-repository recipe",
            });
        }
        return Ok(());
    }
    if url.path().is_some()
        || url.owner() != Some(deployment.id.vendor())
        || url.repository() != Some(deployment.id.name())
    {
        return Err(ConfigError::UnsupportedModelLocator {
            section,
            value: url.to_string(),
            message: "OCR model locator must name the runtime recipe's canonical repository",
        });
    }
    Ok(())
}

fn validate_deployment_name(
    section: &'static str,
    id: &ModelId,
    name: Option<&str>,
) -> Result<(), ConfigError> {
    if name.is_some_and(|name| name.is_empty() || name != name.trim()) {
        return Err(ConfigError::InvalidModelName {
            section,
            model: id.to_string(),
        });
    }
    Ok(())
}

fn validate_optional_default(
    category: &'static str,
    section: &'static str,
    default: Option<&ModelId>,
    models: &[OcrModelDeployment],
) -> Result<(), ConfigError> {
    let Some(default) = default else {
        return Ok(());
    };
    ensure_default_available(
        category,
        section,
        default.as_str(),
        models.iter().filter(|model| model.id == *default).count() == 1,
    )
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

fn parse_stream_segment_duration(
    section: &'static str,
    field: &'static str,
    value: &str,
) -> Result<Duration, ConfigError> {
    let duration = parse_duration(value).map_err(|error| match error {
        ConfigError::InvalidDuration { value, message } => {
            ConfigError::InvalidStreamSegmentDuration {
                section,
                field,
                value,
                message,
            }
        }
        other => other,
    })?;
    if duration.as_millis() > u128::from(u32::MAX) {
        return Err(ConfigError::InvalidStreamSegmentDuration {
            section,
            field,
            value: value.to_string(),
            message: "value is too large for streaming millisecond conversion".to_string(),
        });
    }
    Ok(duration)
}

fn format_duration_for_error(duration: Duration) -> String {
    format!("{}s", duration.as_secs())
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

/// # Errors
///
/// Returns [`ConfigError`] when `value` is not a supported ASR model identifier.
pub fn parse_asr_model(value: &str) -> Result<AsrModel, ConfigError> {
    value
        .parse()
        .map_err(|_| ConfigError::InvalidAsrModelId(value.to_string()))
}

/// # Errors
///
/// Returns [`ConfigError`] when `value` is not a supported TTS model identifier.
pub fn parse_tts_model(value: &str) -> Result<TtsModel, ConfigError> {
    value
        .parse()
        .map_err(|_| ConfigError::InvalidTtsModelId(value.to_string()))
}

fn default_asr_model() -> AsrModel {
    AsrModel::parse("Qwen/Qwen3-ASR-0.6B").expect("default ASR model id is valid")
}

fn default_tts_model() -> TtsModel {
    TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").expect("default TTS model id is valid")
}

fn default_asr_deployment() -> ModelDeployment<AsrModel> {
    let runtime = default_asr_model();
    ModelDeployment {
        id: ModelId::parse(runtime.as_str()).expect("default ASR model id is valid"),
        name: None,
        model: ModelUrl::parse("//Qwen/Qwen3-ASR-0.6B").expect("default ASR model URL is valid"),
        runtime,
    }
}

fn default_tts_deployment() -> ModelDeployment<TtsModel> {
    let runtime = default_tts_model();
    ModelDeployment {
        id: ModelId::parse(runtime.as_str()).expect("default TTS model id is valid"),
        name: None,
        model: ModelUrl::parse("//Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
            .expect("default TTS model URL is valid"),
        runtime,
    }
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
    activity: Option<RawActivity>,
    models: Option<RawModels>,
    services: Option<RawServices>,
    auth: Option<RawAuth>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawActivity {
    enabled: Option<bool>,
    history_capacity: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServer {
    bind: Option<String>,
    cors_allowed_origins: Option<Vec<String>>,
    max_upload_size: Option<String>,
    max_pdf_pages: Option<usize>,
    max_pdf_pixels: Option<u64>,
    max_pdf_output_size: Option<String>,
    max_concurrent_inference: Option<usize>,
    max_websocket_connections: Option<usize>,
    max_pending_websocket_connections: Option<usize>,
    max_websocket_message_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModels {
    dir: Option<PathBuf>,
    source: Option<String>,
    max_loaded: Option<usize>,
    verify_file_integrity: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServices {
    asr: Option<RawModelService>,
    tts: Option<RawTtsService>,
    ocr: Option<RawOcrService>,
    #[serde(rename = "ocr-vl")]
    ocr_vl: Option<RawOcrVlService>,
    llm: Option<RawLlmService>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLlmService {
    enabled: Option<bool>,
    default_model: Option<String>,
    models: Option<Vec<RawLlmModelDeployment>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLlmModelDeployment {
    id: String,
    name: Option<String>,
    model: ModelUrl,
    mmproj_model: Option<ModelUrl>,
    runtime: Option<RawLlmRuntime>,
    chat_template: Option<RawChatTemplate>,
    generation: Option<RawLlmGeneration>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawLlmRuntime {
    context_size: Option<RawContextSize>,
    gpu_layers: Option<RawGpuLayers>,
    parallel_sequences: Option<u32>,
    batch_size: Option<u32>,
    micro_batch_size: Option<u32>,
    threads: Option<i32>,
    request_queue_capacity: Option<usize>,
    event_queue_capacity: Option<usize>,
    queue_timeout: Option<String>,
    generation_timeout: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawContextSize {
    Name(String),
    Tokens(u32),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawGpuLayers {
    Name(String),
    Count(u32),
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawChatTemplate {
    engine: Option<String>,
    template: Option<String>,
    enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawLlmGeneration {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelService {
    enabled: Option<bool>,
    default_model: Option<String>,
    models: Option<Vec<RawModelDeployment>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    stream_chunk_size: Option<f32>,
    stream_target_segment: Option<String>,
    stream_max_segment: Option<String>,
    stream_idle_timeout: Option<String>,
    stream_max_duration: Option<String>,
    max_audio_duration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTtsService {
    enabled: Option<bool>,
    default_model: Option<String>,
    models: Option<Vec<RawModelDeployment>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
    max_length: Option<usize>,
    max_reference_audio_duration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOcrService {
    enabled: Option<bool>,
    default_model: Option<String>,
    models: Option<Vec<RawOcrModelDeployment>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
    max_pixels: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOcrVlService {
    enabled: Option<bool>,
    default_model: Option<String>,
    models: Option<Vec<RawOcrModelDeployment>>,
    idle_timeout: Option<String>,
    max_loaded: Option<usize>,
    device: Option<String>,
    format: Option<String>,
    max_tokens: Option<usize>,
    max_pixels: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelDeployment {
    id: String,
    name: Option<String>,
    model: ModelUrl,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOcrModelDeployment {
    id: String,
    name: Option<String>,
    model: ModelUrl,
    layout_model: Option<ModelUrl>,
    table_structure: Option<RawTableStructure>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableStructure {
    model: ModelUrl,
    dictionary: ModelUrl,
    table_type: String,
    score_threshold: Option<f32>,
    max_structure_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAuth {
    api_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "expected {actual} to be within f32::EPSILON of {expected}"
        );
    }

    #[test]
    fn global_model_ids_include_disabled_service_deployments() {
        let mut config = ServerConfig::default_for_exe(Path::new("/tmp/orchion-server"));
        config.services.tts.models[0].id = config.services.asr.models[0].id.clone();

        let error = validate_model_deployments(&config).unwrap_err();

        assert!(matches!(error, ConfigError::DuplicateModelId { .. }));
    }

    #[test]
    fn explicit_config_path_must_exist() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("missing.toml");

        let error = ServerConfig::load(Some(path.clone())).unwrap_err();

        assert!(
            matches!(error, ConfigError::Read { path: error_path, source }
            if error_path == path && source.kind() == std::io::ErrorKind::NotFound)
        );
    }

    #[test]
    fn cors_environment_value_overrides_toml_origins() {
        let mut config = ServerConfig::from_toml_str(
            r#"
            [server]
            cors_allowed_origins = ["https://toml.example.com"]
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap();

        apply_cors_allowed_origins_override(
            &mut config,
            "https://app.example.com, https://admin.example.com",
        )
        .unwrap();

        assert_eq!(
            config.server.cors_allowed_origins,
            ["https://app.example.com", "https://admin.example.com"]
        );
    }

    #[test]
    fn tts_default_format_is_validated_at_startup() {
        let error = ServerConfig::from_toml_str(
            r#"
            [services.tts]
            format = "wave"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap_err();

        assert!(matches!(error, ConfigError::InvalidTtsFormat { value } if value == "wave"));
    }

    #[test]
    fn asr_stream_chunk_size_defaults_to_two_seconds() {
        let config = ServerConfig::default_for_exe(Path::new("/tmp/orchion-server"));

        assert_f32_close(config.services.asr.stream_chunk_size, 2.0);
    }

    #[test]
    fn asr_stream_max_segment_defaults_to_two_minutes() {
        let config = ServerConfig::default_for_exe(Path::new("/tmp/orchion-server"));

        assert_eq!(
            config.services.asr.stream_max_segment,
            Duration::from_mins(2)
        );
    }

    #[test]
    fn asr_stream_session_limits_have_safe_defaults() {
        let config = ServerConfig::default_for_exe(Path::new("/tmp/orchion-server"));

        assert_eq!(
            config.services.asr.stream_idle_timeout,
            Duration::from_secs(30)
        );
        assert_eq!(
            config.services.asr.stream_max_duration,
            Duration::from_hours(2)
        );
    }

    #[test]
    fn asr_stream_target_segment_defaults_to_twelve_seconds() {
        let config = ServerConfig::default_for_exe(Path::new("/tmp/orchion-server"));

        assert_eq!(
            config.services.asr.stream_target_segment,
            Duration::from_secs(12)
        );
    }

    #[test]
    fn asr_stream_chunk_size_loads_from_config() {
        let config = ServerConfig::from_toml_str(
            r"
            [services.asr]
            stream_chunk_size = 1.5
            ",
            Path::new("/tmp/orchion-server"),
        )
        .unwrap();

        assert_f32_close(config.services.asr.stream_chunk_size, 1.5);
    }

    #[test]
    fn asr_stream_max_segment_loads_from_config() {
        let config = ServerConfig::from_toml_str(
            r#"
            [services.asr]
            stream_max_segment = "90s"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap();

        assert_eq!(
            config.services.asr.stream_max_segment,
            Duration::from_secs(90)
        );
    }

    #[test]
    fn asr_stream_target_segment_loads_from_config() {
        let config = ServerConfig::from_toml_str(
            r#"
            [services.asr]
            stream_target_segment = "15s"
            stream_max_segment = "2m"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap();

        assert_eq!(
            config.services.asr.stream_target_segment,
            Duration::from_secs(15)
        );
    }

    #[test]
    fn asr_stream_max_segment_rejects_zero() {
        let error = ServerConfig::from_toml_str(
            r#"
            [services.asr]
            stream_max_segment = "0s"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ConfigError::InvalidStreamSegmentDuration {
                section: "services.asr",
                field: "stream_max_segment",
                ..
            }
        ));
    }

    #[test]
    fn asr_stream_target_segment_rejects_zero() {
        let error = ServerConfig::from_toml_str(
            r#"
            [services.asr]
            stream_target_segment = "0s"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ConfigError::InvalidStreamSegmentDuration {
                section: "services.asr",
                field: "stream_target_segment",
                ..
            }
        ));
    }

    #[test]
    fn asr_stream_target_segment_rejects_values_above_max_segment() {
        let error = ServerConfig::from_toml_str(
            r#"
            [services.asr]
            stream_target_segment = "130s"
            stream_max_segment = "120s"
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ConfigError::InvalidStreamSegmentDuration {
                section: "services.asr",
                field: "stream_target_segment",
                ..
            }
        ));
    }

    #[test]
    fn llm_parses_exact_main_and_mmproj_profile() {
        let config = ServerConfig::from_toml_str(r#"
            [services.llm]
            enabled = true
            default_model = "qwen/test"

            [[services.llm.models]]
            id = "qwen/test"
            name = "Test Q4"
            model = "//owner/repo/main-Q4.gguf"
            mmproj_model = "//owner/repo/mmproj.gguf"
            runtime = { context_size = 4096, gpu_layers = 0, parallel_sequences = 1, batch_size = 128, micro_batch_size = 64, threads = 4, request_queue_capacity = 2, event_queue_capacity = 3, queue_timeout = "7s", generation_timeout = "2m" }
            chat_template = { engine = "llama_cpp", template = "chatml" }
            generation = { max_tokens = 512, temperature = 0.7, top_p = 0.9, top_k = 40, min_p = 0.05, presence_penalty = 0.1, frequency_penalty = -0.1, repeat_penalty = 1.1 }
        "#, Path::new("/tmp/orchion-server")).unwrap();
        let model = &config.services.llm.models[0];
        assert!(config.services.llm.active());
        assert_eq!(model.id.as_str(), "qwen/test");
        assert!(matches!(
            model.runtime.context_size,
            LlmContextSize::Tokens(4096)
        ));
        assert!(matches!(model.runtime.gpu_layers, LlmGpuLayers::Count(0)));
        assert_eq!(model.runtime.parallel_sequences, 1);
        assert_eq!(model.runtime.queue_timeout, Duration::from_secs(7));
        assert_eq!(model.runtime.generation_timeout, Duration::from_mins(2));
        assert!(model.mmproj_model.is_some());
    }

    #[test]
    fn llm_rejects_repository_wide_download_and_parallel_sequences() {
        for (model, parallel) in [("//owner/repo", 1), ("//owner/repo/main.gguf", 2)] {
            let document = format!(
                r#"
                [services.llm]
                enabled = true
                default_model = "qwen/test"
                [[services.llm.models]]
                id = "qwen/test"
                model = "{model}"
                runtime = {{ parallel_sequences = {parallel} }}
            "#
            );
            assert!(
                ServerConfig::from_toml_str(&document, Path::new("/tmp/orchion-server")).is_err()
            );
        }
    }

    #[test]
    fn llm_rejects_deployment_default_below_responses_minimum() {
        let error = ServerConfig::from_toml_str(
            r#"
            [services.llm]
            enabled = true
            default_model = "qwen/test"
            [[services.llm.models]]
            id = "qwen/test"
            model = "//owner/repo/main.gguf"
            generation = { max_tokens = 15 }
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("generation.max_tokens"));
        assert!(error.to_string().contains("16..=4096"));
    }

    #[test]
    fn llm_accepts_text_subset_jinja_engine() {
        let config = ServerConfig::from_toml_str(
            r#"
            [services.llm]
            enabled = true
            default_model = "qwen/test"
            [[services.llm.models]]
            id = "qwen/test"
            model = "//owner/repo/main.gguf"
            chat_template = { engine = "jinja", enable_thinking = true }
            "#,
            Path::new("/tmp/orchion-server"),
        )
        .unwrap();
        assert_eq!(
            config.services.llm.models[0].chat_template.engine,
            ChatTemplateEngine::Jinja
        );
        assert!(config.services.llm.models[0].chat_template.enable_thinking);
    }
}
