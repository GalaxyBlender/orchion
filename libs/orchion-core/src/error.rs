use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, OrchionError>;

#[derive(Debug, thiserror::Error)]
pub enum OrchionError {
    #[error("invalid model source `{value}`; expected `auto`, `huggingface`, or `modelscope`")]
    InvalidModelSource { value: String },

    #[error("download failed for {repo} from {source_name}: {message}")]
    Download {
        source_name: &'static str,
        repo: String,
        message: String,
    },

    #[error("all download sources failed for {repo}: {messages}")]
    DownloadFallbackExhausted { repo: String, messages: String },

    #[error("cache directory is incomplete: {path}")]
    IncompleteCache { path: PathBuf },

    #[error("blocking task failed: {message}")]
    BlockingTask { message: String },

    #[error("model load failed: {source}")]
    ModelLoad { source: anyhow::Error },

    #[error("inference failed: {source}")]
    Inference { source: anyhow::Error },

    #[error("invalid audio input: {reason}")]
    InvalidAudio { reason: String },

    #[error("resampling failed: {reason}")]
    Resample { reason: String },

    #[error("model {model} does not support {capability}")]
    UnsupportedCapability {
        model: String,
        capability: &'static str,
    },

    #[error("path is not valid UTF-8: {path}")]
    NonUtf8Path { path: PathBuf },
}
