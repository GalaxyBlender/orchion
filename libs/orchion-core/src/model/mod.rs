use std::fmt;
use std::path::{Path, PathBuf};

mod asr;
mod tts;

pub use asr::AsrModel;
pub use tts::TtsModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    Asr,
    Tts,
}

impl ModelCategory {
    pub const fn cache_segment(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Tts => "tts",
        }
    }
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.cache_segment())
    }
}

pub trait ModelSpec: Copy + fmt::Debug + Eq + Send + 'static {
    fn category(self) -> ModelCategory;
    fn cache_key(self) -> &'static str;
    fn huggingface_repo(self) -> &'static str;
    fn modelscope_repo(self) -> &'static str;

    fn cache_path(self, cache_dir: impl AsRef<Path>) -> PathBuf {
        self.huggingface_repo()
            .split('/')
            .fold(cache_dir.as_ref().to_path_buf(), |path, segment| {
                path.join(segment)
            })
    }
}

pub(crate) fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cache_paths_are_repository_scoped() {
        let path = AsrModel::Qwen3Asr06B.cache_path("models");
        assert!(path.ends_with("Qwen/Qwen3-ASR-0.6B"));

        let path = TtsModel::Qwen3Tts06BCustomVoice.cache_path("models");
        assert!(path.ends_with("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    }
}
