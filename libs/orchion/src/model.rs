use std::fmt;
use std::path::{Path, PathBuf};

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
        cache_dir
            .as_ref()
            .join(self.category().cache_segment())
            .join(self.cache_key())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsrModel {
    Qwen3Asr06B,
    Qwen3Asr17B,
}

impl ModelSpec for AsrModel {
    fn category(self) -> ModelCategory {
        ModelCategory::Asr
    }

    fn cache_key(self) -> &'static str {
        match self {
            Self::Qwen3Asr06B => "qwen3-asr-0.6b",
            Self::Qwen3Asr17B => "qwen3-asr-1.7b",
        }
    }

    fn huggingface_repo(self) -> &'static str {
        match self {
            Self::Qwen3Asr06B => "Qwen/Qwen3-ASR-0.6B",
            Self::Qwen3Asr17B => "Qwen/Qwen3-ASR-1.7B",
        }
    }

    fn modelscope_repo(self) -> &'static str {
        self.huggingface_repo()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TtsModel {
    Qwen3Tts06BBase,
    Qwen3Tts06BCustomVoice,
    Qwen3Tts17BBase,
    Qwen3Tts17BCustomVoice,
    Qwen3Tts17BVoiceDesign,
}

impl TtsModel {
    pub const fn supports_voice_cloning(self) -> bool {
        matches!(self, Self::Qwen3Tts06BBase | Self::Qwen3Tts17BBase)
    }

    pub const fn supports_preset_speakers(self) -> bool {
        matches!(
            self,
            Self::Qwen3Tts06BCustomVoice | Self::Qwen3Tts17BCustomVoice
        )
    }

    pub const fn supports_voice_design(self) -> bool {
        matches!(self, Self::Qwen3Tts17BVoiceDesign)
    }
}

impl ModelSpec for TtsModel {
    fn category(self) -> ModelCategory {
        ModelCategory::Tts
    }

    fn cache_key(self) -> &'static str {
        match self {
            Self::Qwen3Tts06BBase => "qwen3-tts-0.6b-base",
            Self::Qwen3Tts06BCustomVoice => "qwen3-tts-0.6b-custom-voice",
            Self::Qwen3Tts17BBase => "qwen3-tts-1.7b-base",
            Self::Qwen3Tts17BCustomVoice => "qwen3-tts-1.7b-custom-voice",
            Self::Qwen3Tts17BVoiceDesign => "qwen3-tts-1.7b-voice-design",
        }
    }

    fn huggingface_repo(self) -> &'static str {
        match self {
            Self::Qwen3Tts06BBase => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts06BCustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts17BBase => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts17BCustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts17BVoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
    }

    fn modelscope_repo(self) -> &'static str {
        self.huggingface_repo()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_models_expose_stable_metadata() {
        assert_eq!(AsrModel::Qwen3Asr06B.category(), ModelCategory::Asr);
        assert_eq!(AsrModel::Qwen3Asr06B.cache_key(), "qwen3-asr-0.6b");
        assert_eq!(
            AsrModel::Qwen3Asr06B.huggingface_repo(),
            "Qwen/Qwen3-ASR-0.6B"
        );
        assert_eq!(
            AsrModel::Qwen3Asr06B.modelscope_repo(),
            "Qwen/Qwen3-ASR-0.6B"
        );
        assert_eq!(AsrModel::Qwen3Asr17B.cache_key(), "qwen3-asr-1.7b");
    }

    #[test]
    fn tts_models_expose_stable_metadata() {
        assert_eq!(TtsModel::Qwen3Tts06BBase.category(), ModelCategory::Tts);
        assert_eq!(TtsModel::Qwen3Tts06BBase.cache_key(), "qwen3-tts-0.6b-base");
        assert_eq!(
            TtsModel::Qwen3Tts17BVoiceDesign.huggingface_repo(),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        );
        assert!(TtsModel::Qwen3Tts17BVoiceDesign.supports_voice_design());
        assert!(!TtsModel::Qwen3Tts17BVoiceDesign.supports_preset_speakers());
        assert!(TtsModel::Qwen3Tts17BBase.supports_voice_cloning());
    }

    #[test]
    fn model_cache_paths_are_category_scoped() {
        let path = AsrModel::Qwen3Asr06B.cache_path("models");
        assert!(path.ends_with("asr/qwen3-asr-0.6b"));

        let path = TtsModel::Qwen3Tts06BCustomVoice.cache_path("models");
        assert!(path.ends_with("tts/qwen3-tts-0.6b-custom-voice"));
    }
}
