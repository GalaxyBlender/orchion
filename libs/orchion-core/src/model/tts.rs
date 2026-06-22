use super::{ModelCategory, ModelSpec};
use crate::{OrchionError, Result};
use std::str::FromStr;

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

impl FromStr for TtsModel {
    type Err = OrchionError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base" => Ok(Self::Qwen3Tts06BBase),
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => Ok(Self::Qwen3Tts06BCustomVoice),
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base" => Ok(Self::Qwen3Tts17BBase),
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" => Ok(Self::Qwen3Tts17BCustomVoice),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" => Ok(Self::Qwen3Tts17BVoiceDesign),
            _ => Err(OrchionError::ModelLoad {
                source: anyhow::anyhow!("unknown TTS model `{value}`"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tts_model_names_and_repositories() {
        let model = TtsModel::from_str("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").unwrap();

        assert_eq!(model, TtsModel::Qwen3Tts17BVoiceDesign);
        assert_eq!(
            model.huggingface_repo(),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        );
    }

    #[test]
    fn rejects_legacy_tts_aliases() {
        assert!(TtsModel::from_str("qwen3-tts-1.7b-voice-design").is_err());
    }

    #[test]
    fn tts_models_expose_stable_metadata() {
        assert_eq!(TtsModel::Qwen3Tts06BBase.category(), ModelCategory::Tts);
        assert!(TtsModel::Qwen3Tts17BVoiceDesign.supports_voice_design());
        assert!(!TtsModel::Qwen3Tts17BVoiceDesign.supports_preset_speakers());
        assert!(TtsModel::Qwen3Tts17BBase.supports_voice_cloning());
    }
}
