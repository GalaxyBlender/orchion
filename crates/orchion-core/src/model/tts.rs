use super::{ModelCategory, ModelSpec, normalize_identifier};
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

impl FromStr for TtsModel {
    type Err = OrchionError;

    fn from_str(value: &str) -> Result<Self> {
        match normalize_identifier(value).as_str() {
            "qwen3-tts-0.6b-base" | "qwen/qwen3-tts-12hz-0.6b-base" => Ok(Self::Qwen3Tts06BBase),
            "qwen3-tts-0.6b-custom-voice" | "qwen/qwen3-tts-12hz-0.6b-customvoice" => {
                Ok(Self::Qwen3Tts06BCustomVoice)
            }
            "qwen3-tts-1.7b-base" | "qwen/qwen3-tts-12hz-1.7b-base" => Ok(Self::Qwen3Tts17BBase),
            "qwen3-tts-1.7b-custom-voice" | "qwen/qwen3-tts-12hz-1.7b-customvoice" => {
                Ok(Self::Qwen3Tts17BCustomVoice)
            }
            "qwen3-tts-1.7b-voice-design" | "qwen/qwen3-tts-12hz-1.7b-voicedesign" => {
                Ok(Self::Qwen3Tts17BVoiceDesign)
            }
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
        assert_eq!(model.cache_key(), "qwen3-tts-1.7b-voice-design");
        assert_eq!(
            model.huggingface_repo(),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        );
    }

    #[test]
    fn tts_models_expose_stable_metadata() {
        assert_eq!(TtsModel::Qwen3Tts06BBase.category(), ModelCategory::Tts);
        assert!(TtsModel::Qwen3Tts17BVoiceDesign.supports_voice_design());
        assert!(!TtsModel::Qwen3Tts17BVoiceDesign.supports_preset_speakers());
        assert!(TtsModel::Qwen3Tts17BBase.supports_voice_cloning());
    }
}
