use super::{
    ModelCapabilities, ModelCategory, ModelDescriptor, ModelId, ModelSpec, ParseModelIdError,
    model_descriptor,
};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TtsModel {
    id: ModelId,
}

impl TtsModel {
    pub fn parse(value: &str) -> Result<Self, ParseModelIdError> {
        Ok(Self {
            id: ModelId::parse(value)?,
        })
    }

    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }

    pub fn descriptor(&self) -> Option<ModelDescriptor> {
        model_descriptor(self.as_str())
            .filter(|descriptor| descriptor.category == ModelCategory::Tts)
    }

    pub fn supports_voice_cloning(&self) -> bool {
        self.descriptor().map_or_else(
            || self.as_str().ends_with("-Base"),
            |descriptor| {
                descriptor
                    .capabilities
                    .contains(ModelCapabilities::TTS_VOICE_CLONING)
            },
        )
    }

    pub fn supports_preset_speakers(&self) -> bool {
        self.descriptor().map_or_else(
            || self.as_str().ends_with("-CustomVoice"),
            |descriptor| {
                descriptor
                    .capabilities
                    .contains(ModelCapabilities::TTS_PRESET_SPEAKERS)
            },
        )
    }

    pub fn supports_voice_design(&self) -> bool {
        self.descriptor().map_or_else(
            || self.as_str().ends_with("-VoiceDesign"),
            |descriptor| {
                descriptor
                    .capabilities
                    .contains(ModelCapabilities::TTS_VOICE_DESIGN)
            },
        )
    }
}

impl AsRef<str> for TtsModel {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for TtsModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ModelSpec for TtsModel {
    fn category(&self) -> ModelCategory {
        ModelCategory::Tts
    }

    fn huggingface_repo(&self) -> &str {
        self.descriptor().map_or(self.as_str(), |descriptor| {
            descriptor.source_locators.hugging_face
        })
    }

    fn modelscope_repo(&self) -> &str {
        self.descriptor().map_or(self.as_str(), |descriptor| {
            descriptor.source_locators.model_scope
        })
    }
}

impl FromStr for TtsModel {
    type Err = ParseModelIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tts_model_names_and_repositories() {
        let model = TtsModel::from_str("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").unwrap();

        assert_eq!(
            model.huggingface_repo(),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        );
    }

    #[test]
    fn accepts_custom_tts_model_ids() {
        let model = TtsModel::from_str("Acme/New-TTS").unwrap();

        assert_eq!(model.huggingface_repo(), "Acme/New-TTS");
        assert!(!model.supports_preset_speakers());
        assert!(!model.supports_voice_cloning());
        assert!(!model.supports_voice_design());
    }

    #[test]
    fn custom_qwen_compatible_model_names_preserve_legacy_capabilities() {
        let base = TtsModel::from_str("Acme/New-TTS-Base").unwrap();
        assert!(base.supports_voice_cloning());
        assert!(!base.supports_preset_speakers());
        assert!(!base.supports_voice_design());

        let custom_voice = TtsModel::from_str("Acme/New-TTS-CustomVoice").unwrap();
        assert!(!custom_voice.supports_voice_cloning());
        assert!(custom_voice.supports_preset_speakers());
        assert!(!custom_voice.supports_voice_design());

        let voice_design = TtsModel::from_str("Acme/New-TTS-VoiceDesign").unwrap();
        assert!(!voice_design.supports_voice_cloning());
        assert!(!voice_design.supports_preset_speakers());
        assert!(voice_design.supports_voice_design());
    }

    #[test]
    fn rejects_invalid_tts_model_ids() {
        assert!(TtsModel::from_str("qwen3-tts-1.7b-voice-design").is_err());
    }

    #[test]
    fn tts_models_expose_stable_metadata() {
        assert_eq!(
            TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
                .unwrap()
                .category(),
            ModelCategory::Tts
        );
        let voice_design = TtsModel::parse("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").unwrap();
        assert!(voice_design.supports_voice_design());
        assert!(!voice_design.supports_preset_speakers());

        let base = TtsModel::parse("Qwen/Qwen3-TTS-12Hz-1.7B-Base").unwrap();
        assert!(base.supports_voice_cloning());
    }
}
