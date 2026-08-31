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
        self.descriptor().is_some_and(|descriptor| {
            descriptor
                .capabilities
                .contains(ModelCapabilities::TTS_VOICE_CLONING)
        })
    }

    pub fn supports_preset_speakers(&self) -> bool {
        self.descriptor().is_some_and(|descriptor| {
            descriptor
                .capabilities
                .contains(ModelCapabilities::TTS_PRESET_SPEAKERS)
        })
    }

    pub fn supports_voice_design(&self) -> bool {
        self.descriptor().is_some_and(|descriptor| {
            descriptor
                .capabilities
                .contains(ModelCapabilities::TTS_VOICE_DESIGN)
        })
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

    fn model_id(&self) -> &str {
        self.as_str()
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
        let model = TtsModel::from_str("alibaba/qwen3-tts-12hz-1.7b-voicedesign").unwrap();

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
    fn unregistered_model_suffixes_do_not_forge_capabilities() {
        for id in [
            "Acme/New-TTS-Base",
            "Acme/New-TTS-CustomVoice",
            "Acme/New-TTS-VoiceDesign",
        ] {
            let model = TtsModel::from_str(id).unwrap();
            assert!(model.descriptor().is_none());
            assert!(!model.supports_voice_cloning());
            assert!(!model.supports_preset_speakers());
            assert!(!model.supports_voice_design());
        }
    }

    #[test]
    fn rejects_invalid_tts_model_ids() {
        assert!(TtsModel::from_str("qwen3-tts-1.7b-voice-design").is_err());
    }

    #[test]
    fn tts_models_expose_stable_metadata() {
        assert_eq!(
            TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-base")
                .unwrap()
                .category(),
            ModelCategory::Tts
        );
        let voice_design = TtsModel::parse("alibaba/qwen3-tts-12hz-1.7b-voicedesign").unwrap();
        assert!(voice_design.supports_voice_design());
        assert!(!voice_design.supports_preset_speakers());

        let base = TtsModel::parse("alibaba/qwen3-tts-12hz-1.7b-base").unwrap();
        assert!(base.supports_voice_cloning());
    }
}
