use std::fmt;
use std::path::{Path, PathBuf};

mod asr;
mod descriptor;
mod id;
mod ocr;
mod tts;
mod url;

pub use asr::AsrModel;
pub use descriptor::{
    ModelCapabilities, ModelCapabilityRequirement, ModelDescriptor, ModelSourceLocators,
    RuntimeProvider, model_descriptor, registered_model_descriptors,
};
pub use id::{ModelId, ParseModelIdError};
pub use ocr::{
    KnownOcrModel, OcrModel, OcrModelAsset, OcrModelAssetKind, OcrModelAssetRole, OcrModelKind,
};
pub use tts::TtsModel;
pub use url::{ModelUrl, ModelUrlSource, ParseModelUrlError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    Asr,
    Tts,
    Ocr,
    OcrVl,
}

impl ModelCategory {
    pub const fn cache_segment(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Tts => "tts",
            Self::Ocr => "ocr",
            Self::OcrVl => "ocr-vl",
        }
    }
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.cache_segment())
    }
}

pub trait ModelSpec: Clone + fmt::Debug + Eq + Send + Sync + 'static {
    fn category(&self) -> ModelCategory;
    fn huggingface_repo(&self) -> &str;
    fn modelscope_repo(&self) -> &str;

    fn required_files(&self) -> &'static [&'static str] {
        &["config.json"]
    }

    fn cache_path(&self, cache_dir: impl AsRef<Path>) -> PathBuf {
        self.huggingface_repo()
            .split('/')
            .fold(cache_dir.as_ref().to_path_buf(), |path, segment| {
                path.join(segment)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cache_paths_are_repository_scoped() {
        let path = AsrModel::parse("Qwen/Qwen3-ASR-0.6B")
            .unwrap()
            .cache_path("models");
        assert!(path.ends_with("Qwen/Qwen3-ASR-0.6B"));

        let path = TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
            .unwrap()
            .cache_path("models");
        assert!(path.ends_with("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    }

    #[test]
    fn registry_separates_canonical_ids_sources_capabilities_and_runtime() {
        let asr = model_descriptor("Qwen/Qwen3-ASR-1.7B").unwrap();
        assert_eq!(asr.canonical_id, "Qwen/Qwen3-ASR-1.7B");
        assert_eq!(asr.category, ModelCategory::Asr);
        assert_eq!(asr.runtime_provider, RuntimeProvider::Qwen3);
        assert_eq!(asr.source_locators.hugging_face, "Qwen/Qwen3-ASR-1.7B");
        assert_eq!(asr.source_locators.model_scope, "Qwen/Qwen3-ASR-1.7B");
        assert!(
            asr.capabilities
                .contains(ModelCapabilities::ASR_TRANSCRIPTION)
        );
        assert!(asr.capabilities.contains(ModelCapabilities::ASR_STREAMING));

        let voice_design = model_descriptor("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").unwrap();
        assert!(
            voice_design
                .capabilities
                .contains(ModelCapabilities::TTS_VOICE_DESIGN)
        );
        assert!(
            !voice_design
                .capabilities
                .contains(ModelCapabilities::TTS_VOICE_CLONING)
        );
    }

    #[test]
    fn registry_contains_only_runtime_supported_qwen_speech_models() {
        let asr_ids = registered_model_descriptors(ModelCategory::Asr)
            .map(|descriptor| descriptor.canonical_id)
            .collect::<Vec<_>>();
        assert_eq!(asr_ids, ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]);

        let tts_ids = registered_model_descriptors(ModelCategory::Tts)
            .map(|descriptor| descriptor.canonical_id)
            .collect::<Vec<_>>();
        assert_eq!(
            tts_ids,
            [
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            ]
        );
    }

    #[test]
    fn unregistered_qwen_compatible_tts_names_have_no_capabilities() {
        let model = TtsModel::parse("Acme/Experimental-TTS-Base").unwrap();

        assert!(model.descriptor().is_none());
        assert!(!model.supports_voice_cloning());
        assert!(!model.supports_preset_speakers());
        assert!(!model.supports_voice_design());
    }

    #[test]
    fn ocr_descriptors_reuse_known_ocr_model_metadata() {
        let descriptor = KnownOcrModel::PaddleOcrVl16.descriptor();

        assert_eq!(descriptor.canonical_id, KnownOcrModel::PaddleOcrVl16.id());
        assert_eq!(descriptor.category, ModelCategory::OcrVl);
        assert_eq!(descriptor.runtime_provider, RuntimeProvider::OarOcr);
        assert!(
            descriptor
                .capabilities
                .contains(ModelCapabilities::OCR_VISION_LANGUAGE)
        );
        assert_eq!(
            model_descriptor(KnownOcrModel::PaddleOcrVl16.id()),
            Some(descriptor)
        );
    }
}
