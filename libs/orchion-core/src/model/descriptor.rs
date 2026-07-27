use super::{KnownOcrModel, ModelCategory, ModelId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeProvider {
    Qwen3,
    OarOcr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelSourceLocators {
    pub hugging_face: &'static str,
    pub model_scope: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ModelCapabilities(u16);

impl ModelCapabilities {
    pub const NONE: Self = Self(0);
    pub const ASR_TRANSCRIPTION: Self = Self(1 << 0);
    pub const ASR_STREAMING: Self = Self(1 << 1);
    pub const TTS_VOICE_CLONING: Self = Self(1 << 2);
    pub const TTS_PRESET_SPEAKERS: Self = Self(1 << 3);
    pub const TTS_VOICE_DESIGN: Self = Self(1 << 4);
    pub const OCR_TEXT: Self = Self(1 << 5);
    pub const OCR_LAYOUT: Self = Self(1 << 6);
    pub const OCR_VISION_LANGUAGE: Self = Self(1 << 7);
    pub const OCR_MARKDOWN: Self = Self(1 << 8);
    pub const OCR_HTML: Self = Self(1 << 9);

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn contains(self, capability: Self) -> bool {
        self.0 & capability.0 == capability.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelCapabilityRequirement {
    pub capability: ModelCapabilities,
    pub requires: ModelCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelDescriptor {
    pub canonical_id: &'static str,
    pub source_locators: ModelSourceLocators,
    pub category: ModelCategory,
    pub capabilities: ModelCapabilities,
    pub requirements: &'static [ModelCapabilityRequirement],
    pub runtime_provider: RuntimeProvider,
}

impl ModelDescriptor {
    #[must_use]
    pub fn effective_capabilities(self, available: ModelCapabilities) -> ModelCapabilities {
        let available = self.capabilities.union(available);
        self.requirements
            .iter()
            .filter(|requirement| available.contains(requirement.requires))
            .fold(self.capabilities, |capabilities, requirement| {
                capabilities.union(requirement.capability)
            })
    }
}

const ASR_CAPABILITIES: ModelCapabilities =
    ModelCapabilities::ASR_TRANSCRIPTION.union(ModelCapabilities::ASR_STREAMING);

const ASR_MODELS: [ModelDescriptor; 2] = [
    qwen_descriptor("Qwen/Qwen3-ASR-0.6B", ModelCategory::Asr, ASR_CAPABILITIES),
    qwen_descriptor("Qwen/Qwen3-ASR-1.7B", ModelCategory::Asr, ASR_CAPABILITIES),
];

const TTS_MODELS: [ModelDescriptor; 5] = [
    qwen_descriptor(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ModelCategory::Tts,
        ModelCapabilities::TTS_VOICE_CLONING,
    ),
    qwen_descriptor(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ModelCategory::Tts,
        ModelCapabilities::TTS_PRESET_SPEAKERS,
    ),
    qwen_descriptor(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ModelCategory::Tts,
        ModelCapabilities::TTS_VOICE_CLONING,
    ),
    qwen_descriptor(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ModelCategory::Tts,
        ModelCapabilities::TTS_PRESET_SPEAKERS,
    ),
    qwen_descriptor(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ModelCategory::Tts,
        ModelCapabilities::TTS_VOICE_DESIGN,
    ),
];

const fn qwen_descriptor(
    canonical_id: &'static str,
    category: ModelCategory,
    capabilities: ModelCapabilities,
) -> ModelDescriptor {
    ModelDescriptor {
        canonical_id,
        source_locators: ModelSourceLocators {
            hugging_face: canonical_id,
            model_scope: canonical_id,
        },
        category,
        capabilities,
        requirements: &[],
        runtime_provider: RuntimeProvider::Qwen3,
    }
}

pub fn model_descriptor(id: &str) -> Option<ModelDescriptor> {
    ASR_MODELS
        .iter()
        .chain(TTS_MODELS.iter())
        .find(|descriptor| descriptor.canonical_id == id)
        .copied()
        .or_else(|| {
            let id = ModelId::parse(id).ok()?;
            KnownOcrModel::from_model_id(&id)
                .ok()
                .map(KnownOcrModel::descriptor)
        })
}

pub fn registered_model_descriptors(
    category: ModelCategory,
) -> impl Iterator<Item = ModelDescriptor> {
    ASR_MODELS
        .iter()
        .chain(TTS_MODELS.iter())
        .copied()
        .chain(
            KnownOcrModel::ALL
                .iter()
                .copied()
                .map(KnownOcrModel::descriptor),
        )
        .filter(move |descriptor| descriptor.category == category)
}
