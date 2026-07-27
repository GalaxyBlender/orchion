#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

pub mod asr;
pub mod device;
pub mod error;
pub mod model;
pub mod ocr;
pub mod tts;

pub use asr::{
    ASR_SAMPLE_RATE, AsrOptions, AsrSegment, AsrStreamingOptions, AsrTimestampGranularity,
    AsrTranscript, prepare_asr_samples,
};
pub use device::{DevicePreference, ParseDevicePreferenceError};
pub use error::{DownloadFailure, OrchionError, Result};
pub use model::{
    AsrModel, KnownOcrModel, ModelCapabilities, ModelCapabilityRequirement, ModelCategory,
    ModelDescriptor, ModelId, ModelSourceLocators, ModelSpec, OcrModel, OcrModelAsset,
    OcrModelAssetKind, OcrModelAssetRole, OcrModelKind, ParseModelIdError, RuntimeProvider,
    TtsModel, model_descriptor, registered_model_descriptors,
};
pub use ocr::{
    OcrLayoutBlock, OcrLimits, OcrOptions, OcrPoint, OcrRegion, OcrResponseFormat, OcrResult,
    OcrTask, OcrUsage,
};
pub use tts::{TtsAudio, TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice, ensure_voice_supported};
