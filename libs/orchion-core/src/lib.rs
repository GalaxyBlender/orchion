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
pub use error::{OrchionError, Result};
pub use model::{
    AsrModel, KnownOcrModel, ModelCategory, ModelHubAsset, ModelHubAssetKind, ModelId, ModelSpec,
    OcrModelKind, ParseModelIdError, TtsModel,
};
pub use ocr::{
    OcrLayoutBlock, OcrOptions, OcrPoint, OcrRegion, OcrResponseFormat, OcrResult, OcrTask,
    OcrUsage,
};
pub use tts::{TtsAudio, TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice, ensure_voice_supported};
