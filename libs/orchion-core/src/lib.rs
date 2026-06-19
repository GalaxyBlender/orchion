#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

pub mod asr;
pub mod device;
pub mod error;
pub mod model;
pub mod tts;

pub use asr::{
    ASR_SAMPLE_RATE, AsrOptions, AsrSegment, AsrStreamingOptions, AsrTimestampGranularity,
    AsrTranscript, prepare_asr_samples,
};
pub use device::{DevicePreference, ParseDevicePreferenceError};
pub use error::{OrchionError, Result};
pub use model::{AsrModel, ModelCategory, ModelSpec, TtsModel};
pub use tts::{TtsAudio, TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice, ensure_voice_supported};
