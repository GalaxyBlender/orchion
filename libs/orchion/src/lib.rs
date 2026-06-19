#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

#[cfg(feature = "asr-qwen3")]
pub mod asr;

#[cfg(feature = "tts-qwen3")]
pub mod tts;

pub use orchion_core::{
    ASR_SAMPLE_RATE, AsrModel, AsrOptions, AsrStreamingOptions, AsrTranscript, DevicePreference,
    ModelCategory, ModelSpec, OrchionError, Result, TtsAudio, TtsLanguage, TtsModel, TtsOptions,
    TtsSpeaker, TtsVoice, ensure_voice_supported, prepare_asr_samples,
};

#[cfg(feature = "audio-ffmpeg")]
pub use orchion_audio::{
    AudioOutputFormat, DecodedAudio, EncodedAudio, FfmpegAudioCodec, decode_audio_bytes,
    encode_tts_audio,
};

#[cfg(feature = "download-all")]
pub use orchion_download::{DownloadSource, ModelDownloader};

#[cfg(feature = "asr-qwen3")]
pub use asr::Asr;

#[cfg(feature = "asr-qwen3")]
pub use orchion_qwen3::AsrStream;

#[cfg(feature = "tts-qwen3")]
pub use tts::Tts;
