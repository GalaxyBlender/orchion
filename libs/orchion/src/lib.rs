#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

#[cfg(feature = "asr-qwen3")]
pub mod asr;

#[cfg(feature = "tts-qwen3")]
pub mod tts;

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub mod ocr;

pub use orchion_core::{
    ASR_SAMPLE_RATE, AsrModel, AsrOptions, AsrSegment, AsrStreamingOptions,
    AsrTimestampGranularity, AsrTranscript, DevicePreference, ModelCategory, ModelId, ModelSpec,
    OcrLayoutBlock, OcrOptions, OcrPoint, OcrRegion, OcrResponseFormat, OcrResult, OcrTask,
    OcrUsage, OrchionError, Result, TtsAudio, TtsLanguage, TtsModel, TtsOptions, TtsSpeaker,
    TtsVoice, ensure_voice_supported, prepare_asr_samples,
};

#[cfg(feature = "audio-ffmpeg")]
pub use orchion_audio::{
    AudioOutputFormat, DecodedAudio, EncodedAudio, FfmpegAudioCodec, decode_audio_bytes,
    decode_audio_file, encode_tts_audio,
};

#[cfg(feature = "download-all")]
pub use orchion_download::{DownloadSource, ModelDownloader};

#[cfg(feature = "asr-qwen3")]
pub use asr::Asr;

#[cfg(feature = "asr-qwen3")]
pub use orchion_qwen3::AsrStream;

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use ocr::Ocr;

#[cfg(feature = "tts-qwen3")]
pub use tts::Tts;
