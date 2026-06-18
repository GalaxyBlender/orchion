#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

#[cfg(any(feature = "asr", feature = "tts"))]
mod blocking;

#[cfg(any(feature = "asr", feature = "tts"))]
pub mod audio;

pub mod error;
pub mod model;

#[cfg(feature = "download")]
pub mod download;

#[cfg(feature = "asr")]
pub mod asr;

#[cfg(feature = "tts")]
pub mod tts;

pub use error::{OrchionError, Result};
pub use model::{AsrModel, ModelCategory, ModelSpec, TtsModel};

#[cfg(feature = "download")]
pub use download::{DownloadSource, ModelDownloader};

#[cfg(feature = "asr")]
pub use asr::{ASR_SAMPLE_RATE, Asr, AsrOptions, AsrStream, AsrStreamingOptions, AsrTranscript};

#[cfg(any(feature = "asr", feature = "tts"))]
pub use audio::{AudioOutputFormat, EncodedAudio, FfmpegAudioCodec};

#[cfg(feature = "asr")]
pub use audio::{DecodedAudio, decode_audio_bytes};

#[cfg(feature = "tts")]
pub use audio::encode_tts_audio;

#[cfg(feature = "tts")]
pub use tts::{Tts, TtsAudio, TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice};
