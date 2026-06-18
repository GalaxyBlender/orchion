#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

#[cfg(any(feature = "asr", feature = "tts"))]
mod blocking;

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
pub use asr::{Asr, AsrOptions, AsrStream, AsrStreamingOptions, AsrTranscript};

#[cfg(feature = "tts")]
pub use tts::{Tts, TtsAudio, TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice};
