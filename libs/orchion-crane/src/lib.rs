//! In-process Crane speech inference for Orchion. The caller owns model
//! provisioning and schedules these blocking operations off async workers.
//!
//! [`Asr`] loads Qwen3-ASR `-hf` weights for batch transcription. [`Tts`]
//! loads Qwen3-TTS `CustomVoice` or Base weights. Neither loader downloads
//! weights or exposes Candle types to callers.
#![allow(clippy::missing_errors_doc)]

mod asr;
mod device;
mod tts;

pub use asr::Asr;
pub use tts::{Tts, TtsOptions, Voice};
