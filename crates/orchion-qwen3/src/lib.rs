#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

mod blocking;

#[cfg(feature = "asr")]
pub mod asr;

#[cfg(feature = "tts")]
pub mod tts;

pub use orchion_core::{OrchionError, Result};

#[cfg(feature = "asr")]
pub use asr::{Asr, AsrStream};

#[cfg(feature = "tts")]
pub use tts::Tts;
