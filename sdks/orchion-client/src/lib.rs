mod client;
mod config;
mod error;

#[cfg(feature = "asr")]
pub mod asr;
#[cfg(feature = "models")]
pub mod models;
#[cfg(feature = "ocr")]
pub mod ocr;
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "tts")]
pub mod tts;

pub use client::Client;
pub use config::ClientConfig;
pub use error::{ClientError, ServerErrorBody, ServerErrorObject};
