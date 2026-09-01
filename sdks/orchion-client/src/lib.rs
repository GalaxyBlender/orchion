//! Typed asynchronous client for Orchion Server.
//!
//! [`Client`] owns shared HTTP transport and exposes domain clients for health, model discovery and
//! residency, Activity, LLM generation and embeddings, ASR, TTS, OCR, and PDF rendering. Each domain is controlled
//! by a same-named Cargo feature; default features enable the complete client.
//!
//! Streaming interfaces expose `next_event` methods instead of a raw wire stream. ASR streaming
//! waits for the server's ready acknowledgement before returning a writable session. Chat
//! completions require the `[DONE]` sentinel, Responses require a completed or incomplete lifecycle
//! event, and Activity treats a clean connection close as a normal end. In-band server errors and
//! premature inference stream closure are returned as structured [`ClientError`] values.
//!
//! The client never automatically retries inference requests or reconnects streams.

mod client;
mod config;
mod error;

#[cfg(any(feature = "activity", feature = "llm"))]
mod sse;

#[cfg(feature = "activity")]
pub mod activity;
#[cfg(feature = "asr")]
pub mod asr;
#[cfg(feature = "health")]
pub mod health;
#[cfg(feature = "llm")]
pub mod llm;
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
pub use error::{ClientError, ServerErrorBody, ServerErrorObject, StreamErrorObject};
