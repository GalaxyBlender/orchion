use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use orchion_protocol::ErrorObject as ServerErrorObject;

/// Error fields carried by a flat server-sent `event:error` frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamErrorObject {
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerErrorBody {
    pub error: ServerErrorObject,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    #[error("invalid client configuration: {message}")]
    InvalidConfig { message: String },
    #[error("failed to build request: {message}")]
    BuildRequest { message: String },
    #[error("request transport failed: {source}")]
    Transport { source: reqwest::Error },
    #[error("server returned HTTP {status}: {message}")]
    Http {
        status: StatusCode,
        message: String,
        error: Option<ServerErrorObject>,
    },
    #[error("failed to decode response: {message}")]
    Decode { message: String },
    #[cfg(feature = "asr")]
    #[error("websocket failed: {message}")]
    WebSocket { message: String },
    #[cfg(any(feature = "asr", feature = "llm"))]
    #[error("streaming server error: {error:?}")]
    StreamingServer { error: ServerErrorObject },
    #[cfg(feature = "llm")]
    #[error("streaming server error: {error:?}")]
    StreamingServerEvent { error: StreamErrorObject },
    #[error("{stream} stream ended before its terminal event")]
    UnexpectedEof { stream: &'static str },
    #[error("{operation} timed out after {timeout:?}")]
    Timeout {
        operation: &'static str,
        timeout: Duration,
    },
    #[cfg(feature = "asr")]
    #[error("streaming session is terminal after {operation} failed")]
    StreamingSessionTerminated { operation: &'static str },
}

impl ClientError {
    #[must_use]
    pub(crate) fn build_request(message: impl Into<String>) -> Self {
        Self::BuildRequest {
            message: message.into(),
        }
    }

    #[must_use]
    pub(crate) fn decode(message: impl Into<String>) -> Self {
        Self::Decode {
            message: message.into(),
        }
    }

    #[cfg(feature = "asr")]
    #[must_use]
    pub(crate) const fn timeout(operation: &'static str, timeout: Duration) -> Self {
        Self::Timeout { operation, timeout }
    }

    #[cfg(feature = "asr")]
    #[must_use]
    pub(crate) const fn streaming_session_terminated(operation: &'static str) -> Self {
        Self::StreamingSessionTerminated { operation }
    }
}

impl From<reqwest::Error> for ClientError {
    fn from(source: reqwest::Error) -> Self {
        Self::Transport { source }
    }
}
