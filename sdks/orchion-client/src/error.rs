use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerErrorBody {
    pub error: ServerErrorObject,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, thiserror::Error)]
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
}

impl From<reqwest::Error> for ClientError {
    fn from(source: reqwest::Error) -> Self {
        Self::Transport { source }
    }
}
