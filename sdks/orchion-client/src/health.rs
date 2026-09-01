use crate::client::{decode_text, ensure_success};
use crate::{Client, ClientError};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Readiness {
    pub status: ReadinessStatus,
    #[serde(default)]
    pub reasons: Vec<ReadinessReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessStatus {
    Ready,
    NotReady,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ReadinessReason {
    pub code: String,
    pub service: Option<String>,
    pub model: Option<String>,
}

/// Client for the server health endpoint.
pub struct HealthClient<'a> {
    client: &'a Client,
}

impl<'a> HealthClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Checks whether the server reports itself healthy.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request fails or the response body is not exactly `ok`
    /// after trimming surrounding whitespace.
    pub async fn check(&self) -> Result<(), ClientError> {
        let response = self.client.get("/healthz")?.send().await?;
        let body = decode_text(response).await?;
        if body.trim() == "ok" {
            Ok(())
        } else {
            Err(ClientError::decode(format!(
                "unexpected health response body: {body:?}"
            )))
        }
    }

    /// Returns the server readiness state. A `503` readiness response is decoded normally.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] for transport failures, unexpected HTTP statuses, or invalid JSON.
    pub async fn ready(&self) -> Result<Readiness, ClientError> {
        let response = self.client.get("/readyz")?.send().await?;
        let response = if response.status().is_success()
            || response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE
        {
            response
        } else {
            ensure_success(response).await?
        };
        let bytes = response.bytes().await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| ClientError::decode(format!("invalid readiness response: {error}")))
    }
}
