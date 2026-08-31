use crate::client::decode_text;
use crate::{Client, ClientError};

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
}
