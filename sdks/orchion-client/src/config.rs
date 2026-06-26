use crate::ClientError;
use reqwest::Url;
use std::time::Duration;

#[derive(Clone)]
pub struct ClientConfig {
    pub base_url: Url,
    pub api_key: Option<String>,
    pub timeout: Duration,
}

impl std::fmt::Debug for ClientConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClientConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl ClientConfig {
    /// Creates a client configuration for the provided base URL.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::InvalidConfig`] when the base URL is invalid or does not use HTTP.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self, ClientError> {
        let base_url = parse_base_url(base_url.as_ref())?;
        Ok(Self {
            base_url,
            api_key: None,
            timeout: Duration::from_mins(5),
        })
    }

    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the request timeout.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::InvalidConfig`] when the timeout is zero.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, ClientError> {
        if timeout.is_zero() {
            return Err(ClientError::InvalidConfig {
                message: "timeout must be greater than zero".to_string(),
            });
        }
        self.timeout = timeout;
        Ok(self)
    }
}

fn parse_base_url(value: &str) -> Result<Url, ClientError> {
    let mut url = Url::parse(value).map_err(|error| ClientError::InvalidConfig {
        message: format!("invalid base URL: {error}"),
    })?;
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(ClientError::InvalidConfig {
                message: format!("unsupported base URL scheme `{scheme}`"),
            });
        }
    }
    if url.cannot_be_a_base() {
        return Err(ClientError::InvalidConfig {
            message: "base URL must be absolute".to_string(),
        });
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_normalizes_base_url_with_trailing_slash() {
        let config = ClientConfig::new("http://localhost:8080").unwrap();

        assert_eq!(config.base_url.as_str(), "http://localhost:8080/");
    }

    #[test]
    fn config_rejects_non_http_base_url() {
        let error = ClientConfig::new("file:///tmp/orchion").unwrap_err();

        assert!(matches!(error, ClientError::InvalidConfig { .. }));
    }

    #[test]
    fn config_rejects_zero_timeout() {
        let error = ClientConfig::new("http://localhost:8080")
            .unwrap()
            .with_timeout(Duration::ZERO)
            .unwrap_err();

        assert!(matches!(error, ClientError::InvalidConfig { .. }));
    }

    #[test]
    fn config_debug_redacts_api_key() {
        let config = ClientConfig::new("http://localhost:8080")
            .unwrap()
            .with_api_key("secret-token");

        let debug = format!("{config:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-token"));
    }
}
