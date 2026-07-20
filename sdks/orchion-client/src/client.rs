use crate::{ClientConfig, ClientError, ServerErrorBody};
use bytes::Bytes;
use reqwest::header::CONTENT_TYPE;
#[cfg(feature = "asr")]
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::{RequestBuilder, Response, Url};
use std::fmt;

/// Shared Orchion API client.
#[derive(Clone)]
pub struct Client {
    config: ClientConfig,
    #[allow(dead_code)]
    http: reqwest::Client,
}

impl fmt::Debug for Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Client")
            .field("base_url", &self.config.base_url)
            .field(
                "api_key",
                &self.config.api_key.as_ref().map(|_| "<redacted>"),
            )
            .field("timeout", &self.config.timeout)
            .field("http", &self.http)
            .finish()
    }
}

impl Client {
    /// Creates a client from a base URL.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the base URL or HTTP client configuration is invalid.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self, ClientError> {
        Self::from_config(ClientConfig::new(base_url)?)
    }

    /// Creates a client from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the underlying HTTP client cannot be built.
    pub fn from_config(config: ClientConfig) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder().timeout(config.timeout).build()?;
        Ok(Self { config, http })
    }

    /// Returns the client configuration.
    #[must_use]
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Returns the models API client.
    #[cfg(feature = "models")]
    #[must_use]
    pub fn models(&self) -> crate::models::ModelsClient<'_> {
        crate::models::ModelsClient::new(self)
    }

    /// Returns the ASR API client.
    #[cfg(feature = "asr")]
    #[must_use]
    pub fn asr(&self) -> crate::asr::AsrClient<'_> {
        crate::asr::AsrClient::new(self)
    }

    /// Returns the TTS API client.
    #[cfg(feature = "tts")]
    #[must_use]
    pub fn tts(&self) -> crate::tts::TtsClient<'_> {
        crate::tts::TtsClient::new(self)
    }

    /// Returns the OCR API client.
    #[cfg(feature = "ocr")]
    #[must_use]
    pub fn ocr(&self) -> crate::ocr::OcrClient<'_> {
        crate::ocr::OcrClient::new(self)
    }

    /// Returns the PDF API client.
    #[cfg(feature = "pdf")]
    #[must_use]
    pub fn pdf(&self) -> crate::pdf::PdfClient<'_> {
        crate::pdf::PdfClient::new(self)
    }

    #[allow(dead_code)]
    pub(crate) fn url(&self, path: &str) -> Result<Url, ClientError> {
        let relative_path = path.strip_prefix('/').unwrap_or(path);
        self.config.base_url.join(relative_path).map_err(|error| {
            ClientError::build_request(format!("invalid request path `{path}`: {error}"))
        })
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, path: &str) -> Result<RequestBuilder, ClientError> {
        Ok(self.authorize(self.http.get(self.url(path)?)))
    }

    #[allow(dead_code)]
    pub(crate) fn post(&self, path: &str) -> Result<RequestBuilder, ClientError> {
        Ok(self.authorize(self.http.post(self.url(path)?)))
    }

    #[allow(dead_code)]
    pub(crate) fn authorize(&self, builder: RequestBuilder) -> RequestBuilder {
        match self.config.api_key.as_deref() {
            Some(api_key) => builder.bearer_auth(api_key),
            None => builder,
        }
    }

    #[cfg(feature = "asr")]
    pub(crate) fn websocket_url(&self, path: &str) -> Result<Url, ClientError> {
        let mut url = self.url(path)?;
        let scheme = match url.scheme() {
            "http" => "ws",
            "https" => "wss",
            scheme => {
                return Err(ClientError::WebSocket {
                    message: format!("unsupported websocket URL scheme `{scheme}`"),
                });
            }
        };
        url.set_scheme(scheme)
            .map_err(|()| ClientError::WebSocket {
                message: format!("failed to set websocket URL scheme `{scheme}`"),
            })?;
        Ok(url)
    }

    #[cfg(feature = "asr")]
    pub(crate) fn websocket_headers(&self) -> Result<HeaderMap, ClientError> {
        let mut headers = HeaderMap::new();
        if let Some(api_key) = self.config.api_key.as_deref() {
            let mut value =
                HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|error| {
                    ClientError::WebSocket {
                        message: format!("invalid Authorization header: {error}"),
                    }
                })?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        Ok(headers)
    }
}

#[allow(dead_code)]
pub(crate) struct BinaryResponse {
    pub(crate) bytes: Bytes,
    pub(crate) content_type: Option<String>,
}

#[allow(dead_code)]
pub(crate) async fn decode_json<T>(response: Response) -> Result<T, ClientError>
where
    T: serde::de::DeserializeOwned,
{
    let response = ensure_success(response).await?;
    response
        .json::<T>()
        .await
        .map_err(|error| ClientError::decode(format!("invalid JSON response: {error}")))
}

#[allow(dead_code)]
pub(crate) async fn decode_text(response: Response) -> Result<String, ClientError> {
    let response = ensure_success(response).await?;
    response
        .text()
        .await
        .map_err(|error| ClientError::decode(format!("invalid text response: {error}")))
}

#[allow(dead_code)]
pub(crate) async fn decode_binary(response: Response) -> Result<BinaryResponse, ClientError> {
    let response = ensure_success(response).await?;
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .map(|value| {
            value.to_str().map(str::to_owned).map_err(|error| {
                ClientError::decode(format!("invalid Content-Type header: {error}"))
            })
        })
        .transpose()?;
    let bytes = response
        .bytes()
        .await
        .map_err(|error| ClientError::decode(format!("invalid binary response: {error}")))?;
    Ok(BinaryResponse {
        bytes,
        content_type,
    })
}

#[allow(dead_code)]
async fn ensure_success(response: Response) -> Result<Response, ClientError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response
        .text()
        .await
        .map_err(|error| ClientError::decode(format!("invalid error response: {error}")))?;

    if let Ok(server_error) = serde_json::from_str::<ServerErrorBody>(&body) {
        let message = server_error.error.message.clone();
        return Err(ClientError::Http {
            status,
            message,
            error: Some(server_error.error),
        });
    }

    let message = if body.is_empty() {
        status
            .canonical_reason()
            .unwrap_or("HTTP request failed")
            .to_string()
    } else {
        body
    };

    Err(ClientError::Http {
        status,
        message,
        error: None,
    })
}
