use crate::{ClientConfig, ClientError, ServerErrorBody};
use bytes::Bytes;
#[cfg(feature = "asr")]
use reqwest::header::{AUTHORIZATION, HeaderValue};
use reqwest::header::{CONTENT_TYPE, HeaderMap};
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
        let http = reqwest::Client::builder()
            .connect_timeout(config.timeout)
            .read_timeout(config.timeout)
            .build()?;
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

    /// Returns the health API client.
    #[cfg(feature = "health")]
    #[must_use]
    pub fn health(&self) -> crate::health::HealthClient<'_> {
        crate::health::HealthClient::new(self)
    }

    /// Returns the activity API client.
    #[cfg(feature = "activity")]
    #[must_use]
    pub fn activity(&self) -> crate::activity::ActivityClient<'_> {
        crate::activity::ActivityClient::new(self)
    }

    /// Returns the LLM API client.
    #[cfg(feature = "llm")]
    #[must_use]
    pub fn llm(&self) -> crate::llm::LlmClient<'_> {
        crate::llm::LlmClient::new(self)
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
        Ok(self
            .authorize(self.http.get(self.url(path)?))
            .timeout(self.config.timeout))
    }

    #[allow(dead_code)]
    pub(crate) fn get_with_path_segment(
        &self,
        path: &str,
        segment: &str,
    ) -> Result<RequestBuilder, ClientError> {
        let mut url = self.url(path)?;
        url.path_segments_mut()
            .map_err(|()| ClientError::build_request("base URL cannot contain path segments"))?
            .pop_if_empty()
            .push(segment);
        Ok(self
            .authorize(self.http.get(url))
            .timeout(self.config.timeout))
    }

    #[allow(dead_code)]
    pub(crate) fn post(&self, path: &str) -> Result<RequestBuilder, ClientError> {
        Ok(self
            .authorize(self.http.post(self.url(path)?))
            .timeout(self.config.timeout))
    }

    #[allow(dead_code)]
    pub(crate) fn delete(&self, path: &str) -> Result<RequestBuilder, ClientError> {
        Ok(self
            .authorize(self.http.delete(self.url(path)?))
            .timeout(self.config.timeout))
    }

    #[cfg(any(feature = "activity", feature = "llm"))]
    #[allow(dead_code)]
    pub(crate) fn stream_get(&self, path: &str) -> Result<RequestBuilder, ClientError> {
        Ok(self.authorize(self.http.get(self.url(path)?)))
    }

    #[cfg(any(feature = "activity", feature = "llm"))]
    #[allow(dead_code)]
    pub(crate) fn stream_post(&self, path: &str) -> Result<RequestBuilder, ClientError> {
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
    pub(crate) headers: HeaderMap,
}

#[allow(dead_code)]
pub(crate) async fn decode_json<T>(response: Response) -> Result<T, ClientError>
where
    T: serde::de::DeserializeOwned,
{
    let response = ensure_success(response).await?;
    let bytes = response.bytes().await?;
    serde_json::from_slice::<T>(&bytes)
        .map_err(|error| ClientError::decode(format!("invalid JSON response: {error}")))
}

#[allow(dead_code)]
pub(crate) async fn decode_text(response: Response) -> Result<String, ClientError> {
    let response = ensure_success(response).await?;
    let bytes = response.bytes().await?;
    String::from_utf8(bytes.to_vec())
        .map_err(|error| ClientError::decode(format!("invalid text response: {error}")))
}

#[allow(dead_code)]
pub(crate) async fn decode_binary(response: Response) -> Result<BinaryResponse, ClientError> {
    let response = ensure_success(response).await?;
    let headers = response.headers().clone();
    let bytes = response.bytes().await?;
    let content_type = headers
        .get(CONTENT_TYPE)
        .map(|value| {
            value.to_str().map(str::to_owned).map_err(|error| {
                ClientError::decode(format!("invalid Content-Type header: {error}"))
            })
        })
        .transpose()?;
    Ok(BinaryResponse {
        bytes,
        content_type,
        headers,
    })
}

#[allow(dead_code)]
pub(crate) async fn ensure_success(response: Response) -> Result<Response, ClientError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = String::from_utf8(response.bytes().await?.to_vec())
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

#[cfg(test)]
mod tests {
    use super::{decode_binary, decode_json, decode_text, ensure_success};
    use crate::ClientError;
    use reqwest::Response;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[tokio::test]
    async fn json_body_disconnect_is_a_transport_error() {
        let response = truncated_response("200 OK").await;

        let error = decode_json::<serde_json::Value>(response)
            .await
            .unwrap_err();

        assert_body_transport(error);
    }

    #[tokio::test]
    async fn text_body_disconnect_is_a_transport_error() {
        let response = truncated_response("200 OK").await;

        let error = decode_text(response).await.unwrap_err();

        assert_body_transport(error);
    }

    #[tokio::test]
    async fn binary_body_disconnect_is_a_transport_error() {
        let response = truncated_response("200 OK").await;

        let Err(error) = decode_binary(response).await else {
            panic!("truncated binary body unexpectedly decoded");
        };

        assert_body_transport(error);
    }

    #[tokio::test]
    async fn error_body_disconnect_is_a_transport_error() {
        let response = truncated_response("500 Internal Server Error").await;

        let error = ensure_success(response).await.unwrap_err();

        assert_body_transport(error);
    }

    #[tokio::test]
    async fn body_timeout_is_a_transport_error_with_the_reqwest_source() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\npartial",
                )
                .unwrap();
            stream.flush().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        });
        let response = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(20))
            .build()
            .unwrap()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();

        let error = decode_text(response).await.unwrap_err();

        match error {
            ClientError::Transport { source } => assert!(source.is_timeout()),
            unexpected => panic!("unexpected error variant: {unexpected:?}"),
        }
    }

    #[tokio::test]
    async fn fully_read_invalid_json_is_a_decode_error() {
        let response = complete_response("200 OK", b"not json").await;

        let error = decode_json::<serde_json::Value>(response)
            .await
            .unwrap_err();

        assert!(matches!(error, ClientError::Decode { .. }));
    }

    #[tokio::test]
    async fn fully_read_invalid_utf8_is_a_decode_error() {
        let response = complete_response("200 OK", &[0xff]).await;

        let error = decode_text(response).await.unwrap_err();

        assert!(matches!(error, ClientError::Decode { .. }));
    }

    fn assert_body_transport(error: ClientError) {
        match error {
            ClientError::Transport { source } => assert!(!source.to_string().is_empty()),
            unexpected => panic!("unexpected error variant: {unexpected:?}"),
        }
    }

    async fn truncated_response(status: &'static str) -> Response {
        raw_response(status, b"partial", 100).await
    }

    async fn complete_response(status: &'static str, body: &'static [u8]) -> Response {
        raw_response(status, body, body.len()).await
    }

    async fn raw_response(
        status: &'static str,
        body: &'static [u8],
        content_length: usize,
    ) -> Response {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });

        reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap()
    }
}
