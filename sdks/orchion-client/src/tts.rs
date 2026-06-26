use crate::client::decode_binary;
use crate::{Client, ClientError};
use bytes::Bytes;
use serde::Serialize;

/// Client for the TTS API.
pub struct TtsClient<'a> {
    client: &'a Client,
}

impl<'a> TtsClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Creates speech audio from text.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or the binary response
    /// cannot be decoded.
    pub async fn create_speech(
        &self,
        request: SpeechRequest,
    ) -> Result<SpeechResponse, ClientError> {
        request.validate()?;

        let response = self
            .client
            .post("/v1/audio/speech")?
            .json(&request)
            .send()
            .await?;
        let response = decode_binary(response).await?;

        Ok(SpeechResponse {
            bytes: response.bytes,
            content_type: response.content_type,
        })
    }
}

/// Text-to-speech request body.
#[derive(Debug, Clone, Serialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<SpeechFormat>,
    pub speed: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

impl SpeechRequest {
    /// Creates a speech request.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            voice: voice.into(),
            response_format: None,
            speed: 1.0,
            language: None,
            voice_prompt: None,
            seed: None,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            max_length: None,
        }
    }

    /// Sets the response audio format.
    #[must_use]
    pub const fn with_response_format(mut self, response_format: SpeechFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Sets the optional speech language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        if self.model.is_empty() {
            return Err(ClientError::build_request("model must not be empty"));
        }

        if self.input.is_empty() {
            return Err(ClientError::build_request("input must not be empty"));
        }

        if self.voice.is_empty() {
            return Err(ClientError::build_request("voice must not be empty"));
        }

        if self.voice.trim().eq_ignore_ascii_case("clone") {
            return Err(ClientError::build_request(
                "clone voice requires multipart reference audio and is not supported by create_speech",
            ));
        }

        Ok(())
    }
}

/// Speech audio format.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFormat {
    Wav,
    Mp3,
    Aac,
    Opus,
    Flac,
    Pcm,
}

/// Binary speech response.
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    pub bytes: Bytes,
    pub content_type: Option<String>,
}
