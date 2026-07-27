use crate::client::decode_binary;
use crate::{Client, ClientError};
use bytes::Bytes;
use reqwest::multipart::{Form, Part};
use serde::Serialize;
use std::path::{Path, PathBuf};

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

    /// Creates speech by cloning a voice from reference audio and its transcript.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, a reference audio path cannot be read,
    /// the multipart request cannot be sent, or the binary response cannot be decoded.
    pub async fn create_voice_clone(
        &self,
        request: VoiceCloneRequest,
    ) -> Result<SpeechResponse, ClientError> {
        let response = self
            .client
            .post("/v1/audio/speech")?
            .multipart(request.into_form().await?)
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
                "clone voice requires multipart reference audio and is not supported by create_speech; use create_voice_clone instead",
            ));
        }

        Ok(())
    }
}

/// Reference audio supplied to a voice clone request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceAudio {
    /// In-memory audio with the filename sent in the multipart file field.
    Bytes { filename: String, bytes: Vec<u8> },
    /// Audio read from a local path when the request is sent.
    Path(PathBuf),
}

impl ReferenceAudio {
    /// Creates in-memory reference audio.
    #[must_use]
    pub fn bytes(filename: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes {
            filename: filename.into(),
            bytes: bytes.into(),
        }
    }

    /// Creates reference audio backed by a local file path.
    #[must_use]
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    async fn into_part(self) -> Result<Part, ClientError> {
        let (filename, bytes) = match self {
            Self::Bytes { filename, bytes } => (filename, bytes),
            Self::Path(path) => {
                let filename = reference_audio_filename(&path)?;
                let bytes = tokio::fs::read(&path).await.map_err(|error| {
                    ClientError::build_request(format!(
                        "failed to read reference audio file `{}`: {error}",
                        path.display()
                    ))
                })?;
                (filename, bytes)
            }
        };

        if filename.is_empty() {
            return Err(ClientError::build_request(
                "reference audio filename must not be empty",
            ));
        }
        if bytes.is_empty() {
            return Err(ClientError::build_request(
                "reference audio bytes must not be empty",
            ));
        }

        Ok(Part::bytes(bytes).file_name(filename))
    }
}

fn reference_audio_filename(path: &Path) -> Result<String, ClientError> {
    path.file_name()
        .filter(|filename| !filename.is_empty())
        .map(|filename| filename.to_string_lossy().into_owned())
        .ok_or_else(|| ClientError::build_request("reference audio path must include a filename"))
}

/// Multipart voice clone request.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceCloneRequest {
    pub model: String,
    pub input: String,
    pub reference_audio: ReferenceAudio,
    pub reference_text: String,
    pub response_format: Option<SpeechFormat>,
    pub speed: f32,
    pub language: Option<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub max_length: Option<usize>,
}

impl VoiceCloneRequest {
    /// Creates a voice clone request.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        reference_audio: ReferenceAudio,
        reference_text: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            reference_audio,
            reference_text: reference_text.into(),
            response_format: None,
            speed: 1.0,
            language: None,
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

    /// Sets the speech speed.
    #[must_use]
    pub const fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Sets the optional speech language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the optional sampling seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the optional sampling temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the optional top-k sampling value.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets the optional top-p sampling value.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the optional repetition penalty.
    #[must_use]
    pub const fn with_repetition_penalty(mut self, repetition_penalty: f64) -> Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }

    /// Sets the optional generated audio length limit.
    #[must_use]
    pub const fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    async fn into_form(self) -> Result<Form, ClientError> {
        if self.model.is_empty() {
            return Err(ClientError::build_request("model must not be empty"));
        }
        if self.input.is_empty() {
            return Err(ClientError::build_request("input must not be empty"));
        }
        if self.reference_text.is_empty() {
            return Err(ClientError::build_request(
                "reference text must not be empty",
            ));
        }

        let reference_audio = self.reference_audio.into_part().await?;
        let mut form = Form::new()
            .text("model", self.model)
            .text("input", self.input)
            .text("voice", "clone")
            .part("reference_audio", reference_audio)
            .text("reference_text", self.reference_text)
            .text("speed", self.speed.to_string());

        if let Some(response_format) = self.response_format {
            form = form.text("response_format", response_format.as_str());
        }
        if let Some(language) = self.language {
            form = form.text("language", language);
        }
        if let Some(seed) = self.seed {
            form = form.text("seed", seed.to_string());
        }
        if let Some(temperature) = self.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(top_k) = self.top_k {
            form = form.text("top_k", top_k.to_string());
        }
        if let Some(top_p) = self.top_p {
            form = form.text("top_p", top_p.to_string());
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            form = form.text("repetition_penalty", repetition_penalty.to_string());
        }
        if let Some(max_length) = self.max_length {
            form = form.text("max_length", max_length.to_string());
        }

        Ok(form)
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

impl SpeechFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Opus => "opus",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        }
    }
}

/// Binary speech response.
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    pub bytes: Bytes,
    pub content_type: Option<String>,
}
