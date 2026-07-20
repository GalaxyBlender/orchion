use crate::client::{decode_json, decode_text};
use crate::{Client, ClientError};
use orchion_core::{OcrLayoutBlock, OcrRegion, OcrUsage};
use reqwest::header::CONTENT_TYPE;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::path::Path;

/// Client for the OCR API.
pub struct OcrClient<'a> {
    client: &'a Client,
}

impl<'a> OcrClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Recognizes text and layout from an image or document.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or the response cannot
    /// be decoded.
    pub async fn recognize(&self, request: OcrRequest) -> Result<OcrResponse, ClientError> {
        let response_format = request.response_format;
        let response = self
            .client
            .post("/v1/ocr")?
            .multipart(request.into_form()?)
            .send()
            .await?;

        match response_format {
            Some(OcrResponseFormat::Json) => {
                let response = decode_json(response).await?;
                Ok(OcrResponse::Json(response))
            }
            Some(
                OcrResponseFormat::Text | OcrResponseFormat::Markdown | OcrResponseFormat::Html,
            ) => {
                let response = decode_text(response).await?;
                Ok(OcrResponse::Text(response))
            }
            None if response_is_json(&response) => {
                let response = decode_json(response).await?;
                Ok(OcrResponse::Json(response))
            }
            None => {
                let response = decode_text(response).await?;
                Ok(OcrResponse::Text(response))
            }
        }
    }
}

/// Multipart OCR request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrRequest {
    pub filename: String,
    pub file_bytes: Vec<u8>,
    pub model: Option<String>,
    pub response_format: Option<OcrResponseFormat>,
    pub task: Option<OcrTask>,
    pub layout_model: Option<String>,
    pub max_tokens: Option<usize>,
}

impl OcrRequest {
    /// Creates an OCR request.
    #[must_use]
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            file_bytes: Vec::new(),
            model: None,
            response_format: None,
            task: None,
            layout_model: None,
            max_tokens: None,
        }
    }

    /// Sets image or document bytes for the multipart file field.
    #[must_use]
    pub fn with_file_bytes(mut self, file_bytes: Vec<u8>) -> Self {
        self.file_bytes = file_bytes;
        self
    }

    /// Reads image or document bytes from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the file cannot be read.
    pub async fn with_file_path(mut self, path: impl AsRef<Path>) -> Result<Self, ClientError> {
        self.file_bytes = tokio::fs::read(path.as_ref()).await.map_err(|error| {
            ClientError::build_request(format!("failed to read OCR file: {error}"))
        })?;
        Ok(self)
    }

    /// Sets the optional OCR model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the optional response format.
    #[must_use]
    pub const fn with_response_format(mut self, response_format: OcrResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Sets the optional OCR task.
    #[must_use]
    pub const fn with_task(mut self, task: OcrTask) -> Self {
        self.task = Some(task);
        self
    }

    /// Sets the optional layout model.
    #[must_use]
    pub fn with_layout_model(mut self, layout_model: impl Into<String>) -> Self {
        self.layout_model = Some(layout_model.into());
        self
    }

    /// Sets the optional token limit.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    fn into_form(self) -> Result<Form, ClientError> {
        if self.filename.is_empty() {
            return Err(ClientError::build_request("filename must not be empty"));
        }

        if self.file_bytes.is_empty() {
            return Err(ClientError::build_request("file bytes must not be empty"));
        }

        let file = Part::bytes(self.file_bytes).file_name(self.filename);
        let mut form = Form::new().part("file", file);

        if let Some(model) = self.model {
            form = form.text("model", model);
        }

        if let Some(response_format) = self.response_format {
            form = form.text("response_format", response_format.as_str());
        }

        if let Some(task) = self.task {
            form = form.text("task", task.as_str());
        }

        if let Some(layout_model) = self.layout_model {
            form = form.text("layout_model", layout_model);
        }

        if let Some(max_tokens) = self.max_tokens {
            form = form.text("max_tokens", max_tokens.to_string());
        }

        Ok(form)
    }
}

fn response_is_json(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
}

/// OCR response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrResponseFormat {
    Json,
    Text,
    Markdown,
    Html,
}

impl OcrResponseFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Html => "html",
        }
    }
}

/// OCR task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrTask {
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

impl OcrTask {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ocr => "ocr",
            Self::Table => "table",
            Self::Formula => "formula",
            Self::Chart => "chart",
            Self::Spotting => "spotting",
            Self::Seal => "seal",
        }
    }
}

/// JSON OCR response body.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OcrJsonResponse {
    pub model: String,
    pub format: OcrResponseFormatValue,
    pub text: String,
    pub markdown: Option<String>,
    pub html: Option<String>,
    #[serde(default)]
    pub regions: Vec<OcrRegion>,
    #[serde(default)]
    pub layout_blocks: Vec<OcrLayoutBlock>,
    pub usage: OcrUsage,
}

/// OCR response format value returned by the JSON endpoint.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrResponseFormatValue {
    Json,
    Text,
    Markdown,
    Html,
}

/// OCR response.
#[derive(Debug, Clone, PartialEq)]
pub enum OcrResponse {
    Json(OcrJsonResponse),
    Text(String),
}
