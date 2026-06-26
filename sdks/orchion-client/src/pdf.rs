use crate::client::decode_binary;
use crate::{Client, ClientError};
use bytes::Bytes;
use reqwest::multipart::{Form, Part};
use std::path::Path;

/// Client for the PDF API.
pub struct PdfClient<'a> {
    client: &'a Client,
}

impl<'a> PdfClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Renders PDF pages as images and returns a zip archive.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request is invalid, cannot be sent, or the binary response
    /// cannot be decoded.
    pub async fn render_images(
        &self,
        request: PdfImagesRequest,
    ) -> Result<PdfImagesResponse, ClientError> {
        let response = self
            .client
            .post("/v1/pdf/images")?
            .multipart(request.into_form()?)
            .send()
            .await?;
        let response = decode_binary(response).await?;

        Ok(PdfImagesResponse {
            bytes: response.bytes,
            content_type: response.content_type,
        })
    }
}

/// Multipart PDF image rendering request.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfImagesRequest {
    pub filename: String,
    pub file_bytes: Vec<u8>,
    pub response_format: Option<PdfImageFormat>,
    pub pages: Option<String>,
    pub scale: Option<f32>,
}

impl PdfImagesRequest {
    /// Creates a PDF image rendering request.
    #[must_use]
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            file_bytes: Vec::new(),
            response_format: None,
            pages: None,
            scale: None,
        }
    }

    /// Sets PDF bytes for the multipart file field.
    #[must_use]
    pub fn with_file_bytes(mut self, file_bytes: Vec<u8>) -> Self {
        self.file_bytes = file_bytes;
        self
    }

    /// Reads PDF bytes from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the file cannot be read.
    pub async fn with_file_path(mut self, path: impl AsRef<Path>) -> Result<Self, ClientError> {
        self.file_bytes = tokio::fs::read(path.as_ref()).await.map_err(|error| {
            ClientError::build_request(format!("failed to read PDF file: {error}"))
        })?;
        Ok(self)
    }

    /// Sets the optional image response format.
    #[must_use]
    pub const fn with_response_format(mut self, response_format: PdfImageFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Sets the optional pages selector.
    #[must_use]
    pub fn with_pages(mut self, pages: impl Into<String>) -> Self {
        self.pages = Some(pages.into());
        self
    }

    /// Sets the optional image scale.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
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

        if let Some(response_format) = self.response_format {
            form = form.text("response_format", response_format.as_str());
        }

        if let Some(pages) = self.pages {
            form = form.text("pages", pages);
        }

        if let Some(scale) = self.scale {
            form = form.text("scale", scale.to_string());
        }

        Ok(form)
    }
}

/// PDF image output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfImageFormat {
    Png,
    Jpeg,
    Webp,
}

impl PdfImageFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }
}

/// Binary PDF image rendering response.
#[derive(Debug, Clone)]
pub struct PdfImagesResponse {
    pub bytes: Bytes,
    pub content_type: Option<String>,
}
