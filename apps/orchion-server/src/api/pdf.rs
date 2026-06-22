use super::openai::ApiError;

pub use orchion_docs::{PageSelection, PdfImageFormat, PdfRenderRequest, RenderedZip};

pub fn render_pdf_to_zip(request: PdfRenderRequest) -> Result<RenderedZip, ApiError> {
    orchion_docs::render_pdf_to_zip(request).map_err(map_pdf_error)
}

pub fn parse_pdf_image_format(value: Option<&str>) -> Result<PdfImageFormat, ApiError> {
    orchion_docs::parse_pdf_image_format(value).map_err(map_pdf_error)
}

pub fn parse_page_selection(value: Option<&str>) -> Result<PageSelection, ApiError> {
    orchion_docs::parse_page_selection(value).map_err(map_pdf_error)
}

pub fn parse_scale(value: Option<&str>) -> Result<f32, ApiError> {
    orchion_docs::parse_scale(value).map_err(map_pdf_error)
}

#[must_use]
pub fn is_pdf_upload(content_type: Option<&str>, file_name: Option<&str>) -> bool {
    orchion_docs::is_pdf_upload(content_type, file_name)
}

fn map_pdf_error(error: orchion_docs::PdfError) -> ApiError {
    if error.is_invalid_request() {
        ApiError::invalid_request(error.to_string(), error.param(), error.code())
    } else {
        ApiError::internal(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_pdf_file_maps_to_openai_invalid_file_shape() {
        let error = map_pdf_error(orchion_docs::PdfError::InvalidPdfFile);

        assert_eq!(error.error.message, "uploaded file must be a valid PDF");
        assert_eq!(error.error.param.as_deref(), Some("file"));
        assert_eq!(error.error.code.as_deref(), Some("invalid_file"));
    }
}
