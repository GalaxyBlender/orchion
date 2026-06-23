use crate::api::http_shared::{authorize, read_text_field, write_multipart_file_to_temp_file};
use crate::api::openai::ApiError;
use crate::api::pdf::{self, PdfRenderRequest};
use crate::infrastructure::orchion::AppState;
use axum::body::Body;
use axum::extract::{Multipart, State};
use axum::http::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use std::sync::Arc;
use tempfile::NamedTempFile;

pub(super) async fn create_pdf_images(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let upload = read_pdf_images_request(multipart).await?;
    let PdfImagesRequest { pdf_file, request } = upload;
    let rendered = tokio::task::spawn_blocking(move || {
        let rendered = pdf::render_pdf_to_zip(request);
        drop(pdf_file);
        rendered
    })
    .await
    .map_err(|error| ApiError::internal(error.to_string()))??;

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/zip"))
        .header(
            CONTENT_DISPOSITION,
            HeaderValue::from_static("attachment; filename=pdf-images.zip"),
        )
        .header("x-pdf-page-count", rendered.page_count.to_string())
        .header("x-pdf-image-count", rendered.file_count.to_string())
        .body(Body::from(rendered.bytes))
        .map_err(|error| ApiError::internal(error.to_string()))
}

struct PdfImagesRequest {
    pdf_file: NamedTempFile,
    request: PdfRenderRequest,
}

async fn read_pdf_images_request(mut multipart: Multipart) -> Result<PdfImagesRequest, ApiError> {
    let mut pdf_file = None;
    let mut response_format = None;
    let mut pages = None;
    let mut scale = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let content_type = field.content_type().map(ToOwned::to_owned);
                let file_name = field.file_name().map(ToOwned::to_owned);
                if !pdf::is_pdf_upload(content_type.as_deref(), file_name.as_deref()) {
                    return Err(ApiError::invalid_request(
                        "uploaded file must be a PDF",
                        Some("file"),
                        Some("invalid_file"),
                    ));
                }
                pdf_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "response_format" => {
                response_format = Some(read_text_field(field, "response_format").await?);
            }
            "pages" => pages = Some(read_text_field(field, "pages").await?),
            "scale" => scale = Some(read_text_field(field, "scale").await?),
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let (pdf_file, pdf_bytes) = pdf_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if pdf_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded PDF file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }

    let request = PdfRenderRequest {
        pdf_path: pdf_file.path().to_path_buf(),
        format: pdf::parse_pdf_image_format(response_format.as_deref())?,
        pages: pdf::parse_page_selection(pages.as_deref())?,
        scale: pdf::parse_scale(scale.as_deref())?,
    };

    Ok(PdfImagesRequest { pdf_file, request })
}
