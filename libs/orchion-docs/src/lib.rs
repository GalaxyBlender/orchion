use image::{DynamicImage, ImageFormat};
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};
use std::collections::BTreeSet;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use thiserror::Error;
use zip::result::ZipError;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

pub type Result<T> = std::result::Result<T, PdfError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum PdfImageFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageSelection {
    All,
    Pages(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct PdfRenderRequest {
    pub pdf_path: PathBuf,
    pub format: PdfImageFormat,
    pub pages: PageSelection,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfRenderLimits {
    pub max_pages: usize,
    pub max_pixels: u64,
    pub max_output_bytes: usize,
}

impl PdfRenderLimits {
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_pages: usize::MAX,
            max_pixels: u64::MAX,
            max_output_bytes: usize::MAX,
        }
    }
}

struct LimitedCursor {
    inner: Cursor<Vec<u8>>,
    max_len: usize,
}

impl LimitedCursor {
    fn new(max_len: usize) -> Self {
        Self {
            inner: Cursor::new(Vec::new()),
            max_len,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.inner.into_inner()
    }
}

impl Write for LimitedCursor {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let position = usize::try_from(self.inner.position()).map_err(|_| output_limit_error())?;
        let end = position
            .checked_add(buffer.len())
            .ok_or_else(output_limit_error)?;
        if end > self.max_len {
            return Err(output_limit_error());
        }
        self.inner.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for LimitedCursor {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

fn output_limit_error() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::FileTooLarge,
        "PDF ZIP output exceeds configured limit",
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedZip {
    pub bytes: Vec<u8>,
    pub page_count: usize,
    pub file_count: usize,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum PdfError {
    #[error("Unsupported response_format. Use png, jpeg, or webp.")]
    UnsupportedImageFormat,
    #[error("Invalid pages. Use comma-separated page numbers or ranges like 1,3-5.")]
    InvalidPages,
    #[error("Invalid pages. Requested page is outside the PDF page count.")]
    PageOutsideDocument,
    #[error("Invalid scale. Use a number from 0.1 to 4.0.")]
    InvalidScale,
    #[error("uploaded file must be a valid PDF")]
    InvalidPdfFile,
    #[error("Requested PDF pages exceed the configured page limit.")]
    TooManyPages,
    #[error("Rendered PDF pages exceed the configured pixel limit.")]
    TooManyPixels,
    #[error("Rendered PDF ZIP exceeds the configured output size limit.")]
    OutputTooLarge,
    #[error("Failed to initialize PDF renderer.")]
    InitializeRenderer,
    #[error("Failed to load PDF page.")]
    LoadPage,
    #[error("Failed to render PDF page.")]
    RenderPage,
    #[error("Failed to encode rendered image.")]
    EncodeImage,
    #[error("Failed to create ZIP archive.")]
    CreateZip,
    #[error("Failed to write ZIP archive.")]
    WriteZip,
    #[error("Failed to finish ZIP archive.")]
    FinishZip,
}

impl PdfError {
    #[must_use]
    pub const fn param(self) -> Option<&'static str> {
        match self {
            Self::UnsupportedImageFormat => Some("response_format"),
            Self::InvalidPages | Self::PageOutsideDocument => Some("pages"),
            Self::InvalidScale => Some("scale"),
            Self::InvalidPdfFile => Some("file"),
            Self::TooManyPages => Some("pages"),
            Self::TooManyPixels => Some("scale"),
            Self::OutputTooLarge => None,
            Self::InitializeRenderer
            | Self::LoadPage
            | Self::RenderPage
            | Self::EncodeImage
            | Self::CreateZip
            | Self::WriteZip
            | Self::FinishZip => None,
        }
    }

    #[must_use]
    pub const fn code(self) -> Option<&'static str> {
        match self {
            Self::InvalidPdfFile => Some("invalid_file"),
            Self::TooManyPages | Self::TooManyPixels | Self::OutputTooLarge => {
                Some("pdf_limit_exceeded")
            }
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_invalid_request(self) -> bool {
        matches!(
            self,
            Self::UnsupportedImageFormat
                | Self::InvalidPages
                | Self::PageOutsideDocument
                | Self::InvalidScale
                | Self::InvalidPdfFile
                | Self::TooManyPages
                | Self::TooManyPixels
                | Self::OutputTooLarge
        )
    }
}

/// Renders the selected PDF pages into a ZIP archive without resource limits.
///
/// # Errors
///
/// Returns [`PdfError`] when PDFium cannot be initialized, the request is invalid, or rendering
/// and ZIP encoding fail.
pub fn render_pdf_to_zip(request: PdfRenderRequest) -> Result<RenderedZip> {
    render_pdf_to_zip_with_limits(request, PdfRenderLimits::unlimited())
}

/// Renders the selected PDF pages into a ZIP archive within the supplied resource limits.
///
/// # Errors
///
/// Returns [`PdfError`] when PDFium cannot be initialized, the request is invalid, a resource
/// limit is exceeded, or rendering and ZIP encoding fail.
pub fn render_pdf_to_zip_with_limits(
    request: PdfRenderRequest,
    limits: PdfRenderLimits,
) -> Result<RenderedZip> {
    let pdfium = bind_pdfium()?;
    let document = pdfium
        .load_pdf_from_file(&request.pdf_path, None)
        .map_err(|_| PdfError::InvalidPdfFile)?;
    let page_count =
        usize::try_from(document.pages().len()).map_err(|_| PdfError::InvalidPdfFile)?;
    let page_indices = selected_page_indices(&request.pages, page_count)?;
    if page_indices.len() > limits.max_pages {
        return Err(PdfError::TooManyPages);
    }
    let mut total_pixels = 0_u64;
    for page_index in &page_indices {
        let page = document
            .pages()
            .get(
                (*page_index)
                    .try_into()
                    .map_err(|_| PdfError::PageOutsideDocument)?,
            )
            .map_err(|_| PdfError::LoadPage)?;
        let width = (f64::from(page.width().value) * f64::from(request.scale)).ceil();
        let height = (f64::from(page.height().value) * f64::from(request.scale)).ceil();
        if !width.is_finite()
            || !height.is_finite()
            || width > u64::MAX as f64
            || height > u64::MAX as f64
        {
            return Err(PdfError::TooManyPixels);
        }
        let page_pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or(PdfError::TooManyPixels)?;
        total_pixels = total_pixels
            .checked_add(page_pixels)
            .ok_or(PdfError::TooManyPixels)?;
        if total_pixels > limits.max_pixels {
            return Err(PdfError::TooManyPixels);
        }
    }
    let mut archive = ZipWriter::new(LimitedCursor::new(limits.max_output_bytes));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let render_config = PdfRenderConfig::new().scale_page_by_factor(request.scale);

    for page_index in &page_indices {
        let page = document
            .pages()
            .get(
                (*page_index)
                    .try_into()
                    .map_err(|_| PdfError::PageOutsideDocument)?,
            )
            .map_err(|_| PdfError::LoadPage)?;
        let image = page
            .render_with_config(&render_config)
            .map_err(|_| PdfError::RenderPage)?
            .as_image()
            .map_err(|_| PdfError::RenderPage)?;
        let bytes = encode_image(&image, request.format)?;

        archive
            .start_file(page_file_name(*page_index + 1, request.format), options)
            .map_err(|error| map_zip_error(error, PdfError::CreateZip))?;
        archive
            .write_all(&bytes)
            .map_err(|error| map_write_error(error, PdfError::WriteZip))?;
    }

    let bytes = archive
        .finish()
        .map_err(|error| map_zip_error(error, PdfError::FinishZip))?
        .into_inner();

    Ok(RenderedZip {
        bytes,
        page_count,
        file_count: page_indices.len(),
    })
}

fn map_zip_error(error: ZipError, fallback: PdfError) -> PdfError {
    match error {
        ZipError::Io(error) => map_write_error(error, fallback),
        _ => fallback,
    }
}

fn map_write_error(error: std::io::Error, fallback: PdfError) -> PdfError {
    if error.kind() == std::io::ErrorKind::FileTooLarge {
        PdfError::OutputTooLarge
    } else {
        fallback
    }
}

pub fn parse_pdf_image_format(value: Option<&str>) -> Result<PdfImageFormat> {
    match value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("png") => Ok(PdfImageFormat::Png),
        Some("jpeg" | "jpg") => Ok(PdfImageFormat::Jpeg),
        Some("webp") => Ok(PdfImageFormat::Webp),
        Some(_) => Err(PdfError::UnsupportedImageFormat),
    }
}

pub fn parse_page_selection(value: Option<&str>) -> Result<PageSelection> {
    parse_page_selection_with_max_pages(value, usize::MAX)
}

/// Parses an explicit page selection while limiting how many unique pages may be requested.
///
/// # Errors
///
/// Returns [`PdfError::InvalidPages`] for malformed selectors and [`PdfError::TooManyPages`] as
/// soon as an explicit selection exceeds `max_pages`.
pub fn parse_page_selection_with_max_pages(
    value: Option<&str>,
    max_pages: usize,
) -> Result<PageSelection> {
    match value.map(str::trim) {
        None | Some("") => Ok(PageSelection::All),
        Some(value) if value.eq_ignore_ascii_case("all") => Ok(PageSelection::All),
        Some(value) => {
            let mut pages = BTreeSet::new();

            for segment in value.split(',') {
                let segment = segment.trim();
                if segment.is_empty() {
                    return Err(PdfError::InvalidPages);
                }

                if let Some((start, end)) = segment.split_once('-') {
                    let start = parse_page_number(start)?;
                    let end = parse_page_number(end)?;
                    if start > end {
                        return Err(PdfError::InvalidPages);
                    }
                    for page in start..=end {
                        pages.insert(page);
                        if pages.len() > max_pages {
                            return Err(PdfError::TooManyPages);
                        }
                    }
                } else {
                    pages.insert(parse_page_number(segment)?);
                    if pages.len() > max_pages {
                        return Err(PdfError::TooManyPages);
                    }
                }
            }

            Ok(PageSelection::Pages(pages.into_iter().collect()))
        }
    }
}

pub fn parse_scale(value: Option<&str>) -> Result<f32> {
    match value.map(str::trim) {
        None | Some("") => Ok(1.0),
        Some(value) => {
            let scale = value.parse::<f32>().map_err(|_| PdfError::InvalidScale)?;
            if scale.is_finite() && (0.1..=4.0).contains(&scale) {
                Ok(scale)
            } else {
                Err(PdfError::InvalidScale)
            }
        }
    }
}

#[must_use]
pub fn is_pdf_upload(content_type: Option<&str>, file_name: Option<&str>) -> bool {
    match content_type
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(content_type) => content_type
            .split(';')
            .next()
            .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/pdf")),
        None => file_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_some_and(|file_name| file_name.to_ascii_lowercase().ends_with(".pdf")),
    }
}

fn bind_pdfium() -> Result<&'static Pdfium> {
    static PDFIUM: OnceLock<std::result::Result<Pdfium, PdfError>> = OnceLock::new();
    PDFIUM
        .get_or_init(initialize_pdfium)
        .as_ref()
        .map_err(|error| *error)
}

fn initialize_pdfium() -> Result<Pdfium> {
    let executable_bindings = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(library_path_in))
        .and_then(|path| Pdfium::bind_to_library(path).ok());
    let bindings = match executable_bindings {
        Some(bindings) => bindings,
        None => Pdfium::bind_to_system_library().map_err(|_| PdfError::InitializeRenderer)?,
    };

    Ok(Pdfium::new(bindings))
}

fn library_path_in(dir: &Path) -> PathBuf {
    dir.join(pdfium_library_file_name())
}

fn pdfium_library_file_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libpdfium.dylib"
    } else if cfg!(target_os = "windows") {
        "pdfium.dll"
    } else {
        "libpdfium.so"
    }
}

fn selected_page_indices(selection: &PageSelection, page_count: usize) -> Result<Vec<usize>> {
    match selection {
        PageSelection::All => Ok((0..page_count).collect()),
        PageSelection::Pages(pages) => pages
            .iter()
            .map(|page| {
                if (1..=page_count).contains(page) {
                    Ok(page - 1)
                } else {
                    Err(PdfError::PageOutsideDocument)
                }
            })
            .collect(),
    }
}

fn encode_image(image: &DynamicImage, format: PdfImageFormat) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(
            &mut bytes,
            match format {
                PdfImageFormat::Png => ImageFormat::Png,
                PdfImageFormat::Jpeg => ImageFormat::Jpeg,
                PdfImageFormat::Webp => ImageFormat::WebP,
            },
        )
        .map_err(|_| PdfError::EncodeImage)?;

    Ok(bytes.into_inner())
}

fn page_file_name(page_number: usize, format: PdfImageFormat) -> String {
    let extension = match format {
        PdfImageFormat::Png => "png",
        PdfImageFormat::Jpeg => "jpg",
        PdfImageFormat::Webp => "webp",
    };

    format!("page-{page_number:04}.{extension}")
}

fn parse_page_number(value: &str) -> Result<usize> {
    let page = value
        .trim()
        .parse::<usize>()
        .map_err(|_| PdfError::InvalidPages)?;
    if page == 0 {
        Err(PdfError::InvalidPages)
    } else {
        Ok(page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as FmtWrite;

    fn assert_pdf_error(error: PdfError, expected: PdfError, message: &str, param: &str) {
        assert_eq!(error, expected);
        assert_eq!(error.to_string(), message);
        assert_eq!(error.param(), Some(param));
    }

    #[test]
    fn parse_pdf_image_format_defaults_to_png_and_accepts_supported_formats() {
        assert_eq!(parse_pdf_image_format(None).unwrap(), PdfImageFormat::Png);
        assert_eq!(
            parse_pdf_image_format(Some("")).unwrap(),
            PdfImageFormat::Png
        );
        assert_eq!(
            parse_pdf_image_format(Some("   ")).unwrap(),
            PdfImageFormat::Png
        );
        assert_eq!(
            parse_pdf_image_format(Some("png")).unwrap(),
            PdfImageFormat::Png
        );
        assert_eq!(
            parse_pdf_image_format(Some("jpeg")).unwrap(),
            PdfImageFormat::Jpeg
        );
        assert_eq!(
            parse_pdf_image_format(Some("jpg")).unwrap(),
            PdfImageFormat::Jpeg
        );
        assert_eq!(
            parse_pdf_image_format(Some("webp")).unwrap(),
            PdfImageFormat::Webp
        );
    }

    #[test]
    fn parse_pdf_image_format_rejects_unsupported_value_with_param() {
        let error = parse_pdf_image_format(Some("gif")).unwrap_err();

        assert_pdf_error(
            error,
            PdfError::UnsupportedImageFormat,
            "Unsupported response_format. Use png, jpeg, or webp.",
            "response_format",
        );
    }

    #[test]
    fn parse_page_selection_defaults_to_all() {
        assert_eq!(parse_page_selection(None).unwrap(), PageSelection::All);
        assert_eq!(parse_page_selection(Some("")).unwrap(), PageSelection::All);
        assert_eq!(
            parse_page_selection(Some("   ")).unwrap(),
            PageSelection::All
        );
        assert_eq!(
            parse_page_selection(Some("all")).unwrap(),
            PageSelection::All
        );
        assert_eq!(
            parse_page_selection(Some("ALL")).unwrap(),
            PageSelection::All
        );
    }

    #[test]
    fn parse_page_selection_parses_ranges_and_deduplicates_ascending() {
        assert_eq!(
            parse_page_selection(Some("1,3-5,4")).unwrap(),
            PageSelection::Pages(vec![1, 3, 4, 5])
        );
    }

    #[test]
    fn parse_page_selection_rejects_ranges_above_limit_without_expanding_them() {
        assert_eq!(
            parse_page_selection_with_max_pages(Some("1-18446744073709551615"), 100).unwrap_err(),
            PdfError::TooManyPages
        );
    }

    #[test]
    fn parse_page_selection_rejects_invalid_selectors_with_pages_param() {
        for value in ["0", "2-1", "abc", "1,,2"] {
            let error = parse_page_selection(Some(value)).unwrap_err();
            assert_pdf_error(
                error,
                PdfError::InvalidPages,
                "Invalid pages. Use comma-separated page numbers or ranges like 1,3-5.",
                "pages",
            );
        }
    }

    #[test]
    fn parse_scale_defaults_and_accepts_inclusive_bounds() {
        assert_eq!(parse_scale(None).unwrap(), 1.0);
        assert_eq!(parse_scale(Some("")).unwrap(), 1.0);
        assert_eq!(parse_scale(Some("   ")).unwrap(), 1.0);
        assert_eq!(parse_scale(Some("0.1")).unwrap(), 0.1);
        assert_eq!(parse_scale(Some("4.0")).unwrap(), 4.0);
    }

    #[test]
    fn parse_scale_rejects_invalid_values_with_scale_param() {
        for value in ["0", "4.1", "NaN", "abc"] {
            let error = parse_scale(Some(value)).unwrap_err();
            assert_pdf_error(
                error,
                PdfError::InvalidScale,
                "Invalid scale. Use a number from 0.1 to 4.0.",
                "scale",
            );
        }
    }

    #[test]
    fn is_pdf_upload_uses_content_type_before_file_name_fallback() {
        assert!(is_pdf_upload(Some("application/pdf"), None));
        assert!(is_pdf_upload(
            Some("Application/PDF; charset=binary"),
            Some("file.txt")
        ));
        assert!(is_pdf_upload(None, Some("file.pdf")));
        assert!(is_pdf_upload(Some("   "), Some("file.PDF")));
        assert!(!is_pdf_upload(Some("text/plain"), Some("file.pdf")));
        assert!(!is_pdf_upload(None, Some("file.png")));
        assert!(!is_pdf_upload(None, None));
    }

    #[test]
    fn page_file_name_uses_padded_one_based_names() {
        assert_eq!(page_file_name(1, PdfImageFormat::Png), "page-0001.png");
        assert_eq!(page_file_name(42, PdfImageFormat::Jpeg), "page-0042.jpg");
        assert_eq!(page_file_name(7, PdfImageFormat::Webp), "page-0007.webp");
    }

    #[test]
    fn selected_page_indices_selects_all_zero_based_indices() {
        assert_eq!(
            selected_page_indices(&PageSelection::All, 3).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn selected_page_indices_converts_explicit_pages_to_zero_based_indices() {
        assert_eq!(
            selected_page_indices(&PageSelection::Pages(vec![1, 3]), 5).unwrap(),
            vec![0, 2]
        );
    }

    #[test]
    fn selected_page_indices_rejects_pages_outside_document_page_count() {
        for selection in [PageSelection::Pages(vec![0]), PageSelection::Pages(vec![4])] {
            let error = selected_page_indices(&selection, 3).unwrap_err();
            assert_pdf_error(
                error,
                PdfError::PageOutsideDocument,
                "Invalid pages. Requested page is outside the PDF page count.",
                "pages",
            );
        }
    }

    #[test]
    fn invalid_pdf_file_uses_invalid_file_error_shape() {
        let error = PdfError::InvalidPdfFile;

        assert_pdf_error(
            error,
            PdfError::InvalidPdfFile,
            "uploaded file must be a valid PDF",
            "file",
        );
        assert_eq!(error.code(), Some("invalid_file"));
    }

    #[test]
    fn render_rejects_selected_pages_above_limit() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), two_page_pdf_bytes()).unwrap();
        let request = PdfRenderRequest {
            pdf_path: file.path().to_path_buf(),
            format: PdfImageFormat::Png,
            pages: PageSelection::All,
            scale: 1.0,
        };

        let error = render_pdf_to_zip_with_limits(
            request,
            PdfRenderLimits {
                max_pages: 1,
                max_pixels: u64::MAX,
                max_output_bytes: usize::MAX,
            },
        )
        .unwrap_err();

        assert_eq!(error, PdfError::TooManyPages);
    }

    #[test]
    fn render_rejects_pages_above_pixel_limit() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), two_page_pdf_bytes()).unwrap();
        let request = PdfRenderRequest {
            pdf_path: file.path().to_path_buf(),
            format: PdfImageFormat::Png,
            pages: PageSelection::All,
            scale: 1.0,
        };

        let error = render_pdf_to_zip_with_limits(
            request,
            PdfRenderLimits {
                max_pages: 2,
                max_pixels: 1,
                max_output_bytes: usize::MAX,
            },
        )
        .unwrap_err();

        assert_eq!(error, PdfError::TooManyPixels);
    }

    #[test]
    fn render_stops_when_zip_exceeds_output_limit() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), two_page_pdf_bytes()).unwrap();
        let request = PdfRenderRequest {
            pdf_path: file.path().to_path_buf(),
            format: PdfImageFormat::Png,
            pages: PageSelection::Pages(vec![1]),
            scale: 1.0,
        };

        let error = render_pdf_to_zip_with_limits(
            request,
            PdfRenderLimits {
                max_pages: 1,
                max_pixels: u64::MAX,
                max_output_bytes: 16,
            },
        )
        .unwrap_err();

        assert_eq!(error, PdfError::OutputTooLarge);
    }

    fn two_page_pdf_bytes() -> Vec<u8> {
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << >> >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << >> >>",
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            writeln!(&mut pdf, "{} 0 obj\n{object}\nendobj", index + 1).unwrap();
        }
        let xref_offset = pdf.len();
        writeln!(&mut pdf, "xref\n0 {}", objects.len() + 1).unwrap();
        writeln!(&mut pdf, "0000000000 65535 f ").unwrap();
        for offset in offsets {
            writeln!(&mut pdf, "{offset:010} 00000 n ").unwrap();
        }
        writeln!(
            &mut pdf,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF",
            objects.len() + 1
        )
        .unwrap();
        pdf.into_bytes()
    }
}
