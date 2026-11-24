use crate::api::activity::ActivityContext;
use crate::api::http_shared::{
    authorize, parse_multipart_value, read_text_field, run_owned, write_multipart_file_to_temp_file,
};
use crate::api::openai::{ApiError, OcrApiFormat, OcrJsonResponse};
use crate::application::ServerApplication;
use crate::application::ocr::{OcrCommand, recognize, resolve_service_choice};
use axum::Json;
use axum::extract::{Extension, Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use orchion::{OcrResponseFormat, OcrTask};
use std::sync::Arc;

#[allow(
    clippy::too_many_lines,
    reason = "multipart parsing and cleanup form one request transaction"
)]
pub(super) async fn create_ocr<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    mut multipart: Multipart,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
    let mut image_file = None;
    let mut model = None;
    let mut response_format = None;
    let mut task = OcrTask::Ocr;
    let mut max_tokens = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                image_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "response_format" => {
                let value = read_text_field(field, "response_format").await?;
                response_format = Some(OcrApiFormat::try_from(value.as_str())?);
            }
            "task" => {
                let value = read_text_field(field, "task").await?;
                task = parse_ocr_task(&value)?;
            }
            "layout_model" => {
                let _ = field.text().await;
                return Err(ApiError::invalid_request(
                    "`layout_model` is not supported; configure the layout model on the deployment",
                    Some("layout_model"),
                    Some("unsupported_ocr_parameter"),
                ));
            }
            "max_tokens" => {
                let value: usize = parse_multipart_value(field, "max_tokens").await?;
                if value == 0 {
                    return Err(ApiError::invalid_request(
                        "`max_tokens` must be greater than 0",
                        Some("max_tokens"),
                        Some("invalid_multipart_field"),
                    ));
                }
                max_tokens = Some(value);
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let model = model.ok_or_else(|| {
        ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )
    })?;
    let (image_file, image_bytes) = image_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if image_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded OCR file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }
    if let Some(Extension(activity)) = &activity {
        activity.set_input_bytes(image_bytes);
        if let Ok(choice) = resolve_service_choice(&state.ocr_policy(), &model) {
            activity.set_model(choice.model().to_string());
        }
    }

    let image_path = image_file.path().to_path_buf();
    let operation_state = Arc::clone(&state);
    let output = run_owned(async move {
        let _image_file = image_file;
        recognize(
            operation_state.as_ref(),
            OcrCommand {
                image_path,
                model,
                response_format: response_format.map(OcrResponseFormat::from),
                task,
                max_tokens,
            },
        )
        .await
        .map_err(ApiError::from)
    })
    .await?;

    let response_format = OcrApiFormat::from(output.format);
    let result = output.result;
    if let Some(Extension(activity)) = &activity {
        activity.set_model(result.model.to_string());
    }
    Ok(ocr_response(response_format, result))
}

fn ocr_response(response_format: OcrApiFormat, result: orchion::OcrResult) -> Response {
    match response_format {
        OcrApiFormat::Json => Json(OcrJsonResponse {
            model: result.model.to_string(),
            format: response_format,
            text: result.text,
            markdown: result.markdown,
            html: result.html,
            regions: result.regions,
            layout_blocks: result.layout_blocks,
            usage: result.usage,
        })
        .into_response(),
        OcrApiFormat::Text => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            result.text,
        )
            .into_response(),
        OcrApiFormat::Markdown => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/markdown; charset=utf-8"),
            )],
            result.markdown.unwrap_or(result.text),
        )
            .into_response(),
        OcrApiFormat::Html => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            )],
            result
                .html
                .unwrap_or_else(|| escaped_text_html(&result.text)),
        )
            .into_response(),
    }
}

fn escaped_text_html(text: &str) -> String {
    let mut html = String::from("<pre>");
    for character in text.chars() {
        match character {
            '&' => html.push_str("&amp;"),
            '<' => html.push_str("&lt;"),
            '>' => html.push_str("&gt;"),
            _ => html.push(character),
        }
    }
    html.push_str("</pre>");
    html
}

#[cfg(test)]
mod response_tests {
    use super::*;

    fn table_result(format: OcrApiFormat) -> orchion::OcrResult {
        orchion::OcrResult {
            model: "paddlepaddle/paddleocr-vl-1.6".parse().unwrap(),
            format: format.into(),
            text: "Product\tTotal\nKeyboard\t98.00".to_string(),
            markdown: Some("| Product | Total |\n| --- | --- |\n| Keyboard | 98.00 |".to_string()),
            html: Some("<table><tr><td>Product</td><td>Total</td></tr><tr><td>Keyboard</td><td>98.00</td></tr></table>".to_string()),
            regions: vec![orchion::OcrRegion {
                text: "Product\tTotal\nKeyboard\t98.00".to_string(),
                confidence: None,
                polygon: Vec::new(),
            }],
            layout_blocks: Vec::new(),
            usage: orchion::OcrUsage { input_pages: 1, output_tokens: None },
        }
    }

    fn document_result(format: OcrApiFormat) -> orchion::OcrResult {
        let mut result = table_result(format);
        result.text = "Project Notes\nReference number 4827.".to_string();
        result.markdown = Some("# Project Notes\n\nReference number 4827.".to_string());
        result.html = Some("<h1>Project Notes</h1>\n<p>Reference number 4827.</p>".to_string());
        result.regions[0].text.clone_from(&result.text);
        result
    }

    async fn assert_response_body(format: OcrApiFormat, result: orchion::OcrResult) -> String {
        let content_type = match format {
            OcrApiFormat::Json => "application/json",
            OcrApiFormat::Text => "text/plain; charset=utf-8",
            OcrApiFormat::Markdown => "text/markdown; charset=utf-8",
            OcrApiFormat::Html => "text/html; charset=utf-8",
        };
        let response = ocr_response(format, result.clone());
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], content_type);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        match format {
            OcrApiFormat::Json => {
                let json: serde_json::Value = serde_json::from_str(&body).unwrap();
                assert_eq!(json["format"], "json");
                assert_eq!(json["text"], result.text);
                assert_eq!(json["regions"][0]["text"], result.regions[0].text);
                assert_eq!(json["markdown"], result.markdown.unwrap());
                assert_eq!(json["html"], result.html.unwrap());
            }
            OcrApiFormat::Text => assert_eq!(body, result.text),
            OcrApiFormat::Markdown => assert_eq!(body, result.markdown.unwrap()),
            OcrApiFormat::Html => assert_eq!(body, result.html.unwrap()),
        }
        body
    }

    #[tokio::test]
    async fn each_ocr_format_selects_its_own_body_and_content_type_for_tables_and_documents() {
        for format in [
            OcrApiFormat::Json,
            OcrApiFormat::Text,
            OcrApiFormat::Markdown,
            OcrApiFormat::Html,
        ] {
            for result in [table_result(format), document_result(format)] {
                assert_response_body(format, result).await;
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires local OCR, layout, SLANet+ models and table dictionary"]
    async fn real_ocr_images_keep_their_format_through_the_http_response_boundary() {
        use orchion::{
            DevicePreference, KnownOcrModel, Ocr, OcrAssets, OcrOptions, TableStructureAssets,
        };
        use std::path::PathBuf;

        let root = PathBuf::from(std::env::var_os("ORCHION_OCR_TEST_MODELS").unwrap());
        let assets = OcrAssets::from_cache_layout(
            KnownOcrModel::PpOcrV6Tiny,
            root.join("PaddlePaddle/PP-OCRv6_tiny"),
            &root,
        )
        .with_layout(Some(
            root.join("PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"),
        ))
        .with_table_structure(Some(TableStructureAssets {
            model: std::env::var_os("ORCHION_OCR_TEST_TABLE_MODEL")
                .unwrap()
                .into(),
            dictionary: std::env::var_os("ORCHION_OCR_TEST_TABLE_DICTIONARY")
                .unwrap()
                .into(),
            table_type: "wireless".to_string(),
            score_threshold: 0.5,
            max_structure_length: 500,
        }));
        let engine = Ocr::load_with_assets_and_device(
            KnownOcrModel::PpOcrV6Tiny.id(),
            assets,
            DevicePreference::Cpu,
        )
        .await
        .unwrap();
        let fixtures =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../libs/orchion-ocr/tests/fixtures");
        let mut images = vec![
            (
                fixtures.join("table.png"),
                vec!["Product", "Keyboard", "98.00", "454.00"],
                true,
            ),
            (
                fixtures.join("merged-table.png"),
                vec!["Product", "Keyboard", "98.00", "454.00"],
                true,
            ),
            (
                fixtures.join("document.png"),
                vec!["Project Notes", "deployment check passed", "4827"],
                false,
            ),
        ];
        if let Some(image) = std::env::var_os("ORCHION_OCR_TEST_IMAGE") {
            images.push((image.into(), vec!["235.00", "14008.00", "2961.00"], true));
        }
        for (image, expected, has_table) in images {
            for format in [
                OcrApiFormat::Json,
                OcrApiFormat::Text,
                OcrApiFormat::Markdown,
            ] {
                let result = engine
                    .recognize_file_with(
                        &image,
                        OcrOptions {
                            response_format: format.into(),
                            layout_model: Some(KnownOcrModel::PpDocLayoutV3.id().parse().unwrap()),
                            ..OcrOptions::default()
                        },
                    )
                    .await
                    .unwrap();
                assert!(!result.text.contains("<table"));
                let markdown = result.markdown.as_deref();
                if let Some(markdown) = markdown {
                    assert!(!markdown.contains("<table"), "{markdown}");
                    assert_eq!(
                        markdown.lines().any(|line| line.starts_with('|')),
                        has_table
                    );
                }
                let body = assert_response_body(format, result).await;
                for text in &expected {
                    assert!(
                        body.contains(text),
                        "{} {format:?}: missing {text:?}: {body}",
                        image.display()
                    );
                }
                if format != OcrApiFormat::Json {
                    assert!(!body.contains("<table"), "{body}");
                }
                println!(
                    "[ocr-http-format] image={} format={format:?} passed",
                    image.display()
                );
            }
        }
    }

    #[tokio::test]
    async fn html_fallback_escapes_literal_recognized_text() {
        let mut result = table_result(OcrApiFormat::Html);
        result.html = None;
        result.text = "A & B <script>literal</script>".to_string();
        let response = ocr_response(OcrApiFormat::Html, result);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            &body[..],
            b"<pre>A &amp; B &lt;script&gt;literal&lt;/script&gt;</pre>"
        );
    }
}

pub(super) fn parse_ocr_task(value: &str) -> Result<OcrTask, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "ocr" => Ok(OcrTask::Ocr),
        "table" => Ok(OcrTask::Table),
        "formula" => Ok(OcrTask::Formula),
        "chart" => Ok(OcrTask::Chart),
        "spotting" => Ok(OcrTask::Spotting),
        "seal" => Ok(OcrTask::Seal),
        _ => Err(ApiError::invalid_request(
            "unsupported OCR task; supported tasks are ocr, table, formula, chart, spotting, and seal",
            Some("task"),
            Some("unsupported_ocr_parameter"),
        )),
    }
}
