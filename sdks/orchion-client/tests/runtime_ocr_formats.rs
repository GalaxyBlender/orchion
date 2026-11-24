//! Opt-in HTTP contract checks against a running server with real OCR models.
#![cfg(feature = "ocr")]

use orchion_client::ocr::OcrJsonResponse;
use reqwest::header::CONTENT_TYPE;
use reqwest::multipart::{Form, Part};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

struct ImageCase {
    path: PathBuf,
    expected: &'static [&'static str],
    table: bool,
}

fn check_content(text: &str, image: &ImageCase) -> Result<(), String> {
    for word in image.expected {
        if !text.contains(word) {
            return Err(format!("missing recognition content {word:?}"));
        }
    }
    Ok(())
}

fn check_markdown(text: &str, image: &ImageCase) -> Result<(), String> {
    check_content(text, image)?;
    if text.to_ascii_lowercase().contains("<table") {
        return Err("Markdown returned an HTML table".to_string());
    }
    let rows = text
        .lines()
        .filter(|line| line.starts_with('|'))
        .collect::<Vec<_>>();
    if image.table {
        if rows.len() < 3 {
            return Err("Markdown lost the table rows".to_string());
        }
        let width = rows[0].matches('|').count();
        if rows.iter().any(|row| row.matches('|').count() != width) {
            return Err("Markdown has inconsistent column counts".to_string());
        }
        let separator = rows[1].trim().trim_matches('|');
        if separator.split('|').any(|cell| {
            let cell = cell.trim().trim_matches(':');
            cell.is_empty() || !cell.chars().all(|character| character == '-')
        }) {
            return Err("Markdown table has an invalid delimiter row".to_string());
        }
    } else if !rows.is_empty() {
        return Err("a non-table image acquired a Markdown table".to_string());
    }
    Ok(())
}

fn check_body(format: &str, body: &str, image: &ImageCase, structured: bool) -> Result<(), String> {
    match format {
        "json" => {
            // Decode the real response with the production SDK's public type.
            let result: OcrJsonResponse =
                serde_json::from_str(body).map_err(|error| error.to_string())?;
            if result.format != orchion_client::ocr::OcrResponseFormatValue::Json {
                return Err("JSON metadata reports a different response format".to_string());
            }
            check_content(&result.text, image)?;
            if result.text.contains("<table")
                || result
                    .regions
                    .iter()
                    .any(|region| region.text.contains("<table"))
            {
                return Err("JSON plain-text fields contain table HTML".to_string());
            }
            if let Some(html) = &result.html {
                check_body("html", html, image, false)?;
            } else if structured {
                return Err("structured JSON is missing its HTML field".to_string());
            }
            if let Some(markdown) = &result.markdown {
                check_markdown(markdown, image)?;
            } else if structured {
                return Err("structured JSON is missing its Markdown field".to_string());
            }
            Ok(())
        }
        "markdown" => check_markdown(body, image),
        "text" => {
            check_content(body, image)?;
            if body.contains("<table") || body.contains("<html") {
                return Err("text response contains HTML table/document markup".to_string());
            }
            if image.table && !body.contains('\t') {
                return Err("text table lost its column separators".to_string());
            }
            Ok(())
        }
        "html" => {
            check_content(body, image)?;
            if !body.trim_start().starts_with('<') || body.contains("<script") {
                return Err("HTML response is not a sanitized HTML document".to_string());
            }
            if body.contains("<table") != image.table {
                return Err("HTML table presence does not match the image".to_string());
            }
            Ok(())
        }
        _ => unreachable!("the test enumerates its response formats"),
    }
}

#[tokio::test]
#[ignore = "set ORCHION_OCR_HTTP_BASE_URL to a running real-model server"]
#[allow(
    clippy::too_many_lines,
    reason = "keep the complete live model/image/task/format matrix visible in one request workflow"
)]
async fn all_ocr_http_formats_for_table_and_non_table_images() {
    let base = std::env::var("ORCHION_OCR_HTTP_BASE_URL").expect("set ORCHION_OCR_HTTP_BASE_URL");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(180))
        .build()
        .unwrap();
    let fixtures =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../libs/orchion-ocr/tests/fixtures");
    let mut images = vec![
        ImageCase {
            path: fixtures.join("table.png"),
            expected: &["Product", "Keyboard", "98.00", "454.00"],
            table: true,
        },
        ImageCase {
            path: fixtures.join("merged-table.png"),
            expected: &["Product", "Keyboard", "98.00", "454.00"],
            table: true,
        },
        ImageCase {
            path: fixtures.join("document.png"),
            expected: &["Project Notes", "deployment check passed", "4827"],
            table: false,
        },
    ];
    if let Some(path) = std::env::var_os("ORCHION_OCR_TEST_IMAGE") {
        images.push(ImageCase {
            path: path.into(),
            expected: &["235.00", "14008.00", "2961.00"],
            table: true,
        });
    }
    let models: Value = client
        .get(format!("{}/v1/models", base.trim_end_matches('/')))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for model in models["data"]
        .as_array()
        .expect("model list must have data")
    {
        if model["type"] != "ocr" {
            continue;
        }
        let id = model["id"].as_str().unwrap();
        let capabilities = model["capabilities"].as_array().unwrap();
        let supports = |name: &str| capabilities.iter().any(|value| value == name);
        let vision_language = supports("ocr_vision_language");
        for image in &images {
            for task in ["ocr", "table"] {
                if task == "table" && (!vision_language || !image.table) {
                    continue;
                }
                for (format, content_type) in [
                    ("json", "application/json"),
                    ("text", "text/plain; charset=utf-8"),
                    ("markdown", "text/markdown; charset=utf-8"),
                    ("html", "text/html; charset=utf-8"),
                ] {
                    let supported = match format {
                        "markdown" => {
                            supports("ocr_markdown") || vision_language && task == "table"
                        }
                        "html" => supports("ocr_html") || vision_language && task == "table",
                        _ => true,
                    };
                    let label = format!(
                        "{id} {} task={task} format={format}",
                        image.path.file_name().unwrap().to_string_lossy()
                    );
                    let part = Part::bytes(std::fs::read(&image.path).unwrap()).file_name(
                        image
                            .path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .into_owned(),
                    );
                    let mut form = Form::new()
                        .part("file", part)
                        .text("model", id.to_string())
                        .text("task", task)
                        .text("response_format", format);
                    if vision_language {
                        form = form.text("max_tokens", "2048");
                    }
                    let response = client
                        .post(format!("{}/v1/ocr", base.trim_end_matches('/')))
                        .multipart(form)
                        .send()
                        .await
                        .unwrap();
                    let status = response.status();
                    let actual_type = response
                        .headers()
                        .get(CONTENT_TYPE)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();
                    let body = response.text().await.unwrap();
                    let check = if supported {
                        if status != reqwest::StatusCode::OK || actual_type != content_type {
                            Err(format!(
                                "expected 200 {content_type}, got {status} {actual_type}"
                            ))
                        } else {
                            check_body(
                                format,
                                &body,
                                image,
                                supports("ocr_layout") || task == "table",
                            )
                        }
                    } else {
                        let error: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                        if status == reqwest::StatusCode::BAD_REQUEST
                            && actual_type == "application/json"
                            && error["error"]["code"] == "unsupported_response_format"
                        {
                            Ok(())
                        } else {
                            Err(format!(
                                "expected unsupported_response_format 400, got {status}"
                            ))
                        }
                    };
                    checked += 1;
                    match check {
                        Ok(()) => println!("[ocr-http-matrix] PASS {label} status={status}"),
                        Err(error) => {
                            println!("[ocr-http-matrix] FAIL {label}: {error}");
                            failures.push(format!("{label}: {error}"));
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "the server exposes no OCR models");
    assert!(
        failures.is_empty(),
        "{} of {checked} cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("[ocr-http-matrix] all {checked} cases passed");
}
