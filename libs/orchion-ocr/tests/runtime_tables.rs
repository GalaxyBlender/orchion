//! Opt-in table screenshot regressions with real OCR models.
#![cfg(feature = "ocr")]

use orchion_core::{
    DevicePreference, KnownOcrModel, OcrOptions, OcrResponseFormat, OcrResult, OrchionError,
};
use orchion_ocr::{OcrAssets, OcrEngine, TableStructureAssets};
use std::path::PathBuf;
use std::time::Instant;

fn model_root() -> PathBuf {
    std::env::var_os("ORCHION_OCR_TEST_MODELS")
        .expect("set ORCHION_OCR_TEST_MODELS to the model cache root")
        .into()
}

fn device() -> DevicePreference {
    match std::env::var("ORCHION_OCR_TEST_DEVICE")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => DevicePreference::Cpu,
        "cuda" => DevicePreference::Cuda(Some(0)),
        other => panic!("unsupported test device: {other}"),
    }
}

fn screenshot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/table.png")
}

fn document() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/document.png")
}

fn merged_screenshot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/merged-table.png")
}

fn assert_markup_fields(result: &OcrResult) {
    assert_eq!(
        result.markdown.is_some(),
        matches!(
            result.format,
            OcrResponseFormat::Json | OcrResponseFormat::Markdown
        )
    );
    assert_eq!(
        result.html.is_some(),
        matches!(
            result.format,
            OcrResponseFormat::Json | OcrResponseFormat::Html
        )
    );
    if let Some(markdown) = &result.markdown {
        assert!(!markdown.contains("<table"), "{markdown}");
        let rows = markdown
            .lines()
            .filter(|line| line.starts_with('|'))
            .collect::<Vec<_>>();
        if !rows.is_empty() {
            assert!(rows.len() >= 2, "{markdown}");
            let width = rows[0].matches('|').count();
            assert!(
                rows.iter().all(|row| row.matches('|').count() == width),
                "{markdown}"
            );
            let separator = rows[1]
                .strip_prefix('|')
                .unwrap()
                .strip_suffix('|')
                .unwrap();
            assert!(
                separator.split('|').all(|cell| {
                    let cell = cell.trim().trim_matches(':');
                    !cell.is_empty() && cell.chars().all(|character| character == '-')
                }),
                "invalid Markdown table delimiter: {markdown}"
            );
        }
    }
}

fn assert_document_result(result: &OcrResult) {
    for text in [
        Some(result.text.as_str()),
        result.markdown.as_deref(),
        result.html.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        for expected in [
            "Project Notes",
            "deployment check passed",
            "4827",
            "Thank you",
        ] {
            assert!(text.contains(expected), "missing {expected:?}: {text}");
        }
        assert!(
            !text.contains("<table"),
            "non-table document gained a table: {text}"
        );
    }
    assert!(
        !result
            .layout_blocks
            .iter()
            .any(|block| block.label == "table")
    );
    assert!(
        !result.regions.is_empty(),
        "document OCR must retain recognized text regions"
    );
    match result.format {
        OcrResponseFormat::Json => {
            assert!(result.markdown.is_some());
            assert!(result.html.is_some());
        }
        OcrResponseFormat::Text => {
            assert!(result.markdown.is_none());
            assert!(result.html.is_none());
        }
        OcrResponseFormat::Markdown => assert!(result.markdown.is_some()),
        OcrResponseFormat::Html => assert!(
            result
                .html
                .as_deref()
                .is_some_and(|html| html.starts_with('<'))
        ),
    }
}

fn assert_content(text: &str) {
    for expected in [
        "Product",
        "Qty",
        "Price",
        "Keyboard",
        "49.00",
        "98.00",
        "Mouse",
        "19.00",
        "57.00",
        "Monitor",
        "299.00",
        "Grand Total",
        "454.00",
    ] {
        assert!(
            text.contains(expected),
            "table is missing {expected:?}: {text}"
        );
    }
}

fn assert_result(result: &OcrResult, has_layout: bool) {
    assert_content(&result.text);
    assert!(
        !result.text.contains("<table"),
        "plain text must not contain table markup: {}",
        result.text
    );
    assert!(!result.text.contains("<html"));
    if let Some(markdown) = &result.markdown {
        assert!(
            !markdown.contains("<table"),
            "Markdown must use pipe tables: {markdown}"
        );
    }
    if has_layout {
        assert!(
            !result.regions.is_empty(),
            "table text regions must not be lost"
        );
        assert_content(
            &result
                .regions
                .iter()
                .map(|region| region.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        assert!(
            result
                .layout_blocks
                .iter()
                .any(|block| block.label == "table")
        );
        assert!(
            result
                .regions
                .iter()
                .all(|region| !region.text.contains("<table"))
        );
    }
    match result.format {
        OcrResponseFormat::Json => {
            assert!(
                result
                    .markdown
                    .as_deref()
                    .is_some_and(|text| text.contains("| Product"))
            );
            let html = result.html.as_deref().unwrap();
            assert!(html.contains("<table>"));
            assert_content(html);
        }
        OcrResponseFormat::Text => {
            assert!(result.markdown.is_none());
            assert!(result.html.is_none());
            assert!(
                result.text.contains('\t'),
                "table columns must stay separated"
            );
        }
        OcrResponseFormat::Markdown => {
            let markdown = result.markdown.as_deref().unwrap();
            assert_content(markdown);
            assert!(
                markdown.contains("| Product"),
                "table must retain columns: {markdown}"
            );
            assert!(markdown.contains("| Keyboard"));
            let rows = markdown
                .lines()
                .filter(|line| line.starts_with('|'))
                .collect::<Vec<_>>();
            assert_eq!(
                rows.len(),
                6,
                "expected five table rows and a header separator"
            );
            assert!(
                rows.iter().all(|line| line.matches('|').count() == 5),
                "expected four columns, including empty cells: {markdown}"
            );
        }
        OcrResponseFormat::Html => {
            let html = result.html.as_deref().unwrap();
            assert_content(html);
            assert!(html.contains("<table>"));
            assert!(!html.contains("&lt;table"));
            assert_eq!(html.matches("<tr>").count(), 5);
            assert_eq!(
                html.matches("<td").count() + html.matches("<th").count(),
                20
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires local OCR, layout, SLANet+ models and table dictionary"]
#[allow(
    clippy::too_many_lines,
    reason = "the complete image/format matrix shares one loaded runtime per model"
)]
async fn traditional_table_screenshot() {
    let root = model_root();
    let table = TableStructureAssets {
        model: std::env::var_os("ORCHION_OCR_TEST_TABLE_MODEL")
            .expect("set ORCHION_OCR_TEST_TABLE_MODEL")
            .into(),
        dictionary: std::env::var_os("ORCHION_OCR_TEST_TABLE_DICTIONARY")
            .expect("set ORCHION_OCR_TEST_TABLE_DICTIONARY")
            .into(),
        table_type: "wireless".into(),
        score_threshold: 0.5,
        max_structure_length: 500,
    };
    let mut images = vec![screenshot(), merged_screenshot(), document()];
    if let Some(image) = std::env::var_os("ORCHION_OCR_TEST_IMAGE") {
        images.push(image.into());
    }
    for (model, directory) in [
        (KnownOcrModel::PpOcrV6Tiny, "PP-OCRv6_tiny"),
        (KnownOcrModel::PpOcrV5Server, "PP-OCRv5_server"),
    ] {
        let assets =
            OcrAssets::from_cache_layout(model, root.join("PaddlePaddle").join(directory), &root)
                .with_layout(Some(
                    root.join("PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"),
                ))
                .with_table_structure(Some(table.clone()));
        let engine = OcrEngine::load_with_assets(model, assets, device())
            .await
            .unwrap();
        for image in &images {
            for format in [
                OcrResponseFormat::Json,
                OcrResponseFormat::Text,
                OcrResponseFormat::Markdown,
                OcrResponseFormat::Html,
            ] {
                let start = Instant::now();
                let result = engine
                    .recognize_file_with(
                        image,
                        OcrOptions {
                            response_format: format,
                            layout_model: Some(KnownOcrModel::PpDocLayoutV3.id().parse().unwrap()),
                            ..OcrOptions::default()
                        },
                    )
                    .await;
                if format == OcrResponseFormat::Html {
                    assert!(
                        matches!(
                            result,
                            Err(OrchionError::UnsupportedCapability {
                                capability: "HTML response format",
                                ..
                            })
                        ),
                        "traditional HTML must be explicitly rejected: {result:?}"
                    );
                    continue;
                }
                let result = result.unwrap();
                assert_markup_fields(&result);
                if *image == screenshot() {
                    assert_result(&result, true);
                } else if *image == document() {
                    assert_document_result(&result);
                } else {
                    if *image == merged_screenshot() {
                        assert_content(&result.text);
                    }
                    assert!(
                        !result.text.trim().is_empty(),
                        "table text must not be lost"
                    );
                    assert!(!result.text.contains("<table"));
                    assert!(
                        result
                            .regions
                            .iter()
                            .all(|region| !region.text.contains("<table"))
                    );
                    assert!(
                        result
                            .layout_blocks
                            .iter()
                            .any(|block| block.label == "table")
                    );
                    if let Some(markdown) = &result.markdown {
                        assert!(markdown.lines().any(|line| line.starts_with('|')));
                        assert!(!markdown.contains("<table"), "{markdown}");
                    }
                    if format == OcrResponseFormat::Json {
                        assert!(result.html.as_deref().unwrap().contains("<table>"));
                    }
                }
                if format == OcrResponseFormat::Text {
                    println!("[ocr-table-text] {:?}", result.text);
                }
                println!(
                    "[ocr-table] model={} image={} format={format:?} seconds={:.3}",
                    model.id(),
                    image.display(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}

#[cfg(feature = "ocr-vl")]
#[tokio::test]
#[ignore = "requires local OCR-VL and layout models; use an optimized build"]
async fn vl_table_screenshot() {
    use orchion_core::OcrTask;

    let root = model_root();
    let model = KnownOcrModel::PaddleOcrVl16;
    let assets =
        OcrAssets::from_cache_layout(model, root.join("PaddlePaddle/PaddleOCR-VL-1.6"), &root)
            .with_layout(Some(
                root.join("PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"),
            ));
    let engine = OcrEngine::load_with_assets(model, assets, device())
        .await
        .unwrap();
    for image in [screenshot(), merged_screenshot(), document()] {
        for task in [OcrTask::Ocr, OcrTask::Table] {
            if image == document() && task == OcrTask::Table {
                continue;
            }
            for format in [
                OcrResponseFormat::Json,
                OcrResponseFormat::Text,
                OcrResponseFormat::Markdown,
                OcrResponseFormat::Html,
            ] {
                let start = Instant::now();
                let result = engine
                    .recognize_file_with(
                        &image,
                        OcrOptions {
                            task,
                            response_format: format,
                            layout_model: (task == OcrTask::Ocr)
                                .then(|| KnownOcrModel::PpDocLayoutV3.id().parse().unwrap()),
                            max_tokens: Some(512),
                        },
                    )
                    .await
                    .unwrap();
                assert_markup_fields(&result);
                if image == document() {
                    assert_document_result(&result);
                } else if image == screenshot() {
                    assert_result(&result, task == OcrTask::Ocr);
                } else {
                    assert_content(&result.text);
                    if let Some(markdown) = &result.markdown {
                        assert_content(markdown);
                        assert!(!markdown.contains("<table"), "{markdown}");
                        assert!(markdown.lines().any(|line| line.starts_with('|')));
                    }
                    if let Some(html) = &result.html {
                        assert_content(html);
                        assert!(html.contains("<table>"));
                    }
                }
                println!(
                    "[ocr-table] model={} image={} task={task:?} format={format:?} seconds={:.3}",
                    model.id(),
                    image.display(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
