#![cfg(any(feature = "ocr", feature = "ocr-vl"))]

use orchion::{
    ModelId, Ocr, OcrAssets, OcrEngine, OcrEngineFuture, OcrLimits, OcrOptions, OcrResponseFormat,
    OcrResult, OcrUsage,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[test]
fn explicit_ocr_assets_do_not_require_a_shared_directory_hierarchy() {
    let assets = OcrAssets::Traditional {
        detector: PathBuf::from("detector.onnx"),
        recognizer: PathBuf::from("runtime/recognizer.onnx"),
        dictionary: PathBuf::from("/opt/dictionaries/ppocr.txt"),
        layout: Some(PathBuf::from("models/layout/inference.onnx")),
    };

    let OcrAssets::Traditional {
        detector,
        recognizer,
        dictionary,
        layout,
    } = assets
    else {
        panic!("expected traditional OCR assets");
    };

    assert_eq!(detector, Path::new("detector.onnx"));
    assert_eq!(recognizer, Path::new("runtime/recognizer.onnx"));
    assert_eq!(dictionary, Path::new("/opt/dictionaries/ppocr.txt"));
    assert_eq!(
        layout.as_deref(),
        Some(Path::new("models/layout/inference.onnx"))
    );
}

#[test]
fn explicit_ocr_vl_assets_keep_main_model_and_layout_independent() {
    let assets = OcrAssets::VisionLanguage {
        model_dir: PathBuf::from("vl"),
        layout: Some(PathBuf::from("/srv/assets/layout.onnx")),
    };

    let OcrAssets::VisionLanguage { model_dir, layout } = assets else {
        panic!("expected OCR-VL assets");
    };

    assert_eq!(model_dir, Path::new("vl"));
    assert_eq!(
        layout.as_deref(),
        Some(Path::new("/srv/assets/layout.onnx"))
    );
}

#[test]
fn explicit_layout_assets_accept_a_top_level_model_path() {
    let assets = OcrAssets::Layout {
        model: PathBuf::from("layout.onnx"),
    };

    let OcrAssets::Layout { model } = assets else {
        panic!("expected layout assets");
    };

    assert_eq!(model, Path::new("layout.onnx"));
}

#[test]
fn cache_layout_assets_use_the_explicit_cache_root() {
    let assets = OcrAssets::from_cache_layout(
        orchion::KnownOcrModel::PpOcrV6Tiny,
        "main-model",
        "configured/cache",
    );
    let OcrAssets::Traditional {
        detector,
        recognizer,
        dictionary,
        layout,
    } = assets
    else {
        panic!("expected traditional OCR assets");
    };
    assert_eq!(
        detector,
        Path::new("configured/cache/PaddlePaddle/PP-OCRv6_tiny_det_onnx/inference.onnx")
    );
    assert_eq!(
        recognizer,
        Path::new("configured/cache/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/inference.onnx")
    );
    assert_eq!(dictionary, Path::new("main-model/ppocrv6_tiny_dict.txt"));
    assert_eq!(layout, None);
}

struct TestOcrEngine {
    model: ModelId,
    limits: Arc<Mutex<Option<OcrLimits>>>,
}

impl OcrEngine for TestOcrEngine {
    fn model(&self) -> &ModelId {
        &self.model
    }

    fn recognize_file_with_limits(
        &self,
        _path: PathBuf,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> OcrEngineFuture<'_, OcrResult> {
        *self.limits.lock().unwrap() = Some(limits);
        let model = self.model.clone();
        Box::pin(async move {
            Ok(OcrResult {
                model,
                format: options.response_format,
                text: "recognized".to_string(),
                markdown: None,
                html: None,
                regions: Vec::new(),
                layout_blocks: Vec::new(),
                usage: OcrUsage {
                    input_pages: 1,
                    output_tokens: None,
                },
            })
        })
    }
}

#[tokio::test]
async fn ocr_facade_dispatches_through_an_object_safe_engine_with_limits() {
    let limits_seen = Arc::new(Mutex::new(None));
    let ocr = Ocr::from_engine(Arc::new(TestOcrEngine {
        model: ModelId::parse("Acme/Test-OCR").unwrap(),
        limits: Arc::clone(&limits_seen),
    }));
    let limits = OcrLimits {
        max_pixels: Some(42),
    };
    let options = OcrOptions {
        response_format: OcrResponseFormat::Text,
        task: orchion::OcrTask::Ocr,
        layout_model: None,
        max_tokens: None,
    };

    let result = ocr
        .recognize_file_with_limits("image.png", options, limits)
        .await
        .unwrap();

    assert_eq!(ocr.model().as_str(), "Acme/Test-OCR");
    assert_eq!(result.text, "recognized");
    assert_eq!(*limits_seen.lock().unwrap(), Some(limits));
}
