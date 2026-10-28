//! Opt-in real-model memory regressions. See `docs/ocr-memory-testing.md`.
#![cfg(all(feature = "ocr", target_os = "linux"))]

#[cfg(feature = "ocr-vl")]
use orchion_core::OcrTask;
use orchion_core::{DevicePreference, KnownOcrModel, OcrOptions, OcrResponseFormat};
use orchion_ocr::{OcrAssets, OcrEngine};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const MIB: u64 = 1024 * 1024;
const GPU_GROWTH_ALLOWANCE: u64 = 128 * MIB;
const CPU_GROWTH_ALLOWANCE: u64 = 256 * MIB;

fn memory_bytes(device: DevicePreference) -> u64 {
    if matches!(device, DevicePreference::Cuda(_)) {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .expect("nvidia-smi is required for CUDA memory tests");
        assert!(output.status.success(), "nvidia-smi failed: {output:?}");
        let pid = std::process::id().to_string();
        return String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| line.split_once(','))
            .filter(|(process, _)| process.trim() == pid)
            .map(|(_, memory)| memory.trim().parse::<u64>().unwrap() * MIB)
            .sum();
    }

    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 1024
}

fn test_assets(model: KnownOcrModel, layout: bool) -> OcrAssets {
    let root = PathBuf::from(
        std::env::var_os("ORCHION_OCR_TEST_MODELS")
            .expect("set ORCHION_OCR_TEST_MODELS to the model cache root"),
    );
    let model_dir = match model {
        KnownOcrModel::PpOcrV5Server => "PaddlePaddle/PP-OCRv5_server",
        KnownOcrModel::PpOcrV6Tiny => "PaddlePaddle/PP-OCRv6_tiny",
        KnownOcrModel::PpDocLayoutV3 => "PaddlePaddle/PP-DocLayoutV3",
        KnownOcrModel::PaddleOcrVl16 => "PaddlePaddle/PaddleOCR-VL-1.6",
        _ => panic!("unsupported test fixture model"),
    };
    OcrAssets::from_cache_layout(model, root.join(model_dir), &root)
        .with_layout(layout.then(|| root.join("PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx")))
}

fn run_case(model: KnownOcrModel, device: DevicePreference, layout: bool, options: &[OcrOptions]) {
    let assets = test_assets(model, layout);
    let images = std::env::split_paths(
        &std::env::var_os("ORCHION_OCR_TEST_IMAGES")
            .expect("set ORCHION_OCR_TEST_IMAGES to one or more image paths"),
    )
    .collect::<Vec<_>>();
    assert!(!images.is_empty(), "at least one image is required");
    let repeats: usize = std::env::var("ORCHION_OCR_TEST_REPEATS")
        .unwrap_or_else(|_| "6".into())
        .parse()
        .unwrap();
    // Spotting upscales the image. Candle's allocator needs a second traversal
    // of the alternating prompt/vision shapes to settle its reusable pool.
    let warmup_rounds = if matches!(device, DevicePreference::Cuda(_))
        && matches!(
            model,
            KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16
        ) {
        2
    } else {
        1
    };
    assert!(repeats > warmup_rounds, "include requests after warmup");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(2)
        .build()
        .unwrap();
    runtime.block_on(async {
        let before_load = memory_bytes(device);
        let engine = OcrEngine::load_with_assets(model, assets, device)
            .await
            .unwrap();
        let allowance = if matches!(device, DevicePreference::Cuda(_)) {
            GPU_GROWTH_ALLOWANCE
        } else {
            CPU_GROWTH_ALLOWANCE
        };
        let mut warmed = None::<u64>;
        let mut warming_peak = 0;
        let mut expected_text = Vec::new();
        for round in 0..repeats {
            for (option_index, option) in options.iter().enumerate() {
                for (image_index, image) in images.iter().enumerate() {
                    let start = Instant::now();
                    let result = engine
                        .recognize_file_with(image, option.clone())
                        .await
                        .unwrap();
                    let elapsed = start.elapsed();
                    let used = memory_bytes(device);
                    let mut text_hash = DefaultHasher::new();
                    result.text.hash(&mut text_hash);
                    if matches!(device, DevicePreference::Cuda(_)) {
                        assert!(used > 0, "CUDA run must not silently fall back to CPU");
                    }
                    println!(
                        "[ocr-memory] model={} device={device:?} round={} task={:?} format={:?} image={} seconds={:.3} memory_mib={} text_hash={:016x} text_bytes={} regions={} blocks={}",
                        model.id(),
                        round + 1,
                        option.task,
                        option.response_format,
                        image.display(),
                        elapsed.as_secs_f64(),
                        used / MIB,
                        text_hash.finish(),
                        result.text.len(),
                        result.regions.len(),
                        result.layout_blocks.len(),
                    );
                    let key = option_index * images.len() + image_index;
                    if round < warmup_rounds {
                        warming_peak = warming_peak.max(used);
                    }
                    if round == 0 {
                        expected_text.push(result.text);
                    } else {
                        assert_eq!(result.text, expected_text[key], "identical request text changed");
                    }
                    if let Some(baseline) = warmed {
                        assert!(
                            used <= baseline + allowance,
                            "memory keeps growing after every input/task was warmed: {} -> {} MiB",
                            baseline / MIB,
                            used / MIB,
                        );
                    }
                }
            }
            if round + 1 == warmup_rounds {
                warmed = Some(warming_peak);
            }
        }
        drop(engine);
        println!(
            "[ocr-memory] model={} dropped before_load_mib={} after_drop_mib={}",
            model.id(),
            before_load / MIB,
            memory_bytes(device) / MIB,
        );
    });
}

fn plain_options() -> Vec<OcrOptions> {
    vec![OcrOptions::default()]
}

fn layout_options() -> Vec<OcrOptions> {
    [OcrResponseFormat::Json, OcrResponseFormat::Markdown]
        .into_iter()
        .map(|response_format| OcrOptions {
            response_format,
            layout_model: Some(KnownOcrModel::PpDocLayoutV3.id().parse().unwrap()),
            ..OcrOptions::default()
        })
        .collect()
}

#[test]
#[ignore = "requires local OCR models and images"]
fn cpu_traditional_server() {
    run_case(
        KnownOcrModel::PpOcrV5Server,
        DevicePreference::Cpu,
        false,
        &plain_options(),
    );
}

#[test]
#[ignore = "requires local OCR models and images"]
fn cpu_traditional_tiny() {
    run_case(
        KnownOcrModel::PpOcrV6Tiny,
        DevicePreference::Cpu,
        false,
        &plain_options(),
    );
}

#[test]
#[ignore = "requires local OCR models and images"]
fn cpu_layout() {
    run_case(
        KnownOcrModel::PpDocLayoutV3,
        DevicePreference::Cpu,
        true,
        &layout_options(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, local OCR models and images"]
fn cuda_traditional_server() {
    run_case(
        KnownOcrModel::PpOcrV5Server,
        DevicePreference::Cuda(Some(0)),
        false,
        &plain_options(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, local OCR models and images"]
fn cuda_traditional_tiny() {
    run_case(
        KnownOcrModel::PpOcrV6Tiny,
        DevicePreference::Cuda(Some(0)),
        false,
        &plain_options(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, local OCR models and images"]
fn cuda_layout() {
    run_case(
        KnownOcrModel::PpDocLayoutV3,
        DevicePreference::Cuda(Some(0)),
        true,
        &layout_options(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, local OCR models and images"]
fn cuda_traditional_layout_server() {
    run_case(
        KnownOcrModel::PpOcrV5Server,
        DevicePreference::Cuda(Some(0)),
        true,
        &layout_options(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA, local OCR models and images"]
fn cuda_traditional_layout_tiny() {
    run_case(
        KnownOcrModel::PpOcrV6Tiny,
        DevicePreference::Cuda(Some(0)),
        true,
        &layout_options(),
    );
}

#[cfg(feature = "ocr-vl")]
fn vl_options() -> Vec<OcrOptions> {
    [
        OcrTask::Ocr,
        OcrTask::Table,
        OcrTask::Formula,
        OcrTask::Chart,
        OcrTask::Spotting,
        OcrTask::Seal,
    ]
    .into_iter()
    .map(|task| OcrOptions {
        task,
        max_tokens: Some(32),
        ..OcrOptions::default()
    })
    .collect()
}

#[cfg(all(feature = "ocr-vl", feature = "cuda"))]
#[test]
#[ignore = "requires CUDA, local OCR-VL models and images"]
fn cuda_vl_tasks() {
    run_case(
        KnownOcrModel::PaddleOcrVl16,
        DevicePreference::Cuda(Some(0)),
        false,
        &vl_options(),
    );
}

#[cfg(feature = "ocr-vl")]
#[test]
#[ignore = "requires local OCR-VL models and images"]
fn cpu_vl_task() {
    run_case(
        KnownOcrModel::PaddleOcrVl16,
        DevicePreference::Cpu,
        false,
        &[OcrOptions {
            max_tokens: Some(4),
            ..OcrOptions::default()
        }],
    );
}

#[cfg(all(feature = "ocr-vl", feature = "cuda"))]
#[test]
#[ignore = "requires CUDA, local OCR-VL models and images"]
fn cuda_vl_layout() {
    let options: Vec<_> = [
        OcrResponseFormat::Json,
        OcrResponseFormat::Markdown,
        OcrResponseFormat::Html,
    ]
    .into_iter()
    .map(|response_format| OcrOptions {
        response_format,
        max_tokens: Some(32),
        layout_model: Some(KnownOcrModel::PpDocLayoutV3.id().parse().unwrap()),
        ..OcrOptions::default()
    })
    .collect();
    run_case(
        KnownOcrModel::PaddleOcrVl16,
        DevicePreference::Cuda(Some(0)),
        true,
        &options,
    );
}
