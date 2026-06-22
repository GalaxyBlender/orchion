#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
use crate::device::ProviderPolicy;
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
use orchion_core::ModelSpec;
#[cfg(feature = "ocr")]
use orchion_core::OcrPoint;
#[cfg(feature = "ocr-vl")]
use orchion_core::OcrTask;
use orchion_core::{DevicePreference, KnownOcrModel, OcrOptions, OcrResult, OrchionError, Result};
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
use orchion_core::{ModelId, OcrLayoutBlock, OcrResponseFormat, OcrUsage};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "ocr-vl")]
const DEFAULT_VL_MAX_TOKENS: usize = 4096;

#[derive(Clone)]
pub enum LoadedOcrRuntime {
    #[cfg(feature = "ocr")]
    Traditional(Arc<TraditionalRuntime>),
    #[cfg(feature = "ocr")]
    Layout(Arc<Mutex<oar_ocr::oarocr::OARStructure>>),
    #[cfg(feature = "ocr-vl")]
    OcrVl(Arc<Mutex<OcrVlRuntime>>),
    #[cfg(any(not(feature = "ocr"), not(feature = "ocr-vl")))]
    Unsupported {
        model: KnownOcrModel,
        capability: &'static str,
    },
}

#[cfg(feature = "ocr")]
pub struct TraditionalRuntime {
    ocr: Mutex<oar_ocr::oarocr::OAROCR>,
    structure: Option<Mutex<oar_ocr::oarocr::OARStructure>>,
}

#[cfg(feature = "ocr-vl")]
pub struct OcrVlRuntime {
    model: oar_ocr_vl::PaddleOcrVl,
    layout_predictor: Option<oar_ocr::predictors::LayoutDetectionPredictor>,
}

pub async fn load_runtime(
    model: KnownOcrModel,
    model_dir: PathBuf,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    tokio::task::spawn_blocking(move || load_runtime_blocking(model, &model_dir, device))
        .await
        .map_err(|error| OrchionError::BlockingTask {
            message: error.to_string(),
        })?
}

fn load_runtime_blocking(
    model: KnownOcrModel,
    model_dir: &Path,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    match model {
        KnownOcrModel::PpOcrV5Mobile
        | KnownOcrModel::PpOcrV5Server
        | KnownOcrModel::PpOcrV6Tiny
        | KnownOcrModel::PpOcrV6Small
        | KnownOcrModel::PpOcrV6Medium => load_traditional_runtime(model, model_dir, device),
        KnownOcrModel::PpDocLayoutV3 => load_layout_runtime(model, model_dir, device),
        KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16 => {
            load_vl_runtime(model, model_dir, device)
        }
    }
}

pub async fn run_ocr(
    model: KnownOcrModel,
    runtime: LoadedOcrRuntime,
    image_path: &Path,
    options: OcrOptions,
) -> Result<OcrResult> {
    let image_path = image_path.to_path_buf();
    tokio::task::spawn_blocking(move || run_ocr_blocking(model, &runtime, &image_path, &options))
        .await
        .map_err(|error| OrchionError::BlockingTask {
            message: error.to_string(),
        })?
}

fn run_ocr_blocking(
    model: KnownOcrModel,
    runtime: &LoadedOcrRuntime,
    image_path: &Path,
    options: &OcrOptions,
) -> Result<OcrResult> {
    match (model, runtime) {
        #[cfg(feature = "ocr")]
        (
            KnownOcrModel::PpOcrV5Mobile
            | KnownOcrModel::PpOcrV5Server
            | KnownOcrModel::PpOcrV6Tiny
            | KnownOcrModel::PpOcrV6Small
            | KnownOcrModel::PpOcrV6Medium,
            LoadedOcrRuntime::Traditional(runtime),
        ) => run_traditional_ocr(model, runtime, image_path, options),
        #[cfg(feature = "ocr")]
        (KnownOcrModel::PpDocLayoutV3, LoadedOcrRuntime::Layout(structure)) => {
            run_layout_ocr(model, structure, image_path, options)
        }
        #[cfg(feature = "ocr-vl")]
        (
            KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16,
            LoadedOcrRuntime::OcrVl(vl),
        ) => run_vl_ocr(model, vl, image_path, options),
        #[cfg(any(not(feature = "ocr"), not(feature = "ocr-vl")))]
        (_, LoadedOcrRuntime::Unsupported { model, capability }) => {
            Err(OrchionError::UnsupportedCapability {
                model: model.id(),
                capability,
            })
        }
        _ => Err(OrchionError::Inference {
            source: anyhow::anyhow!("loaded OCR runtime does not match model `{}`", model.id()),
        }),
    }
}

#[cfg(feature = "ocr")]
fn load_traditional_runtime(
    model: KnownOcrModel,
    model_dir: &Path,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    use oar_ocr::oarocr::OAROCRBuilder;

    let assets = traditional_assets(model, model_dir)?;
    let builder = OAROCRBuilder::new(
        assets.detector.clone(),
        assets.recognizer.clone(),
        assets.dictionary.clone(),
    )
    .ort_session(ort_session_config(ProviderPolicy::for_model(model, device)));
    let ocr = builder.build().map_err(model_load_error)?;
    let structure = load_related_structure_runtime(model_dir, assets, device)?;
    Ok(LoadedOcrRuntime::Traditional(Arc::new(
        TraditionalRuntime {
            ocr: Mutex::new(ocr),
            structure: structure.map(Mutex::new),
        },
    )))
}

#[cfg(not(feature = "ocr"))]
fn load_traditional_runtime(
    model: KnownOcrModel,
    _model_dir: &Path,
    _device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    Ok(LoadedOcrRuntime::Unsupported {
        model,
        capability: "ocr",
    })
}

#[cfg(feature = "ocr")]
fn load_layout_runtime(
    model: KnownOcrModel,
    model_dir: &Path,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    let layout_model = layout_model_path(model_dir)?;
    let structure = build_structure_runtime(model, layout_model, None, device)?;
    Ok(LoadedOcrRuntime::Layout(Arc::new(Mutex::new(structure))))
}

#[cfg(feature = "ocr")]
fn load_related_structure_runtime(
    model_dir: &Path,
    assets: TraditionalAssets,
    device: DevicePreference,
) -> Result<Option<oar_ocr::oarocr::OARStructure>> {
    let layout_dir = related_model_dir(model_dir, KnownOcrModel::PpDocLayoutV3)?;
    let layout_path = layout_model_path(&layout_dir)?;
    if !layout_path.is_file() {
        return Ok(None);
    }

    build_structure_runtime(
        KnownOcrModel::PpDocLayoutV3,
        layout_path,
        Some(assets),
        device,
    )
    .map(Some)
}

#[cfg(feature = "ocr")]
fn build_structure_runtime(
    provider_model: KnownOcrModel,
    layout_model: PathBuf,
    ocr_assets: Option<TraditionalAssets>,
    device: DevicePreference,
) -> Result<oar_ocr::oarocr::OARStructure> {
    use oar_ocr::oarocr::OARStructureBuilder;

    let builder = OARStructureBuilder::new(layout_model)
        .layout_model_name("PP-DocLayout_plus-L")
        .ort_session(ort_session_config(ProviderPolicy::for_model(
            provider_model,
            device,
        )));
    let builder = if let Some(assets) = ocr_assets {
        builder.with_ocr(assets.detector, assets.recognizer, assets.dictionary)
    } else {
        builder
    };
    builder.build().map_err(model_load_error)
}

#[cfg(not(feature = "ocr"))]
fn load_layout_runtime(
    model: KnownOcrModel,
    _model_dir: &Path,
    _device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    Ok(LoadedOcrRuntime::Unsupported {
        model,
        capability: "ocr",
    })
}

#[cfg(feature = "ocr-vl")]
fn load_vl_runtime(
    model: KnownOcrModel,
    model_dir: &Path,
    device_preference: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    use oar_ocr_vl::{PaddleOcrVl, utils::parse_device};

    let provider_policy = ProviderPolicy::for_model(model, device_preference);
    let device = candle_device(provider_policy);
    let candle_device = parse_device(&device).map_err(model_load_error)?;
    let vl = PaddleOcrVl::from_dir(model_dir, candle_device).map_err(model_load_error)?;
    let layout_predictor = load_default_layout_predictor(
        model_dir,
        ProviderPolicy::for_model(KnownOcrModel::PpDocLayoutV3, device_preference),
    )?;

    Ok(LoadedOcrRuntime::OcrVl(Arc::new(Mutex::new(
        OcrVlRuntime {
            model: vl,
            layout_predictor,
        },
    ))))
}

#[cfg(not(feature = "ocr-vl"))]
fn load_vl_runtime(
    model: KnownOcrModel,
    _model_dir: &Path,
    _device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    Ok(LoadedOcrRuntime::Unsupported {
        model,
        capability: "ocr-vl",
    })
}

#[cfg(feature = "ocr")]
fn run_traditional_ocr(
    model: KnownOcrModel,
    runtime: &TraditionalRuntime,
    image_path: &Path,
    options: &OcrOptions,
) -> Result<OcrResult> {
    use oar_ocr::utils::load_image;

    if options.layout_model.is_some() {
        let structure = runtime.structure.as_ref().ok_or_else(|| {
            model_load_error(anyhow::anyhow!(
                "OCR layout model is configured but not loaded for `{}`",
                model.id()
            ))
        })?;
        return run_layout_ocr(model, structure, image_path, options);
    }

    let image = load_image(image_path).map_err(inference_error)?;
    let ocr = runtime
        .ocr
        .lock()
        .map_err(|error| OrchionError::Inference {
            source: anyhow::anyhow!("OCR runtime lock poisoned: {error}"),
        })?;
    let mut pages = ocr.predict(vec![image]).map_err(inference_error)?;
    let page = pages.pop().ok_or_else(|| OrchionError::Inference {
        source: anyhow::anyhow!("OCR returned no pages"),
    })?;

    let regions = page
        .text_regions
        .iter()
        .filter(|region| region.text.is_some())
        .map(|region| {
            orchion_region(
                &region.bounding_box,
                region.text.as_deref().unwrap_or_default(),
                region.confidence,
            )
        })
        .collect::<Vec<_>>();
    let text = page.concatenated_text("\n");

    Ok(base_result(
        model,
        options.response_format,
        text,
        None,
        None,
        regions,
        Vec::new(),
    ))
}

#[cfg(feature = "ocr")]
fn run_layout_ocr(
    model: KnownOcrModel,
    structure: &Mutex<oar_ocr::oarocr::OARStructure>,
    image_path: &Path,
    options: &OcrOptions,
) -> Result<OcrResult> {
    let structure = structure.lock().map_err(|error| OrchionError::Inference {
        source: anyhow::anyhow!("OCR layout runtime lock poisoned: {error}"),
    })?;
    let result = structure.predict(image_path).map_err(inference_error)?;

    let text = result
        .layout_elements
        .iter()
        .filter_map(|element| element.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    let markdown =
        (options.response_format == OcrResponseFormat::Markdown).then(|| result.to_markdown());
    let blocks = result
        .layout_elements
        .iter()
        .map(|element| OcrLayoutBlock {
            label: element
                .label
                .clone()
                .unwrap_or_else(|| format!("{:?}", element.element_type)),
            confidence: Some(element.confidence),
            polygon: polygon_points(&element.bbox),
        })
        .collect();

    Ok(base_result(
        model,
        options.response_format,
        text,
        markdown,
        None,
        Vec::new(),
        blocks,
    ))
}

#[cfg(feature = "ocr-vl")]
fn run_vl_ocr(
    model: KnownOcrModel,
    runtime: &Mutex<OcrVlRuntime>,
    image_path: &Path,
    options: &OcrOptions,
) -> Result<OcrResult> {
    let image = image::open(image_path)
        .map_err(|error| OrchionError::Inference {
            source: anyhow::anyhow!(error),
        })?
        .to_rgb8();
    let max_tokens = options.max_tokens.unwrap_or(DEFAULT_VL_MAX_TOKENS);
    let runtime = runtime.lock().map_err(|error| OrchionError::Inference {
        source: anyhow::anyhow!("OCR-VL runtime lock poisoned: {error}"),
    })?;

    if should_use_vl_layout_pipeline(options) {
        let layout_predictor = runtime.layout_predictor.as_ref().ok_or_else(|| {
            model_load_error(anyhow::anyhow!("OCR-VL layout model is not loaded"))
        })?;
        return run_vl_layout_ocr(model, &runtime, layout_predictor, image, options);
    }

    let task = vl_task(options.task);
    let text = runtime
        .model
        .generate(&[image], &[task], max_tokens)
        .into_iter()
        .next()
        .ok_or_else(|| OrchionError::Inference {
            source: anyhow::anyhow!("OCR-VL returned no results"),
        })?
        .map_err(inference_error)?;
    Ok(base_result(
        model,
        options.response_format,
        text,
        None,
        None,
        Vec::new(),
        Vec::new(),
    ))
}

#[cfg(feature = "ocr-vl")]
fn run_vl_layout_ocr(
    model: KnownOcrModel,
    runtime: &OcrVlRuntime,
    layout_predictor: &oar_ocr::predictors::LayoutDetectionPredictor,
    image: image::RgbImage,
    options: &OcrOptions,
) -> Result<OcrResult> {
    use oar_ocr_vl::{DocParser, DocParserConfig};

    let parser = DocParser::with_config(
        &runtime.model,
        DocParserConfig {
            max_tokens: options.max_tokens.unwrap_or(DEFAULT_VL_MAX_TOKENS),
            ..DocParserConfig::default()
        },
    );
    let structure = parser
        .parse(layout_predictor, image)
        .map_err(inference_error)?;
    let text = structure
        .layout_elements
        .iter()
        .filter_map(|element| element.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    let markdown = (options.response_format == OcrResponseFormat::Markdown)
        .then(|| html_tables_to_markdown(&structure.to_markdown()));
    let html = (options.response_format == OcrResponseFormat::Html).then(|| {
        oar_ocr_vl::utils::to_markdown_openocr(
            &structure.layout_elements,
            &parser.config().markdown_ignore_labels,
            true,
        )
    });
    let blocks = structure
        .layout_elements
        .iter()
        .map(|element| OcrLayoutBlock {
            label: element
                .label
                .clone()
                .unwrap_or_else(|| format!("{:?}", element.element_type)),
            confidence: Some(element.confidence),
            polygon: bbox_points(&element.bbox),
        })
        .collect();

    Ok(base_result(
        model,
        options.response_format,
        text,
        markdown,
        html,
        Vec::new(),
        blocks,
    ))
}

#[cfg(feature = "ocr-vl")]
fn should_use_vl_layout_pipeline(options: &OcrOptions) -> bool {
    options.layout_model.is_some()
        || matches!(
            options.response_format,
            OcrResponseFormat::Markdown | OcrResponseFormat::Html
        )
}

#[cfg(feature = "ocr-vl")]
fn html_tables_to_markdown(input: &str) -> String {
    htmd::convert(input).unwrap_or_else(|_| input.to_string())
}

#[cfg(feature = "ocr-vl")]
fn load_default_layout_predictor(
    vl_model_dir: &Path,
    provider_policy: ProviderPolicy,
) -> Result<Option<oar_ocr::predictors::LayoutDetectionPredictor>> {
    let layout_model = KnownOcrModel::PpDocLayoutV3;
    let layout_dir = related_model_dir(vl_model_dir, layout_model)?;
    let layout_path = layout_model_path(&layout_dir)?;
    if !layout_path.is_file() {
        return Ok(None);
    }

    let predictor = oar_ocr::predictors::LayoutDetectionPredictor::builder()
        .model_name("pp-doclayoutv3")
        .with_ort_config(ort_session_config(provider_policy))
        .build(layout_path)
        .map_err(model_load_error)?;
    Ok(Some(predictor))
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn related_model_dir(vl_model_dir: &Path, model: KnownOcrModel) -> Result<PathBuf> {
    let Some(model_root) = vl_model_dir.parent().and_then(Path::parent) else {
        return Err(model_load_error(anyhow::anyhow!(
            "cannot derive shared model root from OCR model cache path `{}`",
            vl_model_dir.display()
        )));
    };
    Ok(model.cache_path(model_root))
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn model_cache_root(model_dir: &Path) -> Result<&Path> {
    model_dir.parent().and_then(Path::parent).ok_or_else(|| {
        model_load_error(anyhow::anyhow!(
            "cannot derive shared model root from OCR model cache path `{}`",
            model_dir.display()
        ))
    })
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn layout_model_path(model_dir: &Path) -> Result<PathBuf> {
    let root = model_cache_root(model_dir)?;
    let repo = KnownOcrModel::PpDocLayoutV3
        .pp_doclayoutv3_onnx_repo()
        .expect("PP-DocLayoutV3 has an ONNX repo");
    Ok(repo
        .split('/')
        .fold(root.to_path_buf(), |path, segment| path.join(segment))
        .join("inference.onnx"))
}

#[cfg(feature = "ocr")]
struct TraditionalAssets {
    detector: PathBuf,
    recognizer: PathBuf,
    dictionary: PathBuf,
}

#[cfg(feature = "ocr")]
fn traditional_assets(model: KnownOcrModel, model_dir: &Path) -> Result<TraditionalAssets> {
    if model == KnownOcrModel::PpOcrV5Mobile {
        let root = model_cache_root(model_dir)?;
        let registry_dir = root.join("greatv").join("oar-ocr");
        return Ok(TraditionalAssets {
            detector: registry_dir.join("pp-ocrv5_mobile_det.onnx"),
            recognizer: registry_dir.join("pp-ocrv5_mobile_rec.onnx"),
            dictionary: registry_dir.join("ppocrv5_dict.txt"),
        });
    }

    if let (Some(detector_repo), Some(recognizer_repo)) =
        (model.pp_ocr_detector_repo(), model.pp_ocr_recognizer_repo())
    {
        let root = model_cache_root(model_dir)?;
        let detector = detector_repo
            .split('/')
            .fold(root.to_path_buf(), |path, segment| path.join(segment))
            .join("inference.onnx");
        let recognizer = recognizer_repo
            .split('/')
            .fold(root.to_path_buf(), |path, segment| path.join(segment))
            .join("inference.onnx");
        let dictionary = model.dictionary_file().ok_or_else(|| {
            model_load_error(anyhow::anyhow!(
                "missing OCR dictionary metadata for `{}`",
                model.id()
            ))
        })?;
        return Ok(TraditionalAssets {
            detector,
            recognizer,
            dictionary: model_dir.join(dictionary),
        });
    }

    Err(model_load_error(anyhow::anyhow!(
        "missing OCR component repo metadata for `{}`",
        model.id()
    )))
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn ort_session_config(policy: ProviderPolicy) -> oar_ocr::core::config::OrtSessionConfig {
    use oar_ocr::core::config::{OrtExecutionProvider, OrtSessionConfig};

    let provider = match policy {
        ProviderPolicy::OrtCpu => OrtExecutionProvider::CPU,
        ProviderPolicy::OrtCuda(index) => OrtExecutionProvider::CUDA {
            device_id: index.and_then(|value| i32::try_from(value).ok()),
            gpu_mem_limit: None,
            arena_extend_strategy: None,
            cudnn_conv_algo_search: None,
            cudnn_conv_use_max_workspace: None,
        },
        ProviderPolicy::OrtWebGpu => OrtExecutionProvider::WebGPU,
        ProviderPolicy::CandleCpu | ProviderPolicy::CandleCuda(_) | ProviderPolicy::CandleMetal => {
            OrtExecutionProvider::CPU
        }
    };

    OrtSessionConfig::new().with_execution_providers(vec![provider])
}

#[cfg(feature = "ocr-vl")]
fn candle_device(policy: ProviderPolicy) -> String {
    match policy {
        ProviderPolicy::CandleCpu => "cpu".to_string(),
        ProviderPolicy::CandleCuda(None) => "cuda".to_string(),
        ProviderPolicy::CandleCuda(Some(index)) => format!("cuda:{index}"),
        ProviderPolicy::CandleMetal => "metal".to_string(),
        ProviderPolicy::OrtCpu | ProviderPolicy::OrtCuda(_) | ProviderPolicy::OrtWebGpu => {
            "cpu".to_string()
        }
    }
}

#[cfg(feature = "ocr-vl")]
fn vl_task(task: OcrTask) -> oar_ocr_vl::PaddleOcrVlTask {
    match task {
        OcrTask::Ocr => oar_ocr_vl::PaddleOcrVlTask::Ocr,
        OcrTask::Table => oar_ocr_vl::PaddleOcrVlTask::Table,
        OcrTask::Formula => oar_ocr_vl::PaddleOcrVlTask::Formula,
        OcrTask::Chart => oar_ocr_vl::PaddleOcrVlTask::Chart,
        OcrTask::Spotting => oar_ocr_vl::PaddleOcrVlTask::Spotting,
        OcrTask::Seal => oar_ocr_vl::PaddleOcrVlTask::Seal,
    }
}

#[cfg(feature = "ocr")]
fn orchion_region(
    bbox: &oar_ocr::processors::BoundingBox,
    text: &str,
    confidence: Option<f32>,
) -> orchion_core::OcrRegion {
    orchion_core::OcrRegion {
        text: text.to_string(),
        confidence,
        polygon: polygon_points(bbox),
    }
}

#[cfg(feature = "ocr")]
fn polygon_points(bbox: &oar_ocr::processors::BoundingBox) -> Vec<OcrPoint> {
    bbox.points
        .iter()
        .map(|point| OcrPoint {
            x: point.x,
            y: point.y,
        })
        .collect()
}

#[cfg(feature = "ocr-vl")]
fn bbox_points(bbox: &oar_ocr::processors::BoundingBox) -> Vec<orchion_core::OcrPoint> {
    bbox.points
        .iter()
        .map(|point| orchion_core::OcrPoint {
            x: point.x,
            y: point.y,
        })
        .collect()
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn base_result(
    model: KnownOcrModel,
    format: OcrResponseFormat,
    text: String,
    markdown: Option<String>,
    html: Option<String>,
    regions: Vec<orchion_core::OcrRegion>,
    layout_blocks: Vec<OcrLayoutBlock>,
) -> OcrResult {
    OcrResult {
        model: ModelId::parse(model.id()).expect("built-in OCR model IDs are valid"),
        format,
        text,
        markdown,
        html,
        regions,
        layout_blocks,
        usage: OcrUsage {
            input_pages: 1,
            output_tokens: None,
        },
    }
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn model_load_error(error: impl Into<anyhow::Error>) -> OrchionError {
    OrchionError::ModelLoad {
        source: error.into(),
    }
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn inference_error(error: impl Into<anyhow::Error>) -> OrchionError {
    OrchionError::Inference {
        source: error.into(),
    }
}

#[cfg(all(test, feature = "ocr-vl"))]
mod tests {
    use super::*;

    #[test]
    fn ocr_vl_markdown_uses_layout_pipeline() {
        let options = OcrOptions {
            response_format: OcrResponseFormat::Markdown,
            ..OcrOptions::default()
        };

        assert!(should_use_vl_layout_pipeline(&options));
    }

    #[test]
    fn ocr_vl_html_uses_layout_pipeline() {
        let options = OcrOptions {
            response_format: OcrResponseFormat::Html,
            ..OcrOptions::default()
        };

        assert!(should_use_vl_layout_pipeline(&options));
    }

    #[test]
    fn html_tables_are_rendered_as_markdown_tables() {
        let html =
            "<table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>";

        assert_eq!(
            html_tables_to_markdown(html),
            "| Name | Value |\n| ---- | ----- |\n| A    | 1     |"
        );
    }

    #[test]
    fn html_table_div_wrappers_are_removed_from_markdown() {
        let html = "<div style=\"text-align: center;\"><table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table></div>";

        assert_eq!(
            html_tables_to_markdown(html),
            "| Name | Value |\n| ---- | ----- |\n| A    | 1     |"
        );
    }
}

#[cfg(all(test, feature = "ocr"))]
mod ocr_tests {
    use super::*;

    #[test]
    fn pp_ocrv5_mobile_assets_use_oar_registry_filenames() {
        let assets = traditional_assets(
            KnownOcrModel::PpOcrV5Mobile,
            Path::new("models/PaddlePaddle/PP-OCRv5_mobile"),
        )
        .unwrap();

        assert!(
            assets
                .detector
                .ends_with("greatv/oar-ocr/pp-ocrv5_mobile_det.onnx")
        );
        assert!(
            assets
                .recognizer
                .ends_with("greatv/oar-ocr/pp-ocrv5_mobile_rec.onnx")
        );
        assert!(
            assets
                .dictionary
                .ends_with("greatv/oar-ocr/ppocrv5_dict.txt")
        );
    }
}
