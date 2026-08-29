use crate::OcrAssets;
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
use crate::device::{ProviderPolicy, try_provider_candidates};
use image::{ImageReader, Limits};
#[cfg(feature = "ocr")]
use orchion_core::OcrPoint;
#[cfg(feature = "ocr-vl")]
use orchion_core::OcrTask;
use orchion_core::{
    DevicePreference, KnownOcrModel, OcrLimits, OcrOptions, OcrResult, OrchionError, Result,
};
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
use orchion_core::{ModelId, OcrLayoutBlock, OcrResponseFormat, OcrUsage};
use std::path::Path;
#[cfg(feature = "ocr")]
use std::path::PathBuf;
#[cfg(any(feature = "ocr", all(feature = "ocr-vl", not(feature = "cuda"))))]
use std::sync::{Arc, Mutex};

#[cfg(feature = "ocr-vl")]
const DEFAULT_VL_MAX_TOKENS: usize = 4096;

#[derive(Clone)]
pub enum LoadedOcrRuntime {
    #[cfg(feature = "ocr")]
    Traditional(Arc<TraditionalRuntime>),
    #[cfg(feature = "ocr")]
    Layout(Arc<Mutex<oar_ocr::oarocr::OARStructure>>),
    #[cfg(all(feature = "ocr-vl", not(feature = "cuda")))]
    OcrVl(Arc<Mutex<OcrVlRuntime>>),
    #[cfg(all(feature = "ocr-vl", feature = "cuda"))]
    OcrVl(crate::vl_worker::OcrVlWorker),
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
pub(crate) struct OcrVlRuntime {
    model: oar_ocr_vl::PaddleOcrVl,
    layout_predictor: Option<oar_ocr::predictors::LayoutDetectionPredictor>,
}

#[cfg(feature = "ocr-vl")]
struct OcrLayoutSource<'a> {
    predictor: &'a oar_ocr::predictors::LayoutDetectionPredictor,
}

#[cfg(feature = "ocr-vl")]
impl oar_ocr_vl::LayoutSource for OcrLayoutSource<'_> {
    fn detect(
        &self,
        image: &image::RgbImage,
    ) -> std::result::Result<oar_ocr_vl::LayoutDetections, oar_ocr_vl::Error> {
        let result = self
            .predictor
            .predict(vec![image.clone()])
            .map_err(|error| oar_ocr_vl::Error::invalid_input(error.to_string()))?;
        let elements = result
            .elements
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|element| oar_ocr_vl::LayoutDetectionElement {
                bbox: oar_ocr_vl::BoundingBox::new(
                    element
                        .bbox
                        .points
                        .into_iter()
                        .map(|point| oar_ocr_vl::Point::new(point.x, point.y))
                        .collect(),
                ),
                element_type: element.element_type,
                score: element.score,
            })
            .collect();

        Ok(oar_ocr_vl::LayoutDetections::new(elements))
    }
}

pub async fn load_runtime(
    model: KnownOcrModel,
    assets: OcrAssets,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    tokio::task::spawn_blocking(move || load_runtime_blocking(model, &assets, device))
        .await
        .map_err(|error| OrchionError::BlockingTask {
            message: error.to_string(),
        })?
}

fn load_runtime_blocking(
    model: KnownOcrModel,
    assets: &OcrAssets,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    match model {
        KnownOcrModel::PpOcrV5Mobile
        | KnownOcrModel::PpOcrV5Server
        | KnownOcrModel::PpOcrV6Tiny
        | KnownOcrModel::PpOcrV6Small
        | KnownOcrModel::PpOcrV6Medium => load_traditional_runtime(model, assets, device),
        KnownOcrModel::PpDocLayoutV3 => load_layout_runtime(model, assets, device),
        KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16 => {
            load_vl_runtime(model, assets, device)
        }
    }
}

pub async fn run_ocr(
    model: KnownOcrModel,
    runtime: LoadedOcrRuntime,
    image_path: &Path,
    options: OcrOptions,
    limits: OcrLimits,
) -> Result<OcrResult> {
    let image_path = image_path.to_path_buf();
    #[cfg(all(feature = "ocr-vl", feature = "cuda"))]
    if let (
        KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16,
        LoadedOcrRuntime::OcrVl(worker),
    ) = (model, &runtime)
    {
        return worker.run(model, image_path, options, limits).await;
    }

    tokio::task::spawn_blocking(move || {
        run_ocr_blocking(model, &runtime, &image_path, &options, limits)
    })
    .await
    .map_err(|error| OrchionError::BlockingTask {
        message: error.to_string(),
    })?
}

/// Validates an OCR image header and decoder limits without decoding its pixels.
///
/// # Errors
///
/// Returns [`OrchionError::InvalidImage`] when the file is not a supported image or exceeds the
/// pixel limit.
pub fn validate_image_file(image_path: &Path, max_pixels: u64) -> Result<()> {
    let dimensions = image_dimensions(image_path)?;
    validate_pixel_count(dimensions, max_pixels)?;
    let mut reader = image_reader(image_path)?;
    reader.limits(image_limits(max_pixels));
    reader.into_dimensions().map_err(invalid_image_error)?;
    Ok(())
}

fn run_ocr_blocking(
    model: KnownOcrModel,
    runtime: &LoadedOcrRuntime,
    image_path: &Path,
    options: &OcrOptions,
    limits: OcrLimits,
) -> Result<OcrResult> {
    #[cfg(not(feature = "ocr"))]
    let _ = (image_path, options, limits);

    match (model, runtime) {
        #[cfg(feature = "ocr")]
        (
            KnownOcrModel::PpOcrV5Mobile
            | KnownOcrModel::PpOcrV5Server
            | KnownOcrModel::PpOcrV6Tiny
            | KnownOcrModel::PpOcrV6Small
            | KnownOcrModel::PpOcrV6Medium,
            LoadedOcrRuntime::Traditional(runtime),
        ) => run_traditional_ocr(model, runtime, image_path, options, limits),
        #[cfg(feature = "ocr")]
        (KnownOcrModel::PpDocLayoutV3, LoadedOcrRuntime::Layout(structure)) => {
            run_layout_ocr(model, structure, image_path, options, limits)
        }
        #[cfg(all(feature = "ocr-vl", not(feature = "cuda")))]
        (
            KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16,
            LoadedOcrRuntime::OcrVl(vl),
        ) => run_vl_ocr_locked(model, vl, image_path, options, limits),
        #[cfg(any(not(feature = "ocr"), not(feature = "ocr-vl")))]
        (_, LoadedOcrRuntime::Unsupported { model, capability }) => {
            Err(OrchionError::UnsupportedCapability {
                model: model.id().to_string(),
                capability,
            })
        }
        #[cfg(any(feature = "ocr", feature = "ocr-vl"))]
        _ => Err(OrchionError::Inference {
            message: format!("loaded OCR runtime does not match model `{}`", model.id()),
        }),
    }
}

#[cfg(feature = "ocr")]
fn load_traditional_runtime(
    model: KnownOcrModel,
    assets: &OcrAssets,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    use oar_ocr::oarocr::OAROCRBuilder;

    let (assets, layout) = traditional_assets(model, assets)?;
    let ocr = try_ort_provider_candidates(model, device, |provider| {
        OAROCRBuilder::new(
            assets.detector.clone(),
            assets.recognizer.clone(),
            assets.dictionary.clone(),
        )
        .ort_session(ort_session_config(provider))
        .build()
        .map_err(model_load_error)
    })?;
    let structure = load_related_structure_runtime(layout, &assets, device)?;
    Ok(LoadedOcrRuntime::Traditional(Arc::new(
        TraditionalRuntime {
            ocr: Mutex::new(ocr),
            structure: structure.map(Mutex::new),
        },
    )))
}

#[cfg(not(feature = "ocr"))]
#[allow(clippy::unnecessary_wraps)]
fn load_traditional_runtime(
    model: KnownOcrModel,
    _assets: &OcrAssets,
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
    assets: &OcrAssets,
    device: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    let OcrAssets::Layout {
        model: layout_model,
    } = assets
    else {
        return Err(asset_kind_error(model, "layout"));
    };
    let structure = build_structure_runtime(model, layout_model, None, device)?;
    Ok(LoadedOcrRuntime::Layout(Arc::new(Mutex::new(structure))))
}

#[cfg(feature = "ocr")]
fn load_related_structure_runtime(
    layout_model: Option<&Path>,
    assets: &TraditionalAssets,
    device: DevicePreference,
) -> Result<Option<oar_ocr::oarocr::OARStructure>> {
    let Some(layout_model) = layout_model else {
        return Ok(None);
    };

    build_structure_runtime(
        KnownOcrModel::PpDocLayoutV3,
        layout_model,
        Some(assets),
        device,
    )
    .map(Some)
}

#[cfg(feature = "ocr")]
fn build_structure_runtime(
    provider_model: KnownOcrModel,
    layout_model: &Path,
    ocr_assets: Option<&TraditionalAssets>,
    device: DevicePreference,
) -> Result<oar_ocr::oarocr::OARStructure> {
    use oar_ocr::oarocr::OARStructureBuilder;

    try_ort_provider_candidates(provider_model, device, |provider| {
        let builder = OARStructureBuilder::new(layout_model.to_path_buf())
            .layout_model_name(layout_model_name(provider_model))
            .ort_session(ort_session_config(provider));
        let builder = if let Some(assets) = ocr_assets {
            builder.with_ocr(
                assets.detector.clone(),
                assets.recognizer.clone(),
                assets.dictionary.clone(),
            )
        } else {
            builder
        };
        builder.build().map_err(model_load_error)
    })
}

#[cfg(feature = "ocr")]
const fn layout_model_name(_model: KnownOcrModel) -> &'static str {
    "PP-DocLayoutV3"
}

#[cfg(not(feature = "ocr"))]
#[allow(clippy::unnecessary_wraps)]
fn load_layout_runtime(
    model: KnownOcrModel,
    _assets: &OcrAssets,
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
    assets: &OcrAssets,
    device_preference: DevicePreference,
) -> Result<LoadedOcrRuntime> {
    #[cfg(feature = "cuda")]
    return crate::vl_worker::OcrVlWorker::load(model, assets.clone(), device_preference)
        .map(LoadedOcrRuntime::OcrVl);

    #[cfg(not(feature = "cuda"))]
    build_vl_runtime(model, assets, device_preference)
        .map(|runtime| LoadedOcrRuntime::OcrVl(Arc::new(Mutex::new(runtime))))
}

#[cfg(feature = "ocr-vl")]
pub(crate) fn build_vl_runtime(
    model: KnownOcrModel,
    assets: &OcrAssets,
    device_preference: DevicePreference,
) -> Result<OcrVlRuntime> {
    use oar_ocr_vl::{PaddleOcrVl, utils::parse_device};

    let OcrAssets::VisionLanguage { model_dir, layout } = assets else {
        return Err(asset_kind_error(model, "vision-language"));
    };
    let candidates = ProviderPolicy::candidates_for_model(model, device_preference);
    let vl = try_provider_candidates(&candidates, |provider| {
        let device = candle_device(provider);
        let candle_device = parse_device(&device).map_err(model_load_error)?;
        PaddleOcrVl::from_dir(model_dir, candle_device).map_err(model_load_error)
    })?;
    let layout_predictor = load_default_layout_predictor(layout.as_deref(), device_preference)?;

    Ok(OcrVlRuntime {
        model: vl,
        layout_predictor,
    })
}

#[cfg(not(feature = "ocr-vl"))]
#[allow(clippy::unnecessary_wraps)]
fn load_vl_runtime(
    model: KnownOcrModel,
    _assets: &OcrAssets,
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
    limits: OcrLimits,
) -> Result<OcrResult> {
    if options.layout_model.is_some() {
        let structure = runtime.structure.as_ref().ok_or_else(|| {
            model_load_error(anyhow::anyhow!(
                "OCR layout model is configured but not loaded for `{}`",
                model.id()
            ))
        })?;
        return run_layout_ocr(model, structure, image_path, options, limits);
    }

    let image = decode_ocr_image(image_path, limits.max_pixels)?;
    let ocr = runtime
        .ocr
        .lock()
        .map_err(|error| OrchionError::Inference {
            message: format!("OCR runtime lock poisoned: {error}"),
        })?;
    let mut pages = ocr.predict(vec![image]).map_err(inference_error)?;
    let page = pages.pop().ok_or_else(|| OrchionError::Inference {
        message: "OCR returned no pages".to_string(),
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
    limits: OcrLimits,
) -> Result<OcrResult> {
    let structure = structure.lock().map_err(|error| OrchionError::Inference {
        message: format!("OCR layout runtime lock poisoned: {error}"),
    })?;
    let image = decode_ocr_image(image_path, limits.max_pixels)?;
    let result = structure.predict_image(image).map_err(inference_error)?;

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

#[cfg(all(feature = "ocr-vl", not(feature = "cuda")))]
fn run_vl_ocr_locked(
    model: KnownOcrModel,
    runtime: &Mutex<OcrVlRuntime>,
    image_path: &Path,
    options: &OcrOptions,
    limits: OcrLimits,
) -> Result<OcrResult> {
    let image = decode_ocr_image(image_path, limits.max_pixels)?;
    let runtime = runtime.lock().map_err(|error| OrchionError::Inference {
        message: format!("OCR-VL runtime lock poisoned: {error}"),
    })?;
    run_vl_ocr_with_image(model, &runtime, image, options)
}

#[cfg(all(feature = "ocr-vl", feature = "cuda"))]
pub(crate) fn run_vl_ocr(
    model: KnownOcrModel,
    runtime: &OcrVlRuntime,
    image_path: &Path,
    options: &OcrOptions,
    limits: OcrLimits,
) -> Result<OcrResult> {
    let image = decode_ocr_image(image_path, limits.max_pixels)?;
    run_vl_ocr_with_image(model, runtime, image, options)
}

#[cfg(feature = "ocr-vl")]
fn run_vl_ocr_with_image(
    model: KnownOcrModel,
    runtime: &OcrVlRuntime,
    image: image::RgbImage,
    options: &OcrOptions,
) -> Result<OcrResult> {
    let max_tokens = options.max_tokens.unwrap_or(DEFAULT_VL_MAX_TOKENS);

    if should_use_vl_layout_pipeline(options) {
        let layout_predictor = runtime.layout_predictor.as_ref().ok_or_else(|| {
            model_load_error(anyhow::anyhow!("OCR-VL layout model is not loaded"))
        })?;
        return run_vl_layout_ocr(model, runtime, layout_predictor, image, options);
    }

    let task = vl_task(options.task);
    let text = runtime
        .model
        .generate(&[image], &[task], max_tokens)
        .map_err(inference_error)?
        .into_iter()
        .next()
        .ok_or_else(|| OrchionError::Inference {
            message: "OCR-VL returned no results".to_string(),
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
    let layout_source = OcrLayoutSource {
        predictor: layout_predictor,
    };
    let structure = parser
        .parse(&layout_source, image)
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
        render_layout_html(
            structure.layout_elements.iter().map(|element| {
                (
                    element.label.as_deref().unwrap_or("text"),
                    element.text.as_deref().unwrap_or_default(),
                )
            }),
            &parser.config().markdown_ignore_labels,
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

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn decode_ocr_image(image_path: &Path, max_pixels: Option<u64>) -> Result<image::RgbImage> {
    let dimensions = image_dimensions(image_path)?;
    if let Some(max_pixels) = max_pixels {
        validate_pixel_count(dimensions, max_pixels)?;
    }
    let mut reader = image_reader(image_path)?;
    if let Some(max_pixels) = max_pixels {
        reader.limits(image_limits(max_pixels));
    }
    reader
        .decode()
        .map(image::DynamicImage::into_rgb8)
        .map_err(invalid_image_error)
}

fn image_dimensions(image_path: &Path) -> Result<(u32, u32)> {
    image_reader(image_path)?
        .into_dimensions()
        .map_err(invalid_image_error)
}

fn image_reader(image_path: &Path) -> Result<ImageReader<std::io::BufReader<std::fs::File>>> {
    ImageReader::open(image_path)
        .map_err(invalid_image_error)?
        .with_guessed_format()
        .map_err(invalid_image_error)
}

fn validate_pixel_count((width, height): (u32, u32), max_pixels: u64) -> Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        return Err(OrchionError::InvalidImage {
            reason: format!(
                "image contains {pixels} pixels, exceeding the configured limit of {max_pixels}"
            ),
        });
    }
    Ok(())
}

fn image_limits(max_pixels: u64) -> Limits {
    let max_dimension = u32::try_from(max_pixels).unwrap_or(u32::MAX);
    let mut limits = Limits::default();
    limits.max_image_width = Some(max_dimension);
    limits.max_image_height = Some(max_dimension);
    limits.max_alloc = limits
        .max_alloc
        .map(|default| default.min(max_pixels.saturating_mul(16)));
    limits
}

fn invalid_image_error(error: impl std::fmt::Display) -> OrchionError {
    OrchionError::InvalidImage {
        reason: error.to_string(),
    }
}

#[cfg(feature = "ocr-vl")]
fn html_tables_to_markdown(input: &str) -> String {
    htmd::convert(input).unwrap_or_else(|_| input.to_string())
}

#[cfg(feature = "ocr-vl")]
fn render_layout_html<'a>(
    elements: impl IntoIterator<Item = (&'a str, &'a str)>,
    ignore_labels: &[String],
) -> String {
    let mut html = String::new();
    for (label, text) in elements {
        let text = text.trim();
        if text.is_empty() || ignore_labels.iter().any(|ignored| ignored == label) {
            continue;
        }
        let escaped = escape_html(text);
        let block = match label {
            "doc_title" => format!("<h1>{}</h1>", html_lines(&escaped)),
            "paragraph_title" | "abstract_title" | "reference_title" | "content_title" => {
                format!("<h2>{}</h2>", html_lines(&escaped))
            }
            "list" => {
                let items = text
                    .lines()
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .fold(String::new(), |mut items, item| {
                        items.push_str("<li>");
                        items.push_str(&escape_html(item));
                        items.push_str("</li>");
                        items
                    });
                format!("<ul>{items}</ul>")
            }
            "algorithm" => format!("<pre><code>{escaped}</code></pre>"),
            "formula" | "display_formula" | "inline_formula" => {
                format!("<pre data-ocr-kind=\"formula\">{escaped}</pre>")
            }
            "table" => format!("<pre data-ocr-kind=\"table\">{escaped}</pre>"),
            "image" | "figure" | "chart" | "seal" => {
                format!(
                    "<figure><figcaption>{}</figcaption></figure>",
                    html_lines(&escaped)
                )
            }
            _ => format!("<p>{}</p>", html_lines(&escaped)),
        };
        html.push_str(&block);
        html.push('\n');
    }
    html.trim_end().to_string()
}

#[cfg(feature = "ocr-vl")]
fn escape_html(text: &str) -> String {
    text.chars().fold(String::new(), |mut escaped, character| {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            _ => escaped.push(character),
        }
        escaped
    })
}

#[cfg(feature = "ocr-vl")]
fn html_lines(escaped_text: &str) -> String {
    escaped_text.replace('\n', "<br>\n")
}

#[cfg(feature = "ocr-vl")]
fn load_default_layout_predictor(
    layout_model: Option<&Path>,
    device: DevicePreference,
) -> Result<Option<oar_ocr::predictors::LayoutDetectionPredictor>> {
    let Some(layout_model) = layout_model else {
        return Ok(None);
    };

    let predictor =
        try_ort_provider_candidates(KnownOcrModel::PpDocLayoutV3, device, |provider| {
            oar_ocr::predictors::LayoutDetectionPredictor::builder()
                .model_name("pp-doclayoutv3")
                .with_ort_config(ort_session_config(provider))
                .build(layout_model)
                .map_err(model_load_error)
        })?;
    Ok(Some(predictor))
}

#[cfg(feature = "ocr")]
struct TraditionalAssets {
    detector: PathBuf,
    recognizer: PathBuf,
    dictionary: PathBuf,
}

#[cfg(feature = "ocr")]
fn traditional_assets(
    model: KnownOcrModel,
    assets: &OcrAssets,
) -> Result<(TraditionalAssets, Option<&Path>)> {
    let OcrAssets::Traditional {
        detector,
        recognizer,
        dictionary,
        layout,
    } = assets
    else {
        return Err(asset_kind_error(model, "traditional"));
    };
    Ok((
        TraditionalAssets {
            detector: detector.clone(),
            recognizer: recognizer.clone(),
            dictionary: dictionary.clone(),
        },
        layout.as_deref(),
    ))
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn asset_kind_error(model: KnownOcrModel, expected: &str) -> OrchionError {
    model_load_error(anyhow::anyhow!(
        "OCR model `{}` requires a {expected} asset bundle",
        model.id()
    ))
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
            unreachable!("Candle provider passed to ONNX Runtime configuration")
        }
    };

    OrtSessionConfig::new().with_execution_providers(vec![provider])
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn try_ort_provider_candidates<T>(
    model: KnownOcrModel,
    device: DevicePreference,
    mut build: impl FnMut(ProviderPolicy) -> Result<T>,
) -> Result<T> {
    let candidates = ProviderPolicy::candidates_for_model(model, device);
    try_provider_candidates(&candidates, |provider| {
        probe_ort_provider(provider)?;
        build(provider)
    })
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn probe_ort_provider(provider: ProviderPolicy) -> Result<()> {
    match provider {
        ProviderPolicy::OrtCpu => Ok(()),
        ProviderPolicy::OrtCuda(index) => probe_ort_cuda(index),
        ProviderPolicy::OrtWebGpu => probe_ort_webgpu(),
        ProviderPolicy::CandleCpu | ProviderPolicy::CandleCuda(_) | ProviderPolicy::CandleMetal => {
            Err(model_load_error(anyhow::anyhow!(
                "Candle provider passed to ONNX Runtime"
            )))
        }
    }
}

#[cfg(all(any(feature = "ocr", feature = "ocr-vl"), feature = "cuda"))]
fn probe_ort_cuda(index: Option<usize>) -> Result<()> {
    let index = i32::try_from(index.unwrap_or(0))
        .map_err(|error| model_load_error(anyhow::anyhow!("invalid CUDA device index: {error}")))?;
    let provider = ort::ep::CUDA::default()
        .with_device_id(index)
        .build()
        .error_on_failure();
    let builder = ort::session::Session::builder().map_err(|error| {
        model_load_error(anyhow::anyhow!(
            "failed to create ONNX Runtime session builder: {error}"
        ))
    })?;
    builder
        .with_execution_providers([provider])
        .map(|_| ())
        .map_err(|error| {
            model_load_error(anyhow::anyhow!(
                "failed to initialize ONNX Runtime CUDA provider: {error}"
            ))
        })
}

#[cfg(all(any(feature = "ocr", feature = "ocr-vl"), not(feature = "cuda")))]
fn probe_ort_cuda(_index: Option<usize>) -> Result<()> {
    Err(model_load_error(anyhow::anyhow!(
        "ONNX Runtime CUDA provider requested but the cuda feature is not enabled"
    )))
}

#[cfg(all(any(feature = "ocr", feature = "ocr-vl"), feature = "metal"))]
fn probe_ort_webgpu() -> Result<()> {
    let provider = ort::ep::WebGPU::default().build().error_on_failure();
    let builder = ort::session::Session::builder().map_err(|error| {
        model_load_error(anyhow::anyhow!(
            "failed to create ONNX Runtime session builder: {error}"
        ))
    })?;
    builder
        .with_execution_providers([provider])
        .map(|_| ())
        .map_err(|error| {
            model_load_error(anyhow::anyhow!(
                "failed to initialize ONNX Runtime WebGPU provider: {error}"
            ))
        })
}

#[cfg(all(any(feature = "ocr", feature = "ocr-vl"), not(feature = "metal")))]
fn probe_ort_webgpu() -> Result<()> {
    Err(model_load_error(anyhow::anyhow!(
        "ONNX Runtime WebGPU provider requested but the metal feature is not enabled"
    )))
}

#[cfg(feature = "ocr-vl")]
fn candle_device(policy: ProviderPolicy) -> String {
    match policy {
        ProviderPolicy::CandleCpu => "cpu".to_string(),
        ProviderPolicy::CandleCuda(None) => "cuda".to_string(),
        ProviderPolicy::CandleCuda(Some(index)) => format!("cuda:{index}"),
        ProviderPolicy::CandleMetal => "metal".to_string(),
        ProviderPolicy::OrtCpu | ProviderPolicy::OrtCuda(_) | ProviderPolicy::OrtWebGpu => {
            unreachable!("ONNX Runtime provider passed to Candle device selection")
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
fn bbox_points(bbox: &oar_ocr_vl::BoundingBox) -> Vec<orchion_core::OcrPoint> {
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
        message: error.into().to_string(),
    }
}

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
fn inference_error(error: impl Into<anyhow::Error>) -> OrchionError {
    OrchionError::Inference {
        message: error.into().to_string(),
    }
}

#[cfg(all(test, feature = "ocr"))]
mod traditional_tests {
    use super::*;

    #[test]
    fn pp_doclayout_v3_uses_matching_layout_preset() {
        assert_eq!(
            layout_model_name(KnownOcrModel::PpDocLayoutV3),
            "PP-DocLayoutV3"
        );
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
    fn ocr_vl_html_is_semantic_and_escapes_recognized_content() {
        let html = render_layout_html(
            [
                ("doc_title", "Invoice <script>alert(1)</script>"),
                ("text", "Total & tax"),
                ("table", "<table><tr><td>unsafe</td></tr></table>"),
            ],
            &[],
        );

        assert!(html.contains("<h1>Invoice &lt;script&gt;alert(1)&lt;/script&gt;</h1>"));
        assert!(html.contains("<p>Total &amp; tax</p>"));
        assert!(html.contains("data-ocr-kind=\"table\""));
        assert!(html.contains("&lt;table&gt;"));
        assert!(!html.contains("<script>"));
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
