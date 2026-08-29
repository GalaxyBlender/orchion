use super::{
    ModelCapabilities, ModelCapabilityRequirement, ModelCategory, ModelDescriptor, ModelId,
    ModelSourceLocators, ModelSpec, RuntimeProvider,
};
use crate::{OrchionError, Result};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrModelKind {
    TraditionalOcr,
    Layout,
    OcrVl,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OcrModel {
    id: ModelId,
    kind: OcrModelKind,
}

impl OcrModel {
    #[must_use]
    pub const fn new(id: ModelId, kind: OcrModelKind) -> Self {
        Self { id, kind }
    }

    #[must_use]
    pub const fn id(&self) -> &ModelId {
        &self.id
    }

    #[must_use]
    pub const fn kind(&self) -> OcrModelKind {
        self.kind
    }

    pub fn known(&self) -> Option<KnownOcrModel> {
        KnownOcrModel::from_model_id(&self.id)
            .ok()
            .filter(|model| model.kind() == self.kind)
    }
}

impl fmt::Display for OcrModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(formatter)
    }
}

impl ModelSpec for OcrModel {
    fn category(&self) -> ModelCategory {
        match self.kind {
            OcrModelKind::TraditionalOcr | OcrModelKind::Layout => ModelCategory::Ocr,
            OcrModelKind::OcrVl => ModelCategory::OcrVl,
        }
    }

    fn huggingface_repo(&self) -> &str {
        self.id.as_str()
    }

    fn modelscope_repo(&self) -> &str {
        self.id.as_str()
    }

    fn required_files(&self) -> &'static [&'static str] {
        self.known().map_or(&["config.json"], |model| {
            <KnownOcrModel as ModelSpec>::required_files(&model)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnownOcrModel {
    PpOcrV5Mobile,
    PpOcrV5Server,
    PpOcrV6Tiny,
    PpOcrV6Small,
    PpOcrV6Medium,
    PpDocLayoutV3,
    PaddleOcrVl15,
    PaddleOcrVl16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrModelAssetKind {
    RequiredFile,
    PaddleOcrDictionary { output_file: &'static str },
    ModelScopeFile { output_file: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrModelAssetRole {
    Detector,
    Recognizer,
    Dictionary,
    Layout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OcrModelAsset {
    pub repo: &'static str,
    pub file: &'static str,
    pub kind: OcrModelAssetKind,
    pub role: OcrModelAssetRole,
}

const PP_OCRV5_MOBILE_ASSETS: &[OcrModelAsset] = &[
    OcrModelAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_det.onnx",
        kind: OcrModelAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_det.onnx",
        },
        role: OcrModelAssetRole::Detector,
    },
    OcrModelAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_rec.onnx",
        kind: OcrModelAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_rec.onnx",
        },
        role: OcrModelAssetRole::Recognizer,
    },
    OcrModelAsset {
        repo: "greatv/oar-ocr",
        file: "ppocrv5_dict.txt",
        kind: OcrModelAssetKind::ModelScopeFile {
            output_file: "ppocrv5_dict.txt",
        },
        role: OcrModelAssetRole::Dictionary,
    },
];

const PP_OCRV5_SERVER_ASSETS: &[OcrModelAsset] = &[
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv5_server_det_onnx",
        "inference.onnx",
        OcrModelAssetRole::Detector,
    ),
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv5_server_rec_onnx",
        "inference.onnx",
        OcrModelAssetRole::Recognizer,
    ),
    dictionary_asset("PaddlePaddle/PP-OCRv5_server_rec_onnx", "ppocrv5_dict.txt"),
];

const PP_OCRV6_TINY_ASSETS: &[OcrModelAsset] = &[
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        "inference.onnx",
        OcrModelAssetRole::Detector,
    ),
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        "inference.onnx",
        OcrModelAssetRole::Recognizer,
    ),
    dictionary_asset(
        "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        "ppocrv6_tiny_dict.txt",
    ),
];

const PP_OCRV6_SMALL_ASSETS: &[OcrModelAsset] = &[
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "inference.onnx",
        OcrModelAssetRole::Detector,
    ),
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "inference.onnx",
        OcrModelAssetRole::Recognizer,
    ),
    dictionary_asset("PaddlePaddle/PP-OCRv6_small_rec_onnx", "ppocrv6_dict.txt"),
];

const PP_OCRV6_MEDIUM_ASSETS: &[OcrModelAsset] = &[
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        "inference.onnx",
        OcrModelAssetRole::Detector,
    ),
    required_ocr_asset(
        "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        "inference.onnx",
        OcrModelAssetRole::Recognizer,
    ),
    dictionary_asset("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "ppocrv6_dict.txt"),
];

const PP_DOCLAYOUTV3_ASSETS: &[OcrModelAsset] = &[required_ocr_asset(
    "PaddlePaddle/PP-DocLayoutV3_onnx",
    "inference.onnx",
    OcrModelAssetRole::Layout,
)];

const fn required_ocr_asset(
    repo: &'static str,
    file: &'static str,
    role: OcrModelAssetRole,
) -> OcrModelAsset {
    OcrModelAsset {
        repo,
        file,
        kind: OcrModelAssetKind::RequiredFile,
        role,
    }
}

const fn dictionary_asset(repo: &'static str, output_file: &'static str) -> OcrModelAsset {
    OcrModelAsset {
        repo,
        file: "inference.yml",
        kind: OcrModelAssetKind::PaddleOcrDictionary { output_file },
        role: OcrModelAssetRole::Dictionary,
    }
}

impl KnownOcrModel {
    const MARKDOWN_REQUIRES_LAYOUT: ModelCapabilityRequirement = ModelCapabilityRequirement {
        capability: ModelCapabilities::OCR_MARKDOWN,
        requires: ModelCapabilities::OCR_LAYOUT,
    };
    const HTML_REQUIRES_LAYOUT: ModelCapabilityRequirement = ModelCapabilityRequirement {
        capability: ModelCapabilities::OCR_HTML,
        requires: ModelCapabilities::OCR_LAYOUT,
    };
    const MARKDOWN_REQUIREMENTS: [ModelCapabilityRequirement; 1] = [Self::MARKDOWN_REQUIRES_LAYOUT];
    const STRUCTURED_OUTPUT_REQUIREMENTS: [ModelCapabilityRequirement; 2] =
        [Self::MARKDOWN_REQUIRES_LAYOUT, Self::HTML_REQUIRES_LAYOUT];

    pub const ALL: [Self; 8] = [
        Self::PpOcrV5Mobile,
        Self::PpOcrV5Server,
        Self::PpOcrV6Tiny,
        Self::PpOcrV6Small,
        Self::PpOcrV6Medium,
        Self::PpDocLayoutV3,
        Self::PaddleOcrVl15,
        Self::PaddleOcrVl16,
    ];

    pub fn from_model_id(id: &ModelId) -> Result<Self> {
        match id.as_str() {
            "PaddlePaddle/PP-OCRv5_mobile" => Ok(Self::PpOcrV5Mobile),
            "PaddlePaddle/PP-OCRv5_server" => Ok(Self::PpOcrV5Server),
            "PaddlePaddle/PP-OCRv6_tiny" => Ok(Self::PpOcrV6Tiny),
            "PaddlePaddle/PP-OCRv6_small" => Ok(Self::PpOcrV6Small),
            "PaddlePaddle/PP-OCRv6_medium" => Ok(Self::PpOcrV6Medium),
            "PaddlePaddle/PP-DocLayoutV3" => Ok(Self::PpDocLayoutV3),
            "PaddlePaddle/PaddleOCR-VL-1.5" => Ok(Self::PaddleOcrVl15),
            "PaddlePaddle/PaddleOCR-VL-1.6" => Ok(Self::PaddleOcrVl16),
            other => Err(OrchionError::ModelLoad {
                message: format!("unsupported OCR model `{other}`"),
            }),
        }
    }

    pub fn from_traditional_model_id(id: &ModelId) -> Result<Self> {
        let model = Self::from_model_id(id)?;
        if model.is_traditional_ocr() {
            Ok(model)
        } else {
            Err(invalid_ocr_model_kind(id, "traditional OCR model"))
        }
    }

    pub fn from_ocr_vl_model_id(id: &ModelId) -> Result<Self> {
        let model = Self::from_model_id(id)?;
        if model.is_ocr_vl() {
            Ok(model)
        } else {
            Err(invalid_ocr_model_kind(id, "OCR-VL model"))
        }
    }

    pub fn from_layout_model_id(id: &ModelId) -> Result<Self> {
        let model = Self::from_model_id(id)?;
        if model.is_layout_model() {
            Ok(model)
        } else {
            Err(invalid_ocr_model_kind(id, "PaddlePaddle/PP-DocLayoutV3"))
        }
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::PpOcrV5Mobile => "PaddlePaddle/PP-OCRv5_mobile",
            Self::PpOcrV5Server => "PaddlePaddle/PP-OCRv5_server",
            Self::PpOcrV6Tiny => "PaddlePaddle/PP-OCRv6_tiny",
            Self::PpOcrV6Small => "PaddlePaddle/PP-OCRv6_small",
            Self::PpOcrV6Medium => "PaddlePaddle/PP-OCRv6_medium",
            Self::PpDocLayoutV3 => "PaddlePaddle/PP-DocLayoutV3",
            Self::PaddleOcrVl15 => "PaddlePaddle/PaddleOCR-VL-1.5",
            Self::PaddleOcrVl16 => "PaddlePaddle/PaddleOCR-VL-1.6",
        }
    }

    /// Converts built-in metadata to the provider-neutral OCR model key.
    ///
    /// # Panics
    ///
    /// Panics only if a statically declared built-in model ID violates [`ModelId`] syntax.
    pub fn into_model(self) -> OcrModel {
        OcrModel::new(
            ModelId::parse(self.id()).expect("known OCR model id"),
            self.kind(),
        )
    }

    pub const fn kind(self) -> OcrModelKind {
        match self {
            Self::PpOcrV5Mobile
            | Self::PpOcrV5Server
            | Self::PpOcrV6Tiny
            | Self::PpOcrV6Small
            | Self::PpOcrV6Medium => OcrModelKind::TraditionalOcr,
            Self::PpDocLayoutV3 => OcrModelKind::Layout,
            Self::PaddleOcrVl15 | Self::PaddleOcrVl16 => OcrModelKind::OcrVl,
        }
    }

    pub const fn is_traditional_ocr(self) -> bool {
        matches!(self.kind(), OcrModelKind::TraditionalOcr)
    }

    pub const fn is_layout_model(self) -> bool {
        matches!(self.kind(), OcrModelKind::Layout)
    }

    pub const fn is_ocr_vl(self) -> bool {
        matches!(self.kind(), OcrModelKind::OcrVl)
    }

    pub const fn supports_markdown(self) -> bool {
        matches!(self, Self::PaddleOcrVl15 | Self::PaddleOcrVl16)
    }

    pub const fn download_assets(self) -> &'static [OcrModelAsset] {
        match self {
            Self::PpOcrV5Mobile => PP_OCRV5_MOBILE_ASSETS,
            Self::PpOcrV5Server => PP_OCRV5_SERVER_ASSETS,
            Self::PpOcrV6Tiny => PP_OCRV6_TINY_ASSETS,
            Self::PpOcrV6Small => PP_OCRV6_SMALL_ASSETS,
            Self::PpOcrV6Medium => PP_OCRV6_MEDIUM_ASSETS,
            Self::PpDocLayoutV3 => PP_DOCLAYOUTV3_ASSETS,
            Self::PaddleOcrVl15 | Self::PaddleOcrVl16 => &[],
        }
    }

    pub const fn descriptor(self) -> ModelDescriptor {
        let canonical_id = self.id();
        let (capabilities, requirements): (_, &'static [_]) = match self.kind() {
            OcrModelKind::TraditionalOcr => {
                (ModelCapabilities::OCR_TEXT, &Self::MARKDOWN_REQUIREMENTS)
            }
            OcrModelKind::Layout => (ModelCapabilities::OCR_LAYOUT, &[]),
            OcrModelKind::OcrVl => (
                ModelCapabilities::OCR_TEXT.union(ModelCapabilities::OCR_VISION_LANGUAGE),
                &Self::STRUCTURED_OUTPUT_REQUIREMENTS,
            ),
        };
        ModelDescriptor {
            canonical_id,
            source_locators: ModelSourceLocators {
                hugging_face: canonical_id,
                model_scope: canonical_id,
            },
            category: match self.kind() {
                OcrModelKind::TraditionalOcr | OcrModelKind::Layout => ModelCategory::Ocr,
                OcrModelKind::OcrVl => ModelCategory::OcrVl,
            },
            capabilities,
            requirements,
            runtime_provider: RuntimeProvider::OarOcr,
        }
    }

    pub const fn dictionary_file(self) -> Option<&'static str> {
        let assets = self.download_assets();
        let mut index = 0;
        while index < assets.len() {
            if matches!(assets[index].role, OcrModelAssetRole::Dictionary) {
                return match assets[index].kind {
                    OcrModelAssetKind::PaddleOcrDictionary { output_file }
                    | OcrModelAssetKind::ModelScopeFile { output_file } => Some(output_file),
                    OcrModelAssetKind::RequiredFile => Some(assets[index].file),
                };
            }
            index += 1;
        }
        None
    }
}

fn invalid_ocr_model_kind(id: &ModelId, expected: &'static str) -> OrchionError {
    OrchionError::ModelLoad {
        message: format!("OCR model `{id}` is not a {expected}"),
    }
}

impl ModelSpec for KnownOcrModel {
    fn category(&self) -> ModelCategory {
        match (*self).kind() {
            OcrModelKind::TraditionalOcr | OcrModelKind::Layout => ModelCategory::Ocr,
            OcrModelKind::OcrVl => ModelCategory::OcrVl,
        }
    }

    fn huggingface_repo(&self) -> &str {
        (*self).descriptor().source_locators.hugging_face
    }

    fn modelscope_repo(&self) -> &str {
        (*self).descriptor().source_locators.model_scope
    }

    fn required_files(&self) -> &'static [&'static str] {
        match self {
            Self::PpOcrV5Mobile
            | Self::PpOcrV5Server
            | Self::PpOcrV6Tiny
            | Self::PpOcrV6Small
            | Self::PpOcrV6Medium
            | Self::PpDocLayoutV3 => &[],
            Self::PaddleOcrVl15 | Self::PaddleOcrVl16 => &[
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "chat_template.jinja",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_builtin_ocr_model_ids() {
        let id = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        let model = KnownOcrModel::from_model_id(&id).unwrap();
        assert_eq!(model.id(), "PaddlePaddle/PaddleOCR-VL-1.6");
        assert_eq!(model.kind(), OcrModelKind::OcrVl);
        assert!(model.supports_markdown());
    }

    #[test]
    fn traditional_ocr_does_not_support_markdown() {
        let id = ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap();
        let model = KnownOcrModel::from_model_id(&id).unwrap();
        assert_eq!(model.kind(), OcrModelKind::TraditionalOcr);
        assert!(!model.supports_markdown());
    }

    #[test]
    fn resolves_ocr_models_by_expected_capability() {
        let traditional = ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap();
        let ocr_vl = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        let layout = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();

        assert_eq!(
            KnownOcrModel::from_traditional_model_id(&traditional).unwrap(),
            KnownOcrModel::PpOcrV6Tiny
        );
        assert_eq!(
            KnownOcrModel::from_ocr_vl_model_id(&ocr_vl).unwrap(),
            KnownOcrModel::PaddleOcrVl16
        );
        assert_eq!(
            KnownOcrModel::from_layout_model_id(&layout).unwrap(),
            KnownOcrModel::PpDocLayoutV3
        );
        assert!(KnownOcrModel::from_layout_model_id(&traditional).is_err());
        assert!(KnownOcrModel::from_ocr_vl_model_id(&layout).is_err());
    }

    #[test]
    fn structured_ocr_capabilities_are_effective_only_with_layout() {
        let traditional = KnownOcrModel::PpOcrV6Tiny
            .descriptor()
            .effective_capabilities(ModelCapabilities::OCR_LAYOUT);
        assert!(traditional.contains(ModelCapabilities::OCR_MARKDOWN));
        assert!(!traditional.contains(ModelCapabilities::OCR_HTML));

        let ocr_vl = KnownOcrModel::PaddleOcrVl16
            .descriptor()
            .effective_capabilities(ModelCapabilities::OCR_LAYOUT);
        assert!(ocr_vl.contains(ModelCapabilities::OCR_MARKDOWN));
        assert!(ocr_vl.contains(ModelCapabilities::OCR_HTML));
    }

    #[test]
    fn layout_model_does_not_claim_ocr_output_capabilities() {
        let descriptor = KnownOcrModel::PpDocLayoutV3.descriptor();
        let effective = descriptor.effective_capabilities(ModelCapabilities::NONE);

        assert!(effective.contains(ModelCapabilities::OCR_LAYOUT));
        assert!(!effective.contains(ModelCapabilities::OCR_MARKDOWN));
        assert!(!effective.contains(ModelCapabilities::OCR_HTML));
    }
}
