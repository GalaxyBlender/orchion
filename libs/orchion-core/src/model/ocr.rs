use super::{ModelCategory, ModelHubAsset, ModelHubAssetKind, ModelId, ModelSpec};
use crate::{OrchionError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrModelKind {
    TraditionalOcr,
    Layout,
    OcrVl,
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

impl KnownOcrModel {
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
                source: anyhow::anyhow!("unsupported OCR model `{other}`"),
            }),
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

    pub const fn supports_markdown(self) -> bool {
        matches!(self, Self::PaddleOcrVl15 | Self::PaddleOcrVl16)
    }

    pub const fn pp_ocr_detector_repo(self) -> Option<&'static str> {
        match self {
            Self::PpOcrV5Server => Some("PaddlePaddle/PP-OCRv5_server_det_onnx"),
            Self::PpOcrV6Tiny => Some("PaddlePaddle/PP-OCRv6_tiny_det_onnx"),
            Self::PpOcrV6Small => Some("PaddlePaddle/PP-OCRv6_small_det_onnx"),
            Self::PpOcrV6Medium => Some("PaddlePaddle/PP-OCRv6_medium_det_onnx"),
            _ => None,
        }
    }

    pub const fn pp_ocr_recognizer_repo(self) -> Option<&'static str> {
        match self {
            Self::PpOcrV5Server => Some("PaddlePaddle/PP-OCRv5_server_rec_onnx"),
            Self::PpOcrV6Tiny => Some("PaddlePaddle/PP-OCRv6_tiny_rec_onnx"),
            Self::PpOcrV6Small => Some("PaddlePaddle/PP-OCRv6_small_rec_onnx"),
            Self::PpOcrV6Medium => Some("PaddlePaddle/PP-OCRv6_medium_rec_onnx"),
            _ => None,
        }
    }

    pub const fn pp_doclayoutv3_onnx_repo(self) -> Option<&'static str> {
        match self {
            Self::PpDocLayoutV3 => Some("PaddlePaddle/PP-DocLayoutV3_onnx"),
            _ => None,
        }
    }

    pub const fn dictionary_file(self) -> Option<&'static str> {
        match self {
            Self::PpOcrV5Mobile | Self::PpOcrV5Server => Some("ppocrv5_dict.txt"),
            Self::PpOcrV6Tiny => Some("ppocrv6_tiny_dict.txt"),
            Self::PpOcrV6Small | Self::PpOcrV6Medium => Some("ppocrv6_dict.txt"),
            _ => None,
        }
    }
}

const PP_OCRV5_MOBILE_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_det.onnx",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_det.onnx",
        },
    },
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "pp-ocrv5_mobile_rec.onnx",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "pp-ocrv5_mobile_rec.onnx",
        },
    },
    ModelHubAsset {
        repo: "greatv/oar-ocr",
        file: "ppocrv5_dict.txt",
        kind: ModelHubAssetKind::ModelScopeFile {
            output_file: "ppocrv5_dict.txt",
        },
    },
];

const PP_OCRV5_SERVER_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv5_server_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv5_dict.txt",
        },
    },
];

const PP_OCRV6_TINY_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_tiny_dict.txt",
        },
    },
];

const PP_OCRV6_SMALL_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_dict.txt",
        },
    },
];

const PP_OCRV6_MEDIUM_ASSETS: &[ModelHubAsset] = &[
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        file: "inference.onnx",
        kind: ModelHubAssetKind::RequiredFile,
    },
    ModelHubAsset {
        repo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        file: "inference.yml",
        kind: ModelHubAssetKind::PaddleOcrDictionary {
            output_file: "ppocrv6_dict.txt",
        },
    },
];

const PP_DOCLAYOUTV3_ASSETS: &[ModelHubAsset] = &[ModelHubAsset {
    repo: "PaddlePaddle/PP-DocLayoutV3_onnx",
    file: "inference.onnx",
    kind: ModelHubAssetKind::RequiredFile,
}];

impl ModelSpec for KnownOcrModel {
    fn category(self) -> ModelCategory {
        match self.kind() {
            OcrModelKind::TraditionalOcr | OcrModelKind::Layout => ModelCategory::Ocr,
            OcrModelKind::OcrVl => ModelCategory::OcrVl,
        }
    }

    fn cache_key(self) -> &'static str {
        self.id()
    }

    fn huggingface_repo(self) -> &'static str {
        self.id()
    }

    fn modelscope_repo(self) -> &'static str {
        self.id()
    }

    fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::PpOcrV5Mobile => &[],
            Self::PpOcrV5Server => &["ppocrv5_dict.txt"],
            Self::PpOcrV6Tiny => &["ppocrv6_tiny_dict.txt"],
            Self::PpOcrV6Small | Self::PpOcrV6Medium => &["ppocrv6_dict.txt"],
            Self::PpDocLayoutV3 => &[],
            Self::PaddleOcrVl15 | Self::PaddleOcrVl16 => &[
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "chat_template.jinja",
            ],
        }
    }

    fn hub_assets(self) -> &'static [ModelHubAsset] {
        match self {
            Self::PpOcrV5Mobile => PP_OCRV5_MOBILE_ASSETS,
            Self::PpOcrV5Server => PP_OCRV5_SERVER_ASSETS,
            Self::PpOcrV6Tiny => PP_OCRV6_TINY_ASSETS,
            Self::PpOcrV6Small => PP_OCRV6_SMALL_ASSETS,
            Self::PpOcrV6Medium => PP_OCRV6_MEDIUM_ASSETS,
            Self::PpDocLayoutV3 => PP_DOCLAYOUTV3_ASSETS,
            _ => &[],
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
    fn pp_ocrv5_mobile_uses_modelscope_oar_registry_assets() {
        assert_eq!(
            KnownOcrModel::PpOcrV5Mobile.required_files(),
            &[] as &[&str]
        );
        assert_eq!(
            KnownOcrModel::PpOcrV5Mobile.hub_assets(),
            &[
                ModelHubAsset {
                    repo: "greatv/oar-ocr",
                    file: "pp-ocrv5_mobile_det.onnx",
                    kind: ModelHubAssetKind::ModelScopeFile {
                        output_file: "pp-ocrv5_mobile_det.onnx"
                    },
                },
                ModelHubAsset {
                    repo: "greatv/oar-ocr",
                    file: "pp-ocrv5_mobile_rec.onnx",
                    kind: ModelHubAssetKind::ModelScopeFile {
                        output_file: "pp-ocrv5_mobile_rec.onnx"
                    },
                },
                ModelHubAsset {
                    repo: "greatv/oar-ocr",
                    file: "ppocrv5_dict.txt",
                    kind: ModelHubAssetKind::ModelScopeFile {
                        output_file: "ppocrv5_dict.txt"
                    },
                },
            ]
        );
    }
}
