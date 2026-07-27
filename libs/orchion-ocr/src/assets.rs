use orchion_core::{
    KnownOcrModel, OcrModelAsset, OcrModelAssetKind, OcrModelAssetRole, OrchionError, Result,
};
use std::path::{Path, PathBuf};

/// Complete local paths needed to load an OCR runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrAssets {
    /// Detector, recognizer, dictionary, and optional document layout model.
    Traditional {
        detector: PathBuf,
        recognizer: PathBuf,
        dictionary: PathBuf,
        layout: Option<PathBuf>,
    },
    /// Standalone document layout model.
    Layout { model: PathBuf },
    /// OCR-VL model directory and optional document layout model.
    VisionLanguage {
        model_dir: PathBuf,
        layout: Option<PathBuf>,
    },
}

impl OcrAssets {
    #[must_use]
    pub fn with_layout(self, layout: Option<PathBuf>) -> Self {
        match self {
            Self::Traditional {
                detector,
                recognizer,
                dictionary,
                ..
            } => Self::Traditional {
                detector,
                recognizer,
                dictionary,
                layout,
            },
            Self::VisionLanguage { model_dir, .. } => Self::VisionLanguage { model_dir, layout },
            Self::Layout { model } => Self::Layout { model },
        }
    }

    /// Resolves runtime assets from the downloader's stable cache layout.
    ///
    /// # Panics
    ///
    /// Panics if a traditional OCR model descriptor is missing its detector,
    /// recognizer, or dictionary metadata.
    #[must_use]
    pub fn from_cache_layout(
        model: KnownOcrModel,
        model_dir: impl AsRef<Path>,
        cache_root: impl AsRef<Path>,
    ) -> Self {
        let model_dir = model_dir.as_ref();
        let cache_root = cache_root.as_ref();
        let layout = asset_path(
            required_asset(KnownOcrModel::PpDocLayoutV3, OcrModelAssetRole::Layout),
            model_dir,
            cache_root,
        );

        match model {
            KnownOcrModel::PpOcrV5Mobile
            | KnownOcrModel::PpOcrV5Server
            | KnownOcrModel::PpOcrV6Tiny
            | KnownOcrModel::PpOcrV6Small
            | KnownOcrModel::PpOcrV6Medium => Self::Traditional {
                detector: asset_path(
                    required_asset(model, OcrModelAssetRole::Detector),
                    model_dir,
                    cache_root,
                ),
                recognizer: asset_path(
                    required_asset(model, OcrModelAssetRole::Recognizer),
                    model_dir,
                    cache_root,
                ),
                dictionary: asset_path(
                    required_asset(model, OcrModelAssetRole::Dictionary),
                    model_dir,
                    cache_root,
                ),
                layout: layout.is_file().then_some(layout),
            },
            KnownOcrModel::PpDocLayoutV3 => Self::Layout { model: layout },
            KnownOcrModel::PaddleOcrVl15 | KnownOcrModel::PaddleOcrVl16 => Self::VisionLanguage {
                model_dir: model_dir.to_path_buf(),
                layout: layout.is_file().then_some(layout),
            },
        }
    }

    pub(crate) fn from_legacy_cache_layout(model: KnownOcrModel, model_dir: &Path) -> Result<Self> {
        let cache_root =
            model_dir
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| OrchionError::ModelLoad {
                    message: format!(
                        "cannot derive shared model root from OCR model cache path `{}`",
                        model_dir.display()
                    ),
                })?;
        Ok(Self::from_cache_layout(model, model_dir, cache_root))
    }
}

fn required_asset(model: KnownOcrModel, role: OcrModelAssetRole) -> OcrModelAsset {
    model
        .download_assets()
        .iter()
        .find(|asset| asset.role == role)
        .copied()
        .unwrap_or_else(|| panic!("{} descriptor is missing its {role:?} asset", model.id()))
}

fn asset_path(asset: OcrModelAsset, model_dir: &Path, cache_root: &Path) -> PathBuf {
    match asset.kind {
        OcrModelAssetKind::RequiredFile => cached_repo_file(cache_root, asset.repo, asset.file),
        OcrModelAssetKind::PaddleOcrDictionary { output_file } => model_dir.join(output_file),
        OcrModelAssetKind::ModelScopeFile { output_file } => {
            cached_repo_file(cache_root, asset.repo, output_file)
        }
    }
}

fn cached_repo_file(cache_root: &Path, repo: &str, file: &str) -> PathBuf {
    repo.split('/')
        .fold(cache_root.to_path_buf(), |path, segment| path.join(segment))
        .join(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_assets_preserve_downloader_cache_layout() {
        let assets = OcrAssets::from_legacy_cache_layout(
            KnownOcrModel::PpOcrV6Tiny,
            Path::new("models/PaddlePaddle/PP-OCRv6_tiny"),
        )
        .unwrap();

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
            Path::new("models/PaddlePaddle/PP-OCRv6_tiny_det_onnx/inference.onnx")
        );
        assert_eq!(
            recognizer,
            Path::new("models/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/inference.onnx")
        );
        assert_eq!(
            dictionary,
            Path::new("models/PaddlePaddle/PP-OCRv6_tiny/ppocrv6_tiny_dict.txt")
        );
        assert_eq!(layout, None);
    }
}
