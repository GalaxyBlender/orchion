use orchion_core::{KnownOcrModel, OcrModelAsset, OcrModelAssetKind};

pub(crate) type ModelHubAsset = OcrModelAsset;
pub(crate) type ModelHubAssetKind = OcrModelAssetKind;

pub(crate) fn for_model(repo: &str) -> &'static [ModelHubAsset] {
    KnownOcrModel::ALL
        .into_iter()
        .find(|model| model.id() == repo)
        .map_or(&[], KnownOcrModel::download_assets)
}

pub(crate) fn uses_modelscope_file_assets(assets: &[ModelHubAsset]) -> bool {
    assets
        .iter()
        .any(|asset| matches!(asset.kind, ModelHubAssetKind::ModelScopeFile { .. }))
}
