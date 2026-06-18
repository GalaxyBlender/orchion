#![cfg(feature = "download")]

use orchion::{AsrModel, ModelDownloader, Result};

#[tokio::test]
#[ignore = "downloads a real model"]
async fn downloads_real_asr_model() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let path = ModelDownloader::default()
        .download(AsrModel::Qwen3Asr06B, dir.path())
        .await?;

    assert!(path.join(".orchion-complete").exists());
    Ok(())
}
