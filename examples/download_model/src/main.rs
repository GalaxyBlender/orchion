use orchion::{AsrModel, ModelDownloader, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let cache_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models".to_string());
    let path = ModelDownloader::default()
        .download(AsrModel::Qwen3Asr06B, cache_dir)
        .await?;
    println!("downloaded model to {}", path.display());
    Ok(())
}
