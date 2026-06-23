use orchion::{AsrModel, ModelDownloader, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let cache_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models".to_string());
    let model = AsrModel::parse("Qwen/Qwen3-ASR-0.6B").expect("example model id is valid");
    let path = ModelDownloader::default()
        .download(model, cache_dir)
        .await?;
    println!("downloaded model to {}", path.display());
    Ok(())
}
