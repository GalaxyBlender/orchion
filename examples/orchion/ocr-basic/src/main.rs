use orchion::{KnownOcrModel, Ocr, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let image = args.next().expect("usage: ocr-basic <image> [cache_dir]");
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());

    let model = KnownOcrModel::PpOcrV6Tiny.into_model();
    let ocr = Ocr::load_or_download(model, cache_dir).await?;
    let result = ocr.recognize_file(image).await?;

    println!("text: {}", result.text);
    println!("regions: {:#?}", result.regions);
    Ok(())
}
