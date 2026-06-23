use orchion::{Asr, AsrModel, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let audio_path = args
        .next()
        .expect("usage: asr_file <audio.wav> [cache_dir]");
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());

    let model = AsrModel::parse("Qwen/Qwen3-ASR-0.6B").expect("example model id is valid");
    let asr = Asr::load_or_download(model, cache_dir).await?;
    let transcript = asr.transcribe_file(audio_path).await?;
    println!("{}", transcript.text);
    Ok(())
}
