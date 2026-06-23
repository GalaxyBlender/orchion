use orchion::{Result, Tts, TtsLanguage, TtsModel, TtsSpeaker, TtsVoice};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from Orchion.".to_string());
    let output_path = args.next().unwrap_or_else(|| "output.wav".to_string());
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());

    let model =
        TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").expect("example model id is valid");
    let tts = Tts::load_or_download(model, cache_dir).await?;
    tts.synthesize_to_file(
        text,
        TtsVoice::Preset {
            speaker: TtsSpeaker::Ryan,
            language: TtsLanguage::English,
        },
        output_path,
    )
    .await?;
    Ok(())
}
