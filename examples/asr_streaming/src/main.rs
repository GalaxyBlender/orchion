use orchion::{Asr, AsrModel, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let audio_path = args
        .next()
        .expect("usage: asr_streaming <mono.wav> [cache_dir]");
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());

    let mut reader = hound::WavReader::open(audio_path).map_err(|source| {
        orchion::OrchionError::InvalidAudio {
            reason: source.to_string(),
        }
    })?;
    let sample_rate = reader.spec().sample_rate;
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|value| f32::from(value) / f32::from(i16::MAX)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|source| orchion::OrchionError::InvalidAudio {
            reason: source.to_string(),
        })?;

    let asr = Asr::load_or_download(AsrModel::Qwen3Asr06B, cache_dir).await?;
    let mut stream = asr.start_streaming().await?;
    for chunk in samples.chunks(sample_rate as usize) {
        if let Some(partial) = stream.feed(chunk, sample_rate).await? {
            println!("partial: {}", partial.text);
        }
    }
    let final_transcript = stream.finish().await?;
    println!("final: {}", final_transcript.text);
    Ok(())
}
