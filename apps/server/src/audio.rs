use anyhow::Context;
use orchion::TtsAudio;
use std::io::Cursor;
use tempfile::TempPath;

pub fn encode_wav(audio: &TtsAudio) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).context("create WAV writer")?;
        for sample in &audio.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let value = (clamped * f32::from(i16::MAX)) as i16;
            writer.write_sample(value).context("write WAV sample")?;
        }
        writer.finalize().context("finalize WAV writer")?;
    }
    Ok(cursor.into_inner())
}

pub async fn write_temp_file(bytes: &[u8], suffix: &str) -> anyhow::Result<TempPath> {
    let file = tempfile::Builder::new()
        .prefix("orchion-upload-")
        .suffix(suffix)
        .tempfile()
        .context("create temporary upload file")?;
    let path = file.into_temp_path();
    tokio::fs::write(&path, bytes)
        .await
        .context("write temporary upload file")?;
    Ok(path)
}
