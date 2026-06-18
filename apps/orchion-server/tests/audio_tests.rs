use orchion::{AudioOutputFormat, TtsAudio, encode_tts_audio};
use std::process::Command;

#[tokio::test]
async fn encodes_tts_audio_as_wav_bytes() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5], 24_000);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .unwrap();

    assert!(encoded.bytes.starts_with(b"RIFF"));
    assert_eq!(&encoded.bytes[8..12], b"WAVE");
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
