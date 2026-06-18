use orchion::{
    ASR_SAMPLE_RATE, AudioOutputFormat, FfmpegAudioCodec, OrchionError, TtsAudio,
    decode_audio_bytes, encode_tts_audio,
};
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn audio_output_format_parses_openai_values() {
    let cases = [
        ("wav", AudioOutputFormat::Wav, "audio/wav"),
        ("mp3", AudioOutputFormat::Mp3, "audio/mpeg"),
        ("aac", AudioOutputFormat::Aac, "audio/aac"),
        ("opus", AudioOutputFormat::Opus, "audio/ogg"),
        ("flac", AudioOutputFormat::Flac, "audio/flac"),
        ("pcm", AudioOutputFormat::Pcm, "audio/pcm"),
    ];

    for (value, expected, content_type) in cases {
        let format = value.parse::<AudioOutputFormat>().unwrap();
        assert_eq!(format, expected);
        assert_eq!(format.content_type(), content_type);
    }
}

#[tokio::test]
async fn missing_ffmpeg_reports_invalid_audio() {
    let codec = FfmpegAudioCodec::new(PathBuf::from("/definitely/missing/orchion-ffmpeg"));

    let error = codec.decode_for_asr(vec![0, 1, 2, 3]).await.unwrap_err();

    assert!(
        matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("ffmpeg") && reason.contains("not found"))
    );
}

#[tokio::test]
async fn decodes_audio_bytes_to_asr_pcm_with_ffmpeg() {
    if !ffmpeg_available() {
        return;
    }
    let wav = wav_bytes();

    let decoded = decode_audio_bytes(wav).await.unwrap();

    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    assert!(!decoded.samples.is_empty());
}

#[tokio::test]
async fn encodes_tts_audio_as_wav_with_ffmpeg() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5], 24_000);

    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .unwrap();

    assert_eq!(encoded.format, AudioOutputFormat::Wav);
    assert_eq!(encoded.content_type, "audio/wav");
    assert!(encoded.bytes.starts_with(b"RIFF"));
    assert_eq!(&encoded.bytes[8..12], b"WAVE");
}

#[tokio::test]
async fn encoded_wav_preserves_tts_sample_amplitude() {
    if !ffmpeg_available() {
        return;
    }
    let samples = sine_samples(24_000, 440.0, 0.25, 0.1);
    let audio = TtsAudio::new(samples.clone(), 24_000);

    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .unwrap();
    let decoded = wav_f32_samples(&encoded.bytes);

    assert_eq!(decoded.len(), samples.len());
    assert!((rms(&decoded) - rms(&samples)).abs() < 0.01);
    assert!(decoded.iter().all(|sample| sample.abs() <= 0.26));
}

#[tokio::test]
async fn encoded_pcm_is_s16le_without_container() {
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5, 1.0, -1.0], 24_000);

    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Pcm)
        .await
        .unwrap();

    assert_eq!(encoded.format, AudioOutputFormat::Pcm);
    assert_eq!(encoded.content_type, "audio/pcm");
    assert_eq!(encoded.bytes.len(), 10);
    let samples = encoded
        .bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    assert_eq!(samples, vec![0, 16383, -16383, 32767, -32767]);
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn wav_bytes() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for index in 0..2_400 {
            let phase = index as f32 / 24_000.0 * 440.0 * std::f32::consts::TAU;
            let sample = (phase.sin() * f32::from(i16::MAX) * 0.25) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

fn sine_samples(sample_rate: u32, frequency: f32, amplitude: f32, duration_sec: f32) -> Vec<f32> {
    let sample_count = (sample_rate as f32 * duration_sec) as usize;
    (0..sample_count)
        .map(|index| {
            let phase = index as f32 / sample_rate as f32 * frequency * std::f32::consts::TAU;
            phase.sin() * amplitude
        })
        .collect()
}

fn wav_f32_samples(bytes: &[u8]) -> Vec<f32> {
    let cursor = Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).unwrap();
    reader
        .into_samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / f32::from(i16::MAX))
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    let sum = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}
