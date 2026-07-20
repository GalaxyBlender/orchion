use orchion_audio::{
    AudioInputFormat, AudioOutputFormat, FfmpegAudioCodec, StreamingAudioDecoder,
    decode_audio_bytes, decode_pcm_s16le_bytes, encode_tts_audio,
};
use orchion_core::{ASR_SAMPLE_RATE, OrchionError, TtsAudio};
use std::fs;
use std::io::{Cursor, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(unix)]
use std::time::{Duration, Instant};

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

#[test]
fn audio_input_format_parses_streaming_values() {
    assert_eq!(
        "pcm_s16le".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::PcmS16Le
    );
    assert_eq!(
        "webm_opus".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::WebmOpus
    );
    assert_eq!(
        "mp3".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Mp3
    );
    assert_eq!(
        "wav".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Wav
    );
    assert_eq!(
        "auto".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Auto
    );
    assert_eq!(
        "m4a".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::M4a
    );
    assert_eq!(
        "aac".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Aac
    );
    assert_eq!(
        "flac".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Flac
    );
    assert_eq!(
        "ogg".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Ogg
    );
    assert_eq!(
        "opus".parse::<AudioInputFormat>().unwrap(),
        AudioInputFormat::Ogg
    );
}

#[test]
fn decode_pcm_s16le_bytes_rejects_partial_sample() {
    let error = decode_pcm_s16le_bytes(&[0]).unwrap_err();

    assert!(matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("pcm_s16le")));
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
async fn decode_rejects_audio_above_sample_limit() {
    if !ffmpeg_available() {
        return;
    }
    let codec = FfmpegAudioCodec::default();

    let error = codec
        .decode_for_asr_with_max_samples(wav_bytes(), 100)
        .await
        .unwrap_err();

    assert!(
        matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("sample limit"))
    );
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
async fn encoded_wav_is_readable_by_hound() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5], 24_000);

    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .unwrap();
    let mut reader = hound::WavReader::new(Cursor::new(encoded.bytes)).unwrap();
    let samples = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(reader.spec().sample_rate, audio.sample_rate);
    assert_eq!(samples.len(), audio.samples.len());
}

#[tokio::test]
async fn encoded_pcm_is_s16le_without_container() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5, 1.0, -1.0], 24_000);

    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Pcm)
        .await
        .unwrap();

    assert_eq!(encoded.format, AudioOutputFormat::Pcm);
    assert_eq!(encoded.content_type, "audio/pcm");
    assert_eq!(encoded.bytes.len(), 10);
}

#[tokio::test]
async fn streaming_decoder_decodes_mp3_chunks_to_asr_pcm() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(sine_samples(24_000), 24_000);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Mp3)
        .await
        .unwrap();
    let mut decoder = StreamingAudioDecoder::new_for_asr(AudioInputFormat::Mp3, None)
        .await
        .unwrap();
    let mut samples = Vec::new();

    for chunk in encoded.bytes.chunks(257) {
        let decoded = decoder.push(chunk).await.unwrap();
        assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
        samples.extend(decoded.samples);
    }
    let decoded = decoder.finish().await.unwrap();
    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    samples.extend(decoded.samples);

    assert!(!samples.is_empty());
}

#[tokio::test]
async fn streaming_decoder_decodes_webm_opus_chunks_to_asr_pcm() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(sine_samples(24_000), 24_000);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Opus)
        .await
        .unwrap();
    let mut decoder = StreamingAudioDecoder::new_for_asr(AudioInputFormat::WebmOpus, None)
        .await
        .unwrap();
    let mut samples = Vec::new();

    for chunk in encoded.bytes.chunks(257) {
        let decoded = decoder.push(chunk).await.unwrap();
        assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
        samples.extend(decoded.samples);
    }
    let decoded = decoder.finish().await.unwrap();
    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    samples.extend(decoded.samples);

    assert!(!samples.is_empty());
}

#[tokio::test]
async fn streaming_decoder_decodes_wav_chunks_to_asr_pcm() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(sine_samples(24_000), 24_000);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .unwrap();
    let mut decoder = StreamingAudioDecoder::new_for_asr(AudioInputFormat::Wav, None)
        .await
        .unwrap();
    let mut samples = Vec::new();

    for chunk in encoded.bytes.chunks(257) {
        let decoded = decoder.push(chunk).await.unwrap();
        assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
        samples.extend(decoded.samples);
    }
    let decoded = decoder.finish().await.unwrap();
    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    samples.extend(decoded.samples);

    assert!(!samples.is_empty());
}

#[tokio::test]
async fn streaming_decoder_decodes_m4a_chunks_to_asr_pcm() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(sine_samples(24_000), 24_000);
    let encoded = encode_m4a_for_test(&audio);
    let mut decoder = StreamingAudioDecoder::new_for_asr(AudioInputFormat::M4a, None)
        .await
        .unwrap();
    let mut samples = Vec::new();

    for chunk in encoded.chunks(257) {
        let decoded = decoder.push(chunk).await.unwrap();
        assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
        samples.extend(decoded.samples);
    }
    let decoded = decoder.finish().await.unwrap();
    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    samples.extend(decoded.samples);

    assert!(!samples.is_empty());
}

#[tokio::test]
async fn streaming_decoder_decodes_regular_m4a_file_chunks_to_asr_pcm() {
    if !ffmpeg_available() {
        return;
    }
    let audio = TtsAudio::new(sine_samples(24_000), 24_000);
    let encoded = encode_regular_m4a_file_for_test(&audio);
    let mut decoder = StreamingAudioDecoder::new_for_asr(AudioInputFormat::M4a, None)
        .await
        .unwrap();
    let mut samples = Vec::new();

    for chunk in encoded.chunks(257) {
        let decoded = decoder.push(chunk).await.unwrap();
        assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
        samples.extend(decoded.samples);
    }
    let decoded = decoder.finish().await.unwrap();
    assert_eq!(decoded.sample_rate, ASR_SAMPLE_RATE);
    samples.extend(decoded.samples);

    assert!(!samples.is_empty());
}

#[tokio::test]
#[cfg(unix)]
async fn streaming_decoder_rejects_non_empty_input_with_empty_ffmpeg_output() {
    let fake_ffmpeg = fake_successful_ffmpeg_without_output();
    let mut decoder = StreamingAudioDecoder::new_for_asr_with_binary(
        AudioInputFormat::M4a,
        None,
        fake_ffmpeg.clone(),
    )
    .await
    .unwrap();

    decoder.push(b"not empty").await.unwrap();
    let error = decoder.finish().await.unwrap_err();
    fs::remove_file(fake_ffmpeg).unwrap();

    assert!(
        matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("empty sample buffer"))
    );
}

#[tokio::test(flavor = "current_thread")]
#[cfg(unix)]
async fn streaming_decoder_push_yields_when_ffmpeg_stops_reading() {
    let fake_ffmpeg = fake_ffmpeg_with_stalled_stdin();
    let mut decoder = StreamingAudioDecoder::new_for_asr_with_binary(
        AudioInputFormat::M4a,
        None,
        fake_ffmpeg.clone(),
    )
    .await
    .unwrap();
    let input = vec![0_u8; 4 * 1024 * 1024];
    let started_at = Instant::now();

    let result = tokio::time::timeout(Duration::from_millis(50), decoder.push(&input)).await;

    assert!(result.is_err(), "the stalled write unexpectedly completed");
    assert!(
        started_at.elapsed() < Duration::from_millis(500),
        "push blocked the Tokio runtime instead of yielding"
    );
    drop(decoder);
    fs::remove_file(fake_ffmpeg).unwrap();
}

#[tokio::test]
#[cfg(unix)]
async fn streaming_decoder_rejects_ffmpeg_output_before_it_exceeds_sample_limit() {
    let fake_ffmpeg = fake_ffmpeg_with_expanding_output();
    let mut decoder = StreamingAudioDecoder::new_for_asr_with_binary_and_max_samples(
        AudioInputFormat::M4a,
        None,
        fake_ffmpeg.clone(),
        8,
    )
    .await
    .unwrap();

    decoder.push(b"x").await.unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), decoder.finish())
        .await
        .expect("bounded decoder should reject expanding output without blocking")
        .unwrap_err();
    fs::remove_file(fake_ffmpeg).unwrap();

    assert!(matches!(
        error,
        OrchionError::InvalidAudio { reason }
            if reason == "streaming decoded audio exceeded the sample limit"
    ));
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

fn sine_samples(sample_rate: u32) -> Vec<f32> {
    (0..sample_rate)
        .map(|index| {
            let phase = index as f32 / sample_rate as f32 * 440.0 * std::f32::consts::TAU;
            phase.sin() * 0.25
        })
        .collect()
}

fn encode_m4a_for_test(audio: &TtsAudio) -> Vec<u8> {
    let input = audio
        .samples
        .iter()
        .flat_map(|sample| {
            let sample = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            sample.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let mut child = Command::new(Path::new("ffmpeg"))
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            &audio.sample_rate.to_string(),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-acodec",
            "aac",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "ffmpeg m4a encode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn encode_regular_m4a_file_for_test(audio: &TtsAudio) -> Vec<u8> {
    let input = audio
        .samples
        .iter()
        .flat_map(|sample| {
            let sample = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            sample.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "orchion-test-{}-{}.m4a",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut child = Command::new(Path::new("ffmpeg"))
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            &audio.sample_rate.to_string(),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-acodec",
            "aac",
            path.to_string_lossy().as_ref(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "ffmpeg regular m4a encode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    bytes
}

#[cfg(unix)]
fn fake_successful_ffmpeg_without_output() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "orchion-fake-ffmpeg-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, "#!/bin/sh\ncat >/dev/null\nexit 0\n").unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    path
}

#[cfg(unix)]
fn fake_ffmpeg_with_stalled_stdin() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "orchion-stalled-ffmpeg-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, "#!/bin/sh\nsleep 2\nexit 0\n").unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    path
}

#[cfg(unix)]
fn fake_ffmpeg_with_expanding_output() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "orchion-expanding-ffmpeg-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(
        &path,
        "#!/bin/sh\ndd bs=1 count=1 of=/dev/null 2>/dev/null\ndd if=/dev/zero bs=4096 count=4 2>/dev/null\ncat >/dev/null\n",
    )
    .unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    path
}
