#![cfg(feature = "audio-vad")]

use orchion::{ASR_SAMPLE_RATE, AudioVadConfig, AudioVadMode, AudioVadSegment, AudioVadSegmenter};

#[test]
fn audio_vad_api_exposes_stable_names() {
    let config = AudioVadConfig::default();
    let segment = AudioVadSegment {
        start_sample: 16_000,
        end_sample: 32_000,
    };

    assert_eq!(config.frame_duration_ms, 30);
    assert_eq!(config.padding_ms, 300);
    assert_eq!(config.min_speech_ms, 300);
    assert_eq!(config.min_silence_ms, 540);
    assert_eq!(config.merge_gap_ms, 150);
    assert_eq!(config.mode, AudioVadMode::Quality);
    assert_eq!(segment.start_sample, 16_000);
    assert_eq!(segment.end_sample, 32_000);
}

#[test]
fn audio_vad_segmenter_returns_stable_segments() {
    let segmenter = AudioVadSegmenter::new(AudioVadConfig::default());
    let samples = vec![0.0; ASR_SAMPLE_RATE as usize];

    let segments = segmenter.segment(&samples, ASR_SAMPLE_RATE).unwrap();

    assert!(segments.is_empty());
}
