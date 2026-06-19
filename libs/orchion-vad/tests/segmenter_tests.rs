use orchion_core::ASR_SAMPLE_RATE;
use orchion_vad::{AudioSegment, VadConfig, WebRtcVadMode, WebRtcVadSegmenter};

#[test]
fn web_rtc_vad_returns_no_segments_for_silence() {
    let segmenter = WebRtcVadSegmenter::default();
    let samples = vec![0.0; ASR_SAMPLE_RATE as usize];

    let segments = segmenter.segment(&samples, ASR_SAMPLE_RATE).unwrap();

    assert!(segments.is_empty());
}

#[test]
fn post_processing_merges_short_silence_between_speech_runs() {
    let config = VadConfig {
        frame_duration_ms: 30,
        padding_ms: 0,
        min_speech_ms: 60,
        min_silence_ms: 90,
        merge_gap_ms: 60,
        mode: WebRtcVadMode::Quality,
    };
    let frame_samples = 480;
    let flags = [true, true, false, true, true, false, false, false];

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![AudioSegment {
            start_sample: 0,
            end_sample: 5 * frame_samples,
        }]
    );
}

#[test]
fn post_processing_filters_short_speech_runs() {
    let config = VadConfig {
        frame_duration_ms: 30,
        padding_ms: 0,
        min_speech_ms: 90,
        min_silence_ms: 60,
        merge_gap_ms: 0,
        mode: WebRtcVadMode::Quality,
    };
    let frame_samples = 480;
    let flags = [true, false, false, true, true, true];

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![AudioSegment {
            start_sample: 3 * frame_samples,
            end_sample: 6 * frame_samples,
        }]
    );
}

#[test]
fn post_processing_applies_padding_and_clamps_to_audio_bounds() {
    let config = VadConfig {
        frame_duration_ms: 30,
        padding_ms: 30,
        min_speech_ms: 30,
        min_silence_ms: 60,
        merge_gap_ms: 0,
        mode: WebRtcVadMode::Quality,
    };
    let frame_samples = 480;
    let flags = [false, true, false, false];

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![AudioSegment {
            start_sample: 0,
            end_sample: 3 * frame_samples,
        }]
    );
}

#[test]
fn default_post_processing_keeps_clear_pause_between_segments() {
    let config = VadConfig::default();
    let frame_samples = 480;
    let mut flags = Vec::new();
    flags.extend([true; 10]);
    flags.extend([false; 30]);
    flags.extend([true; 10]);

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![
            AudioSegment {
                start_sample: 0,
                end_sample: 20 * frame_samples,
            },
            AudioSegment {
                start_sample: 30 * frame_samples,
                end_sample: 50 * frame_samples,
            },
        ]
    );
}

#[test]
fn default_post_processing_keeps_short_phrase_pause_in_same_segment() {
    let config = VadConfig::default();
    let frame_samples = 480;
    let mut flags = Vec::new();
    flags.extend([true; 12]);
    flags.extend([false; 15]);
    flags.extend([true; 12]);

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![AudioSegment {
            start_sample: 0,
            end_sample: 39 * frame_samples,
        }]
    );
}

#[test]
fn default_post_processing_splits_clear_sentence_pause() {
    let config = VadConfig::default();
    let frame_samples = 480;
    let mut flags = Vec::new();
    flags.extend([true; 12]);
    flags.extend([false; 20]);
    flags.extend([true; 12]);

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![
            AudioSegment {
                start_sample: 0,
                end_sample: 22 * frame_samples,
            },
            AudioSegment {
                start_sample: 22 * frame_samples,
                end_sample: 44 * frame_samples,
            },
        ]
    );
}

#[test]
fn default_post_processing_keeps_brief_pause_in_same_segment() {
    let config = VadConfig::default();
    let frame_samples = 480;
    let mut flags = Vec::new();
    flags.extend([true; 12]);
    flags.extend([false; 4]);
    flags.extend([true; 12]);

    let segments = orchion_vad::segments_from_speech_flags(
        &flags,
        flags.len() * frame_samples,
        frame_samples,
        &config,
    );

    assert_eq!(
        segments,
        vec![AudioSegment {
            start_sample: 0,
            end_sample: 28 * frame_samples,
        }]
    );
}

#[test]
fn rejects_non_asr_sample_rate() {
    let segmenter = WebRtcVadSegmenter::default();
    let error = segmenter.segment(&[0.0; 160], 8_000).unwrap_err();

    assert!(error.to_string().contains("16000"));
}
