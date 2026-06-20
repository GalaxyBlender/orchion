#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

use orchion_core::{ASR_SAMPLE_RATE, OrchionError, Result};
use wavekat_vad::backends::webrtc::WebRtcVad;
pub use wavekat_vad::backends::webrtc::WebRtcVadMode;
use wavekat_vad::{VadError, VoiceActivityDetector};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSegment {
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameSegment {
    start_frame: usize,
    end_frame: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VadConfig {
    pub frame_duration_ms: u32,
    pub padding_ms: u32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub merge_gap_ms: u32,
    pub mode: WebRtcVadMode,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 30,
            padding_ms: 300,
            min_speech_ms: 300,
            min_silence_ms: 540,
            merge_gap_ms: 150,
            mode: WebRtcVadMode::Quality,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WebRtcVadSegmenter {
    config: VadConfig,
}

impl WebRtcVadSegmenter {
    pub fn new(config: VadConfig) -> Self {
        Self { config }
    }

    pub fn segment(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<AudioSegment>> {
        if sample_rate != ASR_SAMPLE_RATE {
            return Err(OrchionError::InvalidAudio {
                reason: format!(
                    "WebRTC VAD requires {ASR_SAMPLE_RATE} Hz audio, got {sample_rate}"
                ),
            });
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let mut vad = WebRtcVad::with_frame_duration(
            ASR_SAMPLE_RATE,
            self.config.mode,
            self.config.frame_duration_ms,
        )
        .map_err(|error| vad_error_into_orchion(&error))?;
        let frame_size = vad.capabilities().frame_size;
        let mut speech_flags = Vec::with_capacity(samples.len().div_ceil(frame_size));
        let mut frame = vec![0_i16; frame_size];

        for chunk in samples.chunks(frame_size) {
            frame.fill(0);
            for (output, sample) in frame.iter_mut().zip(chunk.iter().copied()) {
                *output = f32_to_i16(sample);
            }
            let speech_probability = vad
                .process(&frame, ASR_SAMPLE_RATE)
                .map_err(|error| vad_error_into_orchion(&error))?;
            speech_flags.push(speech_probability > 0.5);
        }

        Ok(segments_from_speech_flags(
            &speech_flags,
            samples.len(),
            frame_size,
            &self.config,
        ))
    }
}

pub fn segments_from_speech_flags(
    speech_flags: &[bool],
    total_samples: usize,
    frame_samples: usize,
    config: &VadConfig,
) -> Vec<AudioSegment> {
    if speech_flags.is_empty() || total_samples == 0 || frame_samples == 0 {
        return Vec::new();
    }

    let min_speech_frames = frames_for_ms(config.min_speech_ms, config.frame_duration_ms).max(1);
    let mut segments = Vec::new();
    let mut index = 0;

    while index < speech_flags.len() {
        if !speech_flags[index] {
            index += 1;
            continue;
        }

        let start_frame = index;
        while index < speech_flags.len() && speech_flags[index] {
            index += 1;
        }
        let end_frame = index;
        if end_frame - start_frame < min_speech_frames {
            continue;
        }

        segments.push(FrameSegment {
            start_frame,
            end_frame,
        });
    }

    let merged_segments = merge_close_segments(segments, config);
    segments_into_audio_segments(
        &merged_segments,
        speech_flags.len(),
        total_samples,
        frame_samples,
        config,
    )
}

fn merge_close_segments(segments: Vec<FrameSegment>, config: &VadConfig) -> Vec<FrameSegment> {
    let mut merged: Vec<FrameSegment> = Vec::with_capacity(segments.len());
    let min_silence_frames = frames_for_ms(config.min_silence_ms, config.frame_duration_ms);
    let merge_gap_frames = frames_for_ms(config.merge_gap_ms, config.frame_duration_ms);

    for segment in segments {
        let Some(previous) = merged.last_mut() else {
            merged.push(segment);
            continue;
        };
        let gap_frames = segment.start_frame.saturating_sub(previous.end_frame);
        if gap_frames < min_silence_frames || gap_frames <= merge_gap_frames {
            previous.end_frame = previous.end_frame.max(segment.end_frame);
        } else {
            merged.push(segment);
        }
    }

    merged
}

fn segments_into_audio_segments(
    segments: &[FrameSegment],
    total_frames: usize,
    total_samples: usize,
    frame_samples: usize,
    config: &VadConfig,
) -> Vec<AudioSegment> {
    let padding_frames = frames_for_ms(config.padding_ms, config.frame_duration_ms);

    segments
        .iter()
        .map(|segment| {
            let padded_start = segment.start_frame.saturating_sub(padding_frames);
            let padded_end = (segment.end_frame + padding_frames).min(total_frames);
            AudioSegment {
                start_sample: padded_start * frame_samples,
                end_sample: (padded_end * frame_samples).min(total_samples),
            }
        })
        .collect()
}

fn frames_for_ms(milliseconds: u32, frame_duration_ms: u32) -> usize {
    if milliseconds == 0 || frame_duration_ms == 0 {
        0
    } else {
        milliseconds.div_ceil(frame_duration_ms) as usize
    }
}

#[allow(clippy::cast_possible_truncation)]
fn f32_to_i16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    if sample <= -1.0 {
        return i16::MIN;
    }
    if sample >= 1.0 {
        return i16::MAX;
    }
    (sample * f32::from(i16::MAX)).round() as i16
}

fn vad_error_into_orchion(error: &VadError) -> OrchionError {
    OrchionError::InvalidAudio {
        reason: format!("WebRTC VAD failed: {error}"),
    }
}
