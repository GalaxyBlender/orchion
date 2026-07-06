use crate::vad_error_into_orchion;
use orchion_core::{ASR_SAMPLE_RATE, OrchionError, Result};
use std::collections::VecDeque;
use wavekat_vad::VoiceActivityDetector;
use wavekat_vad::backends::webrtc::{WebRtcVad, WebRtcVadMode};

const MAX_CANDIDATE_MS: u32 = 60_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingVadConfig {
    pub frame_duration_ms: u32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub max_segment_ms: u32,
    pub speech_padding_ms: u32,
    pub mode: WebRtcVadMode,
}

impl Default for StreamingVadConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 30,
            min_speech_ms: 300,
            min_silence_ms: 500,
            max_segment_ms: 120_000,
            speech_padding_ms: 200,
            mode: WebRtcVadMode::Quality,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamingVadEvent {
    SegmentStarted {
        start_sample: usize,
        samples: Vec<f32>,
    },
    Audio {
        samples: Vec<f32>,
    },
    SegmentFinal {
        start_sample: usize,
        end_sample: usize,
    },
}

pub struct WebRtcStreamingVadEndpoint {
    vad: WebRtcVad,
    state: StreamingVadState,
    pending_samples: Vec<f32>,
}

impl WebRtcStreamingVadEndpoint {
    pub fn new(config: StreamingVadConfig) -> Result<Self> {
        let combined_candidate_ms = validate_streaming_vad_config(&config)?;
        let vad =
            WebRtcVad::with_frame_duration(ASR_SAMPLE_RATE, config.mode, config.frame_duration_ms)
                .map_err(|error| vad_error_into_orchion(&error))?;
        let frame_samples = vad.capabilities().frame_size;
        validate_candidate_sample_budget(&config, combined_candidate_ms, frame_samples)?;

        Ok(Self {
            vad,
            state: StreamingVadState::new(config, frame_samples),
            pending_samples: Vec::new(),
        })
    }

    pub fn push(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<StreamingVadEvent>> {
        if sample_rate != ASR_SAMPLE_RATE {
            return Err(OrchionError::InvalidAudio {
                reason: format!(
                    "WebRTC streaming VAD requires {ASR_SAMPLE_RATE} Hz audio, got {sample_rate}"
                ),
            });
        }

        self.pending_samples.extend_from_slice(samples);

        let frame_samples = self.vad.capabilities().frame_size;
        let mut events = Vec::new();
        while self.pending_samples.len() >= frame_samples {
            let frame = self.pending_samples.drain(..frame_samples).collect();
            events.extend(self.process_frame(frame)?);
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Vec<StreamingVadEvent> {
        let mut events = Vec::new();
        let frame_samples = self.vad.capabilities().frame_size;

        if !self.pending_samples.is_empty() && self.state.should_process_incomplete_frame() {
            let frame = std::mem::take(&mut self.pending_samples);
            let mut vad_frame = frame.clone();
            vad_frame.resize(frame_samples, 0.0);
            let is_speech = self.process_finish_frame(&vad_frame);
            events.extend(self.state.push_flag(is_speech, frame));
        } else {
            self.pending_samples.clear();
        }

        events.extend(self.state.finish());
        events
    }

    fn process_frame(&mut self, frame: Vec<f32>) -> Result<Vec<StreamingVadEvent>> {
        let frame_i16 = frame
            .iter()
            .copied()
            .map(crate::f32_to_i16)
            .collect::<Vec<_>>();
        let speech_probability = self
            .vad
            .process(&frame_i16, ASR_SAMPLE_RATE)
            .map_err(|error| vad_error_into_orchion(&error))?;

        Ok(self.state.push_flag(speech_probability > 0.5, frame))
    }

    fn process_finish_frame(&mut self, frame: &[f32]) -> bool {
        let frame_i16 = frame
            .iter()
            .copied()
            .map(crate::f32_to_i16)
            .collect::<Vec<_>>();
        // finish builds this frame from vad.capabilities().frame_size; if the backend
        // still rejects it, finish cannot return Result, so treat it as non-speech.
        self.vad
            .process(&frame_i16, ASR_SAMPLE_RATE)
            .is_ok_and(|speech_probability| speech_probability > 0.5)
    }
}

fn validate_streaming_vad_config(config: &StreamingVadConfig) -> Result<u32> {
    if config.frame_duration_ms == 0 {
        return Err(invalid_streaming_config(
            "frame_duration_ms must be greater than 0",
        ));
    }
    if config.min_speech_ms == 0 {
        return Err(invalid_streaming_config(
            "min_speech_ms must be greater than 0",
        ));
    }
    if config.min_silence_ms == 0 {
        return Err(invalid_streaming_config(
            "min_silence_ms must be greater than 0",
        ));
    }
    if config.max_segment_ms < config.min_speech_ms {
        return Err(invalid_streaming_config(
            "max_segment_ms must be greater than or equal to min_speech_ms",
        ));
    }
    config
        .speech_padding_ms
        .checked_add(config.min_speech_ms)
        .ok_or_else(|| invalid_streaming_config("speech_padding_ms + min_speech_ms overflowed"))
        .and_then(|combined_candidate_ms| {
            if combined_candidate_ms > MAX_CANDIDATE_MS {
                return Err(invalid_streaming_config(
                    "speech_padding_ms + min_speech_ms exceeds maximum candidate retention",
                ));
            }

            Ok(combined_candidate_ms)
        })
}

fn validate_candidate_sample_budget(
    config: &StreamingVadConfig,
    combined_candidate_ms: u32,
    frame_samples: usize,
) -> Result<()> {
    let candidate_samples = u64::from(combined_candidate_ms) * u64::from(ASR_SAMPLE_RATE) / 1_000;
    if candidate_samples < frame_samples as u64 {
        return Err(invalid_streaming_config(
            "speech_padding_ms + min_speech_ms must allow at least one VAD frame",
        ));
    }

    let min_speech_frames = frames_for_ms(config.min_speech_ms, config.frame_duration_ms).max(1);
    let min_speech_frame_samples = min_speech_frames as u64 * frame_samples as u64;
    if candidate_samples < min_speech_frame_samples {
        return Err(invalid_streaming_config(
            "speech_padding_ms + min_speech_ms must hold the rounded minimum speech frames",
        ));
    }

    Ok(())
}

fn invalid_streaming_config(reason: &'static str) -> OrchionError {
    OrchionError::InvalidAudio {
        reason: format!("invalid streaming VAD config: {reason}"),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StreamingVadState {
    config: StreamingVadConfig,
    frame_samples: usize,
    next_sample: usize,
    phase: StreamingVadPhase,
    pre_speech: VecDeque<BufferedFrame>,
    candidate: VecDeque<BufferedFrame>,
}

impl StreamingVadState {
    pub(crate) fn new(config: StreamingVadConfig, frame_samples: usize) -> Self {
        Self {
            config,
            frame_samples,
            next_sample: 0,
            phase: StreamingVadPhase::Idle,
            pre_speech: VecDeque::new(),
            candidate: VecDeque::new(),
        }
    }

    pub(crate) fn push_flag(
        &mut self,
        is_speech: bool,
        samples: Vec<f32>,
    ) -> Vec<StreamingVadEvent> {
        let frame = BufferedFrame {
            start_sample: self.next_sample,
            samples,
        };
        self.next_sample += frame.samples.len();

        match self.phase.clone() {
            StreamingVadPhase::Idle => self.push_idle(is_speech, frame),
            StreamingVadPhase::CandidateSpeech { speech_frames } => {
                self.push_candidate(is_speech, frame, speech_frames)
            }
            StreamingVadPhase::InSegment {
                start_sample,
                silence_frames,
            } => self.push_segment(is_speech, frame, start_sample, silence_frames),
        }
    }

    pub(crate) fn finish(&mut self) -> Vec<StreamingVadEvent> {
        let events = match self.phase {
            StreamingVadPhase::InSegment { start_sample, .. } => {
                vec![StreamingVadEvent::SegmentFinal {
                    start_sample,
                    end_sample: self.next_sample,
                }]
            }
            StreamingVadPhase::Idle | StreamingVadPhase::CandidateSpeech { .. } => Vec::new(),
        };

        self.phase = StreamingVadPhase::Idle;
        self.candidate.clear();
        self.pre_speech.clear();
        events
    }

    pub(crate) fn retained_sample_count(&self) -> usize {
        self.pre_speech_sample_count() + self.candidate_sample_count()
    }

    fn push_idle(&mut self, is_speech: bool, frame: BufferedFrame) -> Vec<StreamingVadEvent> {
        let mut events = Vec::new();

        if is_speech {
            self.candidate = std::mem::take(&mut self.pre_speech);
            self.candidate.push_back(frame);
            self.trim_candidate();
            self.phase = StreamingVadPhase::CandidateSpeech { speech_frames: 1 };
            events.extend(self.start_segment_if_ready(1));
        } else {
            self.remember_pre_speech(frame);
        }
        events
    }

    fn push_candidate(
        &mut self,
        is_speech: bool,
        frame: BufferedFrame,
        speech_frames: usize,
    ) -> Vec<StreamingVadEvent> {
        let mut events = Vec::new();

        if is_speech {
            let speech_frames = speech_frames + 1;
            self.candidate.push_back(frame);
            self.trim_candidate();
            self.phase = StreamingVadPhase::CandidateSpeech { speech_frames };
            events.extend(self.start_segment_if_ready(speech_frames));
        } else {
            self.candidate.clear();
            self.phase = StreamingVadPhase::Idle;
            self.remember_pre_speech(frame);
        }
        events
    }

    fn push_segment(
        &mut self,
        is_speech: bool,
        frame: BufferedFrame,
        start_sample: usize,
        silence_frames: usize,
    ) -> Vec<StreamingVadEvent> {
        let samples = frame.samples.clone();

        let silence_frames = if is_speech { 0 } else { silence_frames + 1 };
        let segment_frames = self.next_sample.saturating_sub(start_sample) / self.frame_samples;
        let should_finalize = silence_frames >= self.min_silence_frames()
            || (self.max_segment_frames() > 0 && segment_frames >= self.max_segment_frames());

        if should_finalize {
            self.phase = StreamingVadPhase::Idle;
            self.pre_speech.clear();
            self.candidate.clear();
            vec![
                StreamingVadEvent::Audio { samples },
                StreamingVadEvent::SegmentFinal {
                    start_sample,
                    end_sample: self.next_sample,
                },
            ]
        } else {
            self.phase = StreamingVadPhase::InSegment {
                start_sample,
                silence_frames,
            };
            self.remember_pre_speech(frame);
            vec![StreamingVadEvent::Audio { samples }]
        }
    }

    fn start_segment_if_ready(&mut self, speech_frames: usize) -> Vec<StreamingVadEvent> {
        if speech_frames < self.min_speech_frames() {
            return Vec::new();
        }

        let start_sample = self
            .candidate
            .front()
            .map_or(self.next_sample, |frame| frame.start_sample);
        let samples = self
            .candidate
            .iter()
            .flat_map(|frame| frame.samples.iter().copied())
            .collect();
        self.candidate.clear();
        self.phase = StreamingVadPhase::InSegment {
            start_sample,
            silence_frames: 0,
        };

        vec![StreamingVadEvent::SegmentStarted {
            start_sample,
            samples,
        }]
    }

    fn remember_pre_speech(&mut self, frame: BufferedFrame) {
        self.pre_speech.push_back(frame);
        let padding_frames = self.padding_frames();
        while self.pre_speech.len() > padding_frames {
            self.pre_speech.pop_front();
        }
    }

    fn trim_candidate(&mut self) {
        let max_samples = self.max_candidate_samples();
        while self.candidate_sample_count() > max_samples {
            self.candidate.pop_front();
        }
    }

    fn should_process_incomplete_frame(&self) -> bool {
        matches!(
            self.phase,
            StreamingVadPhase::CandidateSpeech { .. } | StreamingVadPhase::InSegment { .. }
        )
    }

    fn pre_speech_sample_count(&self) -> usize {
        self.pre_speech
            .iter()
            .map(|frame| frame.samples.len())
            .sum()
    }

    fn candidate_sample_count(&self) -> usize {
        self.candidate.iter().map(|frame| frame.samples.len()).sum()
    }

    fn padding_frames(&self) -> usize {
        frames_for_ms(self.config.speech_padding_ms, self.config.frame_duration_ms)
    }

    fn min_speech_frames(&self) -> usize {
        frames_for_ms(self.config.min_speech_ms, self.config.frame_duration_ms).max(1)
    }

    fn min_silence_frames(&self) -> usize {
        frames_for_ms(self.config.min_silence_ms, self.config.frame_duration_ms).max(1)
    }

    fn max_segment_frames(&self) -> usize {
        frames_for_ms(self.config.max_segment_ms, self.config.frame_duration_ms)
    }

    fn max_candidate_samples(&self) -> usize {
        if self.config.frame_duration_ms == 0 {
            return 0;
        }

        self.config
            .speech_padding_ms
            .checked_add(self.config.min_speech_ms)
            .map_or(0, |milliseconds| {
                milliseconds as usize * self.frame_samples / self.config.frame_duration_ms as usize
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingVadPhase {
    Idle,
    CandidateSpeech {
        speech_frames: usize,
    },
    InSegment {
        start_sample: usize,
        silence_frames: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct BufferedFrame {
    start_sample: usize,
    samples: Vec<f32>,
}

fn frames_for_ms(milliseconds: u32, frame_duration_ms: u32) -> usize {
    if milliseconds == 0 || frame_duration_ms == 0 {
        0
    } else {
        milliseconds.div_ceil(frame_duration_ms) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::{
        StreamingVadConfig, StreamingVadEvent, StreamingVadState, WebRtcStreamingVadEndpoint,
    };
    use orchion_core::{ASR_SAMPLE_RATE, OrchionError};
    use wavekat_vad::backends::webrtc::WebRtcVadMode;

    fn test_config() -> StreamingVadConfig {
        StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 20,
            min_silence_ms: 30,
            max_segment_ms: 100,
            speech_padding_ms: 10,
            mode: WebRtcVadMode::Quality,
        }
    }

    fn frame(value: f32) -> Vec<f32> {
        vec![value; 160]
    }

    fn voiced_samples(sample_count: usize) -> Vec<f32> {
        (0..sample_count)
            .map(|index| {
                let time = index as f32 / ASR_SAMPLE_RATE as f32;
                (time * 2.0 * std::f32::consts::PI * 180.0).sin() * 0.4
                    + (time * 2.0 * std::f32::consts::PI * 720.0).sin() * 0.15
            })
            .collect()
    }

    fn assert_invalid_config(config: StreamingVadConfig) {
        match WebRtcStreamingVadEndpoint::new(config) {
            Ok(_) => panic!("expected invalid streaming VAD config"),
            Err(error) => assert!(matches!(error, OrchionError::InvalidAudio { .. })),
        }
    }

    #[test]
    fn caption_defaults_match_product_requirement() {
        let config = StreamingVadConfig::default();

        assert_eq!(config.frame_duration_ms, 30);
        assert_eq!(config.min_speech_ms, 300);
        assert_eq!(config.min_silence_ms, 500);
        assert_eq!(config.max_segment_ms, 120_000);
        assert_eq!(config.speech_padding_ms, 200);
        assert_eq!(config.mode, WebRtcVadMode::Quality);
    }

    #[test]
    fn starts_after_minimum_speech_and_keeps_padding_bounded() {
        let mut state = StreamingVadState::new(test_config(), 160);

        assert!(state.push_flag(false, frame(0.0)).is_empty());
        assert!(state.push_flag(true, frame(0.1)).is_empty());
        let events = state.push_flag(true, frame(0.2));

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamingVadEvent::SegmentStarted {
                start_sample: 0,
                samples: vec![0.0; 160]
                    .into_iter()
                    .chain(vec![0.1; 160])
                    .chain(vec![0.2; 160])
                    .collect(),
            }
        );
        assert!(state.retained_sample_count() <= 480);
    }

    #[test]
    fn default_candidate_buffer_does_not_exceed_combined_sample_cap() {
        let config = StreamingVadConfig::default();
        let frame_samples = 480;
        let max_candidate_samples = (config.speech_padding_ms + config.min_speech_ms) as usize
            * ASR_SAMPLE_RATE as usize
            / 1_000;
        let mut state = StreamingVadState::new(config, frame_samples);

        for _ in 0..7 {
            assert!(state.push_flag(false, vec![0.0; frame_samples]).is_empty());
        }

        let mut started_samples = None;
        for _ in 0..10 {
            for event in state.push_flag(true, vec![0.1; frame_samples]) {
                if let StreamingVadEvent::SegmentStarted { samples, .. } = event {
                    started_samples = Some(samples);
                }
            }
        }

        let samples = started_samples.expect("default config should start after minimum speech");
        assert!(samples.len() <= max_candidate_samples);
    }

    #[test]
    fn default_candidate_state_retains_at_most_combined_sample_cap_before_start() {
        let config = StreamingVadConfig::default();
        let frame_samples = 480;
        let max_candidate_samples = (config.speech_padding_ms + config.min_speech_ms) as usize
            * ASR_SAMPLE_RATE as usize
            / 1_000;
        let mut state = StreamingVadState::new(config, frame_samples);

        for _ in 0..7 {
            assert!(state.push_flag(false, vec![0.0; frame_samples]).is_empty());
        }
        for _ in 0..3 {
            assert!(state.push_flag(true, vec![0.1; frame_samples]).is_empty());
        }

        assert!(state.retained_sample_count() <= max_candidate_samples);
    }

    #[test]
    fn finalizes_after_minimum_silence_without_retaining_finished_audio() {
        let mut state = StreamingVadState::new(test_config(), 160);

        let mut events = Vec::new();
        events.extend(state.push_flag(false, frame(0.0)));
        events.extend(state.push_flag(true, frame(0.1)));
        events.extend(state.push_flag(true, frame(0.2)));
        events.extend(state.push_flag(true, frame(0.3)));
        events.extend(state.push_flag(false, frame(0.0)));
        events.extend(state.push_flag(false, frame(0.0)));
        events.extend(state.push_flag(false, frame(0.0)));

        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamingVadEvent::SegmentStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamingVadEvent::Audio { .. }))
        );
        assert!(events.iter().any(|event| {
            matches!(
                event,
                StreamingVadEvent::SegmentFinal {
                    start_sample: 0,
                    end_sample: 1120,
                }
            )
        }));
        assert_eq!(state.retained_sample_count(), 0);
    }

    #[test]
    fn next_segment_after_final_does_not_include_previous_segment_audio() {
        let mut state = StreamingVadState::new(test_config(), 160);

        state.push_flag(false, frame(0.0));
        state.push_flag(true, frame(0.1));
        state.push_flag(true, frame(0.2));
        state.push_flag(true, frame(0.3));
        state.push_flag(false, frame(0.4));
        state.push_flag(false, frame(0.5));
        state.push_flag(false, frame(0.6));

        let mut started_samples = None;
        for event in state.push_flag(true, frame(0.7)) {
            if let StreamingVadEvent::SegmentStarted { samples, .. } = event {
                started_samples = Some(samples);
            }
        }
        for event in state.push_flag(true, frame(0.8)) {
            if let StreamingVadEvent::SegmentStarted { samples, .. } = event {
                started_samples = Some(samples);
            }
        }

        let samples = started_samples.expect("second segment should start");
        assert!(!samples.contains(&0.6));
        assert_eq!(
            samples,
            vec![0.7; 160]
                .into_iter()
                .chain(vec![0.8; 160])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn finish_emits_unpadded_pending_tail_and_real_end_sample() {
        let config = StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 20,
            min_silence_ms: 30,
            max_segment_ms: 1_000,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        };
        let mut endpoint = WebRtcStreamingVadEndpoint::new(config).unwrap();

        let started = endpoint
            .push(&voiced_samples(320), ASR_SAMPLE_RATE)
            .unwrap()
            .into_iter()
            .any(|event| matches!(event, StreamingVadEvent::SegmentStarted { .. }));
        assert!(started, "test waveform must open a segment");

        let tail = voiced_samples(80);
        assert!(endpoint.push(&tail, ASR_SAMPLE_RATE).unwrap().is_empty());

        let events = endpoint.finish();
        assert!(events.iter().any(|event| {
            matches!(event, StreamingVadEvent::Audio { samples } if samples.len() == tail.len())
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                StreamingVadEvent::SegmentFinal {
                    start_sample: 0,
                    end_sample: 400,
                }
            )
        }));
    }

    #[test]
    fn rejects_invalid_streaming_configs() {
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 0,
            min_silence_ms: 30,
            max_segment_ms: 100,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        });
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 20,
            min_silence_ms: 0,
            max_segment_ms: 100,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        });
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 200,
            min_silence_ms: 30,
            max_segment_ms: 100,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        });
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 30,
            min_speech_ms: 1,
            min_silence_ms: 30,
            max_segment_ms: 100,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        });
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: u32::MAX,
            min_silence_ms: 30,
            max_segment_ms: u32::MAX,
            speech_padding_ms: 1,
            mode: WebRtcVadMode::Quality,
        });
    }

    #[test]
    fn accepts_default_streaming_config() {
        assert!(WebRtcStreamingVadEndpoint::new(StreamingVadConfig::default()).is_ok());
    }

    #[test]
    fn rejects_candidate_budget_smaller_than_minimum_speech_frames() {
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 10,
            min_speech_ms: 21,
            min_silence_ms: 30,
            max_segment_ms: 100,
            speech_padding_ms: 0,
            mode: WebRtcVadMode::Quality,
        });
    }

    #[test]
    fn rejects_candidate_budget_above_retention_cap() {
        assert_invalid_config(StreamingVadConfig {
            frame_duration_ms: 30,
            min_speech_ms: 300,
            min_silence_ms: 500,
            max_segment_ms: 60_300,
            speech_padding_ms: 60_000,
            mode: WebRtcVadMode::Quality,
        });
    }

    #[test]
    fn max_segment_ms_forces_a_segment_boundary() {
        let mut state = StreamingVadState::new(test_config(), 160);
        let mut final_count = 0;

        for _ in 0..12 {
            for event in state.push_flag(true, frame(0.5)) {
                if matches!(event, StreamingVadEvent::SegmentFinal { .. }) {
                    final_count += 1;
                }
            }
        }

        assert_eq!(final_count, 1);
        assert!(state.retained_sample_count() <= 320);
    }
}
