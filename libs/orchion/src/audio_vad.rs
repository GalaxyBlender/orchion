use orchion_core::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioVadSegment {
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioVadMode {
    Quality,
    LowBitrate,
    Aggressive,
    VeryAggressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioVadConfig {
    pub frame_duration_ms: u32,
    pub padding_ms: u32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub merge_gap_ms: u32,
    pub mode: AudioVadMode,
}

pub type AudioVadStreamingConfig = orchion_audio_vad::StreamingVadConfig;
pub type AudioVadStreamingEndpoint = orchion_audio_vad::WebRtcStreamingVadEndpoint;
pub type AudioVadStreamingEvent = orchion_audio_vad::StreamingVadEvent;

impl Default for AudioVadConfig {
    fn default() -> Self {
        let config = orchion_audio_vad::VadConfig::default();

        Self {
            frame_duration_ms: config.frame_duration_ms,
            padding_ms: config.padding_ms,
            min_speech_ms: config.min_speech_ms,
            min_silence_ms: config.min_silence_ms,
            merge_gap_ms: config.merge_gap_ms,
            mode: config.mode.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AudioVadSegmenter {
    inner: orchion_audio_vad::WebRtcVadSegmenter,
}

impl AudioVadSegmenter {
    pub fn new(config: AudioVadConfig) -> Self {
        Self {
            inner: orchion_audio_vad::WebRtcVadSegmenter::new(config.into()),
        }
    }

    pub fn segment(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<AudioVadSegment>> {
        self.inner
            .segment(samples, sample_rate)
            .map(|segments| segments.into_iter().map(AudioVadSegment::from).collect())
    }
}

impl From<orchion_audio_vad::AudioSegment> for AudioVadSegment {
    fn from(segment: orchion_audio_vad::AudioSegment) -> Self {
        Self {
            start_sample: segment.start_sample,
            end_sample: segment.end_sample,
        }
    }
}

impl From<AudioVadConfig> for orchion_audio_vad::VadConfig {
    fn from(config: AudioVadConfig) -> Self {
        Self {
            frame_duration_ms: config.frame_duration_ms,
            padding_ms: config.padding_ms,
            min_speech_ms: config.min_speech_ms,
            min_silence_ms: config.min_silence_ms,
            merge_gap_ms: config.merge_gap_ms,
            mode: config.mode.into(),
        }
    }
}

impl From<AudioVadMode> for orchion_audio_vad::WebRtcVadMode {
    fn from(mode: AudioVadMode) -> Self {
        match mode {
            AudioVadMode::Quality => Self::Quality,
            AudioVadMode::LowBitrate => Self::LowBitrate,
            AudioVadMode::Aggressive => Self::Aggressive,
            AudioVadMode::VeryAggressive => Self::VeryAggressive,
        }
    }
}

impl From<orchion_audio_vad::WebRtcVadMode> for AudioVadMode {
    fn from(mode: orchion_audio_vad::WebRtcVadMode) -> Self {
        match mode {
            orchion_audio_vad::WebRtcVadMode::Quality => Self::Quality,
            orchion_audio_vad::WebRtcVadMode::LowBitrate => Self::LowBitrate,
            orchion_audio_vad::WebRtcVadMode::Aggressive => Self::Aggressive,
            orchion_audio_vad::WebRtcVadMode::VeryAggressive => Self::VeryAggressive,
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use crate::{AudioVadStreamingConfig, AudioVadStreamingEndpoint, AudioVadStreamingEvent};
    use orchion_core::ASR_SAMPLE_RATE;

    #[test]
    fn streaming_facade_exports_caption_endpointing_types() {
        let config = AudioVadStreamingConfig::default();
        let mut endpoint = AudioVadStreamingEndpoint::new(config).unwrap();

        let events: Vec<AudioVadStreamingEvent> =
            endpoint.push(&[0.0; 160], ASR_SAMPLE_RATE).unwrap();
        assert!(events.is_empty());
    }
}
