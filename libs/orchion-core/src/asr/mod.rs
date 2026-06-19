mod audio;

use std::str::FromStr;

pub use audio::{ASR_SAMPLE_RATE, prepare_asr_samples};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct AsrTranscript {
    pub text: String,
    pub language: String,
    pub raw_output: String,
    pub segments: Vec<AsrSegment>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct AsrSegment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrTimestampGranularity {
    Segment,
    Word,
}

impl FromStr for AsrTimestampGranularity {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "segment" => Ok(Self::Segment),
            "word" => Ok(Self::Word),
            _ => Err(format!("unsupported timestamp granularity `{value}`")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub max_new_tokens: usize,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            max_new_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrStreamingOptions {
    pub language: Option<String>,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens_streaming: usize,
    pub max_new_tokens_final: usize,
    pub initial_text: Option<String>,
}

impl Default for AsrStreamingOptions {
    fn default() -> Self {
        Self {
            language: None,
            chunk_size_sec: 2.0,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens_streaming: 32,
            max_new_tokens_final: 512,
            initial_text: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_segments_preserve_timing_and_text() {
        let transcript = AsrTranscript {
            text: "hello world".to_string(),
            language: "en".to_string(),
            raw_output: "hello world".to_string(),
            segments: vec![AsrSegment {
                id: 0,
                start: 1.25,
                end: 2.5,
                text: "hello".to_string(),
            }],
        };

        assert_eq!(transcript.segments[0].id, 0);
        assert_eq!(transcript.segments[0].start, 1.25);
        assert_eq!(transcript.segments[0].end, 2.5);
        assert_eq!(transcript.segments[0].text, "hello");
    }

    #[test]
    fn timestamp_granularity_parses_openai_values() {
        assert_eq!(
            "segment".parse::<AsrTimestampGranularity>().unwrap(),
            AsrTimestampGranularity::Segment
        );
        assert_eq!(
            "word".parse::<AsrTimestampGranularity>().unwrap(),
            AsrTimestampGranularity::Word
        );
        assert!("sentence".parse::<AsrTimestampGranularity>().is_err());
    }
}
