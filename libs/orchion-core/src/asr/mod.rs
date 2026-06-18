mod audio;

pub use audio::{ASR_SAMPLE_RATE, prepare_asr_samples};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscript {
    pub text: String,
    pub language: String,
    pub raw_output: String,
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
