use crate::device::{Loaded, inference_error};
use crane_core::{generation::TranscribeOptions, models::qwen3_asr::Model};
use orchion_core::{AsrTranscript, DevicePreference, Result, prepare_asr_samples};
use std::path::Path;

/// Batch transcription with a Qwen3-ASR `-hf` checkpoint.
pub struct Asr(Loaded<Model>);

impl Asr {
    pub fn load(path: impl AsRef<Path>, device: DevicePreference) -> Result<Self> {
        Loaded::open(path.as_ref(), device, Model::new).map(Self)
    }

    /// Transcribe mono PCM, resampling to 16 kHz if needed. Crane detects the
    /// language automatically and does not expose incremental Qwen3-ASR here.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
    ) -> Result<AsrTranscript> {
        let samples = prepare_asr_samples(samples, sample_rate)?;
        let options = TranscribeOptions {
            max_new_tokens,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let raw = self.0.run(|model| {
            model
                .transcribe(&samples, &options)
                .map_err(inference_error)
        })?;
        Ok(parse_transcript(raw))
    }
}

fn parse_transcript(raw_output: String) -> AsrTranscript {
    let (language, text) = raw_output
        .trim()
        .split_once("<asr_text>")
        .and_then(|(prefix, text)| Some((prefix.strip_prefix("language ")?.trim(), text.trim())))
        .unwrap_or(("", raw_output.trim()));
    AsrTranscript {
        text: text.into(),
        language: language.into(),
        raw_output,
        segments: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_raw_output_and_extracts_the_model_language() {
        let raw = "language Chinese<asr_text> 你好。";
        let transcript = parse_transcript(raw.into());
        assert_eq!(transcript.text, "你好。");
        assert_eq!(transcript.language, "Chinese");
        assert_eq!(transcript.raw_output, raw);
    }

    #[test]
    fn does_not_remove_a_separator_from_unstructured_text() {
        let raw = "literal <asr_text> marker";
        let transcript = parse_transcript(raw.into());
        assert_eq!(transcript.text, raw);
        assert_eq!(transcript.language, "");
    }

    #[test]
    fn missing_checkpoint_is_a_model_load_error() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert!(matches!(
            Asr::load(manifest, DevicePreference::Cpu),
            Err(orchion_core::OrchionError::ModelLoad { message }) if message.contains("config.json")
        ));
    }
}
