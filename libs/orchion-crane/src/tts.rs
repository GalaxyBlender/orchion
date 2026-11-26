use crate::device::{Loaded, inference_error};
use candle_core::DType;
use crane_core::{generation::SpeechOptions, models::qwen3_tts::Model};
use orchion_core::{DevicePreference, OrchionError, Result, TtsAudio};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum Voice<'a> {
    /// Speaker identifier from the `CustomVoice` checkpoint (for example `ryan`).
    Preset(&'a str),
    /// Base-model voice cloning. Reference samples must be normalized mono PCM.
    Clone {
        samples: &'a [f32],
        sample_rate: u32,
        text: &'a str,
    },
}

/// Options exposed by the Crane Qwen3-TTS runtime. Seed and `top_k` are not
/// configurable in the upstream model.
#[derive(Debug, Clone, Copy)]
pub struct TtsOptions {
    pub max_frames: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f32,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            max_frames: 2048,
            temperature: 0.7,
            top_p: 0.8,
            repetition_penalty: 1.05,
        }
    }
}

/// Qwen3-TTS `CustomVoice` or Base synthesis.
pub struct Tts(Loaded<Model>);

impl Tts {
    pub fn load(path: impl AsRef<Path>, device: DevicePreference) -> Result<Self> {
        Loaded::open(path.as_ref(), device, Model::new).map(Self)
    }

    #[must_use]
    pub fn reference_sample_rate(&self) -> u32 {
        self.0.model().speaker_encoder_sample_rate()
    }

    /// Generate mono PCM. `language` is a Qwen language name or `"auto"`.
    pub fn synthesize(
        &mut self,
        text: &str,
        language: &str,
        voice: Voice<'_>,
        options: TtsOptions,
    ) -> Result<TtsAudio> {
        if let Voice::Clone { sample_rate, .. } = voice
            && sample_rate != self.reference_sample_rate()
        {
            return Err(OrchionError::InvalidAudio {
                reason: format!(
                    "Crane voice reference must be mono PCM at {} Hz (got {sample_rate} Hz)",
                    self.reference_sample_rate()
                ),
            });
        }
        let options = SpeechOptions {
            max_new_tokens: options.max_frames,
            temperature: options.temperature,
            top_p: Some(options.top_p),
            repetition_penalty: options.repetition_penalty,
            ..Default::default()
        };
        self.0.run(|model| {
            let (waveform, rate) = match voice {
                Voice::Preset(speaker) => {
                    model.generate_speech(text, language, Some(speaker), &options)
                }
                Voice::Clone {
                    samples,
                    text: reference_text,
                    ..
                } => model.generate_voice_clone(text, language, samples, reference_text, &options),
            }
            .map_err(inference_error)?;
            let samples = waveform
                .flatten_all()
                .and_then(|waveform| waveform.to_dtype(DType::F32))
                .and_then(|waveform| waveform.to_vec1())
                .map_err(inference_error)?;
            Ok(TtsAudio::new(samples, rate))
        })
    }
}
