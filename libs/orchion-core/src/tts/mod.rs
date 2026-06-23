use crate::error::{OrchionError, Result};
use crate::model::{ModelSpec, TtsModel};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl TtsAudio {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsOptions {
    pub seed: Option<u64>,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub max_length: usize,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            seed: None,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            repetition_penalty: 1.05,
            max_length: 2048,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsSpeaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsLanguage {
    Auto,
    English,
    Chinese,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsVoice {
    Preset {
        speaker: TtsSpeaker,
        language: TtsLanguage,
    },
    Clone {
        reference_audio: PathBuf,
        reference_text: String,
        language: TtsLanguage,
    },
    Design {
        prompt: String,
        language: TtsLanguage,
    },
}

pub fn ensure_voice_supported(model: &TtsModel, voice: &TtsVoice) -> Result<()> {
    match voice {
        TtsVoice::Preset { .. } if !model.supports_preset_speakers() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo().to_string(),
                capability: "preset speakers",
            })
        }
        TtsVoice::Clone { .. } if !model.supports_voice_cloning() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo().to_string(),
                capability: "voice cloning",
            })
        }
        TtsVoice::Design { .. } if !model.supports_voice_design() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo().to_string(),
                capability: "voice design",
            })
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_capability_checks_match_voice_variants() {
        let preset_model = TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").unwrap();
        let clone_model = TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-Base").unwrap();

        assert!(
            ensure_voice_supported(
                &preset_model,
                &TtsVoice::Preset {
                    speaker: TtsSpeaker::Ryan,
                    language: TtsLanguage::English,
                }
            )
            .is_ok()
        );
        assert!(
            ensure_voice_supported(
                &clone_model,
                &TtsVoice::Preset {
                    speaker: TtsSpeaker::Ryan,
                    language: TtsLanguage::English,
                }
            )
            .is_err()
        );
    }
}
