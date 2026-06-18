use crate::error::{OrchionError, Result};
use crate::model::{ModelSpec, TtsModel};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub type TtsAudio = qwen3_tts::AudioBuffer;
pub type TtsOptions = qwen3_tts::SynthesisOptions;

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

impl From<TtsSpeaker> for qwen3_tts::Speaker {
    fn from(speaker: TtsSpeaker) -> Self {
        match speaker {
            TtsSpeaker::Serena => Self::Serena,
            TtsSpeaker::Vivian => Self::Vivian,
            TtsSpeaker::UncleFu => Self::UncleFu,
            TtsSpeaker::Ryan => Self::Ryan,
            TtsSpeaker::Aiden => Self::Aiden,
            TtsSpeaker::OnoAnna => Self::OnoAnna,
            TtsSpeaker::Sohee => Self::Sohee,
            TtsSpeaker::Eric => Self::Eric,
            TtsSpeaker::Dylan => Self::Dylan,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsLanguage {
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

impl From<TtsLanguage> for qwen3_tts::Language {
    fn from(language: TtsLanguage) -> Self {
        match language {
            TtsLanguage::English => Self::English,
            TtsLanguage::Chinese => Self::Chinese,
            TtsLanguage::Japanese => Self::Japanese,
            TtsLanguage::Korean => Self::Korean,
            TtsLanguage::German => Self::German,
            TtsLanguage::French => Self::French,
            TtsLanguage::Russian => Self::Russian,
            TtsLanguage::Portuguese => Self::Portuguese,
            TtsLanguage::Spanish => Self::Spanish,
            TtsLanguage::Italian => Self::Italian,
        }
    }
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

fn voice_supported(model: TtsModel, voice: &TtsVoice) -> Result<()> {
    match voice {
        TtsVoice::Preset { .. } if !model.supports_preset_speakers() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo(),
                capability: "preset speakers",
            })
        }
        TtsVoice::Clone { .. } if !model.supports_voice_cloning() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo(),
                capability: "voice cloning",
            })
        }
        TtsVoice::Design { .. } if !model.supports_voice_design() => {
            Err(OrchionError::UnsupportedCapability {
                model: model.huggingface_repo(),
                capability: "voice design",
            })
        }
        _ => Ok(()),
    }
}

#[derive(Clone)]
pub struct Tts {
    model: TtsModel,
    engine: Arc<Mutex<qwen3_tts::Qwen3TTS>>,
}

impl Tts {
    pub async fn load(model: TtsModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().to_path_buf();
        crate::blocking::run(move || {
            let path_text = path
                .to_str()
                .ok_or_else(|| OrchionError::NonUtf8Path { path: path.clone() })?;
            let device =
                qwen3_tts::auto_device().map_err(|source| OrchionError::ModelLoad { source })?;
            let device_debug = format!("{device:?}");
            tracing::info!(
                model = ?model,
                device = %qwen3_tts::device_info(&device),
                "TTS device selected"
            );
            tracing::debug!(device_debug, "TTS device details selected");
            let engine = qwen3_tts::Qwen3TTS::from_pretrained(path_text, device)
                .map_err(|source| OrchionError::ModelLoad { source })?;
            Ok(Self {
                model,
                engine: Arc::new(Mutex::new(engine)),
            })
        })
        .await
    }

    #[cfg(feature = "download")]
    pub async fn load_or_download(model: TtsModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = crate::download::ModelDownloader::default()
            .download(model, cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub const fn model(&self) -> TtsModel {
        self.model
    }

    pub async fn synthesize(&self, text: impl AsRef<str>, voice: TtsVoice) -> Result<TtsAudio> {
        self.synthesize_with(text, voice, TtsOptions::default())
            .await
    }

    pub async fn synthesize_with(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> Result<TtsAudio> {
        voice_supported(self.model, &voice)?;
        let text = text.as_ref().to_string();
        let text_len = text.chars().count();
        let engine = Arc::clone(&self.engine);
        crate::blocking::run(move || {
            let started = Instant::now();
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                source: anyhow::anyhow!(error.to_string()),
            })?;
            let audio = match voice {
                TtsVoice::Preset { speaker, language } => engine
                    .synthesize_with_voice(
                        text.as_str(),
                        speaker.into(),
                        language.into(),
                        Some(options),
                    )
                    .map_err(|source| OrchionError::Inference { source }),
                TtsVoice::Clone {
                    reference_audio,
                    reference_text,
                    language,
                } => {
                    let audio = qwen3_tts::AudioBuffer::load(&reference_audio)
                        .map_err(|source| OrchionError::Inference { source })?;
                    let prompt = engine
                        .create_voice_clone_prompt(&audio, Some(reference_text.as_str()))
                        .map_err(|source| OrchionError::Inference { source })?;
                    engine
                        .synthesize_voice_clone(
                            text.as_str(),
                            &prompt,
                            language.into(),
                            Some(options),
                        )
                        .map_err(|source| OrchionError::Inference { source })
                }
                TtsVoice::Design { prompt, language } => engine
                    .synthesize_voice_design(
                        text.as_str(),
                        prompt.as_str(),
                        language.into(),
                        Some(options),
                    )
                    .map_err(|source| OrchionError::Inference { source }),
            }?;
            tracing::debug!(
                text_chars = text_len,
                samples = audio.samples.len(),
                sample_rate = audio.sample_rate,
                elapsed_ms = started.elapsed().as_millis(),
                "TTS synthesis completed"
            );
            Ok(audio)
        })
        .await
    }

    pub async fn synthesize_to_file(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        let output_path = output_path.as_ref().to_path_buf();
        let audio = self.synthesize(text, voice).await?;
        crate::blocking::run(move || {
            audio
                .save(output_path)
                .map_err(|source| OrchionError::Inference { source })
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_mapping_covers_all_public_speakers() {
        let speakers = [
            TtsSpeaker::Serena,
            TtsSpeaker::Vivian,
            TtsSpeaker::UncleFu,
            TtsSpeaker::Ryan,
            TtsSpeaker::Aiden,
            TtsSpeaker::OnoAnna,
            TtsSpeaker::Sohee,
            TtsSpeaker::Eric,
            TtsSpeaker::Dylan,
        ];
        for speaker in speakers {
            let _: qwen3_tts::Speaker = speaker.into();
        }
    }

    #[test]
    fn language_mapping_covers_supported_languages() {
        let languages = [
            TtsLanguage::English,
            TtsLanguage::Chinese,
            TtsLanguage::Japanese,
            TtsLanguage::Korean,
            TtsLanguage::German,
            TtsLanguage::French,
            TtsLanguage::Russian,
            TtsLanguage::Portuguese,
            TtsLanguage::Spanish,
            TtsLanguage::Italian,
        ];
        for language in languages {
            let _: qwen3_tts::Language = language.into();
        }
    }

    #[test]
    fn model_capability_checks_match_voice_variants() {
        assert!(
            voice_supported(
                TtsModel::Qwen3Tts06BCustomVoice,
                &TtsVoice::Preset {
                    speaker: TtsSpeaker::Ryan,
                    language: TtsLanguage::English,
                }
            )
            .is_ok()
        );
        assert!(
            voice_supported(
                TtsModel::Qwen3Tts06BBase,
                &TtsVoice::Preset {
                    speaker: TtsSpeaker::Ryan,
                    language: TtsLanguage::English,
                }
            )
            .is_err()
        );
    }
}
