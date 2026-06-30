use orchion_core::{
    DevicePreference, OrchionError, Result, TtsAudio, TtsLanguage, TtsModel, TtsOptions,
    TtsSpeaker, TtsVoice, ensure_voice_supported,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const VOICE_CLONE_ICL_PREFILL_TOKENS: usize = 9;
const VOICE_CLONE_ICL_CODEC_BOS_TOKENS: usize = 1;
const VOICE_CLONE_ICL_MIN_GENERATED_FRAMES: usize = 75;
const VOICE_CLONE_ICL_EXTRA_CACHE_TOKENS: usize = 256;

#[derive(Clone)]
pub struct Tts {
    model: TtsModel,
    engine: Arc<Mutex<qwen3_tts::Qwen3TTS>>,
}

impl Tts {
    pub async fn load(model: TtsModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    pub async fn load_with_device(
        model: TtsModel,
        model_dir: impl AsRef<Path>,
        preference: DevicePreference,
    ) -> Result<Self> {
        let path = model_dir.as_ref().to_path_buf();
        crate::blocking::run(move || {
            let path_text = path
                .to_str()
                .ok_or_else(|| OrchionError::NonUtf8Path { path: path.clone() })?;
            let resolved = crate::device::resolve_device(preference)?;
            let device_debug = format!("{:?}", resolved.device);
            tracing::info!(
                model = ?model,
                requested_device = %preference,
                device = %resolved.kind,
                "TTS device selected"
            );
            tracing::debug!(device_debug, "TTS device details selected");
            let engine = qwen3_tts::Qwen3TTS::from_pretrained(path_text, resolved.device).map_err(
                |source| OrchionError::ModelLoad {
                    message: source.to_string(),
                },
            )?;
            Ok(Self {
                model,
                engine: Arc::new(Mutex::new(engine)),
            })
        })
        .await
    }

    pub fn model(&self) -> TtsModel {
        self.model.clone()
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
        self.synthesize_upstream(text, voice, options)
            .await
            .map(audio_from_upstream)
    }

    pub async fn synthesize_to_file(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        let output_path = output_path.as_ref().to_path_buf();
        let audio = self
            .synthesize_upstream(text, voice, TtsOptions::default())
            .await?;
        crate::blocking::run(move || {
            audio
                .save(output_path)
                .map_err(|source| OrchionError::Inference {
                    message: source.to_string(),
                })
        })
        .await
    }

    async fn synthesize_upstream(
        &self,
        text: impl AsRef<str>,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> Result<qwen3_tts::AudioBuffer> {
        ensure_voice_supported(&self.model, &voice)?;
        let text = text.as_ref().to_string();
        let text_len = text.chars().count();
        let engine = Arc::clone(&self.engine);
        crate::blocking::run(move || {
            let started = Instant::now();
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                message: error.to_string(),
            })?;
            let audio = match voice {
                TtsVoice::Preset { speaker, language } => engine
                    .synthesize_with_voice(
                        text.as_str(),
                        speaker_to_upstream(speaker),
                        language_to_upstream(language),
                        Some(options_to_upstream(options)),
                    )
                    .map_err(|source| OrchionError::Inference {
                        message: source.to_string(),
                    }),
                TtsVoice::Clone {
                    reference_audio,
                    reference_text,
                    language,
                } => {
                    let audio =
                        qwen3_tts::AudioBuffer::load(&reference_audio).map_err(|source| {
                            OrchionError::Inference {
                                message: source.to_string(),
                            }
                        })?;
                    let prompt = engine
                        .create_voice_clone_prompt(&audio, Some(reference_text.as_str()))
                        .map_err(|source| OrchionError::Inference {
                            message: source.to_string(),
                        })?;
                    validate_voice_clone_icl_prompt(&prompt)?;
                    engine
                        .synthesize_voice_clone(
                            text.as_str(),
                            &prompt,
                            language_to_upstream(language),
                            Some(options_to_upstream(options)),
                        )
                        .map_err(|source| OrchionError::Inference {
                            message: source.to_string(),
                        })
                }
                TtsVoice::Design { prompt, language } => engine
                    .synthesize_voice_design(
                        text.as_str(),
                        prompt.as_str(),
                        language_to_upstream(language),
                        Some(options_to_upstream(options)),
                    )
                    .map_err(|source| OrchionError::Inference {
                        message: source.to_string(),
                    }),
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
}

fn speaker_to_upstream(speaker: TtsSpeaker) -> qwen3_tts::Speaker {
    match speaker {
        TtsSpeaker::Serena => qwen3_tts::Speaker::Serena,
        TtsSpeaker::Vivian => qwen3_tts::Speaker::Vivian,
        TtsSpeaker::UncleFu => qwen3_tts::Speaker::UncleFu,
        TtsSpeaker::Ryan => qwen3_tts::Speaker::Ryan,
        TtsSpeaker::Aiden => qwen3_tts::Speaker::Aiden,
        TtsSpeaker::OnoAnna => qwen3_tts::Speaker::OnoAnna,
        TtsSpeaker::Sohee => qwen3_tts::Speaker::Sohee,
        TtsSpeaker::Eric => qwen3_tts::Speaker::Eric,
        TtsSpeaker::Dylan => qwen3_tts::Speaker::Dylan,
    }
}

fn language_to_upstream(language: TtsLanguage) -> qwen3_tts::Language {
    match language {
        TtsLanguage::Auto => qwen3_tts::Language::English,
        TtsLanguage::English => qwen3_tts::Language::English,
        TtsLanguage::Chinese => qwen3_tts::Language::Chinese,
        TtsLanguage::Japanese => qwen3_tts::Language::Japanese,
        TtsLanguage::Korean => qwen3_tts::Language::Korean,
        TtsLanguage::German => qwen3_tts::Language::German,
        TtsLanguage::French => qwen3_tts::Language::French,
        TtsLanguage::Russian => qwen3_tts::Language::Russian,
        TtsLanguage::Portuguese => qwen3_tts::Language::Portuguese,
        TtsLanguage::Spanish => qwen3_tts::Language::Spanish,
        TtsLanguage::Italian => qwen3_tts::Language::Italian,
    }
}

fn options_to_upstream(options: TtsOptions) -> qwen3_tts::SynthesisOptions {
    let mut upstream = qwen3_tts::SynthesisOptions::default();
    upstream.seed = options.seed;
    upstream.temperature = options.temperature;
    upstream.top_k = options.top_k;
    upstream.top_p = options.top_p;
    upstream.repetition_penalty = options.repetition_penalty;
    upstream.max_length = options.max_length;
    upstream
}

fn audio_from_upstream(audio: qwen3_tts::AudioBuffer) -> TtsAudio {
    TtsAudio::new(audio.samples, audio.sample_rate)
}

fn validate_voice_clone_icl_prompt(prompt: &qwen3_tts::VoiceClonePrompt) -> Result<()> {
    if let Some(ref_codes) = &prompt.ref_codes {
        let reference_frames = ref_codes.dim(0).map_err(|source| OrchionError::Inference {
            message: source.to_string(),
        })?;
        validate_voice_clone_icl_frames(reference_frames)?;
    }
    Ok(())
}

fn validate_voice_clone_icl_frames(reference_frames: usize) -> Result<()> {
    let max_reference_frames = VOICE_CLONE_ICL_MIN_GENERATED_FRAMES
        + VOICE_CLONE_ICL_EXTRA_CACHE_TOKENS
        - VOICE_CLONE_ICL_PREFILL_TOKENS
        - VOICE_CLONE_ICL_CODEC_BOS_TOKENS;

    if reference_frames > max_reference_frames {
        return Err(OrchionError::InvalidAudio {
            reason: format!(
                "voice clone reference audio is too long for ICL prompting; use a shorter reference clip ({reference_frames} encoded frames, maximum {max_reference_frames})"
            ),
        });
    }

    Ok(())
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
            let _ = speaker_to_upstream(speaker);
        }
    }

    #[test]
    fn language_mapping_covers_supported_languages() {
        let languages = [
            TtsLanguage::Auto,
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
            let _ = language_to_upstream(language);
        }
    }

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

    #[test]
    fn device_label_detects_cpu_from_resolver_kind() {
        assert_eq!(crate::device::ResolvedDeviceKind::Cpu.to_string(), "cpu");
    }

    #[test]
    fn exposes_explicit_device_loader_api() {
        let future = Tts::load_with_device(
            TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").unwrap(),
            "models/qwen3-tts-0.6b-custom-voice",
            orchion_core::DevicePreference::Cpu,
        );
        std::mem::drop(future);
    }

    #[test]
    fn rejects_voice_clone_reference_that_exceeds_icl_cache_budget() {
        let result = validate_voice_clone_icl_frames(373);

        let error = result.unwrap_err();
        assert!(matches!(error, OrchionError::InvalidAudio { .. }));
        assert!(
            error
                .to_string()
                .contains("voice clone reference audio is too long")
        );
    }
}
