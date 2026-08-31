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
const LATIN_LANGUAGE_CONFIDENCE_THRESHOLD: f64 = 0.90;

/// Qwen TTS inference engine.
///
/// [`TtsLanguage::Auto`] uses fast script detection for Japanese, Korean, Chinese, and
/// Cyrillic text. Latin text must be detected as a Qwen-supported language with confidence
/// greater than 0.90; otherwise synthesis returns an error and the caller must select a language.
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
                        language_to_upstream(language, &text)?,
                        Some(options_to_upstream(&options)),
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
                            language_to_upstream(language, &text)?,
                            Some(options_to_upstream(&options)),
                        )
                        .map_err(|source| OrchionError::Inference {
                            message: source.to_string(),
                        })
                }
                TtsVoice::Design { prompt, language } => engine
                    .synthesize_voice_design(
                        text.as_str(),
                        prompt.as_str(),
                        language_to_upstream(language, &text)?,
                        Some(options_to_upstream(&options)),
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

fn language_to_upstream(language: TtsLanguage, text: &str) -> Result<qwen3_tts::Language> {
    let language = match language {
        TtsLanguage::Auto => detect_language(text),
        TtsLanguage::English => Ok(qwen3_tts::Language::English),
        TtsLanguage::Chinese => Ok(qwen3_tts::Language::Chinese),
        TtsLanguage::Japanese => Ok(qwen3_tts::Language::Japanese),
        TtsLanguage::Korean => Ok(qwen3_tts::Language::Korean),
        TtsLanguage::German => Ok(qwen3_tts::Language::German),
        TtsLanguage::French => Ok(qwen3_tts::Language::French),
        TtsLanguage::Russian => Ok(qwen3_tts::Language::Russian),
        TtsLanguage::Portuguese => Ok(qwen3_tts::Language::Portuguese),
        TtsLanguage::Spanish => Ok(qwen3_tts::Language::Spanish),
        TtsLanguage::Italian => Ok(qwen3_tts::Language::Italian),
    }?;
    Ok(language)
}

fn detect_language(text: &str) -> Result<qwen3_tts::Language> {
    if text
        .chars()
        .any(|character| matches!(character as u32, 0x3040..=0x30ff))
    {
        Ok(qwen3_tts::Language::Japanese)
    } else if text
        .chars()
        .any(|character| matches!(character as u32, 0xac00..=0xd7af))
    {
        Ok(qwen3_tts::Language::Korean)
    } else if text.chars().any(
        |character| matches!(character as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff),
    ) {
        Ok(qwen3_tts::Language::Chinese)
    } else if text
        .chars()
        .any(|character| matches!(character as u32, 0x0400..=0x052f))
    {
        Ok(qwen3_tts::Language::Russian)
    } else {
        detect_latin_language(text)
    }
}

fn detect_latin_language(text: &str) -> Result<qwen3_tts::Language> {
    let info = whatlang::detect(text).ok_or_else(|| {
        auto_language_error("could not determine a language from the provided text")
    })?;

    if info.script() != whatlang::Script::Latin {
        return Err(auto_language_error(
            "detected a script that Qwen does not support",
        ));
    }
    if info.confidence() <= LATIN_LANGUAGE_CONFIDENCE_THRESHOLD {
        return Err(auto_language_error(
            "did not meet the required confidence threshold",
        ));
    }

    match info.lang() {
        whatlang::Lang::Eng => Ok(qwen3_tts::Language::English),
        whatlang::Lang::Deu => Ok(qwen3_tts::Language::German),
        whatlang::Lang::Fra => Ok(qwen3_tts::Language::French),
        whatlang::Lang::Por => Ok(qwen3_tts::Language::Portuguese),
        whatlang::Lang::Spa => Ok(qwen3_tts::Language::Spanish),
        whatlang::Lang::Ita => Ok(qwen3_tts::Language::Italian),
        _ => Err(auto_language_error(
            "detected a language that Qwen does not support",
        )),
    }
}

fn auto_language_error(reason: &'static str) -> OrchionError {
    OrchionError::Inference {
        message: format!(
            "TTS language auto-detection {reason}; specify the language explicitly and retry"
        ),
    }
}

fn options_to_upstream(options: &TtsOptions) -> qwen3_tts::SynthesisOptions {
    qwen3_tts::SynthesisOptions {
        seed: options.seed,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        repetition_penalty: options.repetition_penalty,
        max_length: options.max_length,
        ..Default::default()
    }
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
            assert!(
                language_to_upstream(
                    language,
                    "The weather is pleasant today, and we are taking a walk through the park."
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn auto_language_detects_non_latin_scripts() {
        assert!(matches!(
            language_to_upstream(TtsLanguage::Auto, "你好"),
            Ok(qwen3_tts::Language::Chinese)
        ));
        assert!(matches!(
            language_to_upstream(TtsLanguage::Auto, "こんにちは"),
            Ok(qwen3_tts::Language::Japanese)
        ));
        assert!(matches!(
            language_to_upstream(TtsLanguage::Auto, "안녕하세요"),
            Ok(qwen3_tts::Language::Korean)
        ));
        assert!(matches!(
            language_to_upstream(TtsLanguage::Auto, "Привет"),
            Ok(qwen3_tts::Language::Russian)
        ));
    }

    #[test]
    fn auto_language_detects_supported_latin_languages() {
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "The weather is pleasant today, and we are taking a walk through the park."
            ),
            Ok(qwen3_tts::Language::English)
        ));
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "Der schnelle braune Fuchs springt über den faulen Hund, während die Sonne hinter den Hügeln untergeht."
            ),
            Ok(qwen3_tts::Language::German)
        ));
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "Le renard brun et rapide saute par-dessus le chien paresseux pendant que le soleil se couche."
            ),
            Ok(qwen3_tts::Language::French)
        ));
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "A rápida raposa marrom salta sobre o cão preguiçoso enquanto o sol se põe atrás das colinas."
            ),
            Ok(qwen3_tts::Language::Portuguese)
        ));
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "El rápido zorro marrón salta sobre el perro perezoso mientras el sol se pone detrás de las colinas."
            ),
            Ok(qwen3_tts::Language::Spanish)
        ));
        assert!(matches!(
            language_to_upstream(
                TtsLanguage::Auto,
                "Questa mattina sono andato al mercato per comprare pane fresco, frutta e verdura. Dopo aver fatto la spesa, ho incontrato alcuni amici e abbiamo bevuto un caffè insieme."
            ),
            Ok(qwen3_tts::Language::Italian)
        ));
    }

    #[test]
    fn auto_language_rejects_ambiguous_latin_text_without_exposing_it() {
        let text = "Hello";
        let error = language_to_upstream(TtsLanguage::Auto, text).unwrap_err();
        let message = error.to_string();

        assert!(matches!(error, OrchionError::Inference { .. }));
        assert!(message.contains("confidence threshold"));
        assert!(message.contains("specify the language explicitly"));
        assert!(!message.contains(text));
    }

    #[test]
    fn auto_language_rejects_unsupported_latin_language_without_exposing_it() {
        let text = "De snelle bruine vos springt over de luie hond terwijl de zon achter de heuvels ondergaat.";
        let error = language_to_upstream(TtsLanguage::Auto, text).unwrap_err();
        let message = error.to_string();

        assert!(matches!(error, OrchionError::Inference { .. }));
        assert!(message.contains("does not support"));
        assert!(message.contains("specify the language explicitly"));
        assert!(!message.contains(text));
    }

    #[test]
    fn model_capability_checks_match_voice_variants() {
        let preset_model = TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap();
        let clone_model = TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-base").unwrap();

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
            TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap(),
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
