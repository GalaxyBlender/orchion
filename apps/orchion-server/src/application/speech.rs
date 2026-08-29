use super::{
    RuntimeError, UseCaseError, finish_owned_file_operation, protect_owned_file_operation,
};
use orchion::{
    AudioOutputFormat, TtsAudio, TtsLanguage, TtsModel, TtsOptions, TtsSpeaker, TtsVoice,
    decode_audio_file_with_max_samples, encode_tts_audio,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::{Duration, Instant};

pub type SpeechRuntimeFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<TtsAudio>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone, Copy)]
pub struct SpeechPolicy {
    pub default_format: AudioOutputFormat,
    pub max_length: usize,
    pub max_reference_audio_duration: Duration,
}

pub trait SpeechRuntime: Send + Sync {
    fn speech_policy(&self) -> SpeechPolicy;
    fn synthesize_speech(
        &self,
        model: TtsModel,
        input: String,
        voice: TtsVoice,
        options: TtsOptions,
    ) -> SpeechRuntimeFuture<'_>;
}

#[derive(Debug, Clone)]
pub struct SpeechCommand {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub output_format: Option<AudioOutputFormat>,
    pub speed: f32,
    pub language: Option<String>,
    pub reference_audio: Option<PathBuf>,
    pub reference_audio_output: Option<PathBuf>,
    pub reference_text: Option<String>,
    pub voice_prompt: Option<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub max_length: Option<usize>,
}

#[derive(Debug)]
pub struct SpeechResult {
    pub bytes: Vec<u8>,
    pub format: AudioOutputFormat,
}

/// # Errors
///
/// Returns [`UseCaseError`] when validation, model loading, synthesis, or encoding fails.
pub async fn synthesize(
    runtime: &impl SpeechRuntime,
    mut command: SpeechCommand,
) -> Result<SpeechResult, UseCaseError> {
    let policy = runtime.speech_policy();
    validate(&command)?;
    validate_request_limit(&command, policy.max_length)?;
    let model = TtsModel::parse(&command.model)
        .map_err(|_| UseCaseError::ModelNotAvailable(command.model.clone()))?;
    let _ = supported_voice(&model, &command)?;

    if let (Some(reference_audio), Some(reference_audio_output)) = (
        command.reference_audio.as_ref(),
        command.reference_audio_output.as_ref(),
    ) {
        transcode_reference_audio(
            reference_audio.clone(),
            reference_audio_output.clone(),
            policy.max_reference_audio_duration,
        )
        .await?;
        command.reference_audio = Some(reference_audio_output.clone());
    }

    let format = command.output_format.unwrap_or(policy.default_format);
    let voice = supported_voice(&model, &command)?;
    let mut options = to_tts_options(&command);
    options.max_length = options.max_length.min(policy.max_length);
    let input = command.input;

    let synthesis_started = Instant::now();
    tracing::debug!("speech synthesis started");
    let audio = runtime
        .synthesize_speech(model, input, voice, options)
        .await
        .map_err(UseCaseError::from)?
        .ok_or_else(|| UseCaseError::ModelNotAvailable(command.model.clone()))?;
    tracing::debug!(
        samples = audio.samples.len(),
        sample_rate = audio.sample_rate,
        elapsed_ms = synthesis_started.elapsed().as_millis(),
        "speech synthesis completed"
    );

    let encode_started = Instant::now();
    tracing::debug!(format = %format, "speech audio encode started");
    let encoded = encode_tts_audio(&audio, format).await?;
    tracing::debug!(
        bytes = encoded.bytes.len(),
        format = %format,
        elapsed_ms = encode_started.elapsed().as_millis(),
        "speech response encoded"
    );
    Ok(SpeechResult {
        bytes: encoded.bytes,
        format,
    })
}

fn supported_voice(model: &TtsModel, command: &SpeechCommand) -> Result<TtsVoice, UseCaseError> {
    let voice = to_tts_voice(command)?;
    orchion::ensure_voice_supported(model, &voice)?;
    Ok(voice)
}

async fn transcode_reference_audio(
    input: PathBuf,
    output: PathBuf,
    max_duration: Duration,
) -> Result<(), UseCaseError> {
    let max_samples = max_audio_samples(max_duration, orchion::ASR_SAMPLE_RATE)?;
    if !protect_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let decoded = decode_audio_file_with_max_samples(input, max_samples).await;
    if finish_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let decoded = decoded.map_err(UseCaseError::ReferenceAudio)?;
    let audio = TtsAudio::new(decoded.samples, decoded.sample_rate);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .map_err(UseCaseError::ReferenceAudio)?;
    if !protect_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let write = tokio::fs::write(output, encoded.bytes).await;
    if finish_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    write.map_err(|error| UseCaseError::Internal(error.to_string()))
}

/// # Errors
///
/// Returns [`UseCaseError`] when a speech parameter is invalid.
pub fn validate(command: &SpeechCommand) -> Result<(), UseCaseError> {
    if command.input.trim().is_empty() {
        return Err(UseCaseError::invalid(
            "`input` must not be empty",
            Some("input"),
            "empty_input",
        ));
    }
    if !command.speed.is_finite() || (command.speed - 1.0).abs() > f32::EPSILON {
        return Err(UseCaseError::invalid(
            "`speed` values other than 1.0 are not currently supported",
            Some("speed"),
            "unsupported_speed",
        ));
    }
    if command
        .temperature
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(UseCaseError::invalid(
            "`temperature` must be greater than 0",
            Some("temperature"),
            "invalid_temperature",
        ));
    }
    if command.top_k.is_some_and(|value| value == 0) {
        return Err(UseCaseError::invalid(
            "`top_k` must be greater than 0",
            Some("top_k"),
            "invalid_top_k",
        ));
    }
    if command
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(UseCaseError::invalid(
            "`top_p` must be between 0 and 1",
            Some("top_p"),
            "invalid_top_p",
        ));
    }
    if command
        .repetition_penalty
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(UseCaseError::invalid(
            "`repetition_penalty` must be greater than 0",
            Some("repetition_penalty"),
            "invalid_repetition_penalty",
        ));
    }
    if command.max_length.is_some_and(|value| value == 0) {
        return Err(UseCaseError::invalid(
            "`max_length` must be greater than 0",
            Some("max_length"),
            "invalid_max_length",
        ));
    }
    Ok(())
}

fn validate_request_limit(command: &SpeechCommand, max_length: usize) -> Result<(), UseCaseError> {
    if command.max_length.is_some_and(|value| value > max_length) {
        return Err(UseCaseError::invalid(
            format!("`max_length` must not exceed the configured limit of {max_length}"),
            Some("max_length"),
            "max_length_exceeded",
        ));
    }
    Ok(())
}

#[must_use]
pub fn to_tts_options(command: &SpeechCommand) -> TtsOptions {
    let defaults = TtsOptions::default();
    TtsOptions {
        seed: Some(command.seed.unwrap_or(42)),
        temperature: command.temperature.unwrap_or(defaults.temperature),
        top_k: command.top_k.unwrap_or(defaults.top_k),
        top_p: command.top_p.unwrap_or(defaults.top_p),
        repetition_penalty: command
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        max_length: command.max_length.unwrap_or(defaults.max_length),
    }
}

/// # Errors
///
/// Returns [`UseCaseError`] when the requested voice or language is invalid.
pub fn to_tts_voice(command: &SpeechCommand) -> Result<TtsVoice, UseCaseError> {
    let language = command
        .language
        .as_deref()
        .map(parse_language)
        .transpose()?
        .unwrap_or(TtsLanguage::Auto);
    match normalize_identifier(&command.voice).as_str() {
        "clone" => {
            let reference_audio = command.reference_audio.as_ref().ok_or_else(|| {
                UseCaseError::invalid(
                    "voice clone requires `reference_audio`",
                    Some("reference_audio"),
                    "missing_required_parameter",
                )
            })?;
            let reference_text = command.reference_text.as_ref().ok_or_else(|| {
                UseCaseError::invalid(
                    "voice clone requires `reference_text`",
                    Some("reference_text"),
                    "missing_required_parameter",
                )
            })?;
            Ok(TtsVoice::Clone {
                reference_audio: reference_audio.clone(),
                reference_text: reference_text.clone(),
                language,
            })
        }
        "design" => {
            let prompt = command.voice_prompt.as_ref().ok_or_else(|| {
                UseCaseError::invalid(
                    "voice design requires `voice_prompt`",
                    Some("voice_prompt"),
                    "missing_required_parameter",
                )
            })?;
            Ok(TtsVoice::Design {
                prompt: prompt.clone(),
                language,
            })
        }
        voice => Ok(TtsVoice::Preset {
            speaker: parse_speaker(voice)?,
            language,
        }),
    }
}

fn parse_speaker(value: &str) -> Result<TtsSpeaker, UseCaseError> {
    match normalize_identifier(value).as_str() {
        "serena" => Ok(TtsSpeaker::Serena),
        "vivian" => Ok(TtsSpeaker::Vivian),
        "uncle-fu" | "unclefu" => Ok(TtsSpeaker::UncleFu),
        "ryan" => Ok(TtsSpeaker::Ryan),
        "aiden" => Ok(TtsSpeaker::Aiden),
        "ono-anna" | "onoanna" => Ok(TtsSpeaker::OnoAnna),
        "sohee" => Ok(TtsSpeaker::Sohee),
        "eric" => Ok(TtsSpeaker::Eric),
        "dylan" => Ok(TtsSpeaker::Dylan),
        _ => Err(UseCaseError::invalid(
            format!("unsupported voice `{value}`"),
            Some("voice"),
            "unsupported_voice",
        )),
    }
}

fn parse_language(value: &str) -> Result<TtsLanguage, UseCaseError> {
    match normalize_identifier(value).as_str() {
        "" | "auto" => Ok(TtsLanguage::Auto),
        "english" | "en" => Ok(TtsLanguage::English),
        "chinese" | "zh" => Ok(TtsLanguage::Chinese),
        "japanese" | "ja" => Ok(TtsLanguage::Japanese),
        "korean" | "ko" => Ok(TtsLanguage::Korean),
        "german" | "de" => Ok(TtsLanguage::German),
        "french" | "fr" => Ok(TtsLanguage::French),
        "russian" | "ru" => Ok(TtsLanguage::Russian),
        "portuguese" | "pt" => Ok(TtsLanguage::Portuguese),
        "spanish" | "es" => Ok(TtsLanguage::Spanish),
        "italian" | "it" => Ok(TtsLanguage::Italian),
        _ => Err(UseCaseError::invalid(
            format!("unsupported language `{value}`"),
            Some("language"),
            "unsupported_language",
        )),
    }
}

fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn max_audio_samples(duration: Duration, sample_rate: u32) -> Result<usize, UseCaseError> {
    let samples = duration
        .as_millis()
        .checked_mul(u128::from(sample_rate))
        .and_then(|samples| samples.checked_add(999))
        .map(|samples| samples / 1000)
        .ok_or_else(|| UseCaseError::Internal("configured audio duration is too large".into()))?;
    usize::try_from(samples)
        .map_err(|_| UseCaseError::Internal("configured audio duration is too large".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command() -> SpeechCommand {
        SpeechCommand {
            model: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice".into(),
            input: "hello".into(),
            voice: "ryan".into(),
            output_format: None,
            speed: 1.0,
            language: None,
            reference_audio: None,
            reference_audio_output: None,
            reference_text: None,
            voice_prompt: None,
            seed: None,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            max_length: None,
        }
    }

    #[test]
    fn validates_configured_length_at_the_application_boundary() {
        let mut command = command();
        command.max_length = Some(65);
        let error = validate_request_limit(&command, 64).unwrap_err();
        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                code: "max_length_exceeded",
                ..
            }
        ));
    }

    #[test]
    fn maps_preset_voice_without_transport_types() {
        assert_eq!(
            to_tts_voice(&command()).unwrap(),
            TtsVoice::Preset {
                speaker: TtsSpeaker::Ryan,
                language: TtsLanguage::Auto,
            }
        );
    }

    #[test]
    fn private_tts_suffixes_cannot_enable_request_capabilities() {
        let cases = [
            ("Acme/Private-TTS-CustomVoice", "ryan"),
            ("Acme/Private-TTS-Base", "clone"),
            ("Acme/Private-TTS-VoiceDesign", "design"),
        ];

        for (model_id, voice) in cases {
            let mut command = command();
            command.model = model_id.into();
            command.voice = voice.into();
            command.reference_audio = Some(PathBuf::from("reference.wav"));
            command.reference_text = Some("reference".into());
            command.voice_prompt = Some("calm voice".into());
            let model = TtsModel::parse(model_id).unwrap();

            let error = supported_voice(&model, &command).unwrap_err();
            assert!(matches!(
                error,
                UseCaseError::Core(orchion::OrchionError::UnsupportedCapability { .. })
            ));
        }
    }
}
