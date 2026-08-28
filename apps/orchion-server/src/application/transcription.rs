use super::{
    RuntimeError, UseCaseError, finish_owned_file_operation, protect_owned_file_operation,
};
use orchion::{AsrModel, AsrOptions, AsrTranscript, decode_audio_file_with_max_samples};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

pub type TranscriptionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<AsrTranscript>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct TranscriptionPolicy {
    pub available_models: Vec<AsrModel>,
    pub max_audio_duration: Duration,
}

pub trait TranscriptionRuntime: Send + Sync {
    fn transcription_policy(&self) -> TranscriptionPolicy;

    fn transcribe(
        &self,
        model: AsrModel,
        samples: Vec<f32>,
        sample_rate: u32,
        options: AsrOptions,
        with_segments: bool,
    ) -> TranscriptionFuture<'_>;
}

#[derive(Debug)]
pub struct TranscriptionCommand {
    pub audio_path: PathBuf,
    pub model: String,
    pub language: Option<String>,
    pub with_segments: bool,
}

#[derive(Debug)]
pub struct TranscriptionResult {
    pub transcript: AsrTranscript,
    pub duration: f64,
}

/// # Errors
///
/// Returns [`UseCaseError`] when decoding, model selection, or transcription fails.
pub async fn transcribe(
    runtime: &impl TranscriptionRuntime,
    command: TranscriptionCommand,
) -> Result<TranscriptionResult, UseCaseError> {
    let policy = runtime.transcription_policy();
    let model = AsrModel::parse(&command.model)
        .map_err(|_| UseCaseError::ModelNotAvailable(command.model.clone()))?;
    if !policy.available_models.contains(&model) {
        return Err(UseCaseError::ModelNotAvailable(command.model));
    }

    let max_samples = max_audio_samples(policy.max_audio_duration, orchion::ASR_SAMPLE_RATE)?;
    if !protect_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let decoded = decode_audio_file_with_max_samples(command.audio_path, max_samples).await;
    if finish_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let decoded = decoded?;
    let duration = transcription_duration(decoded.samples.len(), decoded.sample_rate)?;
    let options = AsrOptions {
        language: command.language,
        ..Default::default()
    };
    let transcript = runtime
        .transcribe(
            model,
            decoded.samples,
            decoded.sample_rate,
            options,
            command.with_segments,
        )
        .await
        .map_err(UseCaseError::from)?
        .ok_or(UseCaseError::ModelNotAvailable(command.model))?;

    Ok(TranscriptionResult {
        transcript,
        duration,
    })
}

fn transcription_duration(sample_count: usize, sample_rate: u32) -> Result<f64, UseCaseError> {
    let sample_rate = usize::try_from(sample_rate)
        .map_err(|_| UseCaseError::Internal("decoded audio sample rate is too large".into()))?;
    if sample_rate == 0 {
        return Err(UseCaseError::Internal(
            "decoded audio sample rate is zero".into(),
        ));
    }
    let whole_seconds = u64::try_from(sample_count / sample_rate)
        .map_err(|_| UseCaseError::Internal("decoded audio duration is too large".into()))?;
    let fractional_nanos = (sample_count % sample_rate)
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_div(sample_rate))
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| UseCaseError::Internal("decoded audio duration is too large".into()))?;
    Ok(Duration::new(whole_seconds, fractional_nanos).as_secs_f64())
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

    #[test]
    fn duration_preserves_fractional_samples() {
        assert!((transcription_duration(24_000, 16_000).unwrap() - 1.5).abs() <= f64::EPSILON);
    }

    #[test]
    fn max_samples_rounds_partial_milliseconds_up() {
        assert_eq!(
            max_audio_samples(Duration::from_millis(1), 16_001).unwrap(),
            17
        );
    }

    #[test]
    fn zero_sample_rate_is_an_internal_error() {
        assert!(matches!(
            transcription_duration(1, 0),
            Err(UseCaseError::Internal(_))
        ));
    }
}
