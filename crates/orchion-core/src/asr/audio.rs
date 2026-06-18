use crate::error::{OrchionError, Result};
use std::borrow::Cow;

pub const ASR_SAMPLE_RATE: u32 = 16_000;

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn prepare_asr_samples(samples: &[f32], sample_rate: u32) -> Result<Cow<'_, [f32]>> {
    if sample_rate == 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "sample_rate must be greater than zero".to_string(),
        });
    }
    if samples.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio samples are empty".to_string(),
        });
    }
    if sample_rate == ASR_SAMPLE_RATE {
        return Ok(Cow::Borrowed(samples));
    }

    let output_len =
        (samples.len() as u64 * u64::from(ASR_SAMPLE_RATE)).div_ceil(u64::from(sample_rate));
    let output_len = usize::try_from(output_len).map_err(|error| OrchionError::Resample {
        reason: error.to_string(),
    })?;
    if output_len == 0 {
        return Err(OrchionError::Resample {
            reason: "resampled audio would be empty".to_string(),
        });
    }

    let mut output = Vec::with_capacity(output_len);
    let ratio = f64::from(sample_rate) / f64::from(ASR_SAMPLE_RATE);
    for index in 0..output_len {
        let source_position = index as f64 * ratio;
        let left_index = source_position.floor() as usize;
        let right_index = (left_index + 1).min(samples.len() - 1);
        let fraction = (source_position - left_index as f64) as f32;
        let left = samples[left_index.min(samples.len() - 1)];
        let right = samples[right_index];
        output.push(left + (right - left) * fraction);
    }

    Ok(Cow::Owned(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_sample_rate() {
        let error = prepare_asr_samples(&[0.0], 0).unwrap_err();
        assert!(
            matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("sample_rate"))
        );
    }

    #[test]
    fn rejects_empty_samples() {
        let error = prepare_asr_samples(&[], ASR_SAMPLE_RATE).unwrap_err();
        assert!(matches!(error, OrchionError::InvalidAudio { reason } if reason.contains("empty")));
    }

    #[test]
    fn keeps_sixteen_kilohertz_samples_borrowed() {
        let input = vec![0.0, 0.25, 0.5];
        let output = prepare_asr_samples(&input, ASR_SAMPLE_RATE).unwrap();
        assert!(matches!(output, Cow::Borrowed(_)));
        assert_eq!(&*output, input.as_slice());
    }

    #[test]
    fn resamples_eight_kilohertz_to_sixteen_kilohertz() {
        let output = prepare_asr_samples(&[0.0, 1.0], 8_000).unwrap();
        assert!(matches!(output, Cow::Owned(_)));
        assert_eq!(output.len(), 4);
        assert!((output[1] - 0.5).abs() < 1e-6);
    }
}
