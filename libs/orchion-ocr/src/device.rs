use orchion_core::{DevicePreference, KnownOcrModel, OcrModelKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPolicy {
    OrtCpu,
    OrtCuda(Option<usize>),
    OrtWebGpu,
    CandleCpu,
    CandleCuda(Option<usize>),
    CandleMetal,
}

impl ProviderPolicy {
    pub fn candidates_for_model(model: KnownOcrModel, device: DevicePreference) -> Vec<Self> {
        match model.kind() {
            OcrModelKind::OcrVl => match device {
                DevicePreference::Auto => Self::auto_vl_candidates(),
                DevicePreference::Cpu => vec![Self::CandleCpu],
                DevicePreference::Metal => vec![Self::CandleMetal],
                DevicePreference::Cuda(index) => vec![Self::CandleCuda(index)],
            },
            OcrModelKind::TraditionalOcr | OcrModelKind::Layout => match device {
                DevicePreference::Auto => Self::auto_onnx_candidates(),
                DevicePreference::Cpu => vec![Self::OrtCpu],
                DevicePreference::Metal => vec![Self::OrtWebGpu],
                DevicePreference::Cuda(index) => vec![Self::OrtCuda(index)],
            },
        }
    }

    fn auto_vl_candidates() -> Vec<Self> {
        vec![
            #[cfg(feature = "cuda")]
            Self::CandleCuda(None),
            #[cfg(feature = "metal")]
            Self::CandleMetal,
            Self::CandleCpu,
        ]
    }

    fn auto_onnx_candidates() -> Vec<Self> {
        vec![
            #[cfg(feature = "cuda")]
            Self::OrtCuda(None),
            #[cfg(feature = "metal")]
            // The upstream `ort` prebuilt runtime currently ships WebGPU builds but no
            // CoreML-specific package. Prefer CoreML once compatible binaries are distributed.
            Self::OrtWebGpu,
            Self::OrtCpu,
        ]
    }
}

pub fn try_provider_candidates<T, E>(
    candidates: &[ProviderPolicy],
    mut attempt: impl FnMut(ProviderPolicy) -> std::result::Result<T, E>,
) -> std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    let mut candidates = candidates.iter().copied().peekable();
    loop {
        let candidate = candidates
            .next()
            .expect("provider candidate lists always contain CPU or one explicit provider");
        match attempt(candidate) {
            Ok(value) => return Ok(value),
            Err(error) => {
                let Some(next) = candidates.peek() else {
                    return Err(error);
                };
                tracing::warn!(
                    provider = ?candidate,
                    fallback_provider = ?next,
                    error = %error,
                    "OCR runtime build failed with provider; trying fallback"
                );
            }
        }
    }
}
