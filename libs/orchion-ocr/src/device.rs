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
    pub fn for_model(model: KnownOcrModel, device: DevicePreference) -> Self {
        match model.kind() {
            OcrModelKind::OcrVl => match device {
                DevicePreference::Auto => Self::auto_vl(),
                DevicePreference::Cpu => Self::CandleCpu,
                DevicePreference::Metal => Self::CandleMetal,
                DevicePreference::Cuda(index) => Self::CandleCuda(index),
            },
            OcrModelKind::TraditionalOcr | OcrModelKind::Layout => match device {
                DevicePreference::Auto => Self::auto_onnx(),
                DevicePreference::Cpu => Self::OrtCpu,
                DevicePreference::Metal => Self::OrtWebGpu,
                DevicePreference::Cuda(index) => Self::OrtCuda(index),
            },
        }
    }

    fn auto_vl() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self::CandleCuda(None)
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            Self::CandleMetal
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        {
            Self::CandleCpu
        }
    }

    fn auto_onnx() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self::OrtCuda(None)
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            // The upstream `ort` prebuilt runtime currently ships WebGPU builds but no
            // CoreML-specific package. Prefer an ORT CoreML policy here once `ort`
            // distributes CoreML-enabled binaries.
            Self::OrtWebGpu
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        {
            Self::OrtCpu
        }
    }
}
