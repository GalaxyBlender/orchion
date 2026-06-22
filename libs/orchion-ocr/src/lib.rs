#[cfg(any(feature = "ocr", feature = "ocr-vl", test))]
mod device;
mod result;
mod runtime;

pub use runtime::OcrEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use orchion_core::{DevicePreference, KnownOcrModel};

    #[test]
    fn traditional_ocr_maps_metal_to_webgpu_policy() {
        let policy =
            device::ProviderPolicy::for_model(KnownOcrModel::PpOcrV6Tiny, DevicePreference::Metal);
        assert_eq!(policy, device::ProviderPolicy::OrtWebGpu);
    }

    #[test]
    fn traditional_ocr_maps_cuda_to_ort_cuda_policy() {
        let policy = device::ProviderPolicy::for_model(
            KnownOcrModel::PpOcrV6Tiny,
            DevicePreference::Cuda(Some(0)),
        );
        assert_eq!(policy, device::ProviderPolicy::OrtCuda(Some(0)));
    }

    #[test]
    fn ocr_vl_maps_metal_to_candle_metal_policy() {
        let policy = device::ProviderPolicy::for_model(
            KnownOcrModel::PaddleOcrVl16,
            DevicePreference::Metal,
        );
        assert_eq!(policy, device::ProviderPolicy::CandleMetal);
    }
}
