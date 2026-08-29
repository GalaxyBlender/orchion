mod assets;
#[cfg(any(feature = "ocr", feature = "ocr-vl", test))]
mod device;
mod result;
mod runtime;
#[cfg(all(feature = "ocr-vl", feature = "cuda"))]
mod vl_worker;

pub use assets::{OcrAssets, TableStructureAssets};
pub use result::validate_image_file;
pub use runtime::OcrEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use orchion_core::{DevicePreference, KnownOcrModel};

    #[test]
    fn traditional_ocr_maps_metal_to_webgpu_policy() {
        let candidates = device::ProviderPolicy::candidates_for_model(
            KnownOcrModel::PpOcrV6Tiny,
            DevicePreference::Metal,
        );
        assert_eq!(candidates, [device::ProviderPolicy::OrtWebGpu]);
    }

    #[test]
    fn traditional_ocr_maps_cuda_to_ort_cuda_policy() {
        let candidates = device::ProviderPolicy::candidates_for_model(
            KnownOcrModel::PpOcrV6Tiny,
            DevicePreference::Cuda(Some(0)),
        );
        assert_eq!(candidates, [device::ProviderPolicy::OrtCuda(Some(0))]);
    }

    #[test]
    fn ocr_vl_maps_metal_to_candle_metal_policy() {
        let candidates = device::ProviderPolicy::candidates_for_model(
            KnownOcrModel::PaddleOcrVl16,
            DevicePreference::Metal,
        );
        assert_eq!(candidates, [device::ProviderPolicy::CandleMetal]);
    }

    #[test]
    fn auto_vl_candidates_follow_compiled_provider_order() {
        let expected = vec![
            #[cfg(feature = "cuda")]
            device::ProviderPolicy::CandleCuda(None),
            #[cfg(feature = "metal")]
            device::ProviderPolicy::CandleMetal,
            device::ProviderPolicy::CandleCpu,
        ];

        assert_eq!(
            device::ProviderPolicy::candidates_for_model(
                KnownOcrModel::PaddleOcrVl16,
                DevicePreference::Auto,
            ),
            expected
        );
    }

    #[test]
    fn auto_onnx_candidates_follow_compiled_provider_order() {
        let expected = vec![
            #[cfg(feature = "cuda")]
            device::ProviderPolicy::OrtCuda(None),
            #[cfg(feature = "metal")]
            device::ProviderPolicy::OrtWebGpu,
            device::ProviderPolicy::OrtCpu,
        ];

        assert_eq!(
            device::ProviderPolicy::candidates_for_model(
                KnownOcrModel::PpOcrV6Tiny,
                DevicePreference::Auto,
            ),
            expected
        );
    }

    #[test]
    fn auto_vl_tries_candidates_in_order_and_falls_back_to_cpu() {
        use device::ProviderPolicy::{CandleCpu, CandleCuda, CandleMetal};

        let candidates = [CandleCuda(None), CandleMetal, CandleCpu];
        let mut attempted = Vec::new();
        let selected = device::try_provider_candidates(&candidates, |candidate| {
            attempted.push(candidate);
            (candidate == CandleCpu)
                .then_some(candidate)
                .ok_or("unavailable")
        })
        .unwrap();

        assert_eq!(attempted, candidates);
        assert_eq!(selected, CandleCpu);
    }

    #[test]
    fn auto_onnx_tries_candidates_in_order_and_falls_back_to_cpu() {
        use device::ProviderPolicy::{OrtCpu, OrtCuda, OrtWebGpu};

        let candidates = [OrtCuda(None), OrtWebGpu, OrtCpu];
        let mut attempted = Vec::new();
        let selected = device::try_provider_candidates(&candidates, |candidate| {
            attempted.push(candidate);
            (candidate == OrtCpu)
                .then_some(candidate)
                .ok_or("unavailable")
        })
        .unwrap();

        assert_eq!(attempted, candidates);
        assert_eq!(selected, OrtCpu);
    }

    #[test]
    fn explicit_provider_failure_is_not_fallback() {
        let candidates = device::ProviderPolicy::candidates_for_model(
            KnownOcrModel::PpOcrV6Tiny,
            DevicePreference::Cuda(Some(2)),
        );
        let mut attempted = Vec::new();
        let result: std::result::Result<(), _> =
            device::try_provider_candidates(&candidates, |candidate| {
                attempted.push(candidate);
                Err("unavailable")
            });

        assert_eq!(result, Err("unavailable"));
        assert_eq!(attempted, [device::ProviderPolicy::OrtCuda(Some(2))]);
    }

    #[cfg(all(feature = "ocr-vl", feature = "cuda"))]
    #[test]
    fn cuda_ocr_engine_handle_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<OcrEngine>();
    }
}
