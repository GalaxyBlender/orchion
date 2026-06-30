use crate::{OrchionError, Result};
use orchion_core::DevicePreference;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedDeviceKind {
    Cpu,
    Metal,
    Cuda(usize),
}

pub(crate) struct ResolvedDevice {
    pub(crate) device: candle_core::Device,
    pub(crate) kind: ResolvedDeviceKind,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaMemoryCandidate {
    ordinal: usize,
    free_bytes: usize,
}

impl fmt::Display for ResolvedDeviceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Metal => formatter.write_str("metal"),
            Self::Cuda(index) => write!(formatter, "cuda{index}"),
        }
    }
}

pub(crate) fn resolve_device(preference: DevicePreference) -> Result<ResolvedDevice> {
    match preference {
        DevicePreference::Auto => Ok(resolve_auto_device()),
        DevicePreference::Cpu => Ok(resolve_cpu_device()),
        DevicePreference::Metal => resolve_metal_device(),
        DevicePreference::Cuda(index) => resolve_cuda_device(index.unwrap_or(0)),
    }
}

fn resolve_auto_device() -> ResolvedDevice {
    if let Some(device) = resolve_cuda_device_with_most_free_memory() {
        return device;
    }

    if let Some(device) = resolve_available_metal_device() {
        return device;
    }

    resolve_cpu_device()
}

fn resolve_cpu_device() -> ResolvedDevice {
    ResolvedDevice {
        device: candle_core::Device::Cpu,
        kind: ResolvedDeviceKind::Cpu,
    }
}

#[cfg(feature = "metal")]
fn resolve_metal_device() -> Result<ResolvedDevice> {
    candle_core::Device::new_metal(0)
        .map(|device| ResolvedDevice {
            device,
            kind: ResolvedDeviceKind::Metal,
        })
        .map_err(|source| OrchionError::ModelLoad {
            message: source.to_string(),
        })
}

#[cfg(not(feature = "metal"))]
fn resolve_metal_device() -> Result<ResolvedDevice> {
    Err(OrchionError::ModelLoad {
        message: "Metal support is not compiled in".to_string(),
    })
}

#[cfg(feature = "metal")]
fn resolve_available_metal_device() -> Option<ResolvedDevice> {
    resolve_metal_device().ok()
}

#[cfg(not(feature = "metal"))]
fn resolve_available_metal_device() -> Option<ResolvedDevice> {
    None
}

#[cfg(feature = "cuda")]
fn resolve_cuda_device(index: usize) -> Result<ResolvedDevice> {
    candle_core::Device::new_cuda(index)
        .map(|device| ResolvedDevice {
            device,
            kind: ResolvedDeviceKind::Cuda(index),
        })
        .map_err(|source| OrchionError::ModelLoad {
            message: source.to_string(),
        })
}

#[cfg(not(feature = "cuda"))]
fn resolve_cuda_device(_index: usize) -> Result<ResolvedDevice> {
    Err(OrchionError::ModelLoad {
        message: "CUDA support is not compiled in".to_string(),
    })
}

#[cfg(feature = "cuda")]
fn resolve_cuda_device_with_most_free_memory() -> Option<ResolvedDevice> {
    cuda_candidate_with_most_free_memory()
        .and_then(|candidate| resolve_cuda_device(candidate.ordinal).ok())
}

#[cfg(not(feature = "cuda"))]
fn resolve_cuda_device_with_most_free_memory() -> Option<ResolvedDevice> {
    None
}

#[cfg(feature = "cuda")]
fn cuda_candidate_with_most_free_memory() -> Option<CudaMemoryCandidate> {
    use candle_core::cuda::cudarc::driver::CudaContext;

    let count = usize::try_from(CudaContext::device_count().ok()?).ok()?;
    (0..count)
        .filter_map(|ordinal| {
            let context = CudaContext::new(ordinal).ok()?;
            let (free, total) = context.mem_get_info().ok()?;
            let _ = total;
            Some(CudaMemoryCandidate {
                ordinal,
                free_bytes: free,
            })
        })
        .max_by_key(|candidate| (candidate.free_bytes, std::cmp::Reverse(candidate.ordinal)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_resolved_device_kinds() {
        assert_eq!(ResolvedDeviceKind::Cpu.to_string(), "cpu");
        assert_eq!(ResolvedDeviceKind::Metal.to_string(), "metal");
        assert_eq!(ResolvedDeviceKind::Cuda(2).to_string(), "cuda2");
    }

    #[test]
    fn explicit_cpu_resolves_without_accelerator() {
        let resolved = resolve_device(DevicePreference::Cpu).unwrap();

        assert_eq!(resolved.kind, ResolvedDeviceKind::Cpu);
        assert!(resolved.device.is_cpu());
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    #[test]
    fn auto_resolves_to_cpu_without_accelerators() {
        let resolved = resolve_device(DevicePreference::Auto).unwrap();

        assert_eq!(resolved.kind, ResolvedDeviceKind::Cpu);
        assert!(resolved.device.is_cpu());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn explicit_cuda_errors_when_not_compiled_in() {
        match resolve_device(DevicePreference::Cuda(None)) {
            Err(OrchionError::ModelLoad { message }) => {
                assert!(message.contains("CUDA support is not compiled in"));
            }
            Err(other) => panic!("expected ModelLoad error, got {other:?}"),
            Ok(_) => panic!("expected explicit CUDA to fail without cuda feature"),
        }
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn explicit_metal_errors_when_not_compiled_in() {
        match resolve_device(DevicePreference::Metal) {
            Err(OrchionError::ModelLoad { message }) => {
                assert!(message.contains("Metal support is not compiled in"));
            }
            Err(other) => panic!("expected ModelLoad error, got {other:?}"),
            Ok(_) => panic!("expected explicit Metal to fail without metal feature"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_candidate_prefers_most_free_memory_then_ordinal() {
        let candidates = [
            CudaMemoryCandidate {
                ordinal: 2,
                free_bytes: 50,
            },
            CudaMemoryCandidate {
                ordinal: 1,
                free_bytes: 90,
            },
            CudaMemoryCandidate {
                ordinal: 0,
                free_bytes: 90,
            },
        ];
        let selected = candidates
            .into_iter()
            .max_by_key(|candidate| (candidate.free_bytes, std::cmp::Reverse(candidate.ordinal)))
            .unwrap();

        assert_eq!(selected.ordinal, 0);
    }
}
