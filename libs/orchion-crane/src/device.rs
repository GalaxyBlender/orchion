use candle_core::{DType, Device};
use orchion_core::{DevicePreference, OrchionError, Result};
use std::path::Path;

pub(crate) struct Loaded<M> {
    model: Option<M>,
    device: Device,
}

impl<M: Send> Loaded<M> {
    pub(crate) fn open<E: std::fmt::Display>(
        path: &Path,
        preference: DevicePreference,
        open: impl FnOnce(&str, &Device, &DType) -> std::result::Result<M, E>,
    ) -> Result<Self> {
        let path = path.to_str().ok_or_else(|| OrchionError::NonUtf8Path {
            path: path.to_path_buf(),
        })?;
        let device = match preference {
            DevicePreference::Cpu => Ok(Device::Cpu),
            DevicePreference::Cuda(index) => Device::new_cuda(index.unwrap_or(0)),
            DevicePreference::Metal => Device::new_metal(0),
            DevicePreference::Auto => Ok(auto_device()),
        }
        .map_err(|error| OrchionError::ModelLoad {
            message: format!("select Crane device {preference}: {error}"),
        })?;
        let dtype = device.bf16_default_to_f32();
        let mut loaded = Self {
            model: None,
            device,
        };
        loaded.model =
            Some(
                open(path, &loaded.device, &dtype).map_err(|error| OrchionError::ModelLoad {
                    message: format!("{path}: {error:#}"),
                })?,
            );
        loaded
            .device
            .synchronize()
            .map_err(|error| OrchionError::ModelLoad {
                message: format!("finish loading {path}: {error}"),
            })?;
        Ok(loaded)
    }

    pub(crate) fn model(&self) -> &M {
        self.model.as_ref().expect("loaded model")
    }

    pub(crate) fn run<T: Send>(
        &mut self,
        run: impl FnOnce(&mut M) -> Result<T> + Send,
    ) -> Result<T> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = &self.device {
            device
                .cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(inference_error)?;
        }
        let model = self.model.as_mut().expect("loaded model");
        let result = self.device.with_context(|| run(model));
        if let Err(error) = self.device.synchronize() {
            if result.is_ok() {
                return Err(inference_error(error));
            }
            tracing::warn!(%error, "failed to finish device work after Crane inference error");
        }
        result
    }
}

impl<M> Drop for Loaded<M> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = &self.device {
            let stream = device.cuda_stream();
            let context = stream.context();
            if let Err(error) = context.check_err() {
                tracing::warn!(%error, "CUDA error before releasing Crane model");
            }
            if let Err(error) = context.bind_to_thread() {
                tracing::warn!(%error, "failed to bind CUDA context before releasing Crane model");
            }
            drop(self.model.take());
            if let Err(error) = context.check_err() {
                tracing::warn!(%error, "failed to release CUDA allocations from Crane model");
            }
            if let Err(error) = stream.synchronize() {
                tracing::warn!(%error, "failed to finish releasing Crane model");
            }
        }
    }
}

fn auto_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    #[cfg(feature = "metal")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    Device::Cpu
}

pub(crate) fn inference_error(error: impl std::fmt::Display) -> OrchionError {
    OrchionError::Inference {
        message: format!("Crane: {error:#}"),
    }
}
