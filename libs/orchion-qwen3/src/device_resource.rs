use candle_core::Device;
use orchion_core::{OrchionError, Result};
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

/// Owns an upstream model or stream state, inside the last shared reference.
/// The cache or a cancelled request can release that reference on any thread.
pub(crate) struct DeviceResource<T> {
    value: Option<T>,
    #[cfg(any(feature = "asr", feature = "cuda"))]
    device: Device,
}

impl<T> DeviceResource<T> {
    pub(crate) fn load(device: &Device, load: impl FnOnce(&Device) -> Result<T>) -> Result<Self> {
        let mut resource = Self {
            value: None,
            #[cfg(any(feature = "asr", feature = "cuda"))]
            device: device.clone(),
        };
        // Even a failed load can have queued frees for partially loaded tensors.
        resource.value = Some(load(device)?);
        Ok(resource)
    }

    #[cfg(feature = "asr")]
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn lock(resource: &Mutex<Self>) -> Result<MutexGuard<'_, Self>> {
        let guard = resource.lock().map_err(|error| OrchionError::Inference {
            message: error.to_string(),
        })?;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = &guard.device {
            device
                .cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(|error| OrchionError::Inference {
                    message: format!("bind CUDA context before inference: {error}"),
                })?;
        }
        Ok(guard)
    }
}

impl<T> Deref for DeviceResource<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.as_ref().expect("device resource is initialized")
    }
}

impl<T> DerefMut for DeviceResource<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("device resource is initialized")
    }
}

#[cfg(feature = "cuda")]
impl<T> Drop for DeviceResource<T> {
    fn drop(&mut self) {
        let Device::Cuda(device) = &self.device else {
            return;
        };
        let stream = device.cuda_stream();
        let context = stream.context();
        // cudarc reports a previously recorded Drop error before binding or
        // synchronizing. Report it separately so it cannot skip either step.
        if let Err(error) = context.check_err() {
            tracing::error!(%error, gpu = context.ordinal(), "CUDA error before releasing resource");
        }
        if let Err(error) = context.bind_to_thread() {
            tracing::error!(%error, gpu = context.ordinal(), "failed to bind CUDA context before releasing resource");
        }
        // cudarc frees default-stream allocations using the calling thread's
        // context. Keep it alive until all queued frees have completed.
        drop(self.value.take());
        if let Err(error) = context.check_err() {
            tracing::error!(%error, gpu = context.ordinal(), "failed to release CUDA resource");
        }
        if let Err(error) = stream.synchronize() {
            tracing::error!(%error, gpu = context.ordinal(), "failed to synchronize CUDA resource release");
        }
    }
}
