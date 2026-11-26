use anyhow::Result;
use orchion_core::DevicePreference;

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::{
    CudaContext, result, sys,
    sys::CUmemPool_attribute_enum::{
        CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    },
};

pub struct Memory {
    #[cfg(feature = "cuda")]
    cuda: Option<std::sync::Arc<CudaContext>>,
}

impl Memory {
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(device: DevicePreference) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        #[cfg(feature = "cuda")]
        let cuda = if let DevicePreference::Cuda(index) = device {
            let context = CudaContext::new(index.unwrap_or(0))?;
            anyhow::ensure!(context.has_async_alloc(), "CUDA memory pool is unavailable");
            Some(context)
        } else {
            None
        };
        Ok(Self {
            #[cfg(feature = "cuda")]
            cuda,
        })
    }

    #[allow(clippy::unnecessary_wraps, unsafe_code)]
    pub fn reset_peak(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.cuda {
            context.synchronize()?;
            let pool = pool(context)?;
            let mut zero = 0_u64;
            // SAFETY: the pool belongs to the live context, and this attribute accepts a uint64_t zero.
            unsafe {
                result::mem_pool::set_attribute(
                    pool,
                    CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                    (&raw mut zero).cast(),
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    pub fn snapshot(&self) -> Result<String> {
        #[cfg(feature = "cuda")]
        if let Some(context) = &self.cuda {
            context.synchronize()?;
            let pool = pool(context)?;
            return Ok(format!(
                "used_mib={:.1} peak_mib={:.1} reserved_mib={:.1}",
                read_mib(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
                read_mib(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
                read_mib(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            ));
        }
        Ok(String::new())
    }
}

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn pool(context: &CudaContext) -> Result<sys::CUmemoryPool> {
    // SAFETY: the device belongs to the borrowed live context.
    Ok(unsafe { result::device::get_mem_pool(context.cu_device()) }?)
}

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
fn read_mib(pool: sys::CUmemoryPool, attribute: sys::CUmemPool_attribute_enum) -> Result<f64> {
    let mut bytes = 0_u64;
    // SAFETY: the benchmark reads only attributes that write a uint64_t.
    unsafe { result::mem_pool::get_attribute(pool, attribute, (&raw mut bytes).cast()) }?;
    Ok(f64::from(u32::try_from(bytes / 1024)?) / 1024.0)
}
