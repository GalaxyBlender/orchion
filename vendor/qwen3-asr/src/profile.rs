use candle_core::Device;
use std::time::Instant;

/// Opt-in synchronized phase timing; not for end-to-end latency measurements.
pub(crate) struct PhaseTimer(Option<Instant>);

impl PhaseTimer {
    pub(crate) fn start() -> Self {
        Self(
            std::env::var_os("ORCHION_ASR_PROFILE")
                .is_some()
                .then(Instant::now),
        )
    }

    pub(crate) fn mark(&mut self, phase: &str, device: &Device) -> candle_core::Result<()> {
        if let Some(start) = self.0 {
            device.synchronize()?;
            log::info!(
                target: "qwen3_asr::profile",
                "{phase}: {:.2}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
            self.0 = Some(Instant::now());
        }
        Ok(())
    }
}
