use crate::TTS_FRAME_LIMIT;
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use crane_core::{generation::SpeechOptions, models::qwen3_tts::Model};
use orchion_core::{DevicePreference, TtsAudio};
use std::path::Path;

/// Benchmark the released streaming API without changing the production adapter.
pub(super) struct Preset {
    model: Model,
    device: Device,
}

impl Preset {
    pub(super) fn load(path: &Path, preference: DevicePreference) -> Result<Self> {
        let path = path.to_str().context("model directory is not UTF-8")?;
        let device = device_for_benchmark(preference)?;
        let model = Model::new(path, &device, &device.bf16_default_to_f32())?;
        device.synchronize()?;
        Ok(Self { model, device })
    }

    pub(super) fn synthesize(&mut self, text: &str) -> Result<TtsAudio> {
        let opts = SpeechOptions {
            max_new_tokens: TTS_FRAME_LIMIT,
            temperature: 0.7,
            top_p: Some(0.8),
            repetition_penalty: 1.05,
            ..Default::default()
        };
        let rate = self.model.sample_rate();
        let result = self.device.with_context(|| -> Result<TtsAudio> {
            let mut samples = Vec::new();
            for chunk in
                self.model
                    .generate_speech_streaming(text, "chinese", Some("vivian"), &opts)?
            {
                samples.extend(chunk?.to_dtype(DType::F32)?.to_vec1::<f32>()?);
            }
            Ok(TtsAudio::new(samples, rate))
        });
        self.model.clear_kv_cache();
        let synchronized = self.device.synchronize();
        let audio = result?;
        synchronized?;
        Ok(audio)
    }
}

pub(super) fn device_for_benchmark(preference: DevicePreference) -> Result<Device> {
    let device = match preference {
        DevicePreference::Cpu => Device::Cpu,
        DevicePreference::Cuda(index) => Device::new_cuda(index.unwrap_or(0))?,
        _ => bail!("expected CPU or CUDA"),
    };
    Ok(device)
}
