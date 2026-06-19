use orchion_core::{
    ASR_SAMPLE_RATE, AsrModel, AsrOptions, AsrStreamingOptions, AsrTranscript, DevicePreference,
    OrchionError, Result, prepare_asr_samples,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Asr {
    model: AsrModel,
    engine: Arc<Mutex<qwen3_asr::AsrInference>>,
}

pub struct AsrStream {
    asr: Asr,
    samples: Vec<f32>,
    options: AsrOptions,
}

impl Asr {
    pub async fn load(model: AsrModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    pub async fn load_with_device(
        model: AsrModel,
        model_dir: impl AsRef<Path>,
        preference: DevicePreference,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        crate::blocking::run(move || {
            let resolved = crate::device::resolve_device(preference)?;
            let device_debug = format!("{:?}", resolved.device);
            tracing::info!(
                model = ?model,
                requested_device = %preference,
                device = %resolved.kind,
                "ASR device selected"
            );
            tracing::debug!(device_debug, "ASR device details selected");
            let engine =
                qwen3_asr::AsrInference::load(&model_dir, resolved.device).map_err(|source| {
                    OrchionError::ModelLoad {
                        source: anyhow::Error::new(source),
                    }
                })?;
            Ok(Self {
                model,
                engine: Arc::new(Mutex::new(engine)),
            })
        })
        .await
    }

    pub const fn model(&self) -> AsrModel {
        self.model
    }

    pub async fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<AsrTranscript> {
        self.transcribe_file_with(path, AsrOptions::default()).await
    }

    pub async fn transcribe_file_with(
        &self,
        path: impl AsRef<Path>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let path = path.as_ref().to_path_buf();
        let engine = Arc::clone(&self.engine);
        crate::blocking::run(move || {
            let path_text = path.to_str().ok_or_else(|| OrchionError::NonUtf8Path {
                path: PathBuf::from(&path),
            })?;
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                source: anyhow::anyhow!(error.to_string()),
            })?;
            engine
                .transcribe(path_text, transcribe_options_into_upstream(options))
                .map(transcript_from_upstream)
                .map_err(|source| OrchionError::Inference {
                    source: anyhow::Error::new(source),
                })
        })
        .await
    }

    pub async fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AsrTranscript> {
        self.transcribe_samples_with(samples, sample_rate, AsrOptions::default())
            .await
    }

    pub async fn transcribe_samples_with(
        &self,
        samples: &[f32],
        sample_rate: u32,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let prepared = prepare_asr_samples(samples, sample_rate)?.into_owned();
        let engine = Arc::clone(&self.engine);
        crate::blocking::run(move || {
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                source: anyhow::anyhow!(error.to_string()),
            })?;
            engine
                .transcribe_samples(&prepared, transcribe_options_into_upstream(options))
                .map(transcript_from_upstream)
                .map_err(|source| OrchionError::Inference {
                    source: anyhow::Error::new(source),
                })
        })
        .await
    }

    pub async fn start_streaming(&self) -> Result<AsrStream> {
        self.start_streaming_with(AsrStreamingOptions::default())
            .await
    }

    pub async fn start_streaming_with(&self, options: AsrStreamingOptions) -> Result<AsrStream> {
        validate_streaming_options(&options)?;
        let asr_options = streaming_options_into_transcribe_options(options);
        Ok(AsrStream {
            asr: self.clone(),
            samples: Vec::new(),
            options: asr_options,
        })
    }
}

impl AsrStream {
    pub async fn feed(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<AsrTranscript>> {
        let prepared = prepare_asr_samples(samples, sample_rate)?.into_owned();
        self.samples.extend(prepared);
        Ok(None)
    }

    pub async fn finish(self) -> Result<AsrTranscript> {
        self.asr
            .transcribe_samples_with(&self.samples, ASR_SAMPLE_RATE, self.options)
            .await
    }
}

fn transcript_from_upstream(result: qwen3_asr::TranscribeResult) -> AsrTranscript {
    AsrTranscript {
        text: result.text,
        language: result.language,
        raw_output: result.raw_output,
        segments: Vec::new(),
    }
}

fn transcribe_options_into_upstream(options: AsrOptions) -> qwen3_asr::TranscribeOptions {
    let mut upstream = qwen3_asr::TranscribeOptions::default();
    upstream.language = options.language;
    upstream.max_new_tokens = options.max_new_tokens;
    upstream
}

fn validate_streaming_options(options: &AsrStreamingOptions) -> Result<()> {
    if !options.chunk_size_sec.is_finite() || options.chunk_size_sec <= 0.0 {
        return Err(OrchionError::InvalidAudio {
            reason: "streaming chunk_size_sec must be finite and greater than zero".to_string(),
        });
    }
    Ok(())
}

fn streaming_options_into_transcribe_options(options: AsrStreamingOptions) -> AsrOptions {
    AsrOptions {
        language: options.language,
        max_new_tokens: options.max_new_tokens_final,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_options_convert_to_upstream_options() {
        let options = AsrOptions {
            language: Some("english".to_string()),
            max_new_tokens: 128,
        };
        let upstream = transcribe_options_into_upstream(options.clone());
        assert_eq!(upstream.language.as_deref(), Some("english"));
        assert_eq!(upstream.max_new_tokens, 128);
    }

    #[test]
    fn streaming_options_reject_non_positive_chunk_size() {
        let options = AsrStreamingOptions {
            chunk_size_sec: 0.0,
            ..Default::default()
        };
        assert!(validate_streaming_options(&options).is_err());
    }

    #[test]
    fn device_label_detects_cuda_index_from_resolver_kind() {
        assert_eq!(
            crate::device::ResolvedDeviceKind::Cuda(3).to_string(),
            "cuda3"
        );
    }

    #[test]
    fn exposes_explicit_device_loader_api() {
        let future = Asr::load_with_device(
            AsrModel::Qwen3Asr06B,
            "models/qwen3-asr-0.6b",
            orchion_core::DevicePreference::Cpu,
        );
        std::mem::drop(future);
    }
}
