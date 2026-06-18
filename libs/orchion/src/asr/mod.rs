mod audio;

use crate::error::{OrchionError, Result};
use crate::model::AsrModel;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub use audio::{ASR_SAMPLE_RATE, prepare_asr_samples};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscript {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

impl From<qwen3_asr::TranscribeResult> for AsrTranscript {
    fn from(result: qwen3_asr::TranscribeResult) -> Self {
        Self {
            text: result.text,
            language: result.language,
            raw_output: result.raw_output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub max_new_tokens: usize,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            max_new_tokens: 512,
        }
    }
}

impl AsrOptions {
    fn into_upstream(self) -> qwen3_asr::TranscribeOptions {
        let mut options =
            qwen3_asr::TranscribeOptions::default().with_max_new_tokens(self.max_new_tokens);
        if let Some(language) = self.language {
            options = options.with_language(language);
        }
        options
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrStreamingOptions {
    pub language: Option<String>,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens_streaming: usize,
    pub max_new_tokens_final: usize,
    pub initial_text: Option<String>,
}

impl Default for AsrStreamingOptions {
    fn default() -> Self {
        Self {
            language: None,
            chunk_size_sec: 2.0,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens_streaming: 32,
            max_new_tokens_final: 512,
            initial_text: None,
        }
    }
}

impl AsrStreamingOptions {
    fn validate(&self) -> Result<()> {
        if !self.chunk_size_sec.is_finite() || self.chunk_size_sec <= 0.0 {
            return Err(OrchionError::InvalidAudio {
                reason: "streaming chunk_size_sec must be finite and greater than zero".to_string(),
            });
        }
        Ok(())
    }

    fn into_upstream(self) -> qwen3_asr::StreamingOptions {
        let mut options = qwen3_asr::StreamingOptions::default()
            .with_chunk_size_sec(self.chunk_size_sec)
            .with_unfixed_chunk_num(self.unfixed_chunk_num)
            .with_unfixed_token_num(self.unfixed_token_num)
            .with_max_new_tokens_streaming(self.max_new_tokens_streaming)
            .with_max_new_tokens_final(self.max_new_tokens_final);
        if let Some(language) = self.language {
            options = options.with_language(language);
        }
        if let Some(initial_text) = self.initial_text {
            options = options.with_initial_text(initial_text);
        }
        options
    }
}

#[derive(Clone)]
pub struct Asr {
    model: AsrModel,
    engine: Arc<Mutex<qwen3_asr::AsrInference>>,
}

pub struct AsrStream {
    engine: Arc<Mutex<qwen3_asr::AsrInference>>,
    state: Option<qwen3_asr::StreamingState>,
}

impl Asr {
    pub async fn load(model: AsrModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        crate::blocking::run(move || {
            let device = qwen3_asr::best_device();
            let device_debug = format!("{device:?}");
            tracing::info!(
                model = ?model,
                device = device_label_from_debug(&device_debug),
                "ASR device selected"
            );
            tracing::debug!(device_debug, "ASR device details selected");
            let engine = qwen3_asr::AsrInference::load(&model_dir, device).map_err(|source| {
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

    #[cfg(feature = "download")]
    pub async fn load_or_download(model: AsrModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = crate::download::ModelDownloader::default()
            .download(model, cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub const fn model(&self) -> AsrModel {
        self.model
    }

    pub async fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<AsrTranscript> {
        self.transcribe_file_with(path, AsrOptions::default()).await
    }

    pub async fn transcribe_audio_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<AsrTranscript> {
        self.transcribe_audio_bytes_with(bytes, AsrOptions::default())
            .await
    }

    pub async fn transcribe_audio_bytes_with(
        &self,
        bytes: impl Into<Vec<u8>>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = crate::audio::decode_audio_bytes(bytes.into()).await?;
        let decode_elapsed = decode_started.elapsed();
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_elapsed.as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.transcribe_samples_with(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|_| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    "ASR inference completed"
                );
            })
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
                .transcribe(path_text, options.into_upstream())
                .map(AsrTranscript::from)
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
                .transcribe_samples(&prepared, options.into_upstream())
                .map(AsrTranscript::from)
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
        options.validate()?;
        let engine = Arc::clone(&self.engine);
        let state_engine = Arc::clone(&engine);
        let state = crate::blocking::run(move || {
            let engine = state_engine
                .lock()
                .map_err(|error| OrchionError::Inference {
                    source: anyhow::anyhow!(error.to_string()),
                })?;
            Ok(engine.init_streaming(options.into_upstream()))
        })
        .await?;
        Ok(AsrStream {
            engine,
            state: Some(state),
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
        let engine = Arc::clone(&self.engine);
        let mut state = self.state.take().ok_or_else(|| OrchionError::Inference {
            source: anyhow::anyhow!("stream already finished"),
        })?;
        let (state, result) = crate::blocking::run(move || {
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                source: anyhow::anyhow!(error.to_string()),
            })?;
            let result = engine
                .feed_audio(&mut state, &prepared)
                .map(|result| result.map(AsrTranscript::from))
                .map_err(|source| OrchionError::Inference {
                    source: anyhow::Error::new(source),
                })?;
            Ok((state, result))
        })
        .await?;
        self.state = Some(state);
        Ok(result)
    }

    pub async fn finish(mut self) -> Result<AsrTranscript> {
        let engine = Arc::clone(&self.engine);
        let mut state = self.state.take().ok_or_else(|| OrchionError::Inference {
            source: anyhow::anyhow!("stream already finished"),
        })?;
        crate::blocking::run(move || {
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                source: anyhow::anyhow!(error.to_string()),
            })?;
            engine
                .finish_streaming(&mut state)
                .map(AsrTranscript::from)
                .map_err(|source| OrchionError::Inference {
                    source: anyhow::Error::new(source),
                })
        })
        .await
    }
}

fn device_label_from_debug(device_debug: &str) -> &'static str {
    if device_debug.contains("Cuda") {
        "cuda"
    } else if device_debug.contains("Metal") {
        "metal"
    } else {
        "cpu"
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
        let upstream = options.clone().into_upstream();
        assert_eq!(upstream.language.as_deref(), Some("english"));
        assert_eq!(upstream.max_new_tokens, 128);
    }

    #[test]
    fn streaming_options_reject_non_positive_chunk_size() {
        let options = AsrStreamingOptions {
            chunk_size_sec: 0.0,
            ..Default::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn device_label_detects_runtime_backend() {
        assert_eq!(device_label_from_debug("Cpu"), "cpu");
        assert_eq!(device_label_from_debug("Cuda(CudaDevice(0))"), "cuda");
        assert_eq!(device_label_from_debug("Metal(MetalDevice(0))"), "metal");
    }
}
