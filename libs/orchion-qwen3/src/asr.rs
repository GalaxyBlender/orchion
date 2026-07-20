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
    engine: Arc<Mutex<qwen3_asr::AsrInference>>,
    state: Arc<Mutex<qwen3_asr::StreamingState>>,
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
                        message: source.to_string(),
                    }
                })?;
            Ok(Self {
                model,
                engine: Arc::new(Mutex::new(engine)),
            })
        })
        .await
    }

    pub fn model(&self) -> AsrModel {
        self.model.clone()
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
                message: error.to_string(),
            })?;
            engine
                .transcribe(path_text, transcribe_options_into_upstream(options))
                .map(transcript_from_upstream)
                .map_err(|source| OrchionError::Inference {
                    message: source.to_string(),
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
                message: error.to_string(),
            })?;
            engine
                .transcribe_samples(&prepared, transcribe_options_into_upstream(options))
                .map(transcript_from_upstream)
                .map_err(|source| OrchionError::Inference {
                    message: source.to_string(),
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
        let upstream_options = streaming_options_into_upstream_options(options);
        let engine = Arc::clone(&self.engine);
        let state = crate::blocking::run({
            let engine = Arc::clone(&engine);
            move || {
                let engine = engine.lock().map_err(|error| OrchionError::Inference {
                    message: error.to_string(),
                })?;
                Ok(engine.init_streaming(upstream_options))
            }
        })
        .await?;
        Ok(AsrStream {
            engine,
            state: Arc::new(Mutex::new(state)),
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
        let state = Arc::clone(&self.state);
        crate::blocking::run(move || {
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                message: error.to_string(),
            })?;
            let mut state = state.lock().map_err(|error| OrchionError::Inference {
                message: error.to_string(),
            })?;
            let transcript = engine
                .feed_audio(&mut state, &prepared)
                .map_err(|source| OrchionError::Inference {
                    message: source.to_string(),
                })?
                .map(transcript_from_upstream);
            Ok(transcript)
        })
        .await
    }

    pub async fn finish(self) -> Result<AsrTranscript> {
        let engine = Arc::clone(&self.engine);
        let state = Arc::clone(&self.state);
        crate::blocking::run(move || {
            let engine = engine.lock().map_err(|error| OrchionError::Inference {
                message: error.to_string(),
            })?;
            let mut state = state.lock().map_err(|error| OrchionError::Inference {
                message: error.to_string(),
            })?;
            engine
                .finish_streaming(&mut state)
                .map(transcript_from_upstream)
                .map_err(|source| OrchionError::Inference {
                    message: source.to_string(),
                })
        })
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
    let chunk_size_samples = (options.chunk_size_sec * ASR_SAMPLE_RATE as f32) as usize;
    if chunk_size_samples == 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "streaming chunk_size_sec must produce at least one sample".to_string(),
        });
    }
    if options.max_new_tokens_streaming == 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "streaming max_new_tokens_streaming must be greater than zero".to_string(),
        });
    }
    if options.max_new_tokens_final == 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "streaming max_new_tokens_final must be greater than zero".to_string(),
        });
    }
    Ok(())
}

fn streaming_options_into_upstream_options(
    options: AsrStreamingOptions,
) -> qwen3_asr::StreamingOptions {
    let mut upstream = qwen3_asr::StreamingOptions::default();
    upstream.language = options.language;
    upstream.chunk_size_sec = options.chunk_size_sec;
    upstream.unfixed_chunk_num = options.unfixed_chunk_num;
    upstream.unfixed_token_num = options.unfixed_token_num;
    upstream.max_new_tokens_streaming = options.max_new_tokens_streaming;
    upstream.max_new_tokens_final = options.max_new_tokens_final;
    upstream.initial_text = options.initial_text;
    upstream
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
    fn streaming_options_reject_zero_token_limits() {
        let streaming_tokens = AsrStreamingOptions {
            max_new_tokens_streaming: 0,
            ..Default::default()
        };
        let final_tokens = AsrStreamingOptions {
            max_new_tokens_final: 0,
            ..Default::default()
        };

        assert!(validate_streaming_options(&streaming_tokens).is_err());
        assert!(validate_streaming_options(&final_tokens).is_err());
    }

    #[test]
    fn streaming_options_convert_to_upstream_streaming_options() {
        let options = AsrStreamingOptions {
            language: Some("zh".to_string()),
            chunk_size_sec: 1.5,
            unfixed_chunk_num: 3,
            unfixed_token_num: 7,
            max_new_tokens_streaming: 48,
            max_new_tokens_final: 256,
            initial_text: Some("previous context".to_string()),
        };

        let upstream = streaming_options_into_upstream_options(options);

        assert_eq!(upstream.language.as_deref(), Some("zh"));
        assert_eq!(upstream.chunk_size_sec, 1.5);
        assert_eq!(upstream.unfixed_chunk_num, 3);
        assert_eq!(upstream.unfixed_token_num, 7);
        assert_eq!(upstream.max_new_tokens_streaming, 48);
        assert_eq!(upstream.max_new_tokens_final, 256);
        assert_eq!(upstream.initial_text.as_deref(), Some("previous context"));
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
            AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap(),
            "models/qwen3-asr-0.6b",
            orchion_core::DevicePreference::Cpu,
        );
        std::mem::drop(future);
    }
}
