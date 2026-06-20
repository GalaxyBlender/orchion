use orchion_core::{AsrModel, AsrOptions, AsrTranscript, DevicePreference, Result};
use std::path::Path;

#[cfg(feature = "vad-webrtc")]
use orchion_core::{ASR_SAMPLE_RATE, AsrSegment};

#[cfg(feature = "audio-ffmpeg")]
use std::path::PathBuf;

#[cfg(feature = "audio-ffmpeg")]
use std::time::Instant;

#[derive(Clone)]
pub struct Asr {
    inner: orchion_qwen3::Asr,
}

#[cfg(all(test, feature = "vad-webrtc"))]
mod tests {
    use super::*;
    use orchion_audio_vad::AudioSegment;
    use orchion_core::AsrSegment;

    #[test]
    fn transcript_from_segment_results_adds_timestamps_and_joins_text() {
        let transcript = transcript_from_segment_results(
            vec![
                (
                    AudioSegment {
                        start_sample: 16_000,
                        end_sample: 32_000,
                    },
                    segment_transcript(" hello ", "en", "raw one"),
                ),
                (
                    AudioSegment {
                        start_sample: 48_000,
                        end_sample: 64_000,
                    },
                    segment_transcript("world", "en", "raw two"),
                ),
            ],
            16_000,
            Some("fallback".to_string()),
        );

        assert_eq!(transcript.text, "hello world");
        assert_eq!(transcript.language, "en");
        assert_eq!(transcript.raw_output, "raw one\nraw two");
        assert_eq!(
            transcript.segments,
            vec![
                AsrSegment {
                    id: 0,
                    start: 1.0,
                    end: 2.0,
                    text: "hello".to_string(),
                },
                AsrSegment {
                    id: 1,
                    start: 3.0,
                    end: 4.0,
                    text: "world".to_string(),
                },
            ]
        );
    }

    #[test]
    fn transcript_from_segment_results_skips_empty_text_segments() {
        let transcript = transcript_from_segment_results(
            vec![
                (
                    AudioSegment {
                        start_sample: 0,
                        end_sample: 16_000,
                    },
                    segment_transcript("", "", ""),
                ),
                (
                    AudioSegment {
                        start_sample: 16_000,
                        end_sample: 32_000,
                    },
                    segment_transcript("kept", "", "raw"),
                ),
            ],
            16_000,
            Some("zh".to_string()),
        );

        assert_eq!(transcript.text, "kept");
        assert_eq!(transcript.language, "zh");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].id, 0);
        assert!((transcript.segments[0].start - 1.0).abs() < f32::EPSILON);
    }

    fn segment_transcript(text: &str, language: &str, raw_output: &str) -> AsrTranscript {
        AsrTranscript {
            text: text.to_string(),
            language: language.to_string(),
            raw_output: raw_output.to_string(),
            segments: Vec::new(),
        }
    }
}

impl Asr {
    pub async fn load(model: AsrModel, model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_device(model, model_dir, DevicePreference::Auto).await
    }

    pub async fn load_with_device(
        model: AsrModel,
        model_dir: impl AsRef<Path>,
        device: DevicePreference,
    ) -> Result<Self> {
        Ok(Self {
            inner: orchion_qwen3::Asr::load_with_device(model, model_dir, device).await?,
        })
    }

    #[cfg(feature = "download-all")]
    pub async fn load_or_download(model: AsrModel, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = orchion_download::ModelDownloader::default()
            .download(model, cache_dir)
            .await?;
        Self::load(model, model_dir).await
    }

    pub const fn model(&self) -> AsrModel {
        self.inner.model()
    }

    pub async fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<AsrTranscript> {
        self.inner.transcribe_file(path).await
    }

    pub async fn transcribe_file_with(
        &self,
        path: impl AsRef<Path>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        self.inner.transcribe_file_with(path, options).await
    }

    pub async fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AsrTranscript> {
        self.inner.transcribe_samples(samples, sample_rate).await
    }

    pub async fn transcribe_samples_with(
        &self,
        samples: &[f32],
        sample_rate: u32,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        self.inner
            .transcribe_samples_with(samples, sample_rate, options)
            .await
    }

    #[cfg(feature = "vad-webrtc")]
    pub async fn transcribe_samples_with_segments(
        &self,
        samples: &[f32],
        sample_rate: u32,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let prepared = orchion_core::prepare_asr_samples(samples, sample_rate)?;
        let segmenter = orchion_audio_vad::WebRtcVadSegmenter::default();
        let audio_segments = segmenter.segment(&prepared, ASR_SAMPLE_RATE)?;
        if audio_segments.is_empty() {
            return Ok(empty_transcript(options.language));
        }

        let mut segment_results = Vec::with_capacity(audio_segments.len());
        for segment in audio_segments {
            let segment_transcript = self
                .inner
                .transcribe_samples_with(
                    &prepared[segment.start_sample..segment.end_sample],
                    ASR_SAMPLE_RATE,
                    options.clone(),
                )
                .await?;
            segment_results.push((segment, segment_transcript));
        }

        Ok(transcript_from_segment_results(
            segment_results,
            ASR_SAMPLE_RATE,
            options.language,
        ))
    }

    #[cfg(feature = "audio-ffmpeg")]
    pub async fn transcribe_audio_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<AsrTranscript> {
        self.transcribe_audio_bytes_with(bytes, AsrOptions::default())
            .await
    }

    #[cfg(feature = "audio-ffmpeg")]
    pub async fn transcribe_audio_bytes_with(
        &self,
        bytes: impl Into<Vec<u8>>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = orchion_audio::decode_audio_bytes(bytes.into()).await?;
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_started.elapsed().as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.inner
            .transcribe_samples_with(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|_| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    "ASR inference completed"
                );
            })
    }

    #[cfg(feature = "audio-ffmpeg")]
    pub async fn transcribe_audio_file_with(
        &self,
        path: impl Into<PathBuf>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = orchion_audio::decode_audio_file(path.into()).await?;
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_started.elapsed().as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.inner
            .transcribe_samples_with(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|_| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    "ASR inference completed"
                );
            })
    }

    #[cfg(all(feature = "audio-ffmpeg", feature = "vad-webrtc"))]
    pub async fn transcribe_audio_bytes_with_segments(
        &self,
        bytes: impl Into<Vec<u8>>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = orchion_audio::decode_audio_bytes(bytes.into()).await?;
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_started.elapsed().as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.transcribe_samples_with_segments(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|transcript| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    segments = transcript.segments.len(),
                    "ASR segmented inference completed"
                );
            })
    }

    #[cfg(all(feature = "audio-ffmpeg", feature = "vad-webrtc"))]
    pub async fn transcribe_audio_file_with_segments(
        &self,
        path: impl Into<PathBuf>,
        options: AsrOptions,
    ) -> Result<AsrTranscript> {
        let decode_started = Instant::now();
        let decoded = orchion_audio::decode_audio_file(path.into()).await?;
        tracing::debug!(
            samples = decoded.samples.len(),
            sample_rate = decoded.sample_rate,
            elapsed_ms = decode_started.elapsed().as_millis(),
            "ASR audio decode completed"
        );
        let inference_started = Instant::now();
        self.transcribe_samples_with_segments(&decoded.samples, decoded.sample_rate, options)
            .await
            .inspect(|transcript| {
                tracing::debug!(
                    elapsed_ms = inference_started.elapsed().as_millis(),
                    segments = transcript.segments.len(),
                    "ASR segmented inference completed"
                );
            })
    }

    pub async fn start_streaming(&self) -> Result<orchion_qwen3::AsrStream> {
        self.inner.start_streaming().await
    }

    pub async fn start_streaming_with(
        &self,
        options: orchion_core::AsrStreamingOptions,
    ) -> Result<orchion_qwen3::AsrStream> {
        self.inner.start_streaming_with(options).await
    }
}

#[cfg(feature = "vad-webrtc")]
#[allow(clippy::cast_precision_loss)]
fn transcript_from_segment_results(
    segment_results: Vec<(orchion_audio_vad::AudioSegment, AsrTranscript)>,
    sample_rate: u32,
    fallback_language: Option<String>,
) -> AsrTranscript {
    let mut text_parts = Vec::new();
    let mut raw_parts = Vec::new();
    let mut language = String::new();
    let mut segments = Vec::new();

    for (audio_segment, transcript) in segment_results {
        let AsrTranscript {
            text,
            language: transcript_language,
            raw_output,
            segments: _,
        } = transcript;
        let segment_text = text.trim().to_string();
        if segment_text.is_empty() {
            continue;
        }
        if language.is_empty() && !transcript_language.is_empty() {
            language = transcript_language;
        }
        let raw_output = raw_output.trim();
        if !raw_output.is_empty() {
            raw_parts.push(raw_output.to_string());
        }
        let id = segments.len();
        segments.push(AsrSegment {
            id,
            start: audio_segment.start_sample as f32 / sample_rate as f32,
            end: audio_segment.end_sample as f32 / sample_rate as f32,
            text: segment_text.clone(),
        });
        text_parts.push(segment_text);
    }

    if language.is_empty() {
        language = fallback_language.unwrap_or_default();
    }

    AsrTranscript {
        text: text_parts.join(" "),
        language,
        raw_output: raw_parts.join("\n"),
        segments,
    }
}

#[cfg(feature = "vad-webrtc")]
fn empty_transcript(language: Option<String>) -> AsrTranscript {
    AsrTranscript {
        text: String::new(),
        language: language.unwrap_or_default(),
        raw_output: String::new(),
        segments: Vec::new(),
    }
}
