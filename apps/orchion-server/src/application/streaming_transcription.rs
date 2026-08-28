use super::model_cache::ModelLease;
use super::resource_policy::InferenceGuard;
use super::{RuntimeError, UseCaseError};
use orchion::{
    ASR_SAMPLE_RATE, Asr, AsrModel, AsrStreamingOptions, AsrTranscript, AudioVadStreamingEvent,
    OrchionError,
};
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};
use tokio::time::timeout;

pub const MAX_CHUNK_SIZE_SEC: f32 = 30.0;
pub const MAX_UNFIXED_CHUNKS: usize = 16;
pub const MAX_UNFIXED_TOKENS: usize = 128;
pub const MAX_NEW_TOKENS: usize = 1_024;
pub const MAX_PROMPT_CHARS: usize = 4_096;

pub type StreamingModelFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<LeasedAsrModel>, RuntimeError>> + Send + 'a>>;

pub trait StreamingTranscriptionRuntime: Send + Sync {
    fn lease_streaming_model(&self, model: AsrModel) -> StreamingModelFuture<'_>;
}

#[derive(Debug, Clone, Copy)]
pub struct TranscriptionStreamLimits {
    pub idle_timeout: Duration,
    pub max_duration: Duration,
    pub max_input_bytes: usize,
}

pub struct TranscriptionStreamBudget {
    started_at: Instant,
    input_bytes: usize,
    decoded_duration: Duration,
}

impl TranscriptionStreamBudget {
    #[must_use]
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            input_bytes: 0,
            decoded_duration: Duration::ZERO,
        }
    }

    /// # Errors
    ///
    /// Returns a use-case error when the session deadline has expired.
    pub fn next_wait(&self, limits: TranscriptionStreamLimits) -> Result<Duration, UseCaseError> {
        Ok(limits.idle_timeout.min(self.remaining(limits)?))
    }

    /// # Errors
    ///
    /// Returns a use-case error when the session deadline has expired.
    pub fn remaining(&self, limits: TranscriptionStreamLimits) -> Result<Duration, UseCaseError> {
        limits
            .max_duration
            .checked_sub(self.started_at.elapsed())
            .ok_or_else(stream_duration_error)
    }

    /// # Errors
    ///
    /// Returns a use-case error when cumulative input exceeds the configured byte limit.
    pub fn record_binary_input(
        &mut self,
        bytes: usize,
        limits: TranscriptionStreamLimits,
    ) -> Result<(), UseCaseError> {
        self.input_bytes = self
            .input_bytes
            .checked_add(bytes)
            .ok_or_else(|| stream_input_too_large_error(limits.max_input_bytes))?;
        if self.input_bytes > limits.max_input_bytes {
            return Err(stream_input_too_large_error(limits.max_input_bytes));
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns a use-case error when cumulative decoded audio exceeds the session limit.
    pub fn record_decoded_samples(
        &mut self,
        samples: usize,
        sample_rate: u32,
        limits: TranscriptionStreamLimits,
    ) -> Result<(), UseCaseError> {
        let duration = decoded_audio_duration(samples, sample_rate)?;
        self.decoded_duration = self
            .decoded_duration
            .checked_add(duration)
            .ok_or_else(|| stream_audio_too_long_error(limits.max_duration))?;
        if self.decoded_duration > limits.max_duration {
            return Err(stream_audio_too_long_error(limits.max_duration));
        }
        Ok(())
    }

    #[must_use]
    pub fn timeout_error(&self, limits: TranscriptionStreamLimits) -> UseCaseError {
        if self.started_at.elapsed() >= limits.max_duration {
            stream_duration_error()
        } else {
            UseCaseError::invalid(
                "transcription stream was idle for too long",
                None,
                "stream_idle_timeout",
            )
        }
    }
}

impl Default for TranscriptionStreamBudget {
    fn default() -> Self {
        Self::new()
    }
}

/// # Errors
///
/// Returns a use-case error when the operation exceeds the session deadline.
pub async fn await_stream_operation<T>(
    budget: &TranscriptionStreamBudget,
    limits: TranscriptionStreamLimits,
    operation: impl Future<Output = T>,
) -> Result<T, UseCaseError> {
    timeout(budget.remaining(limits)?, operation)
        .await
        .map_err(|_| stream_duration_error())
}

/// # Errors
///
/// Returns a use-case error when finishing exceeds the session deadline.
pub async fn await_stream_finish<T>(
    remaining: Result<Duration, UseCaseError>,
    operation: impl Future<Output = T>,
) -> Result<T, UseCaseError> {
    timeout(remaining?, operation)
        .await
        .map_err(|_| stream_duration_error())
}

#[must_use]
pub fn stream_decoder_error(
    error: OrchionError,
    limits: TranscriptionStreamLimits,
) -> UseCaseError {
    if matches!(
        &error,
        OrchionError::InvalidAudio { reason }
            if reason == "streaming decoded audio exceeded the sample limit"
    ) {
        stream_audio_too_long_error(limits.max_duration)
    } else {
        UseCaseError::Core(error)
    }
}

fn decoded_audio_duration(samples: usize, sample_rate: u32) -> Result<Duration, UseCaseError> {
    let sample_rate = usize::try_from(sample_rate)
        .map_err(|_| UseCaseError::Internal("decoded audio sample rate is too large".into()))?;
    if sample_rate == 0 {
        return Err(UseCaseError::Internal(
            "decoded audio sample rate is zero".into(),
        ));
    }
    let seconds = u64::try_from(samples / sample_rate)
        .map_err(|_| UseCaseError::Internal("decoded audio duration is too large".into()))?;
    let nanos = (samples % sample_rate)
        .checked_mul(1_000_000_000)
        .and_then(|remainder| remainder.checked_div(sample_rate))
        .and_then(|nanos| u32::try_from(nanos).ok())
        .ok_or_else(|| UseCaseError::Internal("decoded audio duration is too large".into()))?;
    Ok(Duration::new(seconds, nanos))
}

fn stream_duration_error() -> UseCaseError {
    UseCaseError::invalid(
        "transcription stream exceeded the maximum session duration",
        None,
        "stream_duration_exceeded",
    )
}

fn stream_input_too_large_error(max_input_bytes: usize) -> UseCaseError {
    UseCaseError::invalid(
        format!("transcription stream exceeded the {max_input_bytes} byte input limit"),
        None,
        "stream_input_too_large",
    )
}

fn stream_audio_too_long_error(max_duration: Duration) -> UseCaseError {
    UseCaseError::invalid(
        format!(
            "transcription stream exceeded the {} second audio limit",
            max_duration.as_secs()
        ),
        None,
        "stream_audio_too_long",
    )
}

#[derive(Clone)]
pub struct LeasedAsrModel {
    model: ModelLease<Asr>,
    inference_guard: InferenceGuard,
}

impl LeasedAsrModel {
    #[must_use]
    pub(crate) fn new(model: ModelLease<Asr>, inference_guard: InferenceGuard) -> Self {
        Self {
            model,
            inference_guard,
        }
    }

    /// # Errors
    ///
    /// Returns a use-case error when the model cannot start a streaming operation.
    pub async fn start(
        &self,
        options: AsrStreamingOptions,
    ) -> Result<LeasedAsrStream, UseCaseError> {
        let operation_guard = self.inference_guard.clone();
        let stream = self
            .model
            .run(move |model| async move {
                let _guard = operation_guard;
                model.start_streaming_with(options).await
            })
            .await
            .map_err(|error| {
                UseCaseError::Internal(format!("ASR stream task failed: {error:#}"))
            })??;
        Ok(LeasedAsrStream {
            model: self.clone(),
            stream: Some(stream),
        })
    }
}

pub struct LeasedAsrStream {
    model: LeasedAsrModel,
    stream: Option<orchion::AsrStream>,
}

impl LeasedAsrStream {
    /// # Errors
    ///
    /// Returns a use-case error when streaming inference fails.
    pub async fn feed(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<AsrTranscript>, UseCaseError> {
        let mut stream = self.take_stream()?;
        let samples = samples.to_vec();
        let operation_guard = self.model.inference_guard.clone();
        let (stream, result) = self
            .model
            .model
            .run(move |model| async move {
                let _guard = operation_guard;
                let result = stream.feed(&samples, sample_rate).await;
                drop(model);
                (stream, result)
            })
            .await
            .map_err(|error| {
                UseCaseError::Internal(format!("ASR stream task failed: {error:#}"))
            })?;
        self.stream = Some(stream);
        result.map_err(UseCaseError::Core)
    }

    /// # Errors
    ///
    /// Returns a use-case error when final streaming inference fails.
    pub async fn finish(mut self) -> Result<AsrTranscript, UseCaseError> {
        let stream = self.take_stream()?;
        let operation_guard = self.model.inference_guard.clone();
        self.model
            .model
            .run(move |model| async move {
                let _guard = operation_guard;
                let result = stream.finish().await;
                drop(model);
                result
            })
            .await
            .map_err(|error| UseCaseError::Internal(format!("ASR stream task failed: {error:#}")))?
            .map_err(UseCaseError::Core)
    }

    fn take_stream(&mut self) -> Result<orchion::AsrStream, UseCaseError> {
        self.stream.take().ok_or_else(|| {
            UseCaseError::Core(OrchionError::InvalidAudio {
                reason: "streaming session has already finished".to_string(),
            })
        })
    }
}

pub struct AsrPcmBuffer {
    chunk_size_sec: f32,
    sample_rate: Option<u32>,
    samples: Vec<f32>,
}

impl AsrPcmBuffer {
    #[must_use]
    pub fn new(chunk_size_sec: f32) -> Self {
        Self {
            chunk_size_sec,
            sample_rate: None,
            samples: Vec::new(),
        }
    }

    /// # Errors
    ///
    /// Returns a use-case error for invalid chunk sizing or changing sample rates.
    pub fn push(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<(Vec<f32>, u32)>, UseCaseError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if sample_rate == 0 || !self.chunk_size_sec.is_finite() || self.chunk_size_sec <= 0.0 {
            return Err(UseCaseError::invalid(
                "invalid streaming audio chunk size",
                Some("chunk_size_sec"),
                "invalid_chunk_size",
            ));
        }
        if let Some(current_sample_rate) = self.sample_rate {
            if current_sample_rate != sample_rate {
                return Err(UseCaseError::invalid(
                    "streaming decoded audio sample rate changed",
                    Some("sample_rate"),
                    "invalid_sample_rate",
                ));
            }
        } else {
            self.sample_rate = Some(sample_rate);
        }

        let chunk_samples = checked_sample_count(sample_rate, self.chunk_size_sec)
            .filter(|samples| *samples > 0)
            .ok_or_else(|| {
                UseCaseError::invalid(
                    "invalid streaming audio chunk size",
                    Some("chunk_size_sec"),
                    "invalid_chunk_size",
                )
            })?;

        self.samples.extend_from_slice(samples);
        let mut chunks = Vec::new();
        while self.samples.len() >= chunk_samples {
            chunks.push((self.samples.drain(..chunk_samples).collect(), sample_rate));
        }
        Ok(chunks)
    }

    pub fn drain_remaining(&mut self) -> Option<(Vec<f32>, u32)> {
        if self.samples.is_empty() {
            return None;
        }
        self.sample_rate
            .map(|sample_rate| (std::mem::take(&mut self.samples), sample_rate))
    }
}

fn checked_sample_count(sample_rate: u32, duration_seconds: f32) -> Option<usize> {
    const U64_EXCLUSIVE_UPPER_BOUND: f64 = 18_446_744_073_709_551_616.0;

    let samples = f64::from(sample_rate) * f64::from(duration_seconds);
    if !samples.is_finite() || !(0.0..U64_EXCLUSIVE_UPPER_BOUND).contains(&samples) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let samples = samples as u64;
    usize::try_from(samples).ok()
}

/// # Errors
///
/// Returns a use-case error when a streaming generation option is invalid or unbounded.
pub fn validate_streaming_options(options: &AsrStreamingOptions) -> Result<(), UseCaseError> {
    if !options.chunk_size_sec.is_finite() || options.chunk_size_sec <= 0.0 {
        return Err(UseCaseError::invalid(
            "streaming chunk_size_sec must be finite and greater than zero",
            Some("chunk_size_sec"),
            "invalid_chunk_size",
        ));
    }
    if options.chunk_size_sec > MAX_CHUNK_SIZE_SEC {
        return Err(UseCaseError::invalid(
            format!("streaming chunk_size_sec must not exceed {MAX_CHUNK_SIZE_SEC}"),
            Some("chunk_size_sec"),
            "invalid_chunk_size",
        ));
    }
    if f64::from(options.chunk_size_sec) * f64::from(ASR_SAMPLE_RATE) < 1.0 {
        return Err(UseCaseError::invalid(
            "streaming chunk_size_sec must produce at least one sample",
            Some("chunk_size_sec"),
            "invalid_chunk_size",
        ));
    }
    validate_bounded_nonzero(
        options.max_new_tokens_streaming,
        MAX_NEW_TOKENS,
        "max_new_tokens_streaming",
    )?;
    validate_bounded_nonzero(
        options.max_new_tokens_final,
        MAX_NEW_TOKENS,
        "max_new_tokens_final",
    )?;
    validate_bounded_nonzero(
        options.unfixed_chunk_num,
        MAX_UNFIXED_CHUNKS,
        "unfixed_chunk_num",
    )?;
    validate_bounded_nonzero(
        options.unfixed_token_num,
        MAX_UNFIXED_TOKENS,
        "unfixed_token_num",
    )?;
    if options
        .initial_text
        .as_ref()
        .is_some_and(|prompt| prompt.chars().count() > MAX_PROMPT_CHARS)
    {
        return Err(UseCaseError::invalid(
            format!("streaming prompt must not exceed {MAX_PROMPT_CHARS} characters"),
            Some("prompt"),
            "stream_option_limit_exceeded",
        ));
    }
    Ok(())
}

fn validate_bounded_nonzero(
    value: usize,
    max: usize,
    param: &'static str,
) -> Result<(), UseCaseError> {
    if value == 0 {
        return Err(UseCaseError::invalid(
            format!("streaming {param} must be greater than zero"),
            Some(param),
            "invalid_stream_option",
        ));
    }
    if value > max {
        return Err(UseCaseError::invalid(
            format!("streaming {param} must not exceed {max}"),
            Some(param),
            "stream_option_limit_exceeded",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingTranscriptionEvent {
    Partial {
        segment_id: u64,
        text: String,
    },
    SegmentFinal {
        segment_id: u64,
        text: String,
        start_ms: u64,
        end_ms: u64,
    },
    Completed,
}

pub struct CaptionSession {
    model: LeasedAsrModel,
    streaming_options: AsrStreamingOptions,
    target_segment_millis: u32,
    current_segment: Option<CaptionSegment>,
    next_segment_id: u64,
}

impl CaptionSession {
    #[must_use]
    pub fn new(
        model: LeasedAsrModel,
        streaming_options: AsrStreamingOptions,
        target_segment_millis: u32,
    ) -> Self {
        Self {
            model,
            streaming_options,
            target_segment_millis,
            current_segment: None,
            next_segment_id: 0,
        }
    }

    /// # Errors
    ///
    /// Returns a use-case error when a VAD transition or model operation fails.
    pub async fn apply_vad_events(
        &mut self,
        events: Vec<AudioVadStreamingEvent>,
        sample_rate: u32,
    ) -> Result<Vec<StreamingTranscriptionEvent>, UseCaseError> {
        let mut output = Vec::new();
        for event in events {
            match event {
                AudioVadStreamingEvent::SegmentStarted {
                    start_sample,
                    samples,
                } => {
                    if let Some(segment) = self.current_segment.take() {
                        output.extend(Self::finalize_segment(segment, start_sample).await?);
                    }
                    let segment_id = self.allocate_segment_id()?;
                    let stream = self.model.start(self.streaming_options.clone()).await?;
                    self.current_segment = Some(CaptionSegment::new(
                        segment_id,
                        start_sample,
                        sample_rate,
                        stream,
                        self.streaming_options.chunk_size_sec,
                        self.target_segment_millis,
                    ));
                    output.extend(self.feed_current_segment(&samples, sample_rate).await?);
                }
                AudioVadStreamingEvent::Audio { samples } => {
                    output.extend(self.feed_current_segment(&samples, sample_rate).await?);
                }
                AudioVadStreamingEvent::SegmentFinal { end_sample, .. } => {
                    if let Some(segment) = self.current_segment.take() {
                        output.extend(Self::finalize_segment(segment, end_sample).await?);
                    }
                }
            }
        }
        Ok(output)
    }

    /// # Errors
    ///
    /// Returns a use-case error when final model inference fails.
    pub async fn complete(&mut self) -> Result<Vec<StreamingTranscriptionEvent>, UseCaseError> {
        let mut output = Vec::new();
        if let Some(segment) = self.current_segment.take() {
            let end_sample = segment.last_sample;
            output.extend(Self::finalize_segment(segment, end_sample).await?);
        }
        output.push(StreamingTranscriptionEvent::Completed);
        Ok(output)
    }

    async fn feed_current_segment(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<StreamingTranscriptionEvent>, UseCaseError> {
        let Some(segment) = self.current_segment.as_mut() else {
            return Ok(Vec::new());
        };
        if sample_rate != segment.sample_rate {
            return Err(UseCaseError::invalid(
                "caption segment audio sample rate changed",
                Some("sample_rate"),
                "invalid_sample_rate",
            ));
        }
        segment.last_sample = segment
            .last_sample
            .checked_add(samples.len())
            .ok_or_else(|| {
                UseCaseError::Internal("caption segment sample index overflowed".into())
            })?;
        let chunks = segment.pcm_buffer.push(samples, sample_rate)?;
        let mut output = Vec::new();
        for (chunk_samples, chunk_sample_rate) in chunks {
            if let Some(transcript) = segment
                .stream
                .feed(&chunk_samples, chunk_sample_rate)
                .await?
            {
                let update = segment
                    .text_splitter
                    .observe_partial(&transcript.text, segment.duration_ms());
                let segment_final = update.segment_final.map(ToOwned::to_owned);
                let partial = update.partial.to_string();
                if let Some(text) = segment_final {
                    output.push(segment.final_event(text, segment.last_sample));
                    segment.segment_id = Self::allocate_segment_id_from(&mut self.next_segment_id)?;
                    segment.subtitle_start_sample = segment.last_sample;
                }
                if !partial.trim().is_empty() {
                    output.push(StreamingTranscriptionEvent::Partial {
                        segment_id: segment.segment_id,
                        text: partial,
                    });
                }
            }
        }
        Ok(output)
    }

    async fn finalize_segment(
        mut segment: CaptionSegment,
        end_sample: usize,
    ) -> Result<Vec<StreamingTranscriptionEvent>, UseCaseError> {
        if let Some((samples, sample_rate)) = segment.pcm_buffer.drain_remaining() {
            segment.stream.feed(&samples, sample_rate).await?;
        }
        let segment_id = segment.segment_id;
        let start_ms = sample_index_to_ms(segment.subtitle_start_sample, segment.sample_rate);
        let end_ms = sample_index_to_ms(end_sample, segment.sample_rate);
        let transcript = segment.stream.finish().await?;
        Ok(segment
            .text_splitter
            .flush(&transcript.text)
            .map(|text| {
                vec![StreamingTranscriptionEvent::SegmentFinal {
                    segment_id,
                    text: text.to_string(),
                    start_ms,
                    end_ms,
                }]
            })
            .unwrap_or_default())
    }

    fn allocate_segment_id(&mut self) -> Result<u64, UseCaseError> {
        Self::allocate_segment_id_from(&mut self.next_segment_id)
    }

    fn allocate_segment_id_from(next_segment_id: &mut u64) -> Result<u64, UseCaseError> {
        let segment_id = *next_segment_id;
        *next_segment_id = next_segment_id
            .checked_add(1)
            .ok_or_else(|| UseCaseError::Internal("caption segment id overflowed".into()))?;
        Ok(segment_id)
    }
}

struct CaptionSegment {
    segment_id: u64,
    start_sample: usize,
    subtitle_start_sample: usize,
    last_sample: usize,
    sample_rate: u32,
    stream: LeasedAsrStream,
    pcm_buffer: AsrPcmBuffer,
    text_splitter: CaptionTextSplitter,
}

impl CaptionSegment {
    fn new(
        segment_id: u64,
        start_sample: usize,
        sample_rate: u32,
        stream: LeasedAsrStream,
        chunk_size_sec: f32,
        target_segment_millis: u32,
    ) -> Self {
        Self {
            segment_id,
            start_sample,
            subtitle_start_sample: start_sample,
            last_sample: start_sample,
            sample_rate,
            stream,
            pcm_buffer: AsrPcmBuffer::new(chunk_size_sec),
            text_splitter: CaptionTextSplitter::new(target_segment_millis),
        }
    }

    fn duration_ms(&self) -> u64 {
        sample_index_to_ms(
            self.last_sample.saturating_sub(self.start_sample),
            self.sample_rate,
        )
    }

    fn final_event(&self, text: String, end_sample: usize) -> StreamingTranscriptionEvent {
        StreamingTranscriptionEvent::SegmentFinal {
            segment_id: self.segment_id,
            text,
            start_ms: sample_index_to_ms(self.subtitle_start_sample, self.sample_rate),
            end_ms: sample_index_to_ms(end_sample, self.sample_rate),
        }
    }
}

fn sample_index_to_ms(sample_index: usize, sample_rate: u32) -> u64 {
    assert!(sample_rate > 0, "sample_rate must be greater than zero");
    (sample_index as u64 * 1_000) / u64::from(sample_rate)
}

#[derive(Debug, PartialEq, Eq)]
struct CaptionTextUpdate<'a> {
    segment_final: Option<&'a str>,
    partial: &'a str,
}

struct CaptionTextSplitter {
    target_segment_ms: u32,
    committed_prefix: String,
    candidate_text: Option<String>,
    stable_count: u8,
}

struct CaptionBoundaryCandidate<'a> {
    text: &'a str,
    has_following_text: bool,
}

impl CaptionTextSplitter {
    fn new(target_segment_ms: u32) -> Self {
        Self {
            target_segment_ms,
            committed_prefix: String::new(),
            candidate_text: None,
            stable_count: 0,
        }
    }

    fn observe_partial<'a>(&mut self, text: &'a str, duration_ms: u64) -> CaptionTextUpdate<'a> {
        let uncommitted_text = self.uncommitted_text(text);
        let Some(candidate) = strong_punctuation_candidate_text(uncommitted_text) else {
            self.candidate_text = None;
            self.stable_count = 0;
            return CaptionTextUpdate {
                segment_final: None,
                partial: uncommitted_text,
            };
        };

        if self.candidate_text.as_deref() == Some(candidate.text) {
            self.stable_count = self.stable_count.saturating_add(1);
        } else {
            self.candidate_text = Some(candidate.text.to_string());
            self.stable_count = 1;
        }
        let stable = self.stable_count >= 2;
        let reached_target = duration_ms >= u64::from(self.target_segment_ms);
        if !stable || (!candidate.has_following_text && !reached_target) {
            return CaptionTextUpdate {
                segment_final: None,
                partial: uncommitted_text,
            };
        }

        let final_text = candidate.text;
        let partial = &uncommitted_text[final_text.len()..];
        self.committed_prefix.push_str(final_text);
        self.candidate_text = None;
        self.stable_count = 0;
        CaptionTextUpdate {
            segment_final: Some(final_text),
            partial,
        }
    }

    fn flush<'a>(&mut self, text: &'a str) -> Option<&'a str> {
        let uncommitted_text = self.uncommitted_text(text).trim();
        self.committed_prefix.clear();
        self.candidate_text = None;
        self.stable_count = 0;
        (!uncommitted_text.is_empty()).then_some(uncommitted_text)
    }

    fn uncommitted_text<'a>(&self, text: &'a str) -> &'a str {
        text.trim()
            .strip_prefix(&self.committed_prefix)
            .unwrap_or_else(|| text.trim())
    }
}

fn strong_punctuation_candidate_text(text: &str) -> Option<CaptionBoundaryCandidate<'_>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut terminal_candidate_end = None;
    let mut can_extend_candidate = false;
    for (index, character) in trimmed.char_indices() {
        let character_end = index + character.len_utf8();
        if matches!(character, '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';') {
            terminal_candidate_end = Some(character_end);
            can_extend_candidate = true;
            continue;
        }
        if matches!(
            character,
            '"' | '\'' | '”' | '’' | ')' | ']' | '}' | '）' | '】' | '》' | '」' | '』'
        ) {
            if can_extend_candidate {
                terminal_candidate_end = Some(character_end);
            }
            continue;
        }
        if let Some(candidate_end) = terminal_candidate_end {
            return Some(CaptionBoundaryCandidate {
                text: &trimmed[..candidate_end],
                has_following_text: true,
            });
        }
        can_extend_candidate = false;
    }
    terminal_candidate_end.map(|end| CaptionBoundaryCandidate {
        text: &trimmed[..end],
        has_following_text: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::model_cache::ModelCache;
    use crate::application::resource_policy::ResourcePolicy;
    use orchion::{AsrEngine, AsrEngineFuture, AsrOptions, AsrStreamSession};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn limits(max_duration: Duration, max_input_bytes: usize) -> TranscriptionStreamLimits {
        TranscriptionStreamLimits {
            idle_timeout: Duration::from_secs(30),
            max_duration,
            max_input_bytes,
        }
    }

    fn error_code(error: &UseCaseError) -> Option<&'static str> {
        match error {
            UseCaseError::InvalidRequest { code, .. } => Some(*code),
            _ => None,
        }
    }

    #[test]
    fn budget_enforces_cumulative_wire_and_audio_limits() {
        let mut budget = TranscriptionStreamBudget::new();
        let limits = limits(Duration::from_secs(1), 4);
        budget.record_binary_input(3, limits).unwrap();
        assert_eq!(
            error_code(&budget.record_binary_input(2, limits).unwrap_err()),
            Some("stream_input_too_large")
        );

        let mut budget = TranscriptionStreamBudget::new();
        assert_eq!(
            error_code(
                &budget
                    .record_decoded_samples(
                        usize::try_from(ASR_SAMPLE_RATE).unwrap() + 1,
                        ASR_SAMPLE_RATE,
                        limits
                    )
                    .unwrap_err()
            ),
            Some("stream_audio_too_long")
        );
    }

    #[tokio::test]
    async fn operation_deadline_is_a_domain_error() {
        let budget = TranscriptionStreamBudget::new();
        let error = await_stream_operation(
            &budget,
            limits(Duration::from_millis(5), 1_024),
            tokio::time::sleep(Duration::from_millis(25)),
        )
        .await
        .unwrap_err();
        assert_eq!(error_code(&error), Some("stream_duration_exceeded"));

        let error = await_stream_finish(
            Ok(Duration::from_millis(5)),
            tokio::time::sleep(Duration::from_millis(25)),
        )
        .await
        .unwrap_err();
        assert_eq!(error_code(&error), Some("stream_duration_exceeded"));
    }

    #[test]
    fn validates_generation_bounds_without_wire_types() {
        let mut options = AsrStreamingOptions {
            max_new_tokens_streaming: MAX_NEW_TOKENS + 1,
            ..AsrStreamingOptions::default()
        };
        assert_eq!(
            error_code(&validate_streaming_options(&options).unwrap_err()),
            Some("stream_option_limit_exceeded")
        );
        options.max_new_tokens_streaming = 32;
        options.initial_text = Some("x".repeat(MAX_PROMPT_CHARS + 1));
        assert_eq!(
            error_code(&validate_streaming_options(&options).unwrap_err()),
            Some("stream_option_limit_exceeded")
        );
    }

    #[test]
    fn pcm_buffer_keeps_only_the_tail_for_finish() {
        let mut buffer = AsrPcmBuffer::new(2.0);
        let chunks = buffer.push(&vec![0.0; 80_000], 16_000).unwrap();
        let tail = buffer.drain_remaining().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(tail.0.len(), 16_000);
    }

    #[test]
    fn caption_boundary_commits_stable_punctuation_and_keeps_suffix() {
        let mut splitter = CaptionTextSplitter::new(12_000);
        assert_eq!(
            splitter.observe_partial("中俄北京条约。在这个", 13_000),
            CaptionTextUpdate {
                segment_final: None,
                partial: "中俄北京条约。在这个"
            }
        );
        assert_eq!(
            splitter.observe_partial("中俄北京条约。在这个条约里", 13_500),
            CaptionTextUpdate {
                segment_final: Some("中俄北京条约。"),
                partial: "在这个条约里"
            }
        );
    }

    #[test]
    fn sample_times_use_integer_sample_positions() {
        assert_eq!(sample_index_to_ms(16_000, 16_000), 1_000);
        assert_eq!(sample_index_to_ms(47_999, 16_000), 2_999);
    }

    #[tokio::test]
    async fn caption_session_emits_domain_events_for_stable_boundaries() {
        let model = AsrModel::parse("Acme/Test-ASR").unwrap();
        let cache = ModelCache::new(
            "caption-test",
            vec![model.clone()],
            Duration::from_mins(1),
            1,
            PathBuf::from("test-models"),
        );
        let engine_model = model.clone();
        let lease = cache
            .get_or_load(model, move |_, _| async move {
                Ok(Asr::from_engine(Arc::new(TestAsrEngine {
                    model: engine_model,
                })))
            })
            .await
            .unwrap()
            .unwrap();
        let guard = ResourcePolicy::new(1, 1, 1).acquire_inference().await;
        let options = AsrStreamingOptions {
            chunk_size_sec: 1.0,
            ..AsrStreamingOptions::default()
        };
        let mut session = CaptionSession::new(LeasedAsrModel::new(lease, guard), options, 1_000);

        let first = session
            .apply_vad_events(
                vec![AudioVadStreamingEvent::SegmentStarted {
                    start_sample: 0,
                    samples: vec![0.0],
                }],
                1,
            )
            .await
            .unwrap();
        assert_eq!(
            first,
            vec![StreamingTranscriptionEvent::Partial {
                segment_id: 0,
                text: "hello.".to_string(),
            }]
        );

        let second = session
            .apply_vad_events(
                vec![AudioVadStreamingEvent::Audio { samples: vec![0.0] }],
                1,
            )
            .await
            .unwrap();
        assert_eq!(
            second,
            vec![StreamingTranscriptionEvent::SegmentFinal {
                segment_id: 0,
                text: "hello.".to_string(),
                start_ms: 0,
                end_ms: 2_000,
            }]
        );

        let completed = session.complete().await.unwrap();
        assert_eq!(completed, vec![StreamingTranscriptionEvent::Completed]);
    }

    struct TestAsrEngine {
        model: AsrModel,
    }

    impl AsrEngine for TestAsrEngine {
        fn model(&self) -> AsrModel {
            self.model.clone()
        }

        fn transcribe_file_with(
            &self,
            _path: PathBuf,
            _options: AsrOptions,
        ) -> AsrEngineFuture<'_, AsrTranscript> {
            Box::pin(async { Ok(test_transcript()) })
        }

        fn transcribe_samples_with(
            &self,
            _samples: Vec<f32>,
            _sample_rate: u32,
            _options: AsrOptions,
        ) -> AsrEngineFuture<'_, AsrTranscript> {
            Box::pin(async { Ok(test_transcript()) })
        }

        fn start_streaming_with(
            &self,
            _options: AsrStreamingOptions,
        ) -> AsrEngineFuture<'_, Box<dyn AsrStreamSession>> {
            Box::pin(async { Ok(Box::new(TestAsrStream) as Box<dyn AsrStreamSession>) })
        }
    }

    struct TestAsrStream;

    impl AsrStreamSession for TestAsrStream {
        fn feed(
            &mut self,
            _samples: Vec<f32>,
            _sample_rate: u32,
        ) -> AsrEngineFuture<'_, Option<AsrTranscript>> {
            Box::pin(async { Ok(Some(test_transcript())) })
        }

        fn finish(self: Box<Self>) -> AsrEngineFuture<'static, AsrTranscript> {
            Box::pin(async { Ok(test_transcript()) })
        }
    }

    fn test_transcript() -> AsrTranscript {
        AsrTranscript {
            text: "hello.".to_string(),
            language: "en".to_string(),
            raw_output: String::new(),
            segments: Vec::new(),
        }
    }
}
