use orchion_core::{ASR_SAMPLE_RATE, OrchionError, Result, TtsAudio};
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command as TokioCommand};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputFormat {
    Wav,
    Mp3,
    Aac,
    Opus,
    Flac,
    Pcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioInputFormat {
    Auto,
    PcmS16Le,
    WebmOpus,
    Mp3,
    Wav,
    M4a,
    Aac,
    Flac,
    Ogg,
}

impl AudioInputFormat {
    const fn ffmpeg_input_args(self) -> &'static [&'static str] {
        match self {
            Self::Auto => &[],
            Self::PcmS16Le => &[],
            Self::WebmOpus => &[],
            Self::Mp3 => &[],
            Self::Wav => &[],
            Self::M4a => &[],
            Self::Aac => &[],
            Self::Flac => &[],
            Self::Ogg => &[],
        }
    }
}

impl FromStr for AudioInputFormat {
    type Err = OrchionError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "pcm_s16le" | "pcm" => Ok(Self::PcmS16Le),
            "webm_opus" | "webm" => Ok(Self::WebmOpus),
            "mp3" => Ok(Self::Mp3),
            "wav" => Ok(Self::Wav),
            "m4a" => Ok(Self::M4a),
            "aac" => Ok(Self::Aac),
            "flac" => Ok(Self::Flac),
            "ogg" | "opus" => Ok(Self::Ogg),
            _ => Err(OrchionError::InvalidAudio {
                reason: format!(
                    "unsupported audio input format `{value}`; supported formats are auto, pcm_s16le, webm_opus, mp3, wav, m4a, aac, flac, ogg, and opus"
                ),
            }),
        }
    }
}

impl AudioOutputFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Opus => "opus",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        }
    }

    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Aac => "audio/aac",
            Self::Opus => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Pcm => "audio/pcm",
        }
    }

    const fn ffmpeg_output_args(self) -> &'static [&'static str] {
        match self {
            Self::Wav => &["-acodec", "pcm_s16le", "-f", "wav"],
            Self::Mp3 => &["-acodec", "libmp3lame", "-f", "mp3"],
            Self::Aac => &["-acodec", "aac", "-f", "adts"],
            Self::Opus => &["-acodec", "libopus", "-f", "ogg"],
            Self::Flac => &["-acodec", "flac", "-f", "flac"],
            Self::Pcm => &["-acodec", "pcm_s16le", "-f", "s16le"],
        }
    }
}

impl FromStr for AudioOutputFormat {
    type Err = OrchionError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "aac" => Ok(Self::Aac),
            "opus" => Ok(Self::Opus),
            "flac" => Ok(Self::Flac),
            "pcm" => Ok(Self::Pcm),
            _ => Err(OrchionError::InvalidAudio {
                reason: format!(
                    "unsupported audio output format `{value}`; supported formats are wav, mp3, aac, opus, flac, and pcm"
                ),
            }),
        }
    }
}

impl std::fmt::Display for AudioOutputFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedAudio {
    pub bytes: Vec<u8>,
    pub format: AudioOutputFormat,
    pub content_type: &'static str,
}

pub struct StreamingAudioDecoder {
    inner: StreamingAudioDecoderInner,
}

enum StreamingAudioDecoderInner {
    PcmS16Le {
        sample_rate: u32,
        pending: Vec<u8>,
        output_samples: usize,
        max_output_samples: Option<usize>,
    },
    Ffmpeg(FfmpegStreamingDecoder),
}

struct FfmpegStreamingDecoder {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout_rx: mpsc::Receiver<std::io::Result<Vec<u8>>>,
    stderr_rx: mpsc::UnboundedReceiver<std::io::Result<Vec<u8>>>,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
    pending: Vec<u8>,
    input_bytes: usize,
    output_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegAudioCodec {
    binary: PathBuf,
}

impl Default for FfmpegAudioCodec {
    fn default() -> Self {
        Self::new("ffmpeg")
    }
}

impl FfmpegAudioCodec {
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
        }
    }

    pub async fn decode_for_asr(&self, input: Vec<u8>) -> Result<DecodedAudio> {
        let binary = self.binary.clone();
        run_blocking(move || decode_for_asr_blocking(&binary, input, None)).await
    }

    pub async fn decode_for_asr_with_max_samples(
        &self,
        input: Vec<u8>,
        max_samples: usize,
    ) -> Result<DecodedAudio> {
        let binary = self.binary.clone();
        run_blocking(move || decode_for_asr_blocking(&binary, input, Some(max_samples))).await
    }

    pub async fn decode_file_for_asr(&self, input: impl Into<PathBuf>) -> Result<DecodedAudio> {
        let binary = self.binary.clone();
        let input = input.into();
        run_blocking(move || decode_file_for_asr_blocking(&binary, &input, None)).await
    }

    pub async fn decode_file_for_asr_with_max_samples(
        &self,
        input: impl Into<PathBuf>,
        max_samples: usize,
    ) -> Result<DecodedAudio> {
        let binary = self.binary.clone();
        let input = input.into();
        run_blocking(move || decode_file_for_asr_blocking(&binary, &input, Some(max_samples))).await
    }

    pub async fn encode_tts_samples(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        format: AudioOutputFormat,
    ) -> Result<EncodedAudio> {
        let binary = self.binary.clone();
        run_blocking(move || encode_tts_blocking(&binary, samples, sample_rate, format)).await
    }
}

impl StreamingAudioDecoder {
    pub async fn new_for_asr(format: AudioInputFormat, sample_rate: Option<u32>) -> Result<Self> {
        Self::new_for_asr_inner(format, sample_rate, PathBuf::from("ffmpeg"), None)
    }

    pub async fn new_for_asr_with_max_samples(
        format: AudioInputFormat,
        sample_rate: Option<u32>,
        max_output_samples: usize,
    ) -> Result<Self> {
        Self::new_for_asr_inner(
            format,
            sample_rate,
            PathBuf::from("ffmpeg"),
            Some(max_output_samples),
        )
    }

    pub async fn new_for_asr_with_binary(
        format: AudioInputFormat,
        sample_rate: Option<u32>,
        binary: PathBuf,
    ) -> Result<Self> {
        Self::new_for_asr_inner(format, sample_rate, binary, None)
    }

    pub async fn new_for_asr_with_binary_and_max_samples(
        format: AudioInputFormat,
        sample_rate: Option<u32>,
        binary: PathBuf,
        max_output_samples: usize,
    ) -> Result<Self> {
        Self::new_for_asr_inner(format, sample_rate, binary, Some(max_output_samples))
    }

    fn new_for_asr_inner(
        format: AudioInputFormat,
        sample_rate: Option<u32>,
        binary: PathBuf,
        max_output_samples: Option<usize>,
    ) -> Result<Self> {
        if max_output_samples == Some(0) {
            return Err(OrchionError::InvalidAudio {
                reason: "streaming decoded audio sample limit must be greater than zero"
                    .to_string(),
            });
        }
        match format {
            AudioInputFormat::PcmS16Le => {
                let sample_rate = sample_rate.ok_or_else(|| OrchionError::InvalidAudio {
                    reason: "sample_rate is required for pcm_s16le audio input".to_string(),
                })?;
                if sample_rate == 0 {
                    return Err(OrchionError::InvalidAudio {
                        reason: "sample_rate must be greater than zero".to_string(),
                    });
                }
                Ok(Self {
                    inner: StreamingAudioDecoderInner::PcmS16Le {
                        sample_rate,
                        pending: Vec::new(),
                        output_samples: 0,
                        max_output_samples,
                    },
                })
            }
            AudioInputFormat::Auto
            | AudioInputFormat::WebmOpus
            | AudioInputFormat::Mp3
            | AudioInputFormat::Wav
            | AudioInputFormat::M4a
            | AudioInputFormat::Aac
            | AudioInputFormat::Flac
            | AudioInputFormat::Ogg => Ok(Self {
                inner: StreamingAudioDecoderInner::Ffmpeg(FfmpegStreamingDecoder::start(
                    binary,
                    format,
                    max_output_samples,
                )?),
            }),
        }
    }

    pub async fn push(&mut self, bytes: &[u8]) -> Result<DecodedAudio> {
        match &mut self.inner {
            StreamingAudioDecoderInner::PcmS16Le {
                sample_rate,
                pending,
                output_samples,
                max_output_samples,
            } => {
                let complete_samples = pending
                    .len()
                    .checked_add(bytes.len())
                    .and_then(|bytes| bytes.checked_div(2))
                    .ok_or_else(streaming_sample_limit_error)?;
                let total_samples = output_samples
                    .checked_add(complete_samples)
                    .ok_or_else(streaming_sample_limit_error)?;
                if max_output_samples.is_some_and(|limit| total_samples > limit) {
                    return Err(streaming_sample_limit_error());
                }
                pending.extend_from_slice(bytes);
                let complete_len = pending.len() - (pending.len() % 2);
                let complete = pending.drain(..complete_len).collect::<Vec<_>>();
                *output_samples = total_samples;
                Ok(DecodedAudio {
                    samples: decode_pcm_s16le_complete_bytes(&complete),
                    sample_rate: *sample_rate,
                })
            }
            StreamingAudioDecoderInner::Ffmpeg(decoder) => decoder.push(bytes).await,
        }
    }

    pub async fn finish(mut self) -> Result<DecodedAudio> {
        match &mut self.inner {
            StreamingAudioDecoderInner::PcmS16Le {
                sample_rate,
                pending,
                output_samples: _,
                max_output_samples: _,
            } => {
                if !pending.is_empty() {
                    return Err(OrchionError::InvalidAudio {
                        reason: "pcm_s16le audio input ended with a partial sample".to_string(),
                    });
                }
                Ok(DecodedAudio {
                    samples: Vec::new(),
                    sample_rate: *sample_rate,
                })
            }
            StreamingAudioDecoderInner::Ffmpeg(decoder) => decoder.finish().await,
        }
    }
}

impl FfmpegStreamingDecoder {
    fn start(
        binary: PathBuf,
        format: AudioInputFormat,
        max_output_samples: Option<usize>,
    ) -> Result<Self> {
        let mut args = vec![
            "-hide_banner".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
        ];
        args.extend(
            format
                .ffmpeg_input_args()
                .iter()
                .map(|arg| (*arg).to_string()),
        );
        args.extend([
            "-i".to_string(),
            "pipe:0".to_string(),
            "-vn".to_string(),
            "-ac".to_string(),
            "1".to_string(),
            "-ar".to_string(),
            ASR_SAMPLE_RATE.to_string(),
            "-f".to_string(),
            "f32le".to_string(),
            "pipe:1".to_string(),
        ]);
        let mut command = TokioCommand::new(&binary);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .map_err(|error| ffmpeg_start_error(&binary, error))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "failed to open ffmpeg stdin".to_string(),
            })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "failed to open ffmpeg stdout".to_string(),
            })?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "failed to open ffmpeg stderr".to_string(),
            })?;
        let max_output_bytes = max_output_samples
            .map(|samples| {
                samples
                    .checked_mul(4)
                    .ok_or_else(streaming_sample_limit_error)
            })
            .transpose()?;
        let (stdout_rx, stdout_task) = spawn_bounded_pipe_reader(stdout, max_output_bytes);
        let (stderr_rx, stderr_task) = spawn_limited_pipe_reader(stderr, 64 * 1024);
        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout_rx,
            stderr_rx,
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
            pending: Vec::new(),
            input_bytes: 0,
            output_samples: 0,
        })
    }

    async fn push(&mut self, bytes: &[u8]) -> Result<DecodedAudio> {
        if bytes.is_empty() {
            return Ok(DecodedAudio {
                samples: Vec::new(),
                sample_rate: ASR_SAMPLE_RATE,
            });
        }
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "cannot push audio after decoder input is closed".to_string(),
            })?;
        let write_input = async {
            stdin.write_all(bytes).await?;
            stdin.flush().await
        };
        tokio::pin!(write_input);
        let mut output = Vec::new();
        let mut output_open = true;
        loop {
            if !output_open {
                write_input.await.map_err(ffmpeg_stream_input_error)?;
                break;
            }
            tokio::select! {
                result = &mut write_input => {
                    result.map_err(ffmpeg_stream_input_error)?;
                    break;
                }
                chunk = self.stdout_rx.recv() => {
                    match chunk {
                        Some(chunk) => append_ffmpeg_output(&mut output, chunk)?,
                        None => output_open = false,
                    }
                }
            }
        }
        while let Ok(chunk) = self.stdout_rx.try_recv() {
            append_ffmpeg_output(&mut output, chunk)?;
        }
        self.input_bytes = self
            .input_bytes
            .checked_add(bytes.len())
            .ok_or_else(streaming_sample_limit_error)?;
        self.samples_from_output(output)
    }

    async fn finish(&mut self) -> Result<DecodedAudio> {
        drop(self.stdin.take());
        let mut child = self
            .child
            .take()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "ffmpeg decoder has already finished".to_string(),
            })?;
        let wait_for_child = async {
            child
                .wait()
                .await
                .map_err(|error| OrchionError::InvalidAudio {
                    reason: format!("ffmpeg decode audio stream failed to finish: {error}"),
                })
        };
        let drain_output = async {
            let mut bytes = Vec::new();
            while let Some(chunk) = self.stdout_rx.recv().await {
                append_ffmpeg_output(&mut bytes, chunk)?;
            }
            Ok::<_, OrchionError>(bytes)
        };
        let (status, output) = tokio::try_join!(wait_for_child, drain_output)?;
        let output = self.samples_from_output(output)?;
        self.join_reader_tasks().await?;
        let stderr = self.drain_stderr()?;
        if !status.success() {
            return Err(OrchionError::InvalidAudio {
                reason: ffmpeg_status_error("decode audio stream", status, &stderr),
            });
        }
        if self.input_bytes > 0 && self.output_samples == 0 {
            return Err(OrchionError::InvalidAudio {
                reason: "ffmpeg decoded audio stream to an empty sample buffer".to_string(),
            });
        }
        Ok(output)
    }

    fn samples_from_output(&mut self, bytes: Vec<u8>) -> Result<DecodedAudio> {
        self.pending.extend(bytes);
        let complete_len = self.pending.len() - (self.pending.len() % 4);
        let complete = self.pending.drain(..complete_len).collect::<Vec<_>>();
        let samples = f32_samples_from_le_bytes(&complete)?;
        self.output_samples += samples.len();
        Ok(DecodedAudio {
            samples,
            sample_rate: ASR_SAMPLE_RATE,
        })
    }

    fn drain_stderr(&mut self) -> Result<Vec<u8>> {
        let mut stderr = Vec::new();
        while let Ok(chunk) = self.stderr_rx.try_recv() {
            stderr.extend(chunk.map_err(|error| OrchionError::InvalidAudio {
                reason: format!("failed to read ffmpeg stderr: {error}"),
            })?);
        }
        Ok(stderr)
    }

    async fn join_reader_tasks(&mut self) -> Result<()> {
        if let Some(task) = self.stdout_task.take() {
            task.await.map_err(|_| OrchionError::InvalidAudio {
                reason: "ffmpeg stdout reader task failed".to_string(),
            })?;
        }
        if let Some(task) = self.stderr_task.take() {
            task.await.map_err(|_| OrchionError::InvalidAudio {
                reason: "ffmpeg stderr reader task failed".to_string(),
            })?;
        }
        Ok(())
    }
}

impl Drop for FfmpegStreamingDecoder {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(child) = &mut self.child {
            let _ = child.start_kill();
        }
        if let Some(task) = self.stdout_task.take() {
            task.abort();
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }
}

pub async fn decode_audio_bytes(input: impl Into<Vec<u8>>) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default()
        .decode_for_asr(input.into())
        .await
}

pub async fn decode_audio_bytes_with_max_samples(
    input: impl Into<Vec<u8>>,
    max_samples: usize,
) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default()
        .decode_for_asr_with_max_samples(input.into(), max_samples)
        .await
}

pub async fn decode_audio_file(input: impl Into<PathBuf>) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default().decode_file_for_asr(input).await
}

pub async fn decode_audio_file_with_max_samples(
    input: impl Into<PathBuf>,
    max_samples: usize,
) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default()
        .decode_file_for_asr_with_max_samples(input, max_samples)
        .await
}

pub fn decode_pcm_s16le_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "pcm_s16le byte length must be divisible by 2".to_string(),
        });
    }
    Ok(decode_pcm_s16le_complete_bytes(bytes))
}

pub async fn encode_tts_audio(audio: &TtsAudio, format: AudioOutputFormat) -> Result<EncodedAudio> {
    FfmpegAudioCodec::default()
        .encode_tts_samples(audio.samples.clone(), audio.sample_rate, format)
        .await
}

async fn run_blocking<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| OrchionError::BlockingTask {
            message: error.to_string(),
        })?
}

fn decode_for_asr_blocking(
    binary: &Path,
    input: Vec<u8>,
    max_samples: Option<usize>,
) -> Result<DecodedAudio> {
    if input.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio input bytes are empty".to_string(),
        });
    }

    let input_path = write_temp_audio_input(&input)?;
    let decoded = decode_file_for_asr_blocking(binary, &input_path, max_samples);
    let remove_result = fs::remove_file(&input_path);
    match decoded {
        Ok(decoded) => {
            remove_result.map_err(|error| OrchionError::InvalidAudio {
                reason: format!(
                    "failed to remove temporary audio input `{}`: {error}",
                    input_path.display()
                ),
            })?;
            Ok(decoded)
        }
        Err(error) => {
            let _ = remove_result;
            Err(error)
        }
    }
}

fn decode_file_for_asr_blocking(
    binary: &Path,
    input: &Path,
    max_samples: Option<usize>,
) -> Result<DecodedAudio> {
    if input.as_os_str().is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio input path is empty".to_string(),
        });
    }

    if max_samples == Some(0) {
        return Err(OrchionError::InvalidAudio {
            reason: "audio sample limit must be greater than zero".to_string(),
        });
    }
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-i".to_string(),
        input.to_string_lossy().into_owned(),
    ];
    if let Some(max_samples) = max_samples {
        let probe_samples = max_samples.saturating_add(1);
        let duration_seconds = probe_samples as f64 / f64::from(ASR_SAMPLE_RATE);
        args.extend(["-t".to_string(), format!("{duration_seconds:.9}")]);
    }
    args.extend([
        "-vn".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        "16000".to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "pipe:1".to_string(),
    ]);
    let output = run_ffmpeg_dynamic(binary, &args, Vec::new(), "decode audio input")?;
    let samples = f32_samples_from_le_bytes(&output)?;
    if samples.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "ffmpeg decoded audio to an empty sample buffer".to_string(),
        });
    }
    if let Some(max_samples) = max_samples {
        if samples.len() > max_samples {
            return Err(OrchionError::InvalidAudio {
                reason: format!(
                    "decoded audio exceeds the configured sample limit of {max_samples}"
                ),
            });
        }
    }
    Ok(DecodedAudio {
        samples,
        sample_rate: ASR_SAMPLE_RATE,
    })
}

fn write_temp_audio_input(input: &[u8]) -> Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| OrchionError::InvalidAudio {
            reason: format!("failed to create temporary audio filename: {error}"),
        })?
        .as_nanos();
    path.push(format!(
        "orchion-audio-{}-{unique}.input",
        std::process::id()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| OrchionError::InvalidAudio {
            reason: format!(
                "failed to create temporary audio input `{}`: {error}",
                path.display()
            ),
        })?;
    file.write_all(input)
        .map_err(|error| OrchionError::InvalidAudio {
            reason: format!(
                "failed to write temporary audio input `{}`: {error}",
                path.display()
            ),
        })?;
    Ok(path)
}

fn encode_tts_blocking(
    binary: &Path,
    samples: Vec<f32>,
    sample_rate: u32,
    format: AudioOutputFormat,
) -> Result<EncodedAudio> {
    if sample_rate == 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "sample_rate must be greater than zero".to_string(),
        });
    }
    if samples.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio samples are empty".to_string(),
        });
    }

    if format == AudioOutputFormat::Wav {
        return encode_tts_wav(samples, sample_rate);
    }

    let input = s16_samples_to_le_bytes(&samples);
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "s16le".to_string(),
        "-ar".to_string(),
        sample_rate.to_string(),
        "-ac".to_string(),
        "1".to_string(),
        "-i".to_string(),
        "pipe:0".to_string(),
    ];
    args.extend(
        format
            .ffmpeg_output_args()
            .iter()
            .map(|arg| (*arg).to_string()),
    );
    args.push("pipe:1".to_string());

    let bytes = run_ffmpeg_dynamic(binary, &args, input, "encode TTS audio")?;
    if bytes.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: format!("ffmpeg encoded {format} audio to an empty byte stream"),
        });
    }
    Ok(EncodedAudio {
        bytes,
        format,
        content_type: format.content_type(),
    })
}

fn encode_tts_wav(samples: Vec<f32>, sample_rate: u32) -> Result<EncodedAudio> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).map_err(|error| {
            OrchionError::InvalidAudio {
                reason: format!("failed to start WAV encoder: {error}"),
            }
        })?;
        for sample in samples {
            writer
                .write_sample(sample_to_i16(sample))
                .map_err(|error| OrchionError::InvalidAudio {
                    reason: format!("failed to write WAV sample: {error}"),
                })?;
        }
        writer
            .finalize()
            .map_err(|error| OrchionError::InvalidAudio {
                reason: format!("failed to finalize WAV audio: {error}"),
            })?;
    }
    Ok(EncodedAudio {
        bytes: cursor.into_inner(),
        format: AudioOutputFormat::Wav,
        content_type: AudioOutputFormat::Wav.content_type(),
    })
}

fn run_ffmpeg_dynamic(
    binary: &Path,
    args: &[String],
    input: Vec<u8>,
    operation: &str,
) -> Result<Vec<u8>> {
    let mut child = Command::new(binary)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| ffmpeg_start_error(binary, error))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| OrchionError::InvalidAudio {
            reason: "failed to open ffmpeg stdin".to_string(),
        })?;
    let writer = std::thread::spawn(move || {
        stdin.write_all(&input)?;
        Ok::<(), std::io::Error>(())
    });

    let output = child
        .wait_with_output()
        .map_err(|error| OrchionError::InvalidAudio {
            reason: format!("ffmpeg {operation} failed to finish: {error}"),
        })?;
    let write_result = writer.join().map_err(|_| OrchionError::InvalidAudio {
        reason: format!("ffmpeg {operation} stdin writer panicked"),
    })?;

    if !output.status.success() {
        return Err(OrchionError::InvalidAudio {
            reason: ffmpeg_status_error(operation, output.status, &output.stderr),
        });
    }
    write_result.map_err(|error| OrchionError::InvalidAudio {
        reason: format!(
            "failed to write audio bytes to ffmpeg while trying to {operation}: {error}"
        ),
    })?;
    Ok(output.stdout)
}

fn ffmpeg_start_error(binary: &Path, error: std::io::Error) -> OrchionError {
    if error.kind() == std::io::ErrorKind::NotFound {
        OrchionError::InvalidAudio {
            reason: format!(
                "ffmpeg not found at `{}`; install ffmpeg and ensure it is available on PATH",
                binary.display()
            ),
        }
    } else {
        OrchionError::InvalidAudio {
            reason: format!("failed to start ffmpeg `{}`: {error}", binary.display()),
        }
    }
}

fn ffmpeg_status_error(operation: &str, status: std::process::ExitStatus, stderr: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    if stderr.is_empty() {
        format!("ffmpeg {operation} failed with status {status}")
    } else {
        format!("ffmpeg {operation} failed with status {status}: {stderr}")
    }
}

fn f32_samples_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(OrchionError::InvalidAudio {
            reason: "ffmpeg returned an invalid f32le byte stream".to_string(),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_pcm_s16le_complete_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / f32::from(i16::MAX))
        .collect()
}

fn spawn_bounded_pipe_reader<R>(
    mut reader: R,
    max_bytes: Option<usize>,
) -> (mpsc::Receiver<std::io::Result<Vec<u8>>>, JoinHandle<()>)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let (sender, receiver) = mpsc::channel(8);
    let handle = tokio::spawn(async move {
        let mut buffer = [0_u8; 8192];
        let mut total_bytes = 0_usize;
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(size) => {
                    total_bytes = match total_bytes.checked_add(size) {
                        Some(total_bytes) => total_bytes,
                        None => {
                            let _ = sender
                                .send(Err(std::io::Error::new(
                                    std::io::ErrorKind::FileTooLarge,
                                    "streaming decoded audio exceeded the sample limit",
                                )))
                                .await;
                            break;
                        }
                    };
                    if max_bytes.is_some_and(|limit| total_bytes > limit) {
                        let _ = sender
                            .send(Err(std::io::Error::new(
                                std::io::ErrorKind::FileTooLarge,
                                "streaming decoded audio exceeded the sample limit",
                            )))
                            .await;
                        break;
                    }
                    if sender.send(Ok(buffer[..size].to_vec())).await.is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                    break;
                }
            }
        }
    });
    (receiver, handle)
}

fn spawn_limited_pipe_reader<R>(
    mut reader: R,
    max_retained_bytes: usize,
) -> (
    mpsc::UnboundedReceiver<std::io::Result<Vec<u8>>>,
    JoinHandle<()>,
)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let (sender, receiver) = mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        let mut buffer = [0_u8; 8192];
        let mut retained = 0_usize;
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(size) => {
                    let keep = size.min(max_retained_bytes.saturating_sub(retained));
                    if keep > 0 {
                        retained += keep;
                        if sender.send(Ok(buffer[..keep].to_vec())).is_err() {
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                    break;
                }
            }
        }
    });
    (receiver, handle)
}

fn append_ffmpeg_output(output: &mut Vec<u8>, chunk: std::io::Result<Vec<u8>>) -> Result<()> {
    let chunk = chunk.map_err(|error| {
        if error.kind() == std::io::ErrorKind::FileTooLarge {
            streaming_sample_limit_error()
        } else {
            OrchionError::InvalidAudio {
                reason: format!("failed to read ffmpeg decoded audio stream: {error}"),
            }
        }
    })?;
    output.extend(chunk);
    Ok(())
}

fn ffmpeg_stream_input_error(error: std::io::Error) -> OrchionError {
    OrchionError::InvalidAudio {
        reason: format!("failed to write audio bytes to ffmpeg: {error}"),
    }
}

fn streaming_sample_limit_error() -> OrchionError {
    OrchionError::InvalidAudio {
        reason: "streaming decoded audio exceeded the sample limit".to_string(),
    }
}

fn s16_samples_to_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample_to_i16(*sample).to_le_bytes());
    }
    bytes
}

fn sample_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16
}
