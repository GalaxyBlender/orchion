use orchion_core::{ASR_SAMPLE_RATE, OrchionError, Result, TtsAudio};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::str::FromStr;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

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
    PcmS16Le { sample_rate: u32, pending: Vec<u8> },
    Ffmpeg(FfmpegStreamingDecoder),
}

struct FfmpegStreamingDecoder {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout_rx: mpsc::Receiver<std::io::Result<Vec<u8>>>,
    stderr_rx: mpsc::Receiver<std::io::Result<Vec<u8>>>,
    stdout_thread: Option<std::thread::JoinHandle<()>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
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
        run_blocking(move || decode_for_asr_blocking(&binary, input)).await
    }

    pub async fn decode_file_for_asr(&self, input: impl Into<PathBuf>) -> Result<DecodedAudio> {
        let binary = self.binary.clone();
        let input = input.into();
        run_blocking(move || decode_file_for_asr_blocking(&binary, &input)).await
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
        Self::new_for_asr_with_binary(format, sample_rate, PathBuf::from("ffmpeg")).await
    }

    pub async fn new_for_asr_with_binary(
        format: AudioInputFormat,
        sample_rate: Option<u32>,
        binary: PathBuf,
    ) -> Result<Self> {
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
                inner: StreamingAudioDecoderInner::Ffmpeg(
                    FfmpegStreamingDecoder::start(binary, format).await?,
                ),
            }),
        }
    }

    pub async fn push(&mut self, bytes: &[u8]) -> Result<DecodedAudio> {
        match &mut self.inner {
            StreamingAudioDecoderInner::PcmS16Le {
                sample_rate,
                pending,
            } => {
                pending.extend_from_slice(bytes);
                let complete_len = pending.len() - (pending.len() % 2);
                let complete = pending.drain(..complete_len).collect::<Vec<_>>();
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
    async fn start(binary: PathBuf, format: AudioInputFormat) -> Result<Self> {
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
        let mut child = Command::new(&binary)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
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
        let (stdout_rx, stdout_thread) = spawn_pipe_reader(stdout);
        let (stderr_rx, stderr_thread) = spawn_pipe_reader(stderr);
        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout_rx,
            stderr_rx,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
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
        stdin
            .write_all(bytes)
            .map_err(|error| OrchionError::InvalidAudio {
                reason: format!("failed to write audio bytes to ffmpeg: {error}"),
            })?;
        self.input_bytes += bytes.len();
        stdin.flush().map_err(|error| OrchionError::InvalidAudio {
            reason: format!("failed to flush audio bytes to ffmpeg: {error}"),
        })?;
        self.drain_available_output()
    }

    async fn finish(&mut self) -> Result<DecodedAudio> {
        drop(self.stdin.take());
        let mut child = self
            .child
            .take()
            .ok_or_else(|| OrchionError::InvalidAudio {
                reason: "ffmpeg decoder has already finished".to_string(),
            })?;
        let status = tokio::task::spawn_blocking(move || child.wait())
            .await
            .map_err(|error| OrchionError::BlockingTask {
                message: error.to_string(),
            })?
            .map_err(|error| OrchionError::InvalidAudio {
                reason: format!("ffmpeg decode audio stream failed to finish: {error}"),
            })?;
        let output = self.drain_all_output().await?;
        self.join_reader_threads()?;
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

    fn drain_available_output(&mut self) -> Result<DecodedAudio> {
        let mut bytes = Vec::new();
        while let Ok(chunk) = self.stdout_rx.try_recv() {
            bytes.extend(chunk.map_err(|error| OrchionError::InvalidAudio {
                reason: format!("failed to read ffmpeg decoded audio stream: {error}"),
            })?);
        }
        self.samples_from_output(bytes)
    }

    async fn drain_all_output(&mut self) -> Result<DecodedAudio> {
        let mut bytes = Vec::new();
        while let Ok(chunk) = self.stdout_rx.recv() {
            bytes.extend(chunk.map_err(|error| OrchionError::InvalidAudio {
                reason: format!("failed to read ffmpeg decoded audio stream: {error}"),
            })?);
        }
        self.samples_from_output(bytes)
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

    fn join_reader_threads(&mut self) -> Result<()> {
        if let Some(thread) = self.stdout_thread.take() {
            thread.join().map_err(|_| OrchionError::InvalidAudio {
                reason: "ffmpeg stdout reader panicked".to_string(),
            })?;
        }
        if let Some(thread) = self.stderr_thread.take() {
            thread.join().map_err(|_| OrchionError::InvalidAudio {
                reason: "ffmpeg stderr reader panicked".to_string(),
            })?;
        }
        Ok(())
    }
}

impl Drop for FfmpegStreamingDecoder {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub async fn decode_audio_bytes(input: impl Into<Vec<u8>>) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default()
        .decode_for_asr(input.into())
        .await
}

pub async fn decode_audio_file(input: impl Into<PathBuf>) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default().decode_file_for_asr(input).await
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

fn decode_for_asr_blocking(binary: &Path, input: Vec<u8>) -> Result<DecodedAudio> {
    if input.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio input bytes are empty".to_string(),
        });
    }

    let input_path = write_temp_audio_input(&input)?;
    let decoded = decode_file_for_asr_blocking(binary, &input_path);
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

fn decode_file_for_asr_blocking(binary: &Path, input: &Path) -> Result<DecodedAudio> {
    if input.as_os_str().is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "audio input path is empty".to_string(),
        });
    }

    let args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-i".to_string(),
        input.to_string_lossy().into_owned(),
        "-vn".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        "16000".to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "pipe:1".to_string(),
    ];
    let output = run_ffmpeg_dynamic(binary, &args, Vec::new(), "decode audio input")?;
    let samples = f32_samples_from_le_bytes(&output)?;
    if samples.is_empty() {
        return Err(OrchionError::InvalidAudio {
            reason: "ffmpeg decoded audio to an empty sample buffer".to_string(),
        });
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

fn spawn_pipe_reader<R>(
    mut reader: R,
) -> (
    mpsc::Receiver<std::io::Result<Vec<u8>>>,
    std::thread::JoinHandle<()>,
)
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(size) => {
                    if sender.send(Ok(buffer[..size].to_vec())).is_err() {
                        break;
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
