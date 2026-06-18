use orchion_core::{ASR_SAMPLE_RATE, OrchionError, Result, TtsAudio};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputFormat {
    Wav,
    Mp3,
    Aac,
    Opus,
    Flac,
    Pcm,
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

pub async fn decode_audio_bytes(input: impl Into<Vec<u8>>) -> Result<DecodedAudio> {
    FfmpegAudioCodec::default()
        .decode_for_asr(input.into())
        .await
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

    let output = run_ffmpeg(
        binary,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        input,
        "decode audio input",
    )?;
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

fn run_ffmpeg(binary: &Path, args: &[&str], input: Vec<u8>, operation: &str) -> Result<Vec<u8>> {
    let args = args
        .iter()
        .map(|arg| (*arg).to_string())
        .collect::<Vec<_>>();
    run_ffmpeg_dynamic(binary, &args, input, operation)
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
