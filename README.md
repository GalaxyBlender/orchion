# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion is an easy-to-use async Rust library for speech AI workflows such as ASR and TTS. It includes typed Rust APIs, standalone example binaries, and an OpenAI-compatible HTTP server with Swagger documentation.

Orchion currently supports Qwen3 ASR/TTS models and is designed so additional speech model backends can be added over time. It focuses on practical local inference workflows: model names are represented as Rust enums, model downloads are handled through HuggingFace and ModelScope clients, synchronous upstream inference is wrapped behind async APIs, and the server defaults to platform GPU acceleration when the target platform supports it.

## Highlights

- Easy-to-use async Rust APIs for ASR and TTS workflows.
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/speech` APIs.
- Minimal OpenAI-style extensions for TTS voice clone and voice design.
- `config.toml` based server configuration for model selection and defaults.
- Model downloads from HuggingFace or ModelScope, with automatic fallback support.
- Platform GPU feature defaults in the server crate: Metal on macOS, CUDA on Linux and Windows.
- Swagger UI at `/docs` and OpenAPI JSON at `/api-docs/openapi.json`.

## Workspace Layout

```text
.
├── libs/
│   └── orchion/         # Core Rust library
├── apps/
│   └── server/          # Axum OpenAI-compatible ASR/TTS server
├── examples/
│   ├── asr_file/        # Standalone ASR file example
│   ├── asr_streaming/   # Standalone ASR streaming example
│   ├── download_model/  # Standalone model download example
│   └── tts_preset/      # Standalone preset TTS example
```

## Requirements

- Rust `1.85` or newer.
- A supported Qwen3 ASR/TTS backend through the upstream `qwen3-asr` and `qwen3-tts` crates.
- Optional GPU stack:
  - macOS: Metal-capable device.
  - Linux or Windows: CUDA-capable device and compatible CUDA runtime.

## Quick Start

### Run Tests

```sh
cargo test -p orchion --lib
cargo test -p orchion-server --lib --tests
```

### Run An Example

```sh
cargo run -p orchion-example-download-model -- models
cargo run -p orchion-example-asr-file -- audio.wav models
cargo run -p orchion-example-tts-preset -- "Hello from Orchion" output.wav models
```

### Run The Server

```sh
cargo run -p orchion-server -- --config config.toml
```

If `--config` is omitted, the server looks for `config.toml` beside the executable. If `models.dir` is omitted, models are stored in `models/` beside the executable.

## Core Library

The core crate lives at `libs/orchion` and exposes async Rust APIs for loading, downloading, and running ASR/TTS models.

### Cargo Features

- `default = ["asr", "tts", "download"]`
- `asr`: Qwen3 ASR transcription and streaming wrappers.
- `tts`: Qwen3 TTS preset speaker, voice clone, and voice design wrappers.
- `download`: async model downloads through `hf-hub` and `modelscope`.
- `metal`, `cuda`, `flash-attn`: backend feature opt-ins passed through to upstream crates.

### ASR Example

```rust,no_run
use orchion::{Asr, AsrModel, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let asr = Asr::load_or_download(AsrModel::Qwen3Asr06B, "models").await?;
    let transcript = asr.transcribe_file("audio.wav").await?;
    println!("{}", transcript.text);
    Ok(())
}
```

Streaming accepts mono `f32` samples plus the source sample rate. Orchion automatically resamples chunks to 16 kHz before forwarding them to Qwen3 ASR.

### TTS Example

```rust,no_run
use orchion::{Result, Tts, TtsLanguage, TtsModel, TtsSpeaker, TtsVoice};

#[tokio::main]
async fn main() -> Result<()> {
    let tts = Tts::load_or_download(TtsModel::Qwen3Tts06BCustomVoice, "models").await?;
    tts.synthesize_to_file(
        "Hello from Orchion.",
        TtsVoice::Preset {
            speaker: TtsSpeaker::Ryan,
            language: TtsLanguage::English,
        },
        "output.wav",
    )
    .await?;
    Ok(())
}
```

## OpenAI-Compatible Server

The server crate lives at `apps/server`. It uses Axum and exposes OpenAI-style audio routes.

### Routes

- `GET /healthz`: health check.
- `POST /v1/audio/transcriptions`: OpenAI-style multipart ASR request.
- `POST /v1/audio/speech`: OpenAI-style JSON TTS request.
- `GET /docs`: Swagger UI.
- `GET /api-docs/openapi.json`: OpenAPI document.

### Transcription Request

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.wav \
  -F response_format=json
```

Supported `response_format` values are `json`, `text`, and `verbose_json`. Timestamp granularities are rejected explicitly because the current ASR wrapper does not expose word or segment timestamps.

### Speech Request

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b-custom-voice",
    "input": "Hello from Orchion.",
    "voice": "ryan",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Only `wav` output is currently supported. `speed` must remain `1.0` because the upstream wrapper does not expose speed control.

### Voice Clone Extension

```json
{
  "model": "qwen3-tts-0.6b-custom-voice",
  "input": "Read this with the reference voice.",
  "voice": "clone",
  "reference_audio": "reference.wav",
  "reference_text": "Text spoken in the reference audio.",
  "language": "english",
  "response_format": "wav"
}
```

### Voice Design Extension

```json
{
  "model": "qwen3-tts-0.6b-custom-voice",
  "input": "Read this with a designed voice.",
  "voice": "design",
  "voice_prompt": "A calm narrator with a warm studio tone.",
  "language": "english",
  "response_format": "wav"
}
```

These extension fields keep the request shape close to OpenAI's speech API while exposing Qwen3 TTS capabilities that OpenAI does not define directly.

## Configuration

```toml
[server]
bind = "127.0.0.1:8080"

[models]
dir = "models"
source = "auto"
asr = "qwen3-asr-0.6b"
tts = "qwen3-tts-0.6b-custom-voice"

[defaults.tts]
voice = "ryan"
language = "english"
format = "wav"
```

### Model Sources

`ORCHION_MODEL_SOURCE` controls download routing for the core library:

- `auto` or unset: try HuggingFace first, then ModelScope.
- `huggingface`: use HuggingFace only.
- `modelscope`: use ModelScope only.

When `HF_ENDPOINT` is set, Orchion passes it to the HuggingFace client.

The server also accepts `models.source` in `config.toml` with the same values.

## Development

Useful commands:

```sh
cargo fmt --all -- --check
cargo test -p orchion --lib
cargo test -p orchion-server --lib --tests
cargo check --workspace --exclude orchion-server
cargo check -p orchion-server
```

Real model download tests are kept out of the default test path. Run ignored integration tests explicitly when network access and model storage are available.

## Project Status

Orchion is early-stage software. The public Rust API and server request extensions may change while the project is still stabilizing.

## License

No license file is included yet. Add a `LICENSE` file before distributing this project as open source.
