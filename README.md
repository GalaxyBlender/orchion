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
- Swagger UI at `/docs` and OpenAPI JSON at `/openapi/v1.json`.

## Requirements

- Rust `1.85` or newer.
- `ffmpeg` available on `PATH` for in-memory ASR upload decoding and TTS response encoding.
- A supported Qwen3 ASR/TTS backend through the upstream `qwen3-asr` and `qwen3-tts` crates.
- Optional GPU stack:
  - macOS: Metal-capable device.
  - Linux or Windows: CUDA-capable device and compatible CUDA runtime.

## Quick Start

### Run Tests

```sh
cargo test --workspace --features full,cpu
```

### Run An Example

```sh
cargo run -p orchion --features download-all --example download_model -- models
cargo run -p orchion --features asr-qwen3,download-all,cpu --example asr_file -- audio.wav models
cargo run -p orchion --features tts-qwen3,download-all,cpu --example tts_preset -- "Hello from Orchion" output.wav models
```

## Core Library

The public facade crate lives at `libs/orchion` and exposes async Rust APIs for loading, downloading, and running ASR/TTS models. Domain types live in `libs/orchion-core`, FFmpeg-backed audio conversion lives in `libs/orchion-audio`, model downloads live in `libs/orchion-download`, and Qwen3 runtime adapters live in `libs/orchion-qwen3`.

### Cargo Features

- `default = []`
- `full`: Qwen3 ASR/TTS, FFmpeg audio conversion, and all download providers.
- `asr-qwen3`, `tts-qwen3`: Qwen3 ASR/TTS runtime adapters.
- `audio-ffmpeg`: in-memory audio decode/encode through system `ffmpeg`.
- `download-all`: async model downloads through `hf-hub` and `modelscope`.
- `cpu`, `metal`, `cuda`: backend feature opt-ins passed through to upstream crates.

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

The server crate lives at `apps/orchion-server`. It uses Axum and exposes OpenAI-style audio routes.

### Run The Server

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml
```

The repository includes `apps/orchion-server/config.toml` as a development config. If `--config` is omitted, the server looks for `config.toml` beside the executable. If `models.dir` is omitted, models are stored in `models/` beside the executable.

Logging is controlled by `RUST_LOG`. The server loads `.env` from the executable directory first, then from the current working directory. The repository includes a development `.env`, so `cargo run -p orchion-server -- --config apps/orchion-server/config.toml` emits startup, model loading, download, and request debug logs by default.

### Routes

- `GET /healthz`: health check.
- `GET /v1/models`: OpenAI-style list of configured available models.
- `POST /v1/audio/transcriptions`: OpenAI-style multipart ASR request.
- `POST /v1/audio/speech`: OpenAI-style JSON TTS request.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

### Transcription Request

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.mp3 \
  -F response_format=json
```

Uploaded audio is decoded in memory through system `ffmpeg`; common formats such as `wav`, `mp3`, `m4a`, `flac`, `ogg`, and `webm` work when supported by the installed ffmpeg build. Supported `response_format` values are `json`, `text`, and `verbose_json`. Timestamp granularities are rejected explicitly because the current ASR wrapper does not expose word or segment timestamps.

### Speech Request

Speech synthesis uses `POST /v1/audio/speech`. The `voice` field selects one of three request types: preset voice, voice clone, or voice design.

#### Preset Voice

Preset voice synthesis uses a JSON request and passes a built-in speaker name such as `ryan` as `voice`.

Fields:

- `model`: a TTS model from `models.tts.available`, such as `qwen3-tts-0.6b-custom-voice`.
- `input`: text to synthesize.
- `voice`: built-in speaker name, such as `ryan`.
- `language`: optional synthesis language, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.
- `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_length`: optional Qwen3 TTS sampling controls.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b-custom-voice",
    "input": "Hello from Orchion.",
    "voice": "ryan",
    "seed": 42,
    "response_format": "wav"
  }' \
  --output speech.wav
```

#### Voice Clone

Voice clone uses the same `POST /v1/audio/speech` endpoint, but the request body must be `multipart/form-data` and upload the reference audio directly as a file field. JSON requests do not support voice clone.

Fields:

- `model`: a voice-clone-capable model from `models.tts.available`, such as `qwen3-tts-0.6b-custom-voice`.
- `input`: text to synthesize.
- `voice`: must be `clone`.
- `reference_audio`: reference audio file field, such as `-F reference_audio=@reference.wav`.
- `reference_text`: text spoken in the reference audio.
- `language`: optional language for the reference audio and synthesized text, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -F model=qwen3-tts-0.6b-custom-voice \
  -F input='Read this with the reference voice.' \
  -F voice=clone \
  -F reference_audio=@reference.wav \
  -F reference_text='Text spoken in the reference audio.' \
  -F language=english \
  -F response_format=wav \
  --output cloned.wav
```

#### Voice Design

Voice design uses a JSON request, passes `design` as `voice`, and describes the generated voice through `voice_prompt`.

Fields:

- `model`: a voice-design-capable model from `models.tts.available`, such as `qwen3-tts-1.7b-voice-design`.
- `input`: text to synthesize.
- `voice`: must be `design`.
- `voice_prompt`: text description of the voice.
- `language`: optional synthesis language, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-1.7b-voice-design",
    "input": "Read this with a designed voice.",
    "voice": "design",
    "voice_prompt": "A calm narrator with a warm studio tone.",
    "language": "english",
    "response_format": "wav"
  }' \
  --output designed.wav
```

Supported speech output formats are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`. If `response_format` is omitted, the server uses `[defaults.tts] format` from `config.toml`, which defaults to `wav`. `speed` must remain `1.0` because the upstream wrapper does not expose speed control.

Qwen3 TTS requests also accept sampling controls: `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `max_length`. If `seed` is omitted, Orchion uses `42` to make TTS output reproducible by default. Other sampling fields keep the upstream `qwen3-tts` defaults unless provided. `max_length` is the maximum number of generated codec frames and can be lowered to cap long generations when EOS is delayed.

### Model List Request

```sh
curl http://127.0.0.1:9090/v1/models
```

The response follows the OpenAI model list shape: `object` is `list`, and `data` contains model objects with `id`, `object`, `created`, and `owned_by`. The list is built from `models.asr.available` and `models.tts.available` in `config.toml`.

If `[auth] api_key` is configured, pass it as `Authorization: Bearer <api_key>` for every `/v1/*` request.

## Configuration

```toml
[server]
bind = "127.0.0.1:9090"
max_upload_size = "30M"

[models]
dir = "models"
source = "auto"

[models.asr]
default = "qwen3-asr-0.6b"
available = ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]
idle_timeout = "10m"
max_loaded = 1

[models.tts]
default = "qwen3-tts-0.6b-custom-voice"
available = [
  "qwen3-tts-0.6b-base",
  "qwen3-tts-0.6b-custom-voice",
  "qwen3-tts-1.7b-base",
  "qwen3-tts-1.7b-custom-voice",
  "qwen3-tts-1.7b-voice-design",
]
idle_timeout = "10m"
max_loaded = 1

[auth]
# api_key = "change-me"

[defaults.tts]
format = "wav"
```

`models.asr.available` and `models.tts.available` define the server allowlists. Startup downloads those model files into `models.dir`, but models are loaded into memory lazily when a request asks for them. Requests for models outside the allowlist are rejected immediately. `idle_timeout` unloads inactive models, and `max_loaded` evicts the least recently used loaded model when the cache is full.

`[auth] api_key` is optional. When it is set to a non-empty value, every `/v1/*` route requires `Authorization: Bearer <api_key>`; `/healthz` and `/docs` remain public.

### Model Sources

`ORCHION_MODEL_SOURCE` controls download routing for the core library:

- `auto` or unset: try HuggingFace first, then ModelScope.
- `huggingface`: use HuggingFace only.
- `modelscope`: use ModelScope only.

When `HF_ENDPOINT` is set, Orchion passes it to the HuggingFace client.

The server also accepts `models.source` in `config.toml` with the same values.

`server.max_upload_size` limits request body size for uploads. It defaults to `30M` and accepts bytes or `K`, `M`, `G` suffixes.

### Logging

```dotenv
RUST_LOG=orchion_server=debug,orchion=info,tower_http=debug
```

Set `RUST_LOG` in the environment to override `.env`. If neither is set, the server uses `orchion_server=info,orchion=info,tower_http=debug`.

## Development

Useful commands:

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
```

Real model download tests are kept out of the default test path. Run ignored integration tests explicitly when network access and model storage are available.

## Project Status

Orchion is early-stage software. The public Rust API and server request extensions may change while the project is still stabilizing.
