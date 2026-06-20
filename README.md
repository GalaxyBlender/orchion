# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion provides a unified Rust API library and an out-of-the-box OpenAI-compatible API server for local speech AI workflows. It supports ASR and TTS through typed Rust APIs, standalone examples, and HTTP endpoints that are easy to integrate with existing OpenAI-style clients.

Orchion currently focuses on Qwen3 ASR/TTS models and is structured to support more speech backends over time.

## Highlights

- Unified async Rust APIs for ASR and TTS workflows.
- Ready-to-run OpenAI-compatible API server.
- `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints.
- TTS support for preset voices, voice cloning, and voice design.
- Model downloads through `model-hub` from HuggingFace or ModelScope.
- CPU by default, with optional Metal and CUDA builds.
- Swagger UI at `/docs` and OpenAPI JSON at `/openapi/v1.json`.

## Requirements

- Rust `1.85` or newer.
- `ffmpeg` available on `PATH` for audio decode/encode.
- Enough local disk space for downloaded models.
- Optional GPU runtime for Metal or CUDA acceleration.

## OpenAI-Compatible Server

The server crate lives at `apps/orchion-server` and exposes OpenAI-style audio routes.

### Run The Server

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml
```

`orchion-server` defaults to CPU. Use `--features metal` on macOS, or `--features cuda` on Linux/Windows with a supported CUDA stack. The repository includes `apps/orchion-server/config.toml` as a development config.

### WebUI

Open the React WebUI at `/ui` on the server for ASR/TTS operations, parameter previews, model inspection, and local settings. Debug builds serve `web/dist`; if it is missing, run `bun install` and `bun run build` from `web/`. For frontend iteration, run `bun run dev` from `web/`. Release builds run Bun from `apps/orchion-server/build.rs`, build the SPA, and embed assets in the server binary via `OUT_DIR/ui-dist`. API key and form preferences are stored in browser `localStorage`. Warning: API keys are stored in the browser profile via `localStorage`; do not use or save them on shared or untrusted browsers.

### Routes

- `GET /healthz`: health check.
- `GET /v1/models`: OpenAI-style list of configured models.
- `GET /ui`: React WebUI for ASR/TTS operations, parameter previews, model inspection, and local settings.
- `POST /v1/audio/transcriptions`: OpenAI-style multipart ASR request.
- `POST /v1/audio/speech`: OpenAI-style TTS request.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

### Transcription Request

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment"
```

Uploaded audio is decoded through system `ffmpeg`; common formats such as `wav`, `mp3`, `m4a`, `flac`, `ogg`, and `webm` work when supported by the installed ffmpeg build. Supported `response_format` values are `json`, `text`, `verbose_json`, and `srt`. `timestamp_granularities[]=segment` enables segment timestamps in `verbose_json`; `response_format=srt` returns subtitle cues as `text/plain`. Word-level timestamps are not supported.

### Speech Request

Speech synthesis uses `POST /v1/audio/speech`. The `voice` field selects preset voice, voice clone, or voice design.

#### Preset Voice

Preset voice synthesis uses JSON and passes a built-in speaker name as `voice`.

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

Voice clone uses the same endpoint with `multipart/form-data` and uploads reference audio directly.

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

Voice design uses JSON, passes `design` as `voice`, and describes the generated voice through `voice_prompt`.

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

Supported speech output formats are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`. If `response_format` is omitted, the server uses `[defaults.tts] format` from `config.toml`, which defaults to `wav`. `speed` must remain `1.0` because speed control is not exposed yet.

Qwen3 TTS requests also accept `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `max_length`. If `seed` is omitted, Orchion uses `42` by default. Other sampling fields keep upstream defaults unless provided.

### Model List Request

```sh
curl http://127.0.0.1:9090/v1/models
```

The response follows the OpenAI model list shape: `object` is `list`, and `data` contains model objects with `id`, `object`, `created`, and `owned_by`. The list is built from `models.asr.available` and `models.tts.available` in `config.toml`.

If `[auth] api_key` is configured, pass it as `Authorization: Bearer <api_key>` for every `/v1/*` request.

## Rust Library

The public facade crate lives at `libs/orchion` and exposes async Rust APIs for loading, downloading, and running ASR/TTS models.

### Quick Start

#### Run Tests

```sh
cargo test --workspace --features full,cpu
```

#### Run Examples

```sh
cargo run -p orchion --features download-all --example download_model -- models
cargo run -p orchion --features asr-qwen3,download-all,cpu --example asr_file -- audio.wav models
cargo run -p orchion --features tts-qwen3,download-all,cpu --example tts_preset -- "Hello from Orchion" output.wav models
```

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

### Cargo Features

- `full`: Qwen3 ASR/TTS, FFmpeg audio conversion, and all download providers.
- `asr-qwen3`, `tts-qwen3`: Qwen3 ASR/TTS runtime adapters.
- `audio-ffmpeg`: audio decode/encode through system `ffmpeg`.
- `download-all`: async model downloads through `model-hub` with HuggingFace and ModelScope routing.
- `cpu`, `metal`, `cuda`: backend feature opt-ins.

## Configuration

```toml
[server]
bind = "127.0.0.1:9090"
max_upload_size = "30M"

[models]
dir = "models"
source = "auto"
max_loaded = 2

[models.asr]
default = "qwen3-asr-0.6b"
device = "auto"
available = ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]
idle_timeout = "10m"
max_loaded = 1

[models.tts]
default = "qwen3-tts-0.6b-custom-voice"
device = "auto"
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

`models.asr.available` and `models.tts.available` define the server allowlists. First startup can download all allowlisted model files into `models.dir`; trim `models.*.available` for local development if you do not need every example model. Models are loaded lazily when requested. Requests for models outside the allowlist are rejected. `idle_timeout` unloads inactive models.

Downloaded models use the `model-hub` native repository layout under `models.dir`, for example `models/Qwen/Qwen3-ASR-0.6B`. Orchion writes `.orchion-ready.json` after download and model preparation complete, then uses that manifest plus required local files to skip repeated downloads on later startup.

`models.max_loaded` limits the total resident ASR and TTS models together. `models.asr.max_loaded` and `models.tts.max_loaded` limit each category separately. When any limit is full, the least recently used resident model is evicted. Setting `models.max_loaded = 1` makes ASR and TTS switch residency globally; a request may wait while the evicted category is loaded again, but this does not limit concurrent inference requests.

`models.asr.device` and `models.tts.device` control runtime placement independently. Omitted fields or `auto` prefer CUDA, then Metal, then CPU. When multiple CUDA devices are visible, `auto` chooses the CUDA GPU with the most free memory at model load time. Explicit values include `cpu`, `metal`/`metal0`, `cuda`, `cuda0`, `cuda:0`, `cuda1`, and `cuda:1`.

`[auth] api_key` is optional. When it is set to a non-empty value, every `/v1/*` route requires `Authorization: Bearer <api_key>`; `/healthz` and `/docs` remain public.

### Model Sources

`ORCHION_MODEL_SOURCE` controls download routing for the core library:

- `auto` or unset: try HuggingFace first, then ModelScope.
- `huggingface`: use HuggingFace only.
- `modelscope`: use ModelScope only.

When `HF_ENDPOINT` is set, Orchion uses it for the HuggingFace probe and `model-hub` HuggingFace downloads. The server also accepts `models.source` in `config.toml` with the same values.

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

Real model download tests are ignored by default. Run them explicitly when network access and model storage are available.

## Project Status

Orchion is early-stage software. The public Rust API and server request extensions may change while the project is still stabilizing.
