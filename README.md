# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion provides a unified Rust API library and an OpenAI-compatible server for local speech and document AI workflows. It currently focuses on Qwen3 ASR/TTS and PaddleOCR/OCR-VL, with CPU by default and optional Metal or CUDA builds.

## Highlights

- OpenAI-style HTTP APIs for ASR, TTS, OCR/OCR-VL, and PDF page rendering.
- React WebUI at `/ui` for model-backed local workflows.
- Async Rust APIs and SDK examples.
- Model downloads through `model-hub` from HuggingFace or ModelScope.
- Swagger UI at `/docs` and OpenAPI JSON at `/openapi/v1.json`.

## Requirements

- Rust `1.95` or newer.
- `ffmpeg` available on `PATH` for audio decode/encode.
- Enough local disk space for downloaded models.
- Optional Metal or CUDA runtime for acceleration.

## Run The Server

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.development.toml
```

The development config is under `apps/orchion-server/`. The server defaults to CPU unless a backend feature is enabled.

## WebUI

Open `/ui` on the running server. For frontend development:

```sh
cd web
bun run dev
```

API keys and form preferences are stored in browser `localStorage`; do not save keys on shared or untrusted browsers.

## API Routes

- `GET /healthz`: health check.
- `GET /v1/models`: configured model list.
- `POST /v1/audio/transcriptions`: ASR file transcription.
- `GET /v1/audio/transcriptions/stream`: ASR WebSocket streaming.
- `POST /v1/audio/speech`: TTS.
- `POST /v1/ocr`: OCR and OCR-VL.
- `POST /v1/pdf/images`: PDF page rendering.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

Detailed API docs:

- [ASR](docs/asr.md)
- [ASR streaming protocol](docs/asr-streaming.md)
- [TTS](docs/tts.md)
- [OCR and OCR-VL](docs/ocr.md)
- [PDF rendering](docs/pdf.md)

If `[auth] api_key` is configured, pass `Authorization: Bearer <api_key>` for every `/v1/*` request.

## Rust Library

The facade crate lives at `libs/orchion` and exposes async APIs for loading, downloading, and running ASR/TTS/OCR models.

```rust,no_run
use orchion::{Asr, AsrModel, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let model = AsrModel::parse("Qwen/Qwen3-ASR-0.6B")?;
    let asr = Asr::load_or_download(model, "models").await?;
    let transcript = asr.transcribe_file("audio.wav").await?;
    println!("{}", transcript.text);
    Ok(())
}
```

Useful examples:

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
```

## Configuration

`apps/orchion-server/config.toml` is the full local example. Key sections:

- `[server]`: bind address, upload limit, and PDF page/pixel/output limits.
- `[models]`: model directory, source, and global residency limit.
- `[services.asr]`, `[services.tts]`, `[services.ocr]`, `[services.ocr-vl]`: service enablement, defaults, allowlists, device, and per-service residency. ASR batch audio uses `max_audio_duration`; streaming captions use `stream_target_segment` and `stream_max_segment`; sessions use `stream_idle_timeout` and `stream_max_duration`. TTS uses `max_length` and `max_reference_audio_duration`; OCR-VL uses `max_tokens`.
- `[auth]`: optional API key.

`ORCHION_MODEL_SOURCE` and `models.source` accept `auto`, `huggingface`, or `modelscope`. `RUST_LOG` controls runtime logging.

## Development

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
```

Orchion is early-stage software. The public Rust API and server request extensions may change while the project is still stabilizing.
