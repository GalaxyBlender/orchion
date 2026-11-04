# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion runs speech, document, and language models locally through a unified Rust library, an OpenAI-compatible server, and a browser interface. It supports Qwen3 ASR/TTS, PaddleOCR/OCR-VL, and llama.cpp models.

## Highlights

- Speech recognition, streaming transcription, and speech synthesis.
- OCR, document layout recognition, and PDF page rendering.
- LLM chat, text generation, tool calling, and embeddings.
- WebUI for inference, model management, and request activity.
- Async Rust library and HTTP client SDK.
- Automatic model downloads from Hugging Face or ModelScope.
- Interactive API documentation at `/docs`.

## Requirements

- Rust `1.95` or newer when building from source.
- Bun `1.3.14` available on `PATH` when building the WebUI from source.
- `ffmpeg` available on `PATH` for audio decode/encode.
- Enough local disk space for downloaded models.
- Optional Metal or CUDA runtime for acceleration.

## Run The Server

From a packaged server directory:

```sh
./orchion-server --config config.toml
```

On Windows, use `.\orchion-server.exe --config config.toml`.

To run from source:

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml --models-dir data/models
```

The source commands use the [example configuration](apps/orchion-server/config.toml) and store models in `data/models`. Choose the command for your CPU, Metal, or CUDA backend.

Before accepting requests, the server downloads all models and dependencies configured for enabled services, reusing valid local caches. A download failure prevents startup. Models are loaded into memory on demand; `max_loaded` limits memory residency, not downloads. Disable unneeded services or remove unused models from the configuration to reduce first-startup downloads and disk usage.

## WebUI

Open `/ui` on the running server to transcribe audio, generate speech, recognize documents, chat with models, and manage model loading.

The WebUI can save API keys and form preferences in your browser. Do not save keys on shared or untrusted devices.

## API Routes

- `GET /healthz`: health check.
- `GET /readyz`: readiness check.
- `GET /metrics`: monitoring metrics.
- `GET /v1/models`: list configured models and their capabilities.
- `GET /v1/models/{model}`: retrieve model information.
- `GET /api/models/status`: model loading status.
- `POST /api/models/load`: load a model.
- `POST /api/models/unload`: unload a model.
- `POST /v1/audio/transcriptions`: ASR file transcription.
- `GET /v1/audio/transcriptions/stream`: ASR WebSocket streaming.
- `POST /v1/audio/speech`: TTS.
- `POST /v1/ocr`: OCR and OCR-VL.
- `POST /v1/chat/completions`: chat completions, with streaming support.
- `POST /v1/chat/completions/control`: control reasoning in a chat stream.
- `POST /v1/completions`: text completions.
- `POST /v1/responses`: stateless Responses API, with streaming support.
- `POST /v1/responses/input_tokens`: count Responses input tokens.
- `POST /v1/embeddings`: text and token embeddings.
- `GET|DELETE /v1/stream`: resume or cancel an LLM stream.
- `POST /v1/streams/lookup`: look up resumable streams.
- `POST /v1/pdf/images`: PDF page rendering.
- `GET /api/activity`: request activity, history, and statistics.
- `GET /api/activity/events`: live request activity updates.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

Detailed API docs:

- [Rust SDK and HTTP API coverage](docs/sdk-matrix.md)
- [Resumable LLM streaming](docs/llm-resumable-streaming.md)
- [ASR](docs/asr.md)
- [ASR streaming protocol](docs/asr-streaming.md)
- [TTS](docs/tts.md)
- [OCR and OCR-VL](docs/ocr.md)
- [PDF rendering](docs/pdf.md)

If `[auth] api_key` is configured, pass `Authorization: Bearer <api_key>` for `/v1/*`, `/api/models/*`, `/api/activity*`, and `/metrics` requests.

Request activity shows model IDs, status, timing, and input sizes without storing request bodies, credentials, filenames, or generated output. Live activity also shows client addresses and User-Agent values. History is held in memory and cleared when the server exits. Configure an API key before exposing the server to an untrusted network.

## In-Process Rust SDK

The [`orchion`](libs/orchion) crate embeds model downloading, loading, inference, and streaming in Rust applications. Select the model families and hardware backends you need through Cargo features; the examples below use CPU.

```rust,no_run
use orchion::{Asr, AsrModel, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let model = AsrModel::parse("alibaba/qwen3-asr-0.6b")?;
    let asr = Asr::load_or_download(model, "models").await?;
    let transcript = asr.transcribe_file("audio.wav").await?;
    println!("{}", transcript.text);
    Ok(())
}
```

Native SDK examples:

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
cargo run -p orchion-example-tts-voice-modes --features cpu -- preset "Hello" output.wav models
cargo run -p orchion-example-ocr-basic --features cpu -- image.png models
cargo run -p orchion-example-llm-complete --features cpu -- local/demo hf://org/repo/model.gguf "Hello" models
```

More examples are available under [`examples/orchion`](examples/orchion).

## Server Rust SDK

The [`orchion-client`](sdks/orchion-client) crate is an async Rust client for the server's speech, document, LLM, and management APIs.

```rust,no_run
use orchion_client::{Client, ClientError};

#[tokio::main]
async fn main() -> Result<(), ClientError> {
    let client = Client::new("http://127.0.0.1:9090")?;
    client.health().check().await?;
    for model in client.models().list().await?.data {
        println!("{}: {:?}", model.id, model.capabilities);
    }
    Ok(())
}
```

Server SDK examples:

```sh
cargo run -p orchion-example-client-discovery -- http://127.0.0.1:9090
cargo run -p orchion-example-client-asr-file -- http://127.0.0.1:9090 audio.wav alibaba/qwen3-asr-0.6b
cargo run -p orchion-example-client-tts -- http://127.0.0.1:9090 MODEL "Hello" VOICE output.wav
cargo run -p orchion-example-client-ocr -- http://127.0.0.1:9090 image.png MODEL json
cargo run -p orchion-example-client-llm -- http://127.0.0.1:9090 MODEL "Hello"
```

Set `ORCHION_API_KEY` when the server has authentication enabled. More examples are available under [`examples/orchion-client`](examples/orchion-client).

## Configuration

See [`apps/orchion-server/config.toml`](apps/orchion-server/config.toml) for a complete configuration example.

- `[server]`: listen address, CORS, upload limits, and PDF limits.
- `[models]`: model directory, download source, memory residency limit, and optional file integrity checks.
- `[services.asr]`, `[services.tts]`, `[services.ocr]`, `[services.ocr-vl]`, `[services.llm]`: enable services, select models, and set inference options.
- `[activity]`: request activity and history capacity.
- `[auth]`: API key authentication.

`models.source` accepts `auto`, `huggingface`, or `modelscope`. Valid cached models are reused without network access. `verify_file_integrity = true` enables local SHA-256 checks of cached files; it is disabled by default. LLM models require exact GGUF file URLs, with an optional compatible `mmproj` file for image input.

LLM image inputs must use base64 PNG or JPEG data URLs, rather than remote image URLs.

Environment overrides:

- `ORCHION_MODEL_SOURCE`: model download source.
- `CORS_ALLOWED_ORIGINS`: comma-separated allowed origins, or `*` to allow all origins. The default allows all origins.
- `RUST_LOG`: logging level and filters.
