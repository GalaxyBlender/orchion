# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion provides a unified Rust API library and an OpenAI-compatible server for local speech, document, and text-generation workflows. It supports Qwen3 ASR/TTS, PaddleOCR/OCR-VL, and a text-only llama.cpp runtime.

## Highlights

- OpenAI-style HTTP APIs for ASR, TTS, OCR/OCR-VL, and PDF page rendering.
- React WebUI at `/ui` for model-backed local workflows.
- Async Rust APIs and SDK examples.
- Model downloads through `model-hub` from HuggingFace or ModelScope.
- Swagger UI at `/docs` and OpenAPI JSON at `/openapi/v1.json`.

## Requirements

- Rust `1.95` or newer.
- Bun `1.3.14` available on `PATH` to build the WebUI.
- `ffmpeg` available on `PATH` for audio decode/encode.
- Enough local disk space for downloaded models.
- Optional Metal or CUDA runtime for acceleration.

## Run The Server

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml --models-dir data/models
```

The config is under `apps/orchion-server/`. These development commands override its packaged-model default to use `data/models` from the repository root. The server defaults to CPU unless a backend feature is enabled.

## WebUI

Open `/ui` on the running server. For frontend development:

```sh
cd web
bun run dev
```

API keys and form preferences are stored in browser `localStorage`; do not save keys on shared or untrusted browsers.

## API Routes

- `GET /healthz`: health check.
- `GET /readyz`: public readiness JSON; returns `503` after shutdown starts, for unhealthy resident workers, or after a required default deployment load failure.
- `GET /metrics`: OpenMetrics 1.0 metrics; uses the configured bearer API key when authentication is enabled.
- `GET /v1/models`: configured primary deployment IDs, optional display names, and derived capabilities.
- `GET /v1/models/{model}`: retrieve one configured public model without exposing residency.
- `GET /api/models/status`: configured model runtime residency.
- `POST /api/models/load`: load a configured model runtime.
- `POST /api/models/unload`: unload a configured model runtime.
- `POST /v1/audio/transcriptions`: ASR file transcription.
- `GET /v1/audio/transcriptions/stream`: ASR WebSocket streaming.
- `POST /v1/audio/speech`: TTS.
- `POST /v1/ocr`: OCR and OCR-VL.
- `POST /v1/chat/completions`: indexed JSON/SSE choices with function tools, rich tool/reasoning messages, strict JSON formats, logprobs, logit bias, deployment-bounded `n`, and opt-in `reasoning_control`.
- `POST /v1/chat/completions/control`: authenticated process-local `reasoning_end` control for an armed Chat stream.
- `POST /v1/completions`: legacy raw-prompt JSON/SSE with indexed `n`, logprobs, and logit bias.
- `POST /v1/responses`: stateless JSON or lifecycle SSE with function-call/output items, tools, strict text formats, reasoning items/events, and truthful logprobs where the semantic parser can attribute them; omitted or false `store` is accepted.
- `POST /v1/responses/input_tokens`: count tokens after Responses instructions/input prompt preparation.
- `POST /v1/embeddings`: OpenAI-compatible float or base64 embeddings for text and token inputs.
- `GET|DELETE /v1/stream`: resume/follow or idempotently cancel an owned resumable LLM stream.
- `POST /v1/streams/lookup`: look up explicitly named owned resumable streams.
- `POST /v1/pdf/images`: PDF page rendering.
- `GET /api/activity`: in-flight requests, retained history, and summary statistics.
- `GET /api/activity/events`: authenticated server-sent Activity events.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

Detailed API docs:

- [SDK contract matrix](docs/sdk-matrix.md)
- [Resumable LLM streaming contract](docs/llm-resumable-streaming.md)
- [ASR](docs/asr.md)
- [ASR streaming protocol](docs/asr-streaming.md)
- [TTS](docs/tts.md)
- [OCR and OCR-VL](docs/ocr.md)
- [PDF rendering](docs/pdf.md)

If `[auth] api_key` is configured, pass `Authorization: Bearer <api_key>` for every `/v1/*`, `/api/models/*`, and `/api/activity*` request.

OpenAI chat and Responses request parsing ignores unknown fields and recognized unsupported fields set to `null`. Recognized unsupported non-null values still return parameter-specific errors, and malformed values for known supported fields remain invalid.

Generation deployments advertise bridge-backed tools, parallel tools, JSON formats, logprobs, logit bias, and choice limits from configuration. `chat_template.enable_thinking` controls the omitted-request default but does not advertise reasoning. Set `chat_template.guarantees_reasoning = true` only when the exact deployed template is known to expose separate reasoning output and controllable start/end tags; otherwise reasoning remains conservatively absent from the catalog and requests can still receive a typed runtime rejection.

The Activity view records only allowlisted metadata such as route templates, model IDs, status, timing, and input size. While a request is in flight, Activity clients can also see its peer address and User-Agent; these live-only fields are removed before history is retained. Activity endpoints follow the global API key configuration, so deployments without an API key expose live metadata to clients that can reach the server. Activity never stores request bodies, response bodies, credentials, filenames, or generated output. HTTP timing covers the matched request through response-body completion, including parsing, queueing, and inference; WebSocket timing covers the upgraded session. History is bounded and exists only for the lifetime of the server process.

## In-Process Rust SDK

The `orchion` facade lives at `libs/orchion`. It provides typed loading, downloading, inference, and streaming interfaces for ASR, TTS, OCR/OCR-VL, and text or image LLMs. It is currently workspace-only rather than published to crates.io because not all native adapters are registry-publishable. Heavy domains and hardware backends are opt-in features; examples below select CPU explicitly.

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

`llm-complete` and `llm-streaming-cancel` require an exact GGUF URL. OCR provisioning may download multiple repositories and assembles typed assets before loading. The synchronous `LlmEngine::load` interface is blocking; `LlmEngine::load_deployment` offloads that load from the async runtime.

## Server Rust SDK

The `orchion-client` crate in `sdks/orchion-client` is a typed async client for every public data route listed above. Its default features enable health, models, Activity, LLM, ASR, TTS, OCR, and PDF; applications can disable defaults and select individual domains.

```rust,no_run
use orchion_client::{Client, ClientError};

#[tokio::main]
async fn main() -> Result<(), ClientError> {
    let client = Client::new("http://127.0.0.1:8080")?;
    client.health().check().await?;
    for model in client.models().list().await?.data {
        println!("{}: {:?}", model.id, model.capabilities);
    }
    Ok(())
}
```

Server SDK examples:

```sh
cargo run -p orchion-example-client-discovery -- http://127.0.0.1:8080
cargo run -p orchion-example-client-asr-file -- http://127.0.0.1:8080 audio.wav alibaba/qwen3-asr-0.6b
cargo run -p orchion-example-client-asr-streaming -- http://127.0.0.1:8080 audio.wav alibaba/qwen3-asr-0.6b
cargo run -p orchion-example-client-tts -- http://127.0.0.1:8080 MODEL "Hello" VOICE output.wav
cargo run -p orchion-example-client-ocr -- http://127.0.0.1:8080 image.png MODEL json
cargo run -p orchion-example-client-pdf -- http://127.0.0.1:8080 document.pdf pages.zip
cargo run -p orchion-example-client-llm -- http://127.0.0.1:8080 MODEL "Hello"
cargo run -p orchion-example-client-operations -- http://127.0.0.1:8080 MODEL llm
```

Set `ORCHION_API_KEY` when the server has authentication enabled. `start_streaming` waits for ASR `Ready` before returning. LLM streams require their protocol terminal event; opt-in resumable LLM methods expose the stream ID and last SSE event ID but never retry automatically. Activity streams end normally on server disconnect and do not reconnect automatically. Inference POST requests are never retried automatically.

## Configuration

`apps/orchion-server/config.toml` is the full local example. Key sections:

- `[server]`: bind address, CORS allowed origins, upload limit, and PDF page/pixel/output limits. CORS defaults to all origins (`["*"]`).
- `[activity]`: enable request activity and set the in-memory completed-history capacity (default `500`).
- `[models]`: model directory, source, global residency limit, and file integrity verification. `verify_file_integrity` defaults to `false`; set it to `true` to verify reused model files against the SHA-256 values recorded in their manifest.
- `[services.asr]`, `[services.tts]`, `[services.ocr]`, `[services.ocr-vl]`, `[services.llm]`: service deployments and residency. LLM deployments require exact main GGUF locators and optionally lock an exact mmproj artifact. A configured mmproj publishes static `llm_vision` capability for Chat `image_url` and Responses `input_image`; projector vision support and model compatibility are validated when the worker loads, and load errors are returned if validation fails. Images must be strict `data:image/png;base64,...` or `data:image/jpeg;base64,...` URLs; HTTP(S), files, paths, media parameters, and non-auto detail modes are rejected. Vision limits are configurable with bounded `vision` settings; defaults are 4 images, 10 MiB each, 20 MiB total, 8192 pixels per side, 16,777,216 pixels each, and 33,554,432 pixels total. Generation deployments may set positive `parallel_sequences` when `batch_size` is at least that value; embedding deployments require `parallel_sequences=1` and reject mmproj/vision settings. Multimodal requests require `n=1`, bypass prompt-prefix snapshots, and use exclusive projector prefill before returning to shared continuous decode. Concurrent slots on one loaded model share one global inference permit until the last slot finishes. Generation deployments can opt into worker-local prompt prefix snapshots with `prompt_cache = { enabled = true, max_entries = 4, max_bytes = 268435456, min_prefix_tokens = 32 }`; entries remain text-only, memory-only, and scoped to one loaded worker epoch.
- `[auth]`: optional API key.

`CORS_ALLOWED_ORIGINS` overrides `server.cors_allowed_origins` with a comma-separated origin list, for example `https://app.example.com,https://admin.example.com`; use `*` to allow all origins. `ORCHION_MODEL_SOURCE` and `models.source` accept `auto`, `huggingface`, or `modelscope`. `RUST_LOG` controls runtime logging.

## Development

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
cargo clippy -p orchion-client --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p orchion-client --all-features --no-deps
```

Orchion is early-stage software. The public Rust API and server request extensions may change while the project is still stabilizing.
