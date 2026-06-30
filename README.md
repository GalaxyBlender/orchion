# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion provides a unified Rust API library and an out-of-the-box OpenAI-compatible API server for local speech and document AI workflows. It supports ASR, TTS, and OCR/OCR-VL through typed Rust APIs, standalone examples, and HTTP endpoints that are easy to integrate with existing OpenAI-style clients.

Orchion currently focuses on Qwen3 ASR/TTS plus OCR/OCR-VL document recognition models, and is structured to support more local AI backends over time.

## Highlights

- Unified async Rust APIs for ASR, TTS, and OCR workflows.
- Ready-to-run OpenAI-compatible API server.
- `/v1/audio/transcriptions`, `/v1/audio/speech`, and `/v1/ocr` endpoints.
- TTS support for preset voices, voice cloning, and voice design.
- OCR support for PP-OCRv5, PP-OCRv6, PP-DocLayoutV3, and PaddleOCR-VL 1.5/1.6 model IDs.
- Model downloads through `model-hub` from HuggingFace or ModelScope.
- CPU by default, with optional Metal and CUDA builds.
- Swagger UI at `/docs` and OpenAPI JSON at `/openapi/v1.json`.

## Requirements

- Rust `1.95` or newer.
- `ffmpeg` available on `PATH` for audio decode/encode.
- Enough local disk space for downloaded models.
- Optional GPU runtime for Metal or CUDA acceleration.

## OpenAI-Compatible Server

The server crate lives at `apps/orchion-server` and exposes OpenAI-style audio and OCR routes.

### Run The Server

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.development.toml
```

`orchion-server` defaults to CPU. Use `--features metal` on macOS, or `--features cuda` on Linux/Windows with a supported CUDA stack. The repository includes `apps/orchion-server/config.toml` as a development config.

### WebUI

Open the React WebUI at `/ui` on the server for ASR, TTS, OCR/OCR-VL operations, parameter previews, model inspection, and local settings. For frontend iteration, run `bun run dev` from `web/`. API key and form preferences are stored in browser `localStorage`. Warning: API keys are stored in the browser profile via `localStorage`; do not use or save them on shared or untrusted browsers. The WebUI always calls the current server address and no longer supports a manually entered API base URL.

### Routes

- `GET /healthz`: health check.
- `GET /v1/models`: OpenAI-style list of configured models.
- `GET /ui`: React WebUI for ASR, TTS, OCR/OCR-VL operations, parameter previews, model inspection, and local settings.
- `POST /v1/audio/transcriptions`: OpenAI-style multipart ASR request.
- `POST /v1/audio/speech`: OpenAI-style TTS request.
- `POST /v1/ocr`: OpenAI-style multipart OCR/OCR-VL request.
- `GET /docs`: Swagger UI.
- `GET /openapi/v1.json`: OpenAPI document.

### Transcription Request

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-0.6B \
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

- `model`: a TTS model from `services.tts.available_models`, such as `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
- `input`: text to synthesize.
- `voice`: built-in speaker name, such as `ryan`.
- `language`: optional synthesis language, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.
- `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_length`: optional Qwen3 TTS sampling controls.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
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

- `model`: a voice-clone-capable model from `services.tts.available_models`, such as `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- `input`: text to synthesize.
- `voice`: must be `clone`.
- `reference_audio`: reference audio file field, such as `-F reference_audio=@reference.wav`.
- `reference_text`: text spoken in the reference audio.
- `language`: optional language for the reference audio and synthesized text, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -F model=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
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

- `model`: a voice-design-capable model from `services.tts.available_models`, such as `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- `input`: text to synthesize.
- `voice`: must be `design`.
- `voice_prompt`: text description of the voice.
- `language`: optional synthesis language, such as `english` or `zh`.
- `response_format`: optional output format; supported values are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`.

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "input": "Read this with a designed voice.",
    "voice": "design",
    "voice_prompt": "A calm narrator with a warm studio tone.",
    "language": "english",
    "response_format": "wav"
  }' \
  --output designed.wav
```

Supported speech output formats are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`. If `response_format` is omitted, the server uses `[services.tts] format` from `config.toml`, which defaults to `wav`. `speed` must remain `1.0` because speed control is not exposed yet.

Qwen3 TTS requests also accept `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `max_length`. If `seed` is omitted, Orchion uses `42` by default. Other sampling fields keep upstream defaults unless provided.

### OCR Request

OCR uses `POST /v1/ocr` with `multipart/form-data`. Traditional OCR models return structured text regions and plain text; OCR-VL models can additionally return Markdown when the model supports it.

Fields:

- `file`: image or document image file, such as `-F file=@document.png`.
- `model`: optional model ID in `{vendor}/{name}` format, such as `PaddlePaddle/PP-OCRv6_tiny` or `PaddlePaddle/PaddleOCR-VL-1.6`.
- `response_format`: optional output format; supported values are `json`, `text`, and `markdown`.
- `task`: optional OCR-VL task, such as `ocr`, `table`, `formula`, `chart`, `spotting`, or `seal`.
- `layout_model`: optional OCR-VL layout model; default comes from `[services.ocr-vl] layout_model`.
- `max_tokens`: optional OCR-VL generation limit.

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PP-OCRv6_tiny \
  -F response_format=json
```

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PaddleOCR-VL-1.6 \
  -F response_format=markdown
```

### Model List Request

```sh
curl http://127.0.0.1:9090/v1/models
```

The response follows the OpenAI model list shape: `object` is `list`, and `data` contains model objects with `id`, `object`, `created`, and `owned_by`. The list is built from active `services.asr.available_models`, `services.tts.available_models`, `services.ocr.available_models`, and `services.ocr-vl.available_models` in `config.toml`. Disabled or unconfigured services are omitted from `/v1/models`.

If `[auth] api_key` is configured, pass it as `Authorization: Bearer <api_key>` for every `/v1/*` request.

## Rust Library

The public facade crate lives at `libs/orchion` and exposes async Rust APIs for loading, downloading, and running ASR/TTS/OCR models.

### Quick Start

#### Run Tests

```sh
cargo test --workspace --features full,cpu
```

#### Run Examples

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
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
- `asr`, `tts`, `ocr`, `ocr-vl`: public API gates for ASR, TTS, traditional OCR, and OCR-VL.
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

[services.asr]
enabled = true
default_model = "Qwen/Qwen3-ASR-0.6B"
device = "auto"
available_models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
idle_timeout = "10m"
max_loaded = 1

[services.tts]
enabled = true
default_model = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
device = "auto"
available_models = [
  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]
idle_timeout = "10m"
max_loaded = 1
format = "wav"

[services.ocr]
enabled = false
available_models = []
device = "auto"
idle_timeout = "10m"
max_loaded = 1
format = "json"

[services.ocr-vl]
enabled = false
available_models = []
device = "auto"
idle_timeout = "10m"
max_loaded = 1
format = "markdown"

[auth]
# api_key = "change-me"
```

Enable OCR services explicitly and keep model IDs as `{vendor}/{name}` strings:

```toml
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"
available_models = [
  "PaddlePaddle/PP-OCRv6_tiny",
  "PaddlePaddle/PP-OCRv6_small",
  "PaddlePaddle/PP-OCRv6_medium",
  "PaddlePaddle/PP-OCRv5_mobile",
  "PaddlePaddle/PP-OCRv5_server",
  "PaddlePaddle/PP-DocLayoutV3",
]
device = "auto"
idle_timeout = "10m"
max_loaded = 1
format = "json"

[services.ocr-vl]
enabled = true
default_model = "PaddlePaddle/PaddleOCR-VL-1.6"
available_models = [
  "PaddlePaddle/PaddleOCR-VL-1.5",
  "PaddlePaddle/PaddleOCR-VL-1.6",
]
layout_model = "PaddlePaddle/PP-DocLayoutV3"
device = "auto"
idle_timeout = "10m"
max_loaded = 1
format = "markdown"
```

`services.asr.available_models`, `services.tts.available_models`, `services.ocr.available_models`, and `services.ocr-vl.available_models` define the server allowlists. First startup can download all allowlisted model files into `models.dir`; trim `services.*.available_models` for local development if you do not need every example model. Models are loaded lazily when requested. Requests for models outside the allowlist are rejected. `idle_timeout` unloads inactive models.

`services.asr.enabled`, `services.tts.enabled`, `services.ocr.enabled`, and `services.ocr-vl.enabled` control model downloads and route exposure. OCR services are disabled by default and only become active when `enabled = true` and `available_models` is non-empty. Disabled services are omitted from `/v1/models`; `/v1/ocr` is registered only when OCR or OCR-VL is active.

Downloaded models use the `model-hub` native repository layout under `models.dir`, for example `models/Qwen/Qwen3-ASR-0.6B`. Orchion writes `.orchion-ready.json` after download and model preparation complete, then uses that manifest plus required local files to skip repeated downloads on later startup.

`models.max_loaded` limits the total resident ASR, TTS, OCR, and OCR-VL models together. `services.*.max_loaded` limits each category separately. When any limit is full, the least recently used resident model is evicted. Setting `models.max_loaded = 1` makes services switch residency globally; a request may wait while the evicted category is loaded again, but this does not limit concurrent inference requests.

`services.*.device` controls runtime placement independently. Omitted fields or `auto` prefer CUDA, then Metal, then CPU. When multiple CUDA devices are visible, `auto` chooses the CUDA GPU with the most free memory at model load time. Explicit values include `cpu`, `metal`/`metal0`, `cuda`, `cuda0`, `cuda:0`, `cuda1`, and `cuda:1`.

Device mapping differs by OCR runtime: Traditional ONNX OCR uses `cuda` = ORT CUDA, `metal` = ORT CoreML on Apple platforms, and `cpu` = ORT CPU. OCR-VL uses `cuda` = Candle CUDA, `metal` = Candle Metal, and `cpu` = Candle CPU.

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
