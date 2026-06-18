# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 是一个简单易用的异步 Rust 语音 AI 库，适用于 ASR、TTS 等工作流。它提供类型安全的 Rust API、独立示例二进制程序，以及带 Swagger 文档的 OpenAI 兼容 HTTP 服务。

Orchion 目前支持 Qwen3 ASR/TTS 模型，并预留了后续扩展更多语音模型后端的空间。它关注本地推理的实用工作流：模型名称使用 Rust 枚举表达，模型下载通过 HuggingFace 和 ModelScope 客户端完成，同步上游推理被封装在异步 API 后面，服务端默认使用 CPU，Metal 或 CUDA 后端作为可选特性。

## 亮点

- 简单易用的异步 Rust API，覆盖 ASR/TTS 等语音工作流。
- OpenAI 兼容的 `/v1/audio/transcriptions` 和 `/v1/audio/speech` API。
- 面向 TTS 音色克隆和音色设计的最小 OpenAI 风格扩展。
- 基于 `config.toml` 的服务端模型选择和默认值配置。
- 支持从 HuggingFace 或 ModelScope 下载模型，并支持自动回退。
- 服务端默认使用 CPU，也可选用 macOS Metal 或 Linux/Windows CUDA。
- Swagger UI 位于 `/docs`，OpenAPI JSON 位于 `/openapi/v1.json`。

## 环境要求

- Rust `1.85` 或更高版本。
- `PATH` 中可用的 `ffmpeg`，用于内存中的 ASR 上传解码和 TTS 响应编码。
- 通过上游 `qwen3-asr` 和 `qwen3-tts` crate 支持的 Qwen3 ASR/TTS 后端。
- 可选 GPU 环境：
  - macOS：支持 Metal 的设备。
  - Linux 或 Windows：支持 CUDA 的设备和兼容的 CUDA runtime。

## 快速开始

### 运行测试

```sh
cargo test --workspace --features full,cpu
```

### 运行示例

```sh
cargo run -p orchion --features download-all --example download_model -- models
cargo run -p orchion --features asr-qwen3,download-all,cpu --example asr_file -- audio.wav models
cargo run -p orchion --features tts-qwen3,download-all,cpu --example tts_preset -- "Hello from Orchion" output.wav models
```

## 核心库

公开 facade crate 位于 `libs/orchion`，提供用于加载、下载和运行 ASR/TTS 模型的异步 Rust API。领域类型位于 `libs/orchion-core`，FFmpeg 音频转换位于 `libs/orchion-audio`，模型下载位于 `libs/orchion-download`，Qwen3 运行时适配位于 `libs/orchion-qwen3`。

### Cargo Features

- `default = []`
- `full`：Qwen3 ASR/TTS、FFmpeg 音频转换和全部下载 provider。
- `asr-qwen3`、`tts-qwen3`：Qwen3 ASR/TTS 运行时适配。
- `audio-ffmpeg`：通过系统 `ffmpeg` 做内存音频解码/编码。
- `download-all`：通过 `hf-hub` 和 `modelscope` 异步下载模型。
- `cpu`、`metal`、`cuda`：传递给上游 crate 的后端特性开关。

### ASR 示例

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

流式 ASR 接收单声道 `f32` samples 和源采样率。Orchion 会自动将音频块重采样到 16 kHz 后再传给 Qwen3 ASR。

### TTS 示例

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

## OpenAI 兼容服务

服务端 crate 位于 `apps/orchion-server`。它使用 Axum 并提供 OpenAI 风格的音频接口。

### 运行服务

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml
```

`orchion-server` 默认启用 `cpu` feature。macOS 可追加 `--features metal`，Linux 或 Windows 且 CUDA 环境满足要求时可追加 `--features cuda`；GPU 构建仍会包含 CPU 后端。

仓库内置了 `apps/orchion-server/config.toml` 作为开发配置。如果省略 `--config`，服务会读取可执行文件旁边的 `config.toml`。如果省略 `models.dir`，模型会存储到可执行文件旁边的 `models/`。

日志通过 `RUST_LOG` 控制。服务会先读取可执行文件目录下的 `.env`，再读取当前工作目录下的 `.env`。仓库内置了开发用 `.env`，因此 `cargo run -p orchion-server -- --config apps/orchion-server/config.toml` 默认会输出启动、模型加载、下载和请求 debug 日志。

### 路由

- `GET /healthz`：健康检查。
- `GET /v1/models`：OpenAI 风格的可用模型列表。
- `POST /v1/audio/transcriptions`：OpenAI 风格 multipart ASR 请求。
- `POST /v1/audio/speech`：OpenAI 风格 JSON TTS 请求。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

### 转录请求

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.mp3 \
  -F response_format=json
```

上传音频会通过系统 `ffmpeg` 在内存中解码；只要当前 ffmpeg 构建支持，常见的 `wav`、`mp3`、`m4a`、`flac`、`ogg`、`webm` 等格式都可以使用。支持的 `response_format` 值为 `json`、`text` 和 `verbose_json`。当前 ASR 封装不暴露词级或片段级时间戳，因此 timestamp granularities 会被明确拒绝。

### 语音合成请求

语音合成都使用 `POST /v1/audio/speech`。根据 `voice` 字段分为三类：预设音色、音色克隆和音色设计。

#### 预设音色

预设音色使用 JSON 请求，`voice` 传入内置说话人名称，例如 `ryan`。

字段：

- `model`：必须是 `models.tts.available` 中的 TTS 模型，例如 `qwen3-tts-0.6b-custom-voice`。
- `input`：需要合成的文本。
- `voice`：内置说话人名称，例如 `ryan`。
- `language`：可选，合成语言，例如 `english`、`zh`。
- `response_format`：可选，支持 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。
- `seed`、`temperature`、`top_k`、`top_p`、`repetition_penalty`、`max_length`：可选的 Qwen3 TTS 采样控制参数。

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

#### 音色克隆

音色克隆使用同一个 `POST /v1/audio/speech` 地址，但请求体必须是 `multipart/form-data`，通过文件字段直接上传参考音频；JSON 请求不支持音色克隆。

字段：

- `model`：必须是 `models.tts.available` 中的音色克隆模型，例如 `qwen3-tts-0.6b-custom-voice`。
- `input`：需要合成的文本。
- `voice`：固定传 `clone`。
- `reference_audio`：参考音频文件字段，例如 `-F reference_audio=@reference.wav`。
- `reference_text`：参考音频中说出的文本。
- `language`：可选，参考音频和合成文本语言，例如 `english`、`zh`。
- `response_format`：可选，支持 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。

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

#### 音色设计

音色设计使用 JSON 请求，`voice` 固定传 `design`，并通过 `voice_prompt` 描述要生成的音色。

字段：

- `model`：必须是 `models.tts.available` 中支持音色设计的模型，例如 `qwen3-tts-1.7b-voice-design`。
- `input`：需要合成的文本。
- `voice`：固定传 `design`。
- `voice_prompt`：音色描述文本。
- `language`：可选，合成语言，例如 `english`、`zh`。
- `response_format`：可选，支持 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。

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

语音合成输出支持 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。如果请求未传 `response_format`，服务会使用 `config.toml` 中 `[defaults.tts] format` 的值，默认是 `wav`。`speed` 必须保持为 `1.0`，因为上游封装暂未暴露语速控制。

Qwen3 TTS 请求还支持采样控制参数：`seed`、`temperature`、`top_k`、`top_p`、`repetition_penalty` 和 `max_length`。如果未传 `seed`，Orchion 默认使用 `42`，让 TTS 输出默认可复现。其他采样字段未传时保持上游 `qwen3-tts` 默认值。`max_length` 是生成 codec frames 的上限，可在 EOS 延迟时调低以限制长音频生成。

### 模型列表请求

```sh
curl http://127.0.0.1:9090/v1/models
```

响应保持 OpenAI model list 形状：`object` 为 `list`，`data` 中每个模型对象包含 `id`、`object`、`created` 和 `owned_by`。列表来自 `config.toml` 中的 `models.asr.available` 和 `models.tts.available`。

如果配置了 `[auth] api_key`，所有 `/v1/*` 请求都需要传入 `Authorization: Bearer <api_key>`。

## 配置

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

`models.asr.available` 和 `models.tts.available` 是服务端允许使用的模型列表。启动时会把这些模型文件下载到 `models.dir`，但不会全部加载进内存；请求指定某个模型时才懒加载。请求不在 allowlist 中的模型会立即被拒绝。`idle_timeout` 会卸载空闲模型，`max_loaded` 会在缓存满时按最近最少使用策略卸载已加载模型。

`[auth] api_key` 是可选配置。设置为非空值后，所有 `/v1/*` 路由都要求 `Authorization: Bearer <api_key>`；`/healthz` 和 `/docs` 保持公开。

### 模型来源

`ORCHION_MODEL_SOURCE` 控制核心库的下载来源：

- `auto` 或未设置：先尝试 HuggingFace，再回退到 ModelScope。
- `huggingface`：仅使用 HuggingFace。
- `modelscope`：仅使用 ModelScope。

设置 `HF_ENDPOINT` 时，Orchion 会把它传给 HuggingFace 客户端。

服务端也支持在 `config.toml` 中用 `models.source` 配置相同的值。

`server.max_upload_size` 用于限制上传请求体大小，默认是 `30M`，支持纯字节数以及 `K`、`M`、`G` 后缀。

### 日志

```dotenv
RUST_LOG=orchion_server=debug,orchion=info,tower_http=debug
```

环境变量中的 `RUST_LOG` 优先于 `.env`。如果两者都未设置，服务默认使用 `orchion_server=info,orchion=info,tower_http=debug`。

## 开发

常用命令：

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
```

真实模型下载测试不会进入默认测试路径。如需运行，请在有网络和模型存储空间时显式运行 ignored integration tests。

## 项目状态

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。
