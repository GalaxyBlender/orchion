# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 是一个简单易用的异步 Rust 语音 AI 库，适用于 ASR、TTS 等工作流。它提供类型安全的 Rust API、独立示例二进制程序，以及带 Swagger 文档的 OpenAI 兼容 HTTP 服务。

Orchion 目前支持 Qwen3 ASR/TTS 模型，并预留了后续扩展更多语音模型后端的空间。它关注本地推理的实用工作流：模型名称使用 Rust 枚举表达，模型下载通过 HuggingFace 和 ModelScope 客户端完成，同步上游推理被封装在异步 API 后面，服务端在目标平台支持时默认启用平台 GPU 加速。

## 亮点

- 简单易用的异步 Rust API，覆盖 ASR/TTS 等语音工作流。
- OpenAI 兼容的 `/v1/audio/transcriptions` 和 `/v1/audio/speech` API。
- 面向 TTS 音色克隆和音色设计的最小 OpenAI 风格扩展。
- 基于 `config.toml` 的服务端模型选择和默认值配置。
- 支持从 HuggingFace 或 ModelScope 下载模型，并支持自动回退。
- 服务端 crate 默认启用平台 GPU 特性：macOS 使用 Metal，Linux 和 Windows 使用 CUDA。
- Swagger UI 位于 `/docs`，OpenAPI JSON 位于 `/api-docs/openapi.json`。

## Workspace 结构

```text
.
├── libs/
│   └── orchion/         # Rust 核心库
├── apps/
│   └── server/          # Axum OpenAI 兼容 ASR/TTS 服务
├── examples/
│   ├── asr_file/        # 独立 ASR 文件示例
│   ├── asr_streaming/   # 独立 ASR 流式示例
│   ├── download_model/  # 独立模型下载示例
│   └── tts_preset/      # 独立预设音色 TTS 示例
```

## 环境要求

- Rust `1.85` 或更高版本。
- 通过上游 `qwen3-asr` 和 `qwen3-tts` crate 支持的 Qwen3 ASR/TTS 后端。
- 可选 GPU 环境：
  - macOS：支持 Metal 的设备。
  - Linux 或 Windows：支持 CUDA 的设备和兼容的 CUDA runtime。

## 快速开始

### 运行测试

```sh
cargo test -p orchion --lib
cargo test -p orchion-server --lib --tests
```

### 运行示例

```sh
cargo run -p orchion-example-download-model -- models
cargo run -p orchion-example-asr-file -- audio.wav models
cargo run -p orchion-example-tts-preset -- "Hello from Orchion" output.wav models
```

### 运行服务

```sh
cargo run -p orchion-server -- --config config.toml
```

如果省略 `--config`，服务会读取可执行文件旁边的 `config.toml`。如果省略 `models.dir`，模型会存储到可执行文件旁边的 `models/`。

## 核心库

核心 crate 位于 `libs/orchion`，提供用于加载、下载和运行 ASR/TTS 模型的异步 Rust API。

### Cargo Features

- `default = ["asr", "tts", "download"]`
- `asr`：Qwen3 ASR 转录和流式封装。
- `tts`：Qwen3 TTS 预设音色、音色克隆和音色设计封装。
- `download`：通过 `hf-hub` 和 `modelscope` 异步下载模型。
- `metal`、`cuda`、`flash-attn`：传递给上游 crate 的后端特性开关。

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

服务端 crate 位于 `apps/server`。它使用 Axum 并提供 OpenAI 风格的音频接口。

### 路由

- `GET /healthz`：健康检查。
- `POST /v1/audio/transcriptions`：OpenAI 风格 multipart ASR 请求。
- `POST /v1/audio/speech`：OpenAI 风格 JSON TTS 请求。
- `GET /docs`：Swagger UI。
- `GET /api-docs/openapi.json`：OpenAPI 文档。

### 转录请求

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.wav \
  -F response_format=json
```

支持的 `response_format` 值为 `json`、`text` 和 `verbose_json`。当前 ASR 封装不暴露词级或片段级时间戳，因此 timestamp granularities 会被明确拒绝。

### 语音合成请求

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

当前仅支持 `wav` 输出。`speed` 必须保持为 `1.0`，因为上游封装暂未暴露语速控制。

### 音色克隆扩展

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

### 音色设计扩展

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

这些扩展字段让请求结构尽量接近 OpenAI speech API，同时暴露 Qwen3 TTS 中 OpenAI 未直接定义的能力。

## 配置

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

### 模型来源

`ORCHION_MODEL_SOURCE` 控制核心库的下载来源：

- `auto` 或未设置：先尝试 HuggingFace，再回退到 ModelScope。
- `huggingface`：仅使用 HuggingFace。
- `modelscope`：仅使用 ModelScope。

设置 `HF_ENDPOINT` 时，Orchion 会把它传给 HuggingFace 客户端。

服务端也支持在 `config.toml` 中用 `models.source` 配置相同的值。

## 开发

常用命令：

```sh
cargo fmt --all -- --check
cargo test -p orchion --lib
cargo test -p orchion-server --lib --tests
cargo check --workspace --exclude orchion-server
cargo check -p orchion-server
```

真实模型下载测试不会进入默认测试路径。如需运行，请在有网络和模型存储空间时显式运行 ignored integration tests。

## 项目状态

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。

## 许可证

仓库暂未包含 license 文件。作为开源项目分发前，请先添加 `LICENSE` 文件。
