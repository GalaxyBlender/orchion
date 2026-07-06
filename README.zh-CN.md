# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 提供统一的 Rust API 库和 OpenAI 兼容服务端，面向本地语音与文档 AI 工作流。目前重点支持 Qwen3 ASR/TTS 和 PaddleOCR/OCR-VL，默认 CPU 运行，可选 Metal 或 CUDA 构建。

## 亮点

- OpenAI 风格 HTTP API，覆盖 ASR、TTS、OCR/OCR-VL 和 PDF 页面渲染。
- `/ui` 提供 React WebUI。
- 异步 Rust API 和 SDK 示例。
- 通过 `model-hub` 从 HuggingFace 或 ModelScope 下载模型。
- Swagger UI 位于 `/docs`，OpenAPI JSON 位于 `/openapi/v1.json`。

## 环境要求

- Rust `1.95` 或更高版本。
- `PATH` 中可用的 `ffmpeg`，用于音频解码/编码。
- 足够的本地磁盘空间保存模型文件。
- 如需加速，可准备 Metal 或 CUDA 运行环境。

## 运行服务

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.development.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.development.toml
```

开发配置位于 `apps/orchion-server/`。未启用后端 feature 时，服务默认使用 CPU。

## WebUI

服务运行后打开 `/ui`。前端开发可运行：

```sh
cd web
bun run dev
```

API key 和表单偏好会存储在浏览器 `localStorage`；不要在共享或不可信浏览器中保存 key。

## API 路由

- `GET /healthz`：健康检查。
- `GET /v1/models`：已配置模型列表。
- `POST /v1/audio/transcriptions`：ASR 文件转录。
- `GET /v1/audio/transcriptions/stream`：ASR WebSocket 流式转录。
- `POST /v1/audio/speech`：TTS。
- `POST /v1/ocr`：OCR 和 OCR-VL。
- `POST /v1/pdf/images`：PDF 页面渲染。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

详细 API 文档：

- [ASR](docs/asr.zh-CN.md)
- [ASR 流式协议](docs/asr-streaming.zh-CN.md)
- [TTS](docs/tts.zh-CN.md)
- [OCR 和 OCR-VL](docs/ocr.zh-CN.md)
- [PDF 页面渲染](docs/pdf.zh-CN.md)

如果配置了 `[auth] api_key`，所有 `/v1/*` 请求都需要传入 `Authorization: Bearer <api_key>`。

## Rust 库

公开 facade crate 位于 `libs/orchion`，提供用于加载、下载和运行 ASR/TTS/OCR 模型的异步 API。

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

常用示例：

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
```

## 配置

完整本地配置示例在 `apps/orchion-server/config.toml`。主要配置段：

- `[server]`：监听地址和上传大小限制。
- `[models]`：模型目录、下载来源和全局驻留上限。
- `[services.asr]`、`[services.tts]`、`[services.ocr]`、`[services.ocr-vl]`：服务开关、默认模型、allowlist、运行设备和每类驻留上限。ASR 流式字幕使用 `[services.asr].stream_target_segment = "12s"` 做标点感知软切分，使用 `[services.asr].stream_max_segment = "120s"` 做硬上限。
- `[auth]`：可选 API key。

`ORCHION_MODEL_SOURCE` 和 `models.source` 支持 `auto`、`huggingface`、`modelscope`。`RUST_LOG` 控制运行日志。

## 开发

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
```

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。
