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
- `PATH` 中可用的 Bun `1.3.14`，用于构建 WebUI。
- `PATH` 中可用的 `ffmpeg`，用于音频解码/编码。
- 足够的本地磁盘空间保存模型文件。
- 如需加速，可准备 Metal 或 CUDA 运行环境。

## 运行服务

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml --models-dir data/models
```

配置文件位于 `apps/orchion-server/`。以上开发命令会覆盖发布包使用的默认模型目录，改用仓库根目录下的 `data/models`。未启用后端 feature 时，服务默认使用 CPU。

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
- `GET /api/models/status`：已配置模型的运行时驻留状态。
- `POST /api/models/load`：加载已配置模型的运行时。
- `POST /api/models/unload`：卸载已配置模型的运行时。
- `POST /v1/audio/transcriptions`：ASR 文件转录。
- `GET /v1/audio/transcriptions/stream`：ASR WebSocket 流式转录。
- `POST /v1/audio/speech`：TTS。
- `POST /v1/ocr`：OCR 和 OCR-VL。
- `POST /v1/pdf/images`：PDF 页面渲染。
- `GET /api/activity`：进行中的请求、保留历史和摘要统计。
- `GET /api/activity/events`：需要认证的 Activity 服务端事件流。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

详细 API 文档：

- [ASR](docs/asr.zh-CN.md)
- [ASR 流式协议](docs/asr-streaming.zh-CN.md)
- [TTS](docs/tts.zh-CN.md)
- [OCR 和 OCR-VL](docs/ocr.zh-CN.md)
- [PDF 页面渲染](docs/pdf.zh-CN.md)

如果配置了 `[auth] api_key`，所有 `/v1/*`、`/api/models/*` 和 `/api/activity*` 请求都需要传入 `Authorization: Bearer <api_key>`。

Activity 页面只记录路由模板、模型 ID、状态、耗时和输入大小等白名单元数据。请求进行期间，Activity 客户端还可以看到 peer 地址和 User-Agent；这些仅限实时展示的字段会在写入历史前清除。Activity 接口遵循全局 API key 配置，因此未配置 API key 的部署会向所有能访问服务的客户端开放实时元数据。Activity 不存储请求正文、响应正文、凭据、文件名或生成结果。HTTP 耗时从请求进入匹配路由起，覆盖解析、排队、推理，直到响应 body 完成；WebSocket 耗时覆盖升级后的完整会话。历史数量有上限，并且仅在当前服务进程生命周期内存在。

## Rust 库

公开 facade crate 位于 `libs/orchion`，提供用于加载、下载和运行 ASR/TTS/OCR 模型的异步 API。

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

常用示例：

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
```

## 配置

完整本地配置示例在 `apps/orchion-server/config.toml`。主要配置段：

- `[server]`：监听地址、CORS 允许来源、上传大小限制，以及 PDF 页数、像素和输出大小限制。CORS 默认允许所有来源（`["*"]`）。
- `[activity]`：启用请求活动并设置内存中的已完成历史容量（默认 `500`）。
- `[models]`：模型目录、下载来源、全局驻留上限和文件完整性校验。`verify_file_integrity` 默认是 `false`；设为 `true` 后，复用已下载模型时会按 manifest 中记录的 SHA-256 校验文件。
- `[services.asr]`、`[services.tts]`、`[services.ocr]`、`[services.ocr-vl]`：服务开关、默认模型、allowlist、运行设备和每类驻留上限。ASR 批量音频使用 `max_audio_duration`；流式字幕使用 `stream_target_segment` 和 `stream_max_segment`；会话使用 `stream_idle_timeout` 和 `stream_max_duration`。TTS 使用 `max_length` 和 `max_reference_audio_duration`；OCR-VL 使用 `max_tokens`。
- `[auth]`：可选 API key。

`CORS_ALLOWED_ORIGINS` 使用逗号分隔的来源列表覆盖 `server.cors_allowed_origins`，例如 `https://app.example.com,https://admin.example.com`；使用 `*` 允许所有来源。`ORCHION_MODEL_SOURCE` 和 `models.source` 支持 `auto`、`huggingface`、`modelscope`。`RUST_LOG` 控制运行日志。

## 开发

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
```

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。
