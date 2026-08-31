# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 提供统一的 Rust API 库和 OpenAI 兼容服务端，面向本地语音、文档与文本生成工作流。目前支持 Qwen3 ASR/TTS、PaddleOCR/OCR-VL 和 text-only llama.cpp runtime。

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
- `GET /v1/models`：已配置 primary deployment ID、可选展示名称和派生能力。
- `GET /api/models/status`：已配置模型的运行时驻留状态。
- `POST /api/models/load`：加载已配置模型的运行时。
- `POST /api/models/unload`：卸载已配置模型的运行时。
- `POST /v1/audio/transcriptions`：ASR 文件转录。
- `GET /v1/audio/transcriptions/stream`：ASR WebSocket 流式转录。
- `POST /v1/audio/speech`：TTS。
- `POST /v1/ocr`：OCR 和 OCR-VL。
- `POST /v1/chat/completions`：text-only、单 choice JSON 或 SSE。
- `POST /v1/responses`：无状态 text-only JSON 或生命周期 SSE，必须显式 `store=false`。
- `POST /v1/pdf/images`：PDF 页面渲染。
- `GET /api/activity`：进行中的请求、保留历史和摘要统计。
- `GET /api/activity/events`：需要认证的 Activity 服务端事件流。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

详细 API 文档：

- [SDK 契约矩阵](docs/sdk-matrix.md)
- [ASR](docs/asr.zh-CN.md)
- [ASR 流式协议](docs/asr-streaming.zh-CN.md)
- [TTS](docs/tts.zh-CN.md)
- [OCR 和 OCR-VL](docs/ocr.zh-CN.md)
- [PDF 页面渲染](docs/pdf.zh-CN.md)

如果配置了 `[auth] api_key`，所有 `/v1/*`、`/api/models/*` 和 `/api/activity*` 请求都需要传入 `Authorization: Bearer <api_key>`。

Activity 页面只记录路由模板、模型 ID、状态、耗时和输入大小等白名单元数据。请求进行期间，Activity 客户端还可以看到 peer 地址和 User-Agent；这些仅限实时展示的字段会在写入历史前清除。Activity 接口遵循全局 API key 配置，因此未配置 API key 的部署会向所有能访问服务的客户端开放实时元数据。Activity 不存储请求正文、响应正文、凭据、文件名或生成结果。HTTP 耗时从请求进入匹配路由起，覆盖解析、排队、推理，直到响应 body 完成；WebSocket 耗时覆盖升级后的完整会话。历史数量有上限，并且仅在当前服务进程生命周期内存在。

## 原生 Rust SDK

`orchion` facade 位于 `libs/orchion`，为 ASR、TTS、OCR/OCR-VL 和 text-only LLM 提供类型化的加载、下载、推理与流式接口。由于部分原生 adapter 尚不能从 registry 发布，该 facade 当前仅供 workspace 使用，尚未发布到 crates.io。重型领域和硬件后端均通过 feature 按需启用；以下示例显式选择 CPU。

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

原生 SDK 示例：

```sh
cargo run -p orchion-example-download-model --features cpu -- models
cargo run -p orchion-example-asr-file --features cpu -- audio.wav models
cargo run -p orchion-example-tts-preset --features cpu -- "Hello from Orchion" output.wav models
cargo run -p orchion-example-tts-voice-modes --features cpu -- preset "Hello" output.wav models
cargo run -p orchion-example-ocr-basic --features cpu -- image.png models
cargo run -p orchion-example-llm-complete --features cpu -- local/demo hf://org/repo/model.gguf "Hello" models
```

`llm-complete` 和 `llm-streaming-cancel` 要求传入精确 GGUF URL。OCR provisioning 可能下载多个仓库，并在加载前组装类型化 assets。同步 `LlmEngine::load` 接口会阻塞；`LlmEngine::load_deployment` 会把加载移出 async runtime worker。

## Server Rust SDK

`sdks/orchion-client` 中的 `orchion-client` crate 是上述全部公开数据路由的类型化异步客户端。默认 feature 启用 health、models、Activity、LLM、ASR、TTS、OCR 和 PDF；应用也可以关闭默认 feature 后只选择需要的领域。

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

Server SDK 示例：

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

服务启用认证时设置 `ORCHION_API_KEY`。`start_streaming` 会等待 ASR `Ready` 后再返回。LLM 流必须收到协议终态；Activity 流在服务断开时正常结束，且不会自动重连。推理 POST 请求不会自动重试。

## 配置

完整本地配置示例在 `apps/orchion-server/config.toml`。主要配置段：

- `[server]`：监听地址、CORS 允许来源、上传大小限制，以及 PDF 页数、像素和输出大小限制。CORS 默认允许所有来源（`["*"]`）。
- `[activity]`：启用请求活动并设置内存中的已完成历史容量（默认 `500`）。
- `[models]`：模型目录、下载来源、全局驻留上限和文件完整性校验。`verify_file_integrity` 默认是 `false`；设为 `true` 后，复用已下载模型时会按 manifest 中记录的 SHA-256 校验文件。
- `[services.asr]`、`[services.tts]`、`[services.ocr]`、`[services.ocr-vl]`、`[services.llm]`：服务 deployment 与驻留策略。LLM 主 GGUF 必须是精确文件 locator，可选 mmproj 也会精确锁定；当前仅支持文本和 `parallel_sequences=1`。
- `[auth]`：可选 API key。

`CORS_ALLOWED_ORIGINS` 使用逗号分隔的来源列表覆盖 `server.cors_allowed_origins`，例如 `https://app.example.com,https://admin.example.com`；使用 `*` 允许所有来源。`ORCHION_MODEL_SOURCE` 和 `models.source` 支持 `auto`、`huggingface`、`modelscope`。`RUST_LOG` 控制运行日志。

## 开发

```sh
cargo fmt --all -- --check
cargo test --workspace --features full,cpu
cargo check --workspace
cargo clippy -p orchion-client --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p orchion-client --all-features --no-deps
```

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。
