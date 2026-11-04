# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 通过统一的 Rust 库、OpenAI 兼容服务端和浏览器界面，在本地运行语音、文档与语言模型，支持 Qwen3 ASR/TTS、PaddleOCR/OCR-VL 和 llama.cpp 模型。

## 亮点

- 语音识别、流式转录与语音合成。
- OCR、文档版面识别与 PDF 页面渲染。
- LLM 对话、文本生成、工具调用与向量嵌入。
- WebUI 支持模型推理、模型管理与请求活动查看。
- 异步 Rust 库和 HTTP 客户端 SDK。
- 自动从 Hugging Face 或 ModelScope 下载模型。
- `/docs` 提供交互式 API 文档。

## 环境要求

- 从源码构建时，需要 Rust `1.95` 或更高版本。
- 从源码构建 WebUI 时，需要 `PATH` 中可用的 Bun `1.3.14`。
- `PATH` 中可用的 `ffmpeg`，用于音频解码/编码。
- 足够的本地磁盘空间保存模型文件。
- 如需加速，可准备 Metal 或 CUDA 运行环境。

## 运行服务

在发布包目录中运行：

```sh
./orchion-server --config config.toml
```

Windows 下使用 `.\orchion-server.exe --config config.toml`。

从源码运行：

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml --models-dir data/models
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml --models-dir data/models
```

以上源码命令使用[配置示例](apps/orchion-server/config.toml)，并将模型保存到 `data/models`。可根据硬件选择 CPU、Metal 或 CUDA 对应的命令。

服务会在接收请求前下载所有已启用服务配置的模型及依赖，并复用有效本地缓存；下载失败会阻止启动。模型按需加载到内存，`max_loaded` 限制内存驻留数量，不限制下载数量。可禁用不需要的服务或移除不用的模型，以减少首次启动的下载量和磁盘占用。

## WebUI

服务运行后打开 `/ui`，即可转录音频、生成语音、识别文档、与模型对话，以及管理模型加载。

WebUI 可以在浏览器中保存 API key 和表单偏好。不要在共享或不可信设备中保存 key。

## API 路由

- `GET /healthz`：健康检查。
- `GET /readyz`：就绪状态检查。
- `GET /metrics`：监控指标。
- `GET /v1/models`：列出已配置模型及其能力。
- `GET /v1/models/{model}`：查询模型信息。
- `GET /api/models/status`：模型加载状态。
- `POST /api/models/load`：加载模型。
- `POST /api/models/unload`：卸载模型。
- `POST /v1/audio/transcriptions`：ASR 文件转录。
- `GET /v1/audio/transcriptions/stream`：ASR WebSocket 流式转录。
- `POST /v1/audio/speech`：TTS。
- `POST /v1/ocr`：OCR 和 OCR-VL。
- `POST /v1/chat/completions`：对话补全，支持流式输出。
- `POST /v1/chat/completions/control`：控制对话流的思考过程。
- `POST /v1/completions`：文本补全。
- `POST /v1/responses`：无状态 Responses API，支持流式输出。
- `POST /v1/responses/input_tokens`：统计 Responses 输入 token 数。
- `POST /v1/embeddings`：文本和 token 向量嵌入。
- `GET|DELETE /v1/stream`：恢复或取消 LLM 流。
- `POST /v1/streams/lookup`：查询可恢复的流。
- `POST /v1/pdf/images`：PDF 页面渲染。
- `GET /api/activity`：请求活动、历史与统计。
- `GET /api/activity/events`：实时请求活动更新。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

详细 API 文档：

- [Rust SDK 与 HTTP API 功能对照](docs/sdk-matrix.md)
- [可恢复 LLM 流](docs/llm-resumable-streaming.md)
- [ASR](docs/asr.zh-CN.md)
- [ASR 流式协议](docs/asr-streaming.zh-CN.md)
- [TTS](docs/tts.zh-CN.md)
- [OCR 和 OCR-VL](docs/ocr.zh-CN.md)
- [PDF 页面渲染](docs/pdf.zh-CN.md)

如果配置了 `[auth] api_key`，`/v1/*`、`/api/models/*`、`/api/activity*` 和 `/metrics` 请求需要传入 `Authorization: Bearer <api_key>`。

请求活动展示模型 ID、状态、耗时和输入大小，不存储请求正文、凭据、文件名或生成结果；实时活动还会展示客户端地址和 User-Agent。历史保存在内存中，服务退出后清除。向不可信网络开放服务前，请配置 API key。

## 原生 Rust SDK

[`orchion`](libs/orchion) 库可将模型下载、加载、推理与流式输出集成到 Rust 应用中。通过 Cargo feature 选择需要的模型类型和硬件后端；以下示例使用 CPU。

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

更多示例位于 [`examples/orchion`](examples/orchion)。

## 服务端 Rust 客户端

[`orchion-client`](sdks/orchion-client) 库提供异步 Rust 客户端，用于访问服务端的语音、文档、LLM 和管理接口。

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

客户端示例：

```sh
cargo run -p orchion-example-client-discovery -- http://127.0.0.1:9090
cargo run -p orchion-example-client-asr-file -- http://127.0.0.1:9090 audio.wav alibaba/qwen3-asr-0.6b
cargo run -p orchion-example-client-tts -- http://127.0.0.1:9090 MODEL "Hello" VOICE output.wav
cargo run -p orchion-example-client-ocr -- http://127.0.0.1:9090 image.png MODEL json
cargo run -p orchion-example-client-llm -- http://127.0.0.1:9090 MODEL "Hello"
```

服务启用认证时设置 `ORCHION_API_KEY`。更多示例位于 [`examples/orchion-client`](examples/orchion-client)。

## 配置

完整配置示例见 [`apps/orchion-server/config.toml`](apps/orchion-server/config.toml)。

- `[server]`：监听地址、CORS、上传限制和 PDF 限制。
- `[models]`：模型目录、下载来源、内存驻留上限和可选文件完整性校验。
- `[services.asr]`、`[services.tts]`、`[services.ocr]`、`[services.ocr-vl]`、`[services.llm]`：启用服务、选择模型和设置推理选项。
- `[activity]`：请求活动与历史容量。
- `[auth]`：API key 认证。

`models.source` 支持 `auto`、`huggingface` 和 `modelscope`。有效模型缓存无需联网即可复用。`verify_file_integrity = true` 可启用缓存文件的本地 SHA-256 校验，默认关闭。LLM 模型需要精确的 GGUF 文件 URL；如需图像输入，可额外配置兼容的 `mmproj` 文件。

LLM 图像输入必须使用 PNG 或 JPEG 的 base64 data URL，不支持远程图片 URL。

环境变量覆盖：

- `ORCHION_MODEL_SOURCE`：模型下载来源。
- `CORS_ALLOWED_ORIGINS`：逗号分隔的允许来源，或用 `*` 允许所有来源；默认允许所有来源。
- `RUST_LOG`：日志级别与过滤规则。
