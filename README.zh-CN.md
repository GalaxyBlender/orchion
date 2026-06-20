# Orchion

[English](README.md) | [简体中文](README.zh-CN.md)

Orchion 提供统一的 Rust API 库和开箱即用的 OpenAI 兼容 API Server，面向本地语音 AI 工作流。它支持通过类型安全的 Rust API、独立示例和易于接入 OpenAI 风格客户端的 HTTP 接口使用 ASR 和 TTS。

Orchion 目前聚焦 Qwen3 ASR/TTS 模型，同时保留了后续扩展更多语音模型后端的空间。

## 亮点

- 统一的异步 Rust API，覆盖 ASR 和 TTS 工作流。
- 开箱即用的 OpenAI 兼容 API Server。
- `/v1/audio/transcriptions` 和 `/v1/audio/speech` 接口。
- TTS 支持预设音色、音色克隆和音色设计。
- 支持从 HuggingFace 或 ModelScope 下载模型。
- 默认使用 CPU，可选 Metal 或 CUDA 构建。
- Swagger UI 位于 `/docs`，OpenAPI JSON 位于 `/openapi/v1.json`。

## 环境要求

- Rust `1.85` 或更高版本。
- `PATH` 中可用的 `ffmpeg`，用于音频解码/编码。
- 足够的本地磁盘空间用于存放模型文件。
- 如需加速，可准备 Metal 或 CUDA GPU 运行环境。

## OpenAI 兼容服务

服务端 crate 位于 `apps/orchion-server`，提供 OpenAI 风格的音频接口。

### 运行服务

```sh
cargo run -p orchion-server -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features metal -- --config apps/orchion-server/config.toml
cargo run -p orchion-server --features cuda -- --config apps/orchion-server/config.toml
```

`orchion-server` 默认使用 CPU。macOS 可使用 `--features metal`，Linux/Windows 且 CUDA 环境可用时可使用 `--features cuda`。仓库内置了 `apps/orchion-server/config.toml` 作为开发配置。

### WebUI

在服务器的 `/ui` 打开 React WebUI，可用于 ASR/TTS 操作、参数预览、模型检查和本地设置。Debug 构建会服务 `web/dist`；如果目录缺失，请在 `web/` 下运行 `bun install` 和 `bun run build`。前端迭代可在 `web/` 下运行 `bun run dev`。Release 构建会从 `apps/orchion-server/build.rs` 运行 Bun、构建 SPA，并通过 `OUT_DIR/ui-dist` 将资源嵌入服务端二进制。API key 和表单偏好会存储在浏览器 `localStorage`。警告：API key 会通过 `localStorage` 存储在浏览器配置档案中；不要在共享或不可信浏览器中使用或保存 API key。

### 路由

- `GET /healthz`：健康检查。
- `GET /ui`：用于 ASR/TTS 操作、参数预览、模型检查和本地设置的 React WebUI。
- `GET /v1/models`：OpenAI 风格的可用模型列表。
- `POST /v1/audio/transcriptions`：OpenAI 风格 multipart ASR 请求。
- `POST /v1/audio/speech`：OpenAI 风格 TTS 请求。
- `GET /docs`：Swagger UI。
- `GET /openapi/v1.json`：OpenAPI 文档。

### 转录请求

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b \
  -F file=@audio.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment"
```

上传音频会通过系统 `ffmpeg` 解码；只要当前 ffmpeg 构建支持，常见的 `wav`、`mp3`、`m4a`、`flac`、`ogg`、`webm` 等格式都可以使用。支持的 `response_format` 值为 `json`、`text`、`verbose_json` 和 `srt`。`timestamp_granularities[]=segment` 会在 `verbose_json` 中启用片段时间戳；`response_format=srt` 会以 `text/plain` 返回字幕 cues。暂不支持词级时间戳。

### 语音合成请求

语音合成都使用 `POST /v1/audio/speech`。根据 `voice` 字段分为预设音色、音色克隆和音色设计。

#### 预设音色

预设音色使用 JSON 请求，`voice` 传入内置说话人名称。

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

音色克隆使用同一个接口，请求体为 `multipart/form-data`，并直接上传参考音频。

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

语音合成输出支持 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。如果请求未传 `response_format`，服务会使用 `config.toml` 中 `[defaults.tts] format` 的值，默认是 `wav`。`speed` 必须保持为 `1.0`，因为暂未暴露语速控制。

Qwen3 TTS 请求还支持 `seed`、`temperature`、`top_k`、`top_p`、`repetition_penalty` 和 `max_length`。如果未传 `seed`，Orchion 默认使用 `42`。其他采样字段未传时保持上游默认值。

### 模型列表请求

```sh
curl http://127.0.0.1:9090/v1/models
```

响应保持 OpenAI model list 形状：`object` 为 `list`，`data` 中每个模型对象包含 `id`、`object`、`created` 和 `owned_by`。列表来自 `config.toml` 中的 `models.asr.available` 和 `models.tts.available`。

如果配置了 `[auth] api_key`，所有 `/v1/*` 请求都需要传入 `Authorization: Bearer <api_key>`。

## Rust 库

公开 facade crate 位于 `libs/orchion`，提供用于加载、下载和运行 ASR/TTS 模型的异步 Rust API。

### 快速开始

#### 运行测试

```sh
cargo test --workspace --features full,cpu
```

#### 运行示例

```sh
cargo run -p orchion --features download-all --example download_model -- models
cargo run -p orchion --features asr-qwen3,download-all,cpu --example asr_file -- audio.wav models
cargo run -p orchion --features tts-qwen3,download-all,cpu --example tts_preset -- "Hello from Orchion" output.wav models
```

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

### Cargo Features

- `full`：Qwen3 ASR/TTS、FFmpeg 音频转换和全部下载来源。
- `asr-qwen3`、`tts-qwen3`：Qwen3 ASR/TTS 运行时适配。
- `audio-ffmpeg`：通过系统 `ffmpeg` 解码和编码音频。
- `download-all`：通过 HuggingFace 和 ModelScope 异步下载模型。
- `cpu`、`metal`、`cuda`：后端特性开关。

## 配置

```toml
[server]
bind = "127.0.0.1:9090"
max_upload_size = "30M"

[models]
dir = "models"
source = "auto"
max_loaded = 2

[models.asr]
default = "qwen3-asr-0.6b"
device = "auto"
available = ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]
idle_timeout = "10m"
max_loaded = 1

[models.tts]
default = "qwen3-tts-0.6b-custom-voice"
device = "auto"
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

`models.asr.available` 和 `models.tts.available` 是服务端允许使用的模型列表。首次启动可能会把 allowlist 中的全部模型文件下载到 `models.dir`；本地开发时如果不需要示例里的所有模型，可以精简 `models.*.available`。模型会在请求指定时懒加载。请求不在 allowlist 中的模型会被拒绝。`idle_timeout` 会卸载空闲模型。

`models.max_loaded` 限制 ASR 和 TTS 加起来的总驻留模型数。`models.asr.max_loaded` 和 `models.tts.max_loaded` 分别限制单个类别的驻留模型数。任一限制达到上限时，会按最近最少使用策略卸载已驻留模型。设置 `models.max_loaded = 1` 后，ASR/TTS 会在全局范围内切换驻留；如果对应类别已被卸载，请求会等待模型重新加载，但这不是并发推理请求数限制。

`models.asr.device` 和 `models.tts.device` 分别控制 ASR/TTS 的运行设备。省略该字段或设置为 `auto` 时，会优先选择 CUDA，其次 Metal，最后 CPU；如果可见多张 CUDA 显卡，`auto` 会在模型加载时选择当前剩余显存最多的 CUDA 设备。显式值支持 `cpu`、`metal`/`metal0`、`cuda`、`cuda0`、`cuda:0`、`cuda1` 和 `cuda:1`。

`[auth] api_key` 是可选配置。设置为非空值后，所有 `/v1/*` 路由都要求 `Authorization: Bearer <api_key>`；`/healthz` 和 `/docs` 保持公开。

### 模型来源

`ORCHION_MODEL_SOURCE` 控制核心库的下载来源：

- `auto` 或未设置：先尝试 HuggingFace，再回退到 ModelScope。
- `huggingface`：仅使用 HuggingFace。
- `modelscope`：仅使用 ModelScope。

设置 `HF_ENDPOINT` 时，Orchion 会把它传给 HuggingFace 客户端。服务端也支持在 `config.toml` 中用 `models.source` 配置相同的值。

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

真实模型下载测试默认会被忽略。如需运行，请在具备网络和模型存储空间时显式执行。

## 项目状态

Orchion 仍处于早期阶段。项目稳定前，公开 Rust API 和服务端请求扩展都可能调整。
