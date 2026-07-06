# ASR 流式 WebSocket 协议

[English](asr-streaming.md)

端点：`/v1/audio/transcriptions/stream`

第一条 WebSocket 消息必须是 JSON start 消息。收到 `ready` 后，二进制消息承载 `input_audio_format` 声明格式的音频字节。JSON 控制消息 `{"type":"end"}` 用于通知服务端 flush 当前流。

## Live Transcript 模式

不传 `mode` 时使用 Live Transcript 模式。

Start 消息：

```json
{
  "type": "start",
  "model": "Qwen/Qwen3-ASR-Flash",
  "input_audio_format": "pcm_s16le",
  "sample_rate": 16000
}
```

事件：

- `ready`：服务端已接受 start 消息。
- `partial`：连续流的中间转录结果，形状为 `{"type":"partial","text":"..."}`。
- `final`：客户端发送 `end` 后的最终转录结果，形状为 `{"type":"final","text":"..."}`。
- `error`：结构化错误对象。

## Captions 模式

发送 `"mode":"caption"` 时使用 Captions 模式。一个 WebSocket 是一个字幕会话。客户端连续发送音频；服务端负责端点检测并输出稳定字幕片段。

使用服务端 endpointing 默认值的 Start 消息：

Captions 模式下使用 `pcm_s16le` 时，必须传 `sample_rate`，且值必须为 `16000`。

```json
{
  "type": "start",
  "mode": "caption",
  "model": "Qwen/Qwen3-ASR-Flash",
  "input_audio_format": "pcm_s16le",
  "sample_rate": 16000
}
```

显式 endpointing 的 Start 消息：

Endpointing 值单位为毫秒。`min_speech_ms + speech_padding_ms` 必须小于等于 `60000`，且候选窗口必须能覆盖按 30 ms VAD 帧向上取整后的最小语音窗口。字幕切分还会使用服务端 duration 字符串配置：`[services.asr].stream_target_segment = "12s"` 是标点感知软目标；`[services.asr].stream_max_segment = "120s"` 是硬上限。

```json
{
  "type": "start",
  "mode": "caption",
  "model": "Qwen/Qwen3-ASR-Flash",
  "input_audio_format": "pcm_s16le",
  "sample_rate": 16000,
  "endpointing": {
    "min_speech_ms": 300,
    "min_silence_ms": 500,
    "speech_padding_ms": 200
  }
}
```

Captions 事件：

- `ready`：服务端已接受 start 消息。
- `partial`：当前字幕片段的中间文本，形状为 `{"type":"partial","segment_id":0,"text":"..."}`。
- `segment_final`：稳定字幕片段，形状为 `{"type":"segment_final","segment_id":0,"text":"...","start_ms":0,"end_ms":1840}`。`start_ms` 和 `end_ms` 是可选时间字段。
- `completed`：客户端发送 `end` 后，整个 WebSocket 会话已结束。形状为 `{"type":"completed"}`。
- `error`：结构化错误对象。

`final` 只用于 Live Transcript 模式。Captions 模式中，句子或字幕片段完成用 `segment_final` 表示，整个流结束用 `completed` 表示。
