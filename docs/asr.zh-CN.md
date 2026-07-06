# ASR API

[English](asr.md)

Orchion 通过文件上传和 WebSocket 流式接口提供 OpenAI 风格的语音识别能力。

## 文件转录

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-0.6B \
  -F file=@audio.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment"
```

支持的音频格式取决于本机安装的 `ffmpeg` 构建。常见格式包括 `wav`、`mp3`、`m4a`、`flac`、`ogg` 和 `webm`。

支持的 `response_format`：

- `json`
- `text`
- `verbose_json`
- `srt`

`timestamp_granularities[]=segment` 会在 `verbose_json` 中启用片段时间戳；暂不支持词级时间戳。

## 流式识别

WebSocket 流式识别使用 `GET /v1/audio/transcriptions/stream`。

流式识别支持两种输出模式：

- Live transcript：不传 `mode`，消费连续 `partial` 和结束时的 `final` 事件。
- Captions：发送 `mode: "caption"`，消费 `partial`、`segment_final` 和 `completed` 事件。

完整流式协议见 [asr-streaming.zh-CN.md](asr-streaming.zh-CN.md)。
