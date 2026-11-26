# ASR API

[English](asr.md)

Orchion 通过文件上传和 WebSocket 流式接口提供 OpenAI 风格的语音识别能力。

## 文件转录

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=alibaba/qwen3-asr-0.6b \
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

### CUDA 显存

CUDA 文件转写会把音频编码器的卷积拆成小批次执行，不需要把输入拆成多个推理请求。返回前会同步 CUDA，让临时显存可被回收，模型保持加载。此优化不改变 WebSocket 流式识别。长录音仍可能达到显存上限，因为注意力和解码仍需要当前转写输入的上下文。

## 流式识别

WebSocket 流式识别使用 `GET /v1/audio/transcriptions/stream`。

流式识别支持两种输出模式：

- Live transcript：不传 `mode`，消费连续 `partial` 和结束时的 `final` 事件。
- Captions：发送 `mode: "caption"`，消费 `partial`、`segment_final` 和 `completed` 事件。

完整流式协议见 [asr-streaming.zh-CN.md](asr-streaming.zh-CN.md)。
