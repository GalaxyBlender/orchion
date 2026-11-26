# ASR API

[简体中文](asr.zh-CN.md)

Orchion exposes OpenAI-style speech recognition through file upload and WebSocket streaming.

## File Transcription

```sh
curl http://127.0.0.1:9090/v1/audio/transcriptions \
  -F model=alibaba/qwen3-asr-0.6b \
  -F file=@audio.mp3 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=segment"
```

Supported audio formats depend on the installed `ffmpeg` build. Common formats include `wav`, `mp3`, `m4a`, `flac`, `ogg`, and `webm`.

Supported `response_format` values:

- `json`
- `text`
- `verbose_json`
- `srt`

`timestamp_granularities[]=segment` enables segment timestamps in `verbose_json`. Word-level timestamps are not supported.

### CUDA memory

On CUDA, file transcription runs the audio encoder's convolution in bounded
batches without splitting the input into separate model requests. Temporary GPU
allocations can be reclaimed before the request returns; the model stays
loaded. This does not change WebSocket streaming. Long recordings can still hit
GPU memory limits because attention and decoding retain context for the input
being transcribed.

## Streaming

Use `GET /v1/audio/transcriptions/stream` for WebSocket streaming.

Streaming supports two output modes:

- Live transcript: omit `mode` and consume continuous `partial` plus terminal `final` events.
- Captions: send `mode: "caption"` and consume `partial`, `segment_final`, and `completed` events.

See [asr-streaming.md](asr-streaming.md) for the full streaming protocol.
