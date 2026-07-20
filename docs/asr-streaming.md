# ASR Streaming WebSocket Protocol

[简体中文](asr-streaming.zh-CN.md)

Endpoint: `/v1/audio/transcriptions/stream`

The first WebSocket message must be a JSON start message. Binary messages after `ready` carry audio bytes in the declared `input_audio_format`. A JSON control message `{"type":"end"}` tells the server to flush the current stream.

When API key authentication is enabled, non-browser clients may use the `Authorization: Bearer <api_key>` handshake header. Browser clients may instead include `"api_key":"<api_key>"` in the start message. Authentication completes before the server loads a model.

The server closes streams that exceed `[services.asr].stream_idle_timeout` or `[services.asr].stream_max_duration`. Total binary input is limited by `[server].max_upload_size`, decoded audio cannot exceed `stream_max_duration`, and `chunk_size_sec` cannot exceed 30 seconds.

## Live Transcript Mode

Live transcript mode is selected when `mode` is absent.

Start message:

```json
{
  "type": "start",
  "model": "Qwen/Qwen3-ASR-Flash",
  "input_audio_format": "pcm_s16le",
  "sample_rate": 16000
}
```

Events:

- `ready`: server accepted the start message.
- `partial`: interim transcript for the continuous stream. Shape: `{"type":"partial","text":"..."}`.
- `final`: final transcript after client sends `end`. Shape: `{"type":"final","text":"..."}`.
- `error`: structured error object.

## Caption Mode

Caption mode is selected with `"mode":"caption"`. One WebSocket is one subtitle session. The client continuously sends audio; the server owns endpointing and emits stable subtitle segments.

Start message using server endpointing defaults:

For `pcm_s16le` in caption mode, `sample_rate` is required and must be `16000`.

```json
{
  "type": "start",
  "mode": "caption",
  "model": "Qwen/Qwen3-ASR-Flash",
  "input_audio_format": "pcm_s16le",
  "sample_rate": 16000
}
```

Start message with explicit endpointing:

Endpointing values are milliseconds. `min_speech_ms + speech_padding_ms` must be `60000` or less, and the candidate window must cover the rounded 30 ms VAD speech frame window. Caption segmentation also uses server configuration duration strings: `[services.asr].stream_target_segment = "12s"` is the punctuation-aware soft target, and `[services.asr].stream_max_segment = "120s"` is the hard limit.

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

Caption events:

- `ready`: server accepted the start message.
- `partial`: interim text for the current subtitle fragment. Shape: `{"type":"partial","segment_id":0,"text":"..."}`.
- `segment_final`: stable subtitle fragment. Shape: `{"type":"segment_final","segment_id":0,"text":"...","start_ms":0,"end_ms":1840}`. `start_ms` and `end_ms` are optional timing fields.
- `completed`: whole WebSocket session is finished after client sends `end`. Shape: `{"type":"completed"}`.
- `error`: structured error object.

`final` is only used by live transcript mode. In caption mode, sentence or subtitle completion is represented by `segment_final`; whole-stream completion is represented by `completed`.
